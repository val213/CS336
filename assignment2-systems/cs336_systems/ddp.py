import torch
import torch.distributed as dist
import torch.nn as nn


class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module

        # 1. broadcast rank 0 的参数给所有 rank（保证起点一致）
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. 按反向顺序把参数分桶
        # 反向顺序：让 backward 最先算出梯度的参数最先填桶，尽早触发 all-reduce
        params = [p for p in self.module.parameters() if p.requires_grad]
        params = list(reversed(params))

        self.buckets: list[list[nn.Parameter]] = []   # 每个桶包含哪些参数
        self.param_to_bucket: dict[nn.Parameter, int] = {}  # 每个参数属于哪个桶

        # 遍历 params，按 bucket_size_mb 分桶，填充 self.buckets 和 self.param_to_bucket
        current_bucket = []
        current_size = 0
        for param in params:
            param_size = param.data.nbytes / 1024 / 1024  # bytes → MB

            # 桶非空且加上这个参数会超限，就封桶、开新桶
            if current_bucket and current_size + param_size > bucket_size_mb:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0

            current_bucket.append(param)
            current_size += param_size

        # 最后一个桶别忘了加进去
        if current_bucket:
            self.buckets.append(current_bucket)
        
        for bucket_idx, bucket in enumerate(self.buckets):
            for param in bucket:
                self.param_to_bucket[param] = bucket_idx

        # 3. 每个桶已就绪的参数计数（hook 里用来判断桶是否满了）
        self.bucket_ready_count: list[int] = [0] * len(self.buckets)

        # 4. 每个桶的 all-reduce handle，以及 flat grad tensor（用于写回）
        # 格式: [(handle, flat_grad, bucket_idx), ...]
        self.bucket_handles: list[tuple] = []

        # 5. 为每个参数注册 gradient hook
        self.hooks = []
        for param in params:
            handle = param.register_post_accumulate_grad_hook(self.hook)
            self.hooks.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def hook(self, param: nn.Parameter):
        # 找到这个参数所在的桶
        bucket_idx = self.param_to_bucket[param]
        self.bucket_ready_count[bucket_idx] += 1

        # 检查桶是否满了（ready_count == 桶里参数数量）
        if self.bucket_ready_count[bucket_idx] == len(self.buckets[bucket_idx]):
            # 满了就把桶里所有梯度 flatten + cat 成一个大 tensor，发起异步 all-reduce
            flat_grad = torch.cat([p.grad.view(-1) for p in self.buckets[bucket_idx]])
            handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
            # 把 (handle, flat_grad, bucket_idx) 存进 self.bucket_handles
            self.bucket_handles.append((handle, flat_grad, bucket_idx))

    def finish_gradient_synchronization(self):
        # 等待所有桶的 all-reduce 完成，把结果写回各参数的 .grad
        world_size = dist.get_world_size()

        for handle, flat_grad, bucket_idx in self.bucket_handles:
            handle.wait()
            # flat_grad 除以 world_size，然后把数据切回各参数的 .grad
            flat_grad /= world_size
            offset = 0
            for param in self.buckets[bucket_idx]:
                param_size = param.grad.numel()
                param.grad.data.copy_(flat_grad[offset:offset+param_size].view_as(param.grad))
                offset += param_size

        self.bucket_handles.clear()

    def reset_buckets(self):
        # 每个 step 开始时重置计数器，准备下一轮
        self.bucket_ready_count = [0] * len(self.buckets)
        self.bucket_handles.clear()
