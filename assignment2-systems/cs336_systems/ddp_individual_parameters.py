import torch.distributed as dist
import torch.nn as nn
import torch

class DDPIndividualParameters(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []
        self.hooks = []
        # 1. broadcast rank 0 的参数给所有 rank
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # 2. 为每个参数注册 gradient hook
        for param in self.module.parameters():
            if param.requires_grad:
                handle = param.register_post_accumulate_grad_hook(self.hook)
                # 3. 存一个列表记录异步 all-reduce 的 handle
                self.hooks.append(handle)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        # 等待所有 handle 完成，然后除以 world_size
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        # all-reduce 是求和，还需要除以 world_size 变成平均
        world_size = dist.get_world_size()
        for param in self.module.parameters():
            if param.grad is not None:
                param.grad /= world_size

    def hook(self, param):
        if param.grad is not None:
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)  # 存起来，之后 wait