import torch
import torch.distributed as dist

class ShardedOptimizer:
    def __init__(self, params: list[torch.Tensor], optimizer_cls, **kwargs):
        # 1. 把所有 params 存起来
        # 2. 算出自己的 shard（哪些 param 归我管）
        # 3. 只用自己的 shard 创建内部 optimizer
        self.params = list(params)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.shard = self.params[self.rank::self.world_size]
        self.optimizer = optimizer_cls(self.shard, **kwargs)                             
    def zero_grad(self):                                                                          
        # 对所有 params 清零
        for param in self.params:                                 
            param.grad = None

    def step(self):
        # 只 step 自己的 shard
        # 然后遍历所有 params，各自从 owner broadcast
        self.optimizer.step()
        for i, param in enumerate(self.params):
            owner = i % self.world_size
            dist.broadcast(param.data, src=owner)
        
        
    