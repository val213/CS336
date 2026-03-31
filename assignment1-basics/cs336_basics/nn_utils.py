from typing import Iterable
import torch
from torch import Tensor


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def rmsnorm(x: Tensor, weights: Tensor, eps: float) -> Tensor:
    # 第一步：计算 x 每个位置沿最后一维的均方值，保持维度
    RMS= torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    # 第二步：除以 RMS（加 eps 再开方）
    x = x / RMS
    # 第三步：乘以 weights
    x *= weights
    return x

def softmax(x: Tensor, dim: int) -> Tensor:
    # 先减最大值保证数值稳定
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    # 计算交叉熵损失
    logits_max = logits.max(dim=-1, keepdim=True).values
    log_softmax = logits - logits_max - torch.log(torch.sum(torch.exp(logits - logits_max), dim=-1, keepdim=True))
    return -torch.gather(log_softmax, -1, targets.unsqueeze(-1)).mean()

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float = 1.0) -> None:
    # 先转成 list，避免生成器只能遍历一次的问题
    params = [p for p in parameters if p.grad is not None]

    # 计算所有梯度拼在一起的 global L2 norm
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params))

    # 如果超标，等比例缩小所有梯度
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(clip_coef)