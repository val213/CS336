import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.data

                # 从 state 里取或初始化 m, v, t
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0

                # 按公式更新 m, v, t，然后更新参数
                state["m"] = state["m"]*beta1 + (1-beta1)*g
                state["v"] = state["v"]*beta2 + (1-beta2)*g**2
                state["t"] += 1

                # 偏置校正
                m_hat = state["m"] / (1 - beta1 ** state["t"])
                v_hat = state["v"] / (1 - beta2 ** state["t"])

                # 自适应学习率更新参数
                p.data = p.data - lr * m_hat / (torch.sqrt(v_hat) + eps)

                # 权重衰减
                p.data = p.data - lr * wd * p.data

                return p.data


def get_lr_cosine_schedule(it: int, max_learning_rate: float, min_learning_rate: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (max_learning_rate - min_learning_rate) * (1 + math.cos(progress * math.pi)) / 2
    else:
        return min_learning_rate