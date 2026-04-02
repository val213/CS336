import torch
from torch import Tensor


class FlashAttentionPyTorch(torch.autograd.Function):
    """
    FlashAttention2 的 PyTorch 实现。

    核心思想：把 Q、K、V 分块，用 online softmax 逐块计算 attention，
    避免实例化完整的 (seq_len, seq_len) 注意力矩阵，从而节省显存。

    Forward 只保存 Q、K、V、O、L（logsumexp），不保存中间的 S 和 P。
    Backward 用保存的 L 重新计算 P，同样分块完成梯度计算。
    """

    @staticmethod
    def forward(ctx, Q: Tensor, K: Tensor, V: Tensor, is_causal: bool = False) -> Tensor:
        """
        分块计算 attention 输出 O，并保存 logsumexp L 供 backward 使用。

        Args:
            Q, K, V: shape (batch, seq_len, d)
            is_causal: 是否使用因果掩码（上三角 mask）

        Returns:
            O: attention 输出，shape (batch, seq_len, d)
        """
        batch, seq_len, d = Q.shape
        scale = d ** -0.5
        block_size = 32
        num_blocks = (seq_len + block_size - 1) // block_size

        O = torch.zeros_like(Q)                                          # (batch, seq_len, d)
        L = torch.zeros(batch, seq_len, device=Q.device, dtype=Q.dtype) # (batch, seq_len)

        for i in range(num_blocks):
            # --- 取第 i 个 Q 块 ---
            q_start, q_end = i * block_size, min((i + 1) * block_size, seq_len)
            Qi = Q[:, q_start:q_end, :]   # (batch, Br, d)
            Br = q_end - q_start

            # 初始化当前 Q 块的 online softmax 统计量
            mi = torch.full((batch, Br), float('-inf'), device=Q.device, dtype=Q.dtype)  # 当前最大值
            li = torch.zeros((batch, Br), device=Q.device, dtype=Q.dtype)                # 当前 sum(exp)
            Oi = torch.zeros((batch, Br, d), device=Q.device, dtype=Q.dtype)             # 当前累积输出

            for j in range(num_blocks):
                # --- 取第 j 个 K/V 块 ---
                k_start, k_end = j * block_size, min((j + 1) * block_size, seq_len)
                Kj = K[:, k_start:k_end, :]   # (batch, Bc, d)
                Vj = V[:, k_start:k_end, :]   # (batch, Bc, d)

                # S = Q_i @ K_j^T / sqrt(d)，shape: (batch, Br, Bc)
                Sij = (Qi @ Kj.transpose(-2, -1)) * scale

                if is_causal:
                    pass  # TODO: 因果掩码

                # --- Online softmax 更新 ---
                # 1. 更新当前块的行最大值
                mi_old = mi.clone()
                mi = torch.maximum(mi_old, Sij.max(dim=-1).values)  # (batch, Br)

                # 2. 计算当前块的 exp（减去新最大值，数值稳定）
                Pij = torch.exp(Sij - mi.unsqueeze(-1))              # (batch, Br, Bc)

                # 3. rescale 旧的统计量（因为 mi 变大了，之前的基准需要修正）
                rescale = torch.exp(mi_old - mi)                      # (batch, Br)
                li = li * rescale + Pij.sum(dim=-1)                   # (batch, Br)
                Oi = Oi * rescale.unsqueeze(-1) + Pij @ Vj            # (batch, Br, d)

            # --- 归一化，写回结果 ---
            # 此时 Oi 是未归一化的累积输出，除以 li 才是真正的 attention 输出
            O[:, q_start:q_end, :] = Oi / li.unsqueeze(-1)
            # L = m + log(l) = logsumexp，供 backward 重建 softmax 用
            L[:, q_start:q_end] = mi + torch.log(li)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: Tensor) -> tuple[Tensor, Tensor, Tensor, None]:
        """
        分块计算 dQ、dK、dV。
        用保存的 L 重建 softmax，不需要存完整的 S 和 P。

        Args:
            dO: loss 对 O 的梯度，shape (batch, seq_len, d)

        Returns:
            dQ, dK, dV: 各输入的梯度
            None: is_causal 是 bool，不需要梯度
        """
        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal

        batch, seq_len, d = Q.shape
        scale = d ** -0.5
        block_size = 32
        num_blocks = (seq_len + block_size - 1) // block_size

        # D[i] = rowsum(dO_i ⊙ O_i)，softmax backward 公式里的归一化标量
        # 物理含义：梯度流过 softmax 时每行需要减去的修正量
        D = (dO * O).sum(dim=-1)   # (batch, seq_len)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        for i in range(num_blocks):
            q_start, q_end = i * block_size, min((i + 1) * block_size, seq_len)
            Qi  = Q[:, q_start:q_end, :]    # (batch, Br, d)
            dOi = dO[:, q_start:q_end, :]   # (batch, Br, d)
            Li  = L[:, q_start:q_end]       # (batch, Br)
            Di  = D[:, q_start:q_end]       # (batch, Br)
            dQi = torch.zeros_like(Qi)      # 当前 Q 块的梯度累积

            for j in range(num_blocks):
                k_start, k_end = j * block_size, min((j + 1) * block_size, seq_len)
                Kj = K[:, k_start:k_end, :]   # (batch, Bc, d)
                Vj = V[:, k_start:k_end, :]   # (batch, Bc, d)

                # 用 L 重建 softmax（不需要存 P）
                # P_ij = exp(S_ij - L_i)，等价于归一化后的 softmax 值
                Sij = (Qi @ Kj.transpose(-2, -1)) * scale   # (batch, Br, Bc)
                Pij = torch.exp(Sij - Li.unsqueeze(-1))      # (batch, Br, Bc)

                # dV = P^T @ dO
                dVj = Pij.transpose(-2, -1) @ dOi            # (batch, Bc, d)

                # dP = dO @ V^T
                dPij = dOi @ Vj.transpose(-2, -1)            # (batch, Br, Bc)

                # softmax backward: dS = P ⊙ (dP - D)
                # D 需要 broadcast 到 (batch, Br, Bc)
                dSij = Pij * (dPij - Di.unsqueeze(-1))        # (batch, Br, Bc)

                # dQ += dS @ K,  dK += dS^T @ Q
                dQi += dSij @ Kj * scale                      # (batch, Br, d)
                dKj  = dSij.transpose(-2, -1) @ Qi * scale    # (batch, Bc, d)

                # dK 和 dV 跨所有 i 块累加
                dK[:, k_start:k_end, :] += dKj
                dV[:, k_start:k_end, :] += dVj

            dQ[:, q_start:q_end, :] = dQi

        return dQ, dK, dV, None
