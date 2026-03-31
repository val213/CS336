from torch._tensor import Tensor


import torch
from torch import Tensor

from cs336_basics.nn_utils import silu, rmsnorm, softmax, cross_entropy, gradient_clipping


def linear(in_features: Tensor, weights: Tensor) -> Tensor:   
    return in_features @ weights.T

def embedding(weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids, :]

def swiglu(x: Tensor, W1: Tensor, W2: Tensor, W3: Tensor) -> Tensor:
    gate = linear(x, W1)
    val = linear(x, W3)
    hidden = silu(gate) * val
    return linear(hidden, W2)

def rope(x: Tensor, token_positions: Tensor | None, dim: int, theta: float) -> Tensor:
    if token_positions is None:
        seq_len = x.shape[-2]
        token_positions = torch.arange(seq_len, device=x.device)

    # i = [0, 1, 2, ..., d_k/2 - 1]
    i = torch.arange(dim // 2, device=x.device)
    # 每对维度的基础频率
    freqs = 1.0 / (theta ** (i / (dim // 2))) # (d_k/2,)
    # 每个位置、每对维度的角度
    angles = token_positions[..., None] * freqs  # (..., seq_len, d_k/2)

    # 把 x 拆成相邻的两两一对
    x1 = x[..., 0::2]   # 偶数维度  (..., seq_len, d_k/2)
    x2 = x[..., 1::2]   # 奇数维度  (..., seq_len, d_k/2)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x1_out = x1 * cos - x2 * sin
    x2_out = x1 * sin + x2 * cos

    # x1_new 放偶数位，x2_new 放奇数位
    out = torch.stack([x1_out, x2_out], dim=-1)  # (..., seq_len, d_k/2, 2)
    out = out.flatten(-2)                          # (..., seq_len, d_k)
    return out

def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5)  
    if mask is not None:                                                                                                                                                                           
      scores = scores.masked_fill(mask == False, float('-inf'))
    return softmax(scores, dim=-1) @ v

def multihead_self_attention(x: Tensor, Wq: Tensor, Wk: Tensor, Wv: Tensor, Wout: Tensor, num_heads: int) -> Tensor:
    #  第一步： 三个线性投影，把 x 投影成 Q、K、V
    q = linear(x, Wq)
    k = linear(x, Wk)
    v = linear(x, Wv)

    # 第二步：把 Q、K、V 拆成 num_heads 个头，每个头的维度是 d_k
    q = q.view(q.size(0), q.size(1), num_heads, -1).transpose(1, 2)
    k = k.view(k.size(0), k.size(1), num_heads, -1).transpose(1, 2)
    v = v.view(v.size(0), v.size(1), num_heads, -1).transpose(1, 2)

    # 第三步：对每个头做 scaled dot product attention
    seq_len = x.shape[-2]                                           
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
    batch, _, seq, _ = attn_output.shape   
    attn_output = attn_output.transpose(1, 2)          # (batch, seq, num_heads, d_k)
    attn_output = attn_output.reshape(batch, seq, -1)  # (batch, seq, d_model)

    # 第四步：把 num_heads 个头拼接起来，得到最终的输出
    return linear(attn_output, Wout)

def multihead_self_attention_with_rope(x: Tensor, Wq: Tensor, Wk: Tensor, Wv: Tensor, Wout: Tensor, num_heads: int, dim: int, theta: float, token_positions: Tensor | None = None) -> Tensor:
    q = linear(x, Wq)
    k = linear(x, Wk)
    v = linear(x, Wv)

    q = q.view(q.size(0), q.size(1), num_heads, -1).transpose(1, 2)
    k = k.view(k.size(0), k.size(1), num_heads, -1).transpose(1, 2)
    v = v.view(v.size(0), v.size(1), num_heads, -1).transpose(1, 2)

    # 和 multihead_self_attention 唯一的区别是这里在做 attention 之前，对 Q 和 K 施加 RoPE 位置编码，V 不需要。
    head_dim = dim // num_heads                                                                                                                                                                    
    q = rope(q, token_positions, head_dim, theta)                   
    k = rope(k, token_positions, head_dim, theta)

    seq_len = x.shape[-2]                                           
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    attn_output = scaled_dot_product_attention(q, k, v, mask=mask)
    batch, _, seq, _ = attn_output.shape   
    attn_output = attn_output.transpose(1, 2)          # (batch, seq, num_heads, d_k)
    attn_output = attn_output.reshape(batch, seq, -1)  # (batch, seq, d_model)

    return linear(attn_output, Wout)

def transformer_block(x: Tensor, weights: dict[str, Tensor],num_heads: int, d_model: int, d_ff: int, max_seq_len: int, theta: float,) -> Tensor:
    residual = x
    x = rmsnorm(x, weights["ln1.weight"],1e-5)
    attn_output = multihead_self_attention_with_rope(x, weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"], weights["attn.output_proj.weight"], num_heads, d_model, theta, None)
    x = residual + attn_output

    residual = x
    x = rmsnorm(x, weights["ln2.weight"],1e-5)
    ff_output = swiglu(x, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])
    x = residual + ff_output
    return x

def transformer_lm(
    in_indices: Tensor,              # (batch, seq_len) 整数 token id
    weights: dict[str, Tensor],      # 模型所有层的权重
    vocab_size: int,                 # 词表大小（用不上，但接口保留）
    num_layers: int,                 # Transformer Block 的层数
    num_heads: int,                  # 注意力头数
    d_model: int,                    # 模型隐层维度
    d_ff: int,                       # FFN 中间层维度
    rope_theta: float,               # RoPE 的 theta 参数
) -> Tensor:                         # (batch, seq_len, vocab_size) logits
    # 1. Embedding: 把 token id 转成 embedding
    x = embedding(weights["token_embeddings.weight"], in_indices)

    # 2. 循环 N 个 Transformer Block
    for i in range(num_layers):
          layer_weights = {
              k.replace(f"layers.{i}.", ""): v
              for k, v in weights.items()
              if k.startswith(f"layers.{i}.")
          }
          x = transformer_block(x, layer_weights, num_heads, d_model, d_ff, 0, rope_theta)

    # 3. 最后的 RMSNorm
    x = rmsnorm(x, weights["ln_final.weight"], 1e-5)

    # 4. LM Head：投影到 vocab_size
    return linear(x, weights["lm_head.weight"])