"""
CS336 作业2 - Transformer 模型基准测试脚本

用法示例：
    # 基础基准测试（仅前向传播）
    uv run python scripts/benchmark_transformer.py --model-size small --mode forward

    # 前向+反向传播
    uv run python scripts/benchmark_transformer.py --model-size medium --mode forward_backward

    # 混合精度（BF16）
    uv run python scripts/benchmark_transformer.py --model-size large --mixed-precision

    # 内存分析
    uv run python scripts/benchmark_transformer.py --model-size 2.7B --memory-profile

    # 使用 nsys 性能分析（在命令前加 nsys profile）
    uv run nsys profile -o result python scripts/benchmark_transformer.py --model-size small

nsys CLI 在 WSL（Ubuntu）上的安装方法：
    # 方法一：通过 NVIDIA CUDA 官方 apt 仓库安装 nsight-systems-cli
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get install -y nsight-systems-cli

    # 方法二：如果已安装完整 CUDA Toolkit（包含 nsys）
    # nsys 通常位于 /usr/local/cuda/bin/nsys 或 /opt/nvidia/nsight-systems/*/bin/nsys
    # 确保该路径已在 PATH 中：
    export PATH=/usr/local/cuda/bin:$PATH

    # 验证安装
    nsys --version
"""

import argparse
import timeit
from contextlib import nullcontext

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model as basics_model

# ─────────────────────────────────────────────
# 模型配置表（见作业说明 §1.1.2）
# ─────────────────────────────────────────────
MODEL_CONFIGS = {
    "small": dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B":   dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}

VOCAB_SIZE = 10_000
BATCH_SIZE = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark CS336 Transformer model")

    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=list(MODEL_CONFIGS.keys()),
        help="预设模型大小（small/medium/large/xl/2.7B）",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
        help="序列（上下文）长度",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="forward_backward",
        choices=["forward", "forward_backward"],
        help="benchmark 模式：仅前向 或 前向+反向传播",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="预热步骤数（不计入计时）",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="计时步骤数",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="启用 BF16 混合精度（torch.autocast）",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="使用 torch.compile 编译整个模型",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="启用 PyTorch 内存分析器，输出 memory_snapshot.pickle",
    )
    parser.add_argument(
        "--memory-snapshot-path",
        type=str,
        default="memory_snapshot.pickle",
        help="内存快照输出路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备（cuda 或 cpu）",
    )
    parser.add_argument(
        "--rope-theta",
        type=float,
        default=10000.0,
        help="RoPE theta 参数",
    )
    return parser.parse_args()


def build_model(args) -> torch.nn.Module:
    """根据命令行参数构建 Transformer 模型。"""
    cfg = MODEL_CONFIGS[args.model_size]
    model = basics_model.BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=args.rope_theta,
    )
    model = model.to(args.device)
    if args.compile:
        print("正在使用 torch.compile 编译模型…")
        model = torch.compile(model)
    return model


def make_batch(args) -> torch.Tensor:
    """生成随机整数输入 token 批次。"""
    return torch.randint(
        0,
        VOCAB_SIZE,
        (BATCH_SIZE, args.context_length),
        device=args.device,
    )


def run_step(model, batch, mode: str, autocast_ctx):
    """执行一步前向（或前向+反向）传播，返回损失值。"""
    with autocast_ctx:
        logits = model(batch)
        # 使用移位后的 token 作为目标（语言模型标准做法）
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            batch[:, 1:].reshape(-1),
        )

    if mode == "forward_backward":
        loss.backward()

    return loss


def main():
    args = parse_args()
    device = args.device

    print(f"设备: {device}")
    print(f"模型大小: {args.model_size}  配置: {MODEL_CONFIGS[args.model_size]}")
    print(f"上下文长度: {args.context_length}  批大小: {BATCH_SIZE}")
    print(f"模式: {args.mode}  预热步: {args.warmup_steps}  计时步: {args.steps}")
    print(f"混合精度: {args.mixed_precision}  torch.compile: {args.compile}")
    print("-" * 60)

    model = build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 选择 autocast 上下文
    if args.mixed_precision:
        autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    # ─── 预热阶段 ───────────────────────────────────────
    print(f"执行 {args.warmup_steps} 步预热…")
    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            optimizer.zero_grad()
            batch = make_batch(args)
            run_step(model, batch, args.mode, autocast_ctx)
            if device == "cuda":
                torch.cuda.synchronize()

    # ─── 可选：开启内存分析 ──────────────────────────────
    if args.memory_profile:
        print("开始记录 GPU 内存历史…")
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    # ─── 计时阶段 ───────────────────────────────────────
    timings = []

    with nvtx.range("benchmark"):
        for step in range(args.steps):
            optimizer.zero_grad()
            batch = make_batch(args)

            with nvtx.range(f"step_{step}"):
                t0 = timeit.default_timer()

                with nvtx.range("forward" if args.mode == "forward" else "forward_backward"):
                    run_step(model, batch, args.mode, autocast_ctx)

                if args.mode == "forward_backward":
                    with nvtx.range("optimizer_step"):
                        optimizer.step()

                if device == "cuda":
                    torch.cuda.synchronize()

                t1 = timeit.default_timer()

            elapsed_ms = (t1 - t0) * 1000
            timings.append(elapsed_ms)
            print(f"  步骤 {step + 1:3d}/{args.steps}: {elapsed_ms:.2f} ms")

    # ─── 可选：保存内存快照 ──────────────────────────────
    if args.memory_profile:
        snapshot_path = args.memory_snapshot_path
        print(f"保存内存快照到 {snapshot_path} …")
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"内存快照已保存。可将其拖拽到 https://pytorch.org/memory_viz 进行可视化。")

    # ─── 打印统计结果 ────────────────────────────────────
    import statistics

    mean_ms = statistics.mean(timings)
    std_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0

    print("-" * 60)
    print(f"平均耗时: {mean_ms:.2f} ms  标准差: {std_ms:.2f} ms")
    print(f"平均吞吐: {1000 / mean_ms:.2f} 步/秒")

    # 如果安装了 pandas，以 Markdown 表格输出结果
    try:
        import pandas as pd

        cfg = MODEL_CONFIGS[args.model_size]
        df = pd.DataFrame(
            [
                {
                    "model_size": args.model_size,
                    "d_model": cfg["d_model"],
                    "num_layers": cfg["num_layers"],
                    "context_length": args.context_length,
                    "mode": args.mode,
                    "mixed_precision": args.mixed_precision,
                    "mean_ms": round(mean_ms, 2),
                    "std_ms": round(std_ms, 2),
                }
            ]
        )
        print("\n结果（Markdown 表格）：")
        print(df.to_markdown(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
