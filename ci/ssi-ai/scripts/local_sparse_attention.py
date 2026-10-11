"""Causal local sparse attention and a small validation/benchmark harness."""

from __future__ import annotations

import argparse
import time
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


class LocalSparseAttention(nn.Module):
    """Causal sliding-window self-attention with bounded query chunks."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 128,
        chunk_size: int = 128,
        num_sink_tokens: int = 0,
        attention_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        for name, value in (("d_model", d_model), ("n_heads", n_heads), ("chunk_size", chunk_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (("window_size", window_size), ("num_sink_tokens", num_sink_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.num_sink_tokens = num_sink_tokens
        self.attention_dropout = attention_dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, length, _ = x.shape
        qkv = self.qkv_proj(x).reshape(batch, length, 3, self.n_heads, self.head_dim)
        return qkv.permute(2, 0, 3, 1, 4).unbind(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"Expected (batch, length, {self.d_model}), received {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("Input must be floating point")
        batch, length, _ = x.shape
        if length == 0:
            return self.out_proj(x)

        q, k, v = self._project(x)
        outputs: list[torch.Tensor] = []
        dropout_p = self.attention_dropout if self.training else 0.0
        for start in range(0, length, self.chunk_size):
            end = min(start + self.chunk_size, length)
            local_start = max(0, start - self.window_size)
            prefix_end = min(self.num_sink_tokens, local_start)
            local_positions = torch.arange(local_start, end, device=x.device)

            if prefix_end:
                key_positions = torch.cat((torch.arange(prefix_end, device=x.device), local_positions))
                kc = torch.cat((k[:, :, :prefix_end], k[:, :, local_start:end]), dim=2)
                vc = torch.cat((v[:, :, :prefix_end], v[:, :, local_start:end]), dim=2)
            else:
                key_positions = local_positions
                kc = k[:, :, local_start:end]
                vc = v[:, :, local_start:end]

            query_positions = torch.arange(start, end, device=x.device)
            distance = query_positions[:, None] - key_positions[None, :]
            allowed = (distance >= 0) & ((distance <= self.window_size) | (key_positions[None, :] < self.num_sink_tokens))
            outputs.append(F.scaled_dot_product_attention(q[:, :, start:end], kc, vc, attn_mask=allowed, dropout_p=dropout_p, is_causal=False))

        context = torch.cat(outputs, dim=2).transpose(1, 2).reshape(batch, length, self.d_model)
        return self.out_proj(context)


def dense_causal_local_attention(model: LocalSparseAttention, x: torch.Tensor) -> torch.Tensor:
    """Compute the same local/sink pattern densely for small correctness tests."""
    q, k, v = model._project(x)
    positions = torch.arange(x.shape[1], device=x.device)
    distance = positions[:, None] - positions[None, :]
    allowed = (distance >= 0) & ((distance <= model.window_size) | (positions[None, :] < model.num_sink_tokens))
    context = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False)
    return model.out_proj(context.transpose(1, 2).reshape(x.shape[0], x.shape[1], model.d_model))


def self_test() -> None:
    torch.manual_seed(0)
    model = LocalSparseAttention(32, 4, window_size=3, chunk_size=5, num_sink_tokens=2).eval()
    x = torch.randn(2, 13, 32)
    y = model(x)
    assert y.shape == x.shape
    reference = dense_causal_local_attention(model, x)
    torch.testing.assert_close(y, reference, rtol=1e-5, atol=1e-5)

    changed = x.clone()
    changed[:, 9:] += 100
    torch.testing.assert_close(model(x)[:, :9], model(changed)[:, :9], rtol=1e-5, atol=1e-5)
    for bad in ((31, 4), (32, 0)):
        try:
            LocalSparseAttention(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid constructor arguments were accepted")
    print("self-test: PASS")


def benchmark_attention(seq_lengths: Iterable[int], device: torch.device, iterations: int, warmups: int) -> None:
    model = LocalSparseAttention(256, 8, window_size=128, chunk_size=256, num_sink_tokens=4).to(device).eval()
    print(f"device={device}, torch={torch.__version__}, dtype=float32")
    print(f"{'seq_len':>8} | {'latency_ms':>12} | {'peak_cuda_mib':>15}")
    print("-" * 44)
    for seq_len in seq_lengths:
        x = torch.randn(2, seq_len, 256, device=device)
        with torch.inference_mode():
            for _ in range(warmups):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            for _ in range(iterations):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            else:
                peak = 0.0
        latency = (time.perf_counter() - start) * 1000.0 / iterations
        print(f"{seq_len:8d} | {latency:12.2f} | {peak:15.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    if not args.self_test and not args.benchmark:
        parser.error("choose --self-test or --benchmark")
    if args.self_test:
        self_test()
    if args.benchmark:
        benchmark_attention(args.seq_lengths, torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.iterations, args.warmups)


if __name__ == "__main__":
    main()
