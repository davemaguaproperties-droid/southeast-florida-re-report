#!/usr/bin/env python3
"""Daily validation and benchmark runner for the ssi-ai skill.

Examples:
  python ssi_daily.py --mode check
  python ssi_daily.py --mode benchmark --compare-dense
  python ssi_daily.py --mode all --output-dir ./ssi-artifacts
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from local_sparse_attention import (  # noqa: E402
    LocalSparseAttention,
    dense_causal_local_attention,
)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_checks() -> dict[str, Any]:
    torch.manual_seed(0)
    model = LocalSparseAttention(32, 4, window_size=3, chunk_size=5, num_sink_tokens=2).eval()
    x = torch.randn(2, 13, 32)
    with torch.inference_mode():
        y = model(x)
        reference = dense_causal_local_attention(model, x)
        changed = x.clone()
        changed[:, 9:] += 100
        past_error = (model(x)[:, :9] - model(changed)[:, :9]).abs().max().item()
    dense_error = (y - reference).abs().max().item()
    if tuple(y.shape) != tuple(x.shape):
        raise AssertionError(f"shape mismatch: input={tuple(x.shape)} output={tuple(y.shape)}")
    if dense_error > 1e-5:
        raise AssertionError(f"dense-reference mismatch: {dense_error}")
    if past_error > 1e-5:
        raise AssertionError(f"causal leakage: {past_error}")
    for bad in ((31, 4), (32, 0)):
        try:
            LocalSparseAttention(*bad)
        except ValueError:
            continue
        raise AssertionError(f"invalid constructor accepted: {bad}")
    return {
        "status": "PASS",
        "shape": list(y.shape),
        "dense_reference_max_abs_error": dense_error,
        "causal_past_max_abs_error": past_error,
        "invalid_constructor_checks": "PASS",
    }


def measure(fn, x: torch.Tensor, device: torch.device, warmups: int, iterations: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmups):
            fn(x)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(iterations):
            fn(x)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000 / iterations
        peak_mib = (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0
    return elapsed_ms, peak_mib


def run_benchmark(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    model = LocalSparseAttention(
        args.d_model,
        args.n_heads,
        window_size=args.window_size,
        chunk_size=args.chunk_size,
        num_sink_tokens=args.num_sink_tokens,
    ).to(device).eval()
    rows: list[dict[str, Any]] = []
    for seq_len in args.seq_lengths:
        x = torch.randn(args.batch_size, seq_len, args.d_model, device=device)
        local_ms, local_peak = measure(model, x, device, args.warmups, args.iterations)
        row: dict[str, Any] = {
            "seq_len": seq_len,
            "local_ms": local_ms,
            "local_peak_cuda_mib": local_peak,
        }
        if args.compare_dense:
            if seq_len > args.max_dense_seq_len:
                row["dense_ms"] = None
                row["dense_peak_cuda_mib"] = None
                row["dense_skipped"] = f"sequence length exceeds --max-dense-seq-len={args.max_dense_seq_len}"
            else:
                dense_ms, dense_peak = measure(lambda q: dense_causal_local_attention(model, q), x, device, args.warmups, args.iterations)
                row["dense_ms"] = dense_ms
                row["dense_peak_cuda_mib"] = dense_peak
        rows.append(row)
    return rows


def write_artifacts(args: argparse.Namespace, payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    rows = payload.get("benchmark", [])
    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_dir / "benchmark.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    if args.plot and rows:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("plot skipped: matplotlib is not installed", file=sys.stderr)
        else:
            comparable = [r for r in rows if r.get("dense_ms") is not None]
            if not comparable:
                print("plot skipped: no dense comparison rows", file=sys.stderr)
            else:
                seq = [r["seq_len"] for r in comparable]
                local = [r["local_ms"] for r in comparable]
                dense = [r["dense_ms"] for r in comparable]
                plt.figure(figsize=(9, 5.5), dpi=150)
                plt.plot(seq, local, "o-", linewidth=2, label="Local sparse attention")
                plt.plot(seq, dense, "o-", linewidth=2, label="Dense masked attention")
                plt.xlabel("Sequence length (tokens)")
                plt.ylabel("Execution time per call (ms)")
                plt.title("SSi-AI execution time: local vs. dense attention")
                plt.grid(axis="y", alpha=0.3)
                plt.legend(frameon=False)
                plt.tight_layout()
                plt.savefig(output_dir / "attention_timing.png", bbox_inches="tight")
                plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily SSi-AI validation and benchmark runner")
    parser.add_argument("--mode", choices=("check", "benchmark", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("ssi-artifacts"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-sink-tokens", type=int, default=2)
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--compare-dense", action="store_true")
    parser.add_argument("--max-dense-seq-len", type=int, default=2048)
    parser.add_argument("--plot", action="store_true", help="write attention_timing.png when --compare-dense is enabled")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = choose_device(args.device)
    payload: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "torch": torch.__version__,
        "device": str(device),
        "dtype": "float32",
        "config": vars(args) | {"output_dir": str(args.output_dir)},
    }
    if args.mode in ("check", "all"):
        payload["checks"] = run_checks()
        print("checks: PASS")
    if args.mode in ("benchmark", "all"):
        payload["benchmark"] = run_benchmark(args, device)
        print(f"benchmark: {len(payload['benchmark'])} sequence lengths on {device}")
    write_artifacts(args, payload, args.output_dir)
    print(f"artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
