#!/usr/bin/env python3
"""Check current SSi-AI timings against a rolling median of prior results.

Example:
  python check_rolling_regression.py \
    --current artifacts/current/results.json \
    --history-glob 'artifacts/history/**/results.json' \
    --output artifacts/current/regression_report.json \
    --threshold 0.25 \
    --min-samples 3

Exit codes:
  0: no regression detected, or insufficient history (informational pass)
  1: one or more regressions detected
  2: invalid arguments or unreadable/malformed input
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


def load_results(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def mad(values: list[float], center: float) -> float:
    return median([abs(value - center) for value in values])


def collect_history(patterns: list[str], current: Path) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(Path(path).resolve() for path in glob.glob(pattern, recursive=True))
    return sorted(path for path in paths if path.is_file() and path != current.resolve())


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SSi-AI latency against rolling historical medians")
    parser.add_argument("--current", type=Path, required=True, help="current run results.json")
    parser.add_argument("--history-glob", action="append", required=True, help="glob for prior results.json files; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="write JSON regression report here")
    parser.add_argument("--metric", default="local_ms", help="benchmark field to evaluate")
    parser.add_argument("--threshold", type=float, default=0.25, help="allowed slowdown as a fraction; 0.25 means 25%%")
    parser.add_argument("--min-samples", type=int, default=3, help="minimum prior samples required for a hard decision")
    parser.add_argument("--min-absolute-ms", type=float, default=0.0, help="ignore slowdowns smaller than this absolute amount")
    parser.add_argument("--max-seq-len", type=int, default=None, help="only evaluate sequence lengths up to this value")
    args = parser.parse_args()

    if args.threshold < 0 or args.min_samples < 1 or args.min_absolute_ms < 0:
        parser.error("threshold must be nonnegative, min-samples must be positive, and min-absolute-ms must be nonnegative")
    if not args.current.is_file():
        print(f"error: current file does not exist: {args.current}", file=sys.stderr)
        return 2

    try:
        current_data = load_results(args.current)
        history_paths = collect_history(args.history_glob, args.current)
        history_data = [(path, load_results(path)) for path in history_paths]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    current_rows = current_data.get("benchmark", [])
    if not isinstance(current_rows, list):
        print("error: current results.json has no list-valued benchmark field", file=sys.stderr)
        return 2

    current_by_seq = {
        int(row["seq_len"]): float(row[args.metric])
        for row in current_rows
        if isinstance(row, dict) and row.get("seq_len") is not None and row.get(args.metric) is not None
    }
    if args.max_seq_len is not None:
        current_by_seq = {seq: value for seq, value in current_by_seq.items() if seq <= args.max_seq_len}

    comparisons: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    for seq_len, current_value in sorted(current_by_seq.items()):
        historical_values: list[float] = []
        sources: list[str] = []
        for path, data in history_data:
            for row in data.get("benchmark", []):
                if not isinstance(row, dict):
                    continue
                if int(row.get("seq_len", -1)) != seq_len or row.get(args.metric) is None:
                    continue
                value = float(row[args.metric])
                if math.isfinite(value) and value >= 0:
                    historical_values.append(value)
                    sources.append(str(path))
                break
        baseline = median(historical_values) if historical_values else None
        allowed = (baseline * (1 + args.threshold)) if baseline is not None else None
        absolute_delta = (current_value - baseline) if baseline is not None else None
        is_regression = (
            baseline is not None
            and len(historical_values) >= args.min_samples
            and current_value > allowed
            and absolute_delta > args.min_absolute_ms
        )
        comparison = {
            "seq_len": seq_len,
            "current_ms": current_value,
            "historical_samples": len(historical_values),
            "historical_median_ms": baseline,
            "historical_mad_ms": mad(historical_values, baseline) if baseline is not None else None,
            "allowed_ms": allowed,
            "absolute_delta_ms": absolute_delta,
            "slowdown_fraction": ((current_value / baseline) - 1) if baseline and baseline > 0 else None,
            "decision": "REGRESSION" if is_regression else ("PASS" if len(historical_values) >= args.min_samples else "INSUFFICIENT_HISTORY"),
            "history_sources": sources,
        }
        comparisons.append(comparison)
        if is_regression:
            regressions.append(comparison)

    report = {
        "status": "FAIL" if regressions else "PASS",
        "metric": args.metric,
        "threshold_fraction": args.threshold,
        "min_samples": args.min_samples,
        "min_absolute_ms": args.min_absolute_ms,
        "current_file": str(args.current),
        "history_files_considered": [str(path) for path in history_paths],
        "comparisons": comparisons,
        "regressions": regressions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for comparison in comparisons:
        print(
            f"seq={comparison['seq_len']} current={comparison['current_ms']:.3f}ms "
            f"median={comparison['historical_median_ms'] if comparison['historical_median_ms'] is not None else 'n/a'} "
            f"samples={comparison['historical_samples']} decision={comparison['decision']}"
        )
    print(f"report: {args.output}")
    if regressions:
        print(f"performance regressions detected: {len(regressions)}", file=sys.stderr)
        return 1
    if not history_paths:
        print("no historical results found; treating as informational pass", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
