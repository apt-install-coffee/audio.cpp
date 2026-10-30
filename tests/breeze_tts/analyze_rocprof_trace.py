#!/usr/bin/env python3
"""Turn a rocprofv3 kernel-dispatch trace into per-kernel attribution tables.

Reads the CSV files rocprofv3 writes under `--output-directory` and prints:

  * a per-kernel table (total ms, call count, mean us) sorted by total time;
  * a coarse per-category table, mapping kernel names onto the audio.cpp
    pipeline stage they belong to, so the split between the backbone decode,
    the depth decoder, the vocoder and the cache-copy overhead is visible.

Usage:
    python3 tests/breeze_tts/analyze_rocprof_trace.py tests/breeze_tts/results/<run>/rocprof
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


# Kernel-name fragments -> pipeline stage. Ordered: the first match wins.
CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    # The identity bf16 re-round of the whole KV cache (O2 in the plan).
    ("cache_round_bf16", ("round_bf16", "round_BF16", "ROUND_BF16")),
    # Conversion of the cache into the F16 buffer the flash kernel wants.
    ("cache_convert_f16", ("convert_unary", "cpy_f32_f16", "to_fp16", "cvt_f32_f16")),
    # The flash-attention / tile attention kernels.
    ("attention", ("fattn", "flash_attn", "mm_fattn", "attn")),
    # Quantised matrix-vector / matrix-matrix.
    ("matmul", ("mul_mat", "mmvq", "mmq", "mul_mat_id", "dequantize", "gemm")),
    # Elementwise / reduction / softmax / sampling helpers.
    ("elementwise", ("unary", "add", "mul", "rope", "norm", "soft_max", "argsort",
                     "top_k", "argmax", "cumsum", "scale", "cpd", "cpy", "cont",
                     "fill", "transpose", "get_rows", "diag_mask")),
    # convolutions (the vocoder).
    ("conv", ("conv_transpose", "conv2d", "conv_1d", "im2col", "col2im", "conv")),
]


def classify(kernel_name: str) -> str:
    lowered = kernel_name.lower()
    for category, fragments in CATEGORY_RULES:
        if any(fragment.lower() in lowered for fragment in fragments):
            return category
    return "other"


def find_trace_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*.csv")):
        try:
            head = path.open("r", encoding="utf-8", errors="replace").readline()
        except OSError:
            continue
        lowered = head.lower()
        if "kernel" in lowered and ("duration" in lowered or "end_time" in lowered):
            files.append(path)
    return files


def read_dispatch_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(handle, dialect=dialect)
        return [row for row in reader if row]


def pick(row: dict[str, str], *names: str) -> str | None:
    lowered = {key.lower(): value for key, value in row.items() if key}
    for name in names:
        if name in lowered:
            return lowered[name]
    for key, value in lowered.items():
        if any(name in key for name in names):
            return value
    return None


def duration_us(row: dict[str, str]) -> float:
    raw = pick(row, "duration", "duration_us", "Duration")
    if raw is not None and raw.strip():
        # rocprof reports nanoseconds in a "Duration" column.
        try:
            return float(raw.strip()) / 1000.0
        except ValueError:
            pass
    # Newer rocprofv3 builds only write the start and end timestamps.
    start = pick(row, "start_timestamp", "Start_Timestamp")
    end = pick(row, "end_timestamp", "End_Timestamp")
    if start and end:
        try:
            return (float(end.strip()) - float(start.strip())) / 1000.0
        except ValueError:
            return 0.0
    return 0.0


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    root = Path(argv[0])
    files = find_trace_files(root)
    if not files:
        print(f"no kernel dispatch CSV found under {root}")
        return 1

    per_kernel: dict[str, list[float]] = defaultdict(list)
    for path in files:
        for row in read_dispatch_rows(path):
            name = pick(row, "kernel_name", "Kernel_Name", "name")
            if not name:
                continue
            per_kernel[name.strip()].append(duration_us(row))

    per_category: dict[str, list[float]] = defaultdict(list)
    for name, durations in per_kernel.items():
        per_category[classify(name)].extend(durations)

    total_us = sum(sum(d) for d in per_kernel.values())

    print(f"trace files      : {len(files)}")
    print(f"distinct kernels : {len(per_kernel)}")
    print(f"total kernel time: {total_us / 1000.0:.3f} ms")
    print()

    print("=== per kernel (total ms desc, top 30) ===")
    print(f"{'total_ms':>10} {'calls':>8} {'mean_us':>10}  kernel")
    ranked = sorted(per_kernel.items(), key=lambda kv: -sum(kv[1]))
    for name, durations in ranked[:30]:
        total = sum(durations)
        print(f"{total / 1000.0:10.3f} {len(durations):8d} {total / len(durations):10.2f}  {name[:110]}")

    print()
    print("=== per category ===")
    print(f"{'total_ms':>10} {'share':>8} {'calls':>8}  category")
    for category, durations in sorted(per_category.items(), key=lambda kv: -sum(kv[1])):
        total = sum(durations)
        share = (total / total_us * 100.0) if total_us else 0.0
        print(f"{total / 1000.0:10.3f} {share:7.2f}% {len(durations):8d}  {category}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
