#!/usr/bin/env python3
"""Print a compact comparison table for Drama 3 benchmark manifests.

Usage:
    python3 tests/fish_drama3/compare_runs.py results/*/benchmark_manifest.json

Each argument may be a manifest JSON file or a directory containing one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STAGES = (
    ("drama3.setup_ms", "setup"),
    ("drama3.embedding_ms", "embed"),
    ("drama3.backbone_prefill_ms", "b.prefill"),
    ("drama3.tower_prefill_ms", "t.prefill"),
    ("drama3.lm_head_ms", "lm_head"),
    ("drama3.sampling_ms", "sample"),
    ("drama3.tower_ms", "tower"),
    ("drama3.backbone_step_ms", "b.step"),
    ("drama3.codec_ms", "codec"),
)
EXTRA = (
    ("drama3.offload_prefill_attention_ms", "pf.attn"),
    ("drama3.offload_prefill_experts_ms", "pf.exp"),
    ("disk_experts.bytes_read", "disk_read"),
    ("disk_experts.expert_loads", "loads"),
    ("disk_experts.cache_hits", "hits"),
    ("disk_experts.stage_ms", "stage_ms"),
)


def load(path: Path) -> dict:
    if path.is_dir():
        path = path / "benchmark_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def short_options(options: list[str]) -> str:
    parts = []
    for option in options:
        key, _, value = option.partition("=")
        parts.append(f"{key.replace('fish_drama3.', '')}={value}")
    return " ".join(parts)


def main(argv: list[str]) -> int:
    paths = [Path(argument) for argument in argv] or [Path(".")]
    rows = [load(path) for path in paths]
    rows.sort(key=lambda row: (row.get("rtf_median") or 9e9))

    header = f"{'label':<18}{'RTF':>7}{'xRT':>7}{'ms/frame':>9}  " + \
             "".join(f"{name:>9}" for _, name in STAGES)
    print(header)
    print("-" * len(header))
    for row in rows:
        per_frame = row.get("stage_ms_per_frame", {})
        line = (f"{row.get('label', '?'):<18}"
                f"{(row.get('rtf_median') or 0):>7.3f}"
                f"{(row.get('x_realtime_median') or 0):>7.2f}"
                f"{(row.get('ms_per_frame_median') or 0):>9.2f}  ")
        line += "".join(f"{per_frame.get(key, 0.0):>9.2f}" for key, _ in STAGES)
        print(line)
    print()
    for row in rows:
        per_frame = row.get("stage_ms_per_frame", {})
        extras = {key: per_frame.get(key, 0.0) for key, _ in EXTRA if key in per_frame}
        if not extras:
            continue
        print(f"{row.get('label', '?')}: " + ", ".join(
            f"{name}={extras[key]:.0f}" for key, name in EXTRA if key in extras))
    print()
    for row in rows:
        print(f"{row.get('label', '?'):<18} {short_options(row.get('session_options', []))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
