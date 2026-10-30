#!/usr/bin/env python3
"""Compare Breeze-TTS-2 ROCm benchmark runs.

Prints a table of the runs found under a results directory, with the raw RTF
and a length-normalised figure. The normalised column matters because the
fused-depth and batched-CFG paths change the sampled token stream, so they can
generate a different number of frames for the same prompt. Comparing raw RTF
across runs with different frame counts overstates (or understates) the change;
the per-frame cost is the honest comparison.

Usage:
    python3 tests/breeze_tts/compare_runs.py tests/breeze_tts/results
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TIMING_RE = re.compile(
    r"^\[TIMING ts=[0-9A-Za-z_.-]+\]\s+(?P<key>\S+)\s+(?P<value>[-+0-9.eE]+)\s*$"
)

FRAME_SECONDS = 1920.0 / 24000.0  # BreezeTTS emits 1920 samples per frame at 24 kHz.


def stage_medians(timing_log: Path) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    if not timing_log.exists():
        return {}
    for raw in timing_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.match(raw.strip())
        if match is None:
            continue
        try:
            values.setdefault(match.group("key"), []).append(float(match.group("value")))
        except ValueError:
            continue
    import statistics
    return {k: statistics.median(v) for k, v in values.items()}


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    root = Path(argv[0])
    rows = []
    for manifest_path in sorted(root.glob("*/benchmark_manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        measured = manifest.get("measured") or []
        if not measured:
            continue
        run_dir = manifest_path.parent
        stages = stage_medians(run_dir / "audiocpp_timing.log")
        ar_ms = stages.get("breeze_tts.ar.total_ms")
        depth_ms = stages.get("breeze_tts.ar.depth_ms")
        backbone_ms = stages.get("breeze_tts.ar.backbone_cond_decode_ms")
        audio_s = sum(e["audio_duration_ms"] for e in measured) / 1000.0
        requests = len(measured)
        frames = audio_s / requests / FRAME_SECONDS
        rows.append({
            "label": manifest.get("label", run_dir.name),
            "dir": run_dir.name,
            "guidance": manifest.get("guidance_scale"),
            "rtf": manifest["rtf_median"],
            "frames": frames,
            "ar_ms": ar_ms,
            "ar_ms_per_frame": (ar_ms / frames) if ar_ms and frames else None,
            "depth_ms_per_frame": (depth_ms / frames) if depth_ms and frames else None,
            "backbone_ms_per_frame": (backbone_ms / frames) if backbone_ms and frames else None,
        })

    if not rows:
        print(f"no benchmark manifests under {root}")
        return 1

    print(f"{'label':22s} {'cfg':>4} {'rtf':>8} {'frames':>7} {'ar_ms':>9} "
          f"{'ar/frame':>9} {'depth/f':>9} {'bb/f':>8}")
    for row in rows:
        ar = f"{row['ar_ms']:.0f}" if row["ar_ms"] else "-"
        arf = f"{row['ar_ms_per_frame']:.2f}" if row["ar_ms_per_frame"] else "-"
        dpf = f"{row['depth_ms_per_frame']:.2f}" if row["depth_ms_per_frame"] else "-"
        bbf = f"{row['backbone_ms_per_frame']:.2f}" if row["backbone_ms_per_frame"] else "-"
        print(f"{row['label']:22s} {row['guidance']:4.1f} {row['rtf']:8.4f} {row['frames']:7.1f} "
              f"{ar:>9} {arf:>9} {dpf:>9} {bbf:>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
