#!/usr/bin/env python3
"""Interleaved A/B driver for Breeze-TTS-2 ROCm runs.

Consumer cards drift (clock, temperature, and the driver's own scheduling
noise), so two configurations are measured alternately - A, B, A, B, ... - and
compared on their medians instead of on a single pair of runs.

The comparison is done on *per frame* cost, not on the raw RTF: a change that
alters the sampled token stream can generate a different number of frames for
the same prompt, and the raw RTF would then measure the frame count rather
than the speed.

Usage:

    python3 tests/breeze_tts/ab_run.py \
        --binary build-rocm-gfx1101/bin/audiocpp_cli \
        --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
        --out-root tests/breeze_tts/results \
        --label-a nofuse --option-a breeze_tts.fused_depth=false \
        --label-b fuse   --option-b breeze_tts.fused_depth=true \
        --rounds 2 --iterations 5
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

TIMING_RE = re.compile(
    r"^\[(?:TIMING|TRACE) ts=[0-9A-Za-z_.-]+\]\s+(?P<key>\S+)\s+(?P<value>[-+0-9.eE]+)\s*$"
)
FRAME_SECONDS = 1920.0 / 24000.0


def timing_series(path: Path) -> dict[str, list[float]]:
    """Return the per-request timing series, in the order the requests ran.

    The values of one key are appended once per request (the warm-up request
    first), so the series can be paired with the manifest's ``measured`` list
    after dropping the warm-up entry.
    """
    values: dict[str, list[float]] = {}
    if not path.exists():
        return {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.match(raw.strip())
        if match is None:
            continue
        try:
            values.setdefault(match.group("key"), []).append(float(match.group("value")))
        except ValueError:
            continue
    return values


def run_one(args, label: str, session_options: list[str], binary: str = "") -> dict:
    out_dir = Path(args.out_root) / f"{args.label_prefix}{label}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_hip_breeze.py")),
        "--binary", binary or args.binary,
        "--model", args.model,
        "--out-dir", str(out_dir),
        "--label", label,
        "--guidance-scale", str(args.guidance_scale),
        "--iterations", str(args.iterations),
        "--warmup", str(args.warmup),
        "--max-tokens", str(args.max_tokens),
    ]
    for option in session_options:
        cmd += ["--session-option", option]
    if args.case_name:
        cmd += ["--case-name", args.case_name]
    print(f"[ab] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    manifest = json.loads((out_dir / "benchmark_manifest.json").read_text())
    series = timing_series(out_dir / "audiocpp_timing.log")
    measured = manifest["measured"]

    def aligned(key: str) -> list[float]:
        # The log starts with the warm-up request; drop it so the series lines
        # up with the manifest's measured list.
        values = series.get(key, [])
        return values[1:1 + len(measured)]

    ar = aligned("breeze_tts.ar.total_ms")
    depth = aligned("breeze_tts.ar.depth_ms")
    # BreezeTTS emits 1920 samples (80 ms) per audio frame.
    frames = [m["audio_duration_ms"] / (FRAME_SECONDS * 1000.0) for m in measured]
    per_frame = [(a / f) if f else 0.0 for a, f in zip(ar, frames)]
    depth_per_frame = [(d / f) if f else 0.0 for d, f in zip(depth, frames)]
    stage_series = {k: aligned(f"breeze_tts.depth.{k}_ms") for k in ("logits", "sample", "gather", "decode")}
    stages = {
        k: statistics.median([(v / f) if f else 0.0 for v, f in zip(stage_series[k], frames)])
        if stage_series[k] else 0.0
        for k in stage_series
    }
    rtfs = [m["rtf"] for m in measured]
    return {
        "label": label,
        "rtf_median": statistics.median(rtfs),
        "rtf_min": min(rtfs),
        "frames": statistics.median(frames) if frames else 0.0,
        "ar_ms": statistics.median(ar) if ar else 0.0,
        "depth_ms": statistics.median(depth) if depth else 0.0,
        "ms_per_frame": statistics.median(per_frame) if per_frame else 0.0,
        "depth_ms_per_frame": statistics.median(depth_per_frame) if depth_per_frame else 0.0,
        "stages": stages,
        "sha": [m.get("audio_sha256") for m in measured],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="binary for side A")
    parser.add_argument("--binary-b", default="", help="binary for side B (defaults to --binary)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--label-prefix", default="ab_")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--option-a", action="append", default=[])
    parser.add_argument("--option-b", action="append", default=[])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--case-name", default="")
    args = parser.parse_args()

    results: dict[str, list[dict]] = {"a": [], "b": []}
    for round_index in range(args.rounds):
        for side, label, options in (
            ("a", args.label_a, args.option_a),
            ("b", args.label_b, args.option_b),
        ):
            binary = args.binary_b if side == "b" else ""
            results[side].append(run_one(args, f"{label}_r{round_index + 1}", options, binary))

    print("\n=== interleaved A/B summary ===")
    header = f"{'side':>4} {'rtf_med':>9} {'rtf_min':>9} {'frames':>8} {'ar_ms':>10} {'ms/frame':>10} {'depth_ms/f':>12}"
    print(header)
    for side in ("a", "b"):
        rtf = statistics.median([r["rtf_median"] for r in results[side]])
        rtf_min = statistics.median([r["rtf_min"] for r in results[side]])
        frames = statistics.median([r["frames"] for r in results[side]])
        ar_ms = statistics.median([r["ar_ms"] for r in results[side]])
        per_frame = statistics.median([r["ms_per_frame"] for r in results[side]])
        depth = statistics.median([r["depth_ms_per_frame"] for r in results[side]])
        print(f"{side:>4} {rtf:9.4f} {rtf_min:9.4f} {frames:8.1f} {ar_ms:10.1f} {per_frame:10.2f} {depth:12.2f}")

    print("\n-- per-frame depth stages (ms) --")
    for side in ("a", "b"):
        stages = {k: statistics.median([r["stages"][k] for r in results[side]])
                  for k in ("logits", "sample", "gather", "decode")}
        print(f"{side:>4} " + "  ".join(f"{k}={v:7.3f}" for k, v in stages.items()))

    a = results["a"][-1]
    b = results["b"][-1]
    print("\nsha256 A:", a["sha"])
    print("sha256 B:", b["sha"])
    print("bit-identical:", a["sha"] == b["sha"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
