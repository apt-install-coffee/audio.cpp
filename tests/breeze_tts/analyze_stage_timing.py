#!/usr/bin/env python3
"""Summarise the audio.cpp per-stage timing log for a Breeze-TTS-2 run.

The framework writes lines of the form

    [TIMING ts=1761234567.123] breeze_tts.ar.backbone_cond_decode_ms 812.34

This script aggregates the BreezeTTS stage keys (median over the logged
occurrences) so the AR loop can be split into backbone prefill, backbone
decode, depth decode, vocoder and prompt time.

Usage:
    python3 tests/breeze_tts/analyze_stage_timing.py tests/breeze_tts/results/<run>/audiocpp_timing.log
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path


# Logical stage -> the timing keys that feed it.
STAGES: list[tuple[str, tuple[str, ...]]] = [
    ("prompt (text+speech encoders)", ("breeze_tts.generate.prompt_ms",)),
    ("backbone prefill (cond)", ("breeze_tts.ar.backbone_cond_prefill_ms",)),
    ("backbone prefill (uncond)", ("breeze_tts.ar.backbone_uncond_prefill_ms",)),
    ("backbone AR decode (cond)", ("breeze_tts.ar.backbone_cond_decode_ms",)),
    ("backbone AR decode (uncond)", ("breeze_tts.ar.backbone_uncond_decode_ms",)),
    ("depth decode (15 codebooks)", ("breeze_tts.ar.depth_ms",)),
    ("AR loop total", ("breeze_tts.ar.total_ms",)),
    ("vocoder (speech decoder)", ("breeze_tts.speech_decoder.streaming_total_ms",)),
    ("generate total", ("breeze_tts.generate.total_ms",)),
]

TIMING_RE = None


def parse(path: Path) -> dict[str, list[float]]:
    import re

    global TIMING_RE
    if TIMING_RE is None:
        TIMING_RE = re.compile(r"^\[TIMING ts=[0-9A-Za-z_.-]+\]\s+(?P<key>\S+)\s+(?P<value>[-+0-9.eE]+)\s*$")

    values: dict[str, list[float]] = defaultdict(list)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.match(raw.strip())
        if match is None:
            continue
        try:
            values[match.group("key")].append(float(match.group("value")))
        except ValueError:
            continue
    return values


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"timing log not found: {path}")
        return 1
    values = parse(path)

    print(f"timing log: {path}")
    print(f"distinct timing keys: {len(values)}")
    print()
    print(f"{'median_ms':>11} {'min_ms':>10} {'max_ms':>10} {'n':>4}  stage")
    for stage, keys in STAGES:
        collected = [v for key in keys for v in values.get(key, [])]
        if not collected:
            continue
        print(f"{statistics.median(collected):11.2f} {min(collected):10.2f} "
              f"{max(collected):10.2f} {len(collected):4d}  {stage}")

    # Anything else the run logged, so no signal is silently dropped.
    known = {key for _, keys in STAGES for key in keys}
    extra = {k: v for k, v in values.items() if k not in known and "breeze" in k}
    if extra:
        print()
        print("--- other breeze_tts timing keys ---")
        for key in sorted(extra):
            collected = extra[key]
            print(f"{statistics.median(collected):11.2f} {min(collected):10.2f} "
                  f"{max(collected):10.2f} {len(collected):4d}  {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
