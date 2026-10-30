#!/usr/bin/env python3
"""Quality gate for Breeze-TTS-2 ROCm changes.

Transcribes the WAVs produced by a benchmark run with an ASR model and
compares the transcript against the prompt that was sent. Reports the
word-error rate (English) and character-error rate (CJK) per request and as a
corpus mean, so a speedup can be checked against "did it still say the right
words".

This is the Tier-B gate: changes that legitimately reorder floating-point
accumulation (batched CFG) or change the random-draw mapping (fused depth)
cannot be bit-identical, so they are judged on intelligibility instead.

Usage:
    python3 tests/breeze_tts/quality_gate.py \
        --binary build-rocm-gfx1101/bin/audiocpp_cli \
        --asr-model models/Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf \
        tests/breeze_tts/results/baseline_cfg1.0 \
        tests/breeze_tts/results/full_cfg1.0
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


PUNCTUATION = re.compile(r"[^\w一-鿿\s]")
CJK = re.compile(r"[一-鿿]")


def normalise(text: str) -> str:
    text = text.lower()
    text = PUNCTUATION.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str) -> tuple[float, str]:
    """Return (error rate, unit) for the pair, choosing WER or CER by script."""
    if CJK.search(reference) or CJK.search(hypothesis):
        units_ref = list(reference.replace(" ", ""))
        units_hyp = list(hypothesis.replace(" ", ""))
        unit = "CER"
    else:
        units_ref = reference.split()
        units_hyp = hypothesis.split()
        unit = "WER"
    if not units_ref:
        return 0.0, unit
    return levenshtein(units_ref, units_hyp) / len(units_ref), unit


def transcribe(binary: Path, asr_model: Path, wav: Path, backend: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
        out_path = Path(handle.name)
    try:
        result = subprocess.run(
            [str(binary), "--task", "asr", "--family", "citrinet_asr",
             "--model", str(asr_model), "--backend", backend,
             "--audio", str(wav), "--text-out", str(out_path)],
            check=False, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"asr failed for {wav}:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}",
                  file=sys.stderr)
            return ""
        return out_path.read_text(encoding="utf-8", errors="replace").strip()
    finally:
        out_path.unlink(missing_ok=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="benchmark run directories")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--case-file",
                        default=str(Path(__file__).with_name("breeze_tts_warm_bench_cases.txt")))
    parser.add_argument("--case-name", default="default")
    args = parser.parse_args(argv)

    binary = Path(args.binary).resolve()
    asr_model = Path(args.asr_model).resolve()

    print(f"{'run':24s} {'req':>4} {'unit':>4} {'err%':>7}  transcript")
    summary = []
    for run in args.runs:
        run_dir = Path(run)
        # The prompt that produced each request is recorded in the request
        # sequence, so the reference is exact rather than reconstructed.
        sequence = json.loads((run_dir / "request_sequence.json").read_text())
        rates = []
        for item in sequence:
            request_id = item["id"]
            wav = run_dir / "audio" / f"{request_id}.wav"
            if not wav.exists():
                continue
            reference = normalise(item.get("text", ""))
            if not reference:
                continue
            if CJK.search(reference):
                print(f"{run_dir.name:24s} {request_id[:12]:>12} {'--':>4} {'skipped':>7}  "
                      f"(CJK prompt, English-only ASR)")
                continue
            hypothesis = normalise(transcribe(binary, asr_model, wav, args.backend))
            rate, unit = error_rate(reference, hypothesis)
            rates.append(rate)
            print(f"{run_dir.name:24s} {request_id[:12]:>12} {unit:>4} {rate * 100:6.2f}%  "
                  f"{hypothesis[:70]}")
        if rates:
            summary.append((run_dir.name, sum(rates) / len(rates), len(rates)))
    print()
    print(f"{'run':24s} {'mean_err%':>10} {'n':>4}")
    for name, mean, count in summary:
        print(f"{name:24s} {mean * 100:9.2f}% {count:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
