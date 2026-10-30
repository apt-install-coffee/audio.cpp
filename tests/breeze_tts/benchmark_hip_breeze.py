#!/usr/bin/env python3
"""Reproducible ROCm/HIP benchmark for Breeze-TTS-2.

Drives `audiocpp_cli` with a `--request-sequence` so that every measured
request runs inside one process (warm graphs, warm allocator, warm pools) and
records, per request, the wall time, the generated audio duration and the RTF.

The run manifest captures everything needed to compare two runs later: the
binary/model/prompt hashes, the git commit, the ROCm and driver versions, the
GPU clock before and after the run, and the full GGML/HIP/ROCBLAS environment.

Usage (from the repository root):

    python3 tests/breeze_tts/benchmark_hip_breeze.py \
        --binary build-rocm-gfx1101/bin/audiocpp_cli \
        --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
        --out-dir tests/breeze_tts/results/baseline \
        --guidance-scale 1.0

    # classifier-free-guidance regime (the feature that costs the most)
    python3 tests/breeze_tts/benchmark_hip_breeze.py ... --guidance-scale 3.0

The script is deliberately driver-only: it does not reimplement the model, it
measures the shipped binary exactly the way a user would run it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CASE_FILE = Path(__file__).with_name("breeze_tts_warm_bench_cases.txt")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        )
        return bool(out.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return False


def read_cases(path: Path, case_name: str) -> list[str]:
    """Read the `[name]` section of a warm-bench case catalog."""
    cases: dict[str, list[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            cases.setdefault(current, [])
            continue
        if current is not None:
            cases[current].append(line)
    if case_name not in cases:
        raise SystemExit(
            f"unknown case {case_name!r} in {path}; available: {', '.join(sorted(cases))}"
        )
    return cases[case_name]


def gpu_clock() -> dict[str, str]:
    """Sample the GPU clock so a throttled run can be flagged."""
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        return {}
    try:
        out = subprocess.run(
            [rocm_smi, "--showclocks", "--showtemp", "--showuse"],
            check=False, capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    clocks: dict[str, str] = {}
    for line in out.splitlines():
        if "SCLK" in line or "MCLK" in line:
            match = re.search(r"([0-9]+)Mhz", line)
            if match:
                clocks.setdefault("sclk_mhz" if "SCLK" in line else "mclk_mhz", match.group(1))
        if "Temperature" in line:
            match = re.search(r":\s*([0-9.]+)", line)
            if match:
                clocks.setdefault("temp_c", match.group(1))
    return clocks


def gpu_busy_percent() -> int:
    """Return the GPU utilisation in percent, or -1 when it cannot be read."""
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi is None:
        return -1
    try:
        out = subprocess.run(
            [rocm_smi, "--showuse"],
            check=False, capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return -1
    match = re.search(r"GPU use \(%\):\s*([0-9]+)", out)
    return int(match.group(1)) if match else -1


def wait_for_gpu_idle(max_wait_s: float = 600.0, threshold: int = 20) -> int:
    """Wait until the GPU is quiet enough to measure.

    Another process (a server, another agent, a background job) can use the
    same GPU.  Measuring while it does makes the numbers meaningless, so the
    harness waits for the utilisation to drop below the threshold first and
    records what it saw.
    """
    deadline = time.monotonic() + max_wait_s
    while True:
        busy = gpu_busy_percent()
        if busy < 0 or busy < threshold:
            return busy
        if time.monotonic() >= deadline:
            print(f"[bench] warning: GPU still {busy}% busy after {max_wait_s:.0f}s",
                  file=sys.stderr)
            return busy
        time.sleep(5.0)


def gpu_architecture() -> str:
    rocminfo = shutil.which("rocminfo")
    if rocminfo is None:
        return "unknown"
    try:
        out = subprocess.run(
            [rocminfo], check=False, capture_output=True, text=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"gfx\d+[a-z]*", out)
    return match.group(0) if match else "unknown"


def rocm_version() -> str:
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        return "unknown"
    try:
        out = subprocess.run(
            [hipcc, "--version"], check=False, capture_output=True, text=True, timeout=120,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    match = re.search(r"HIP version:\s*(\S+)", out)
    return match.group(1) if match else "unknown"


def interesting_env() -> dict[str, str]:
    prefixes = ("GGML_", "HIP_", "ROCBLAS_", "HIPBLASLT_", "HSA_", "AMD_")
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith(prefixes)}


METRIC_RE = re.compile(r"^metrics\[(?P<id>.+)\]\.(?P<key>[a-z_.0-9]+)=(?P<value>[-+0-9.eE]+)$")


def parse_metrics(stdout: str) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for line in stdout.splitlines():
        match = METRIC_RE.match(line.strip())
        if match is None:
            continue
        bucket = metrics.setdefault(match.group("id"), {})
        try:
            bucket[match.group("key")] = float(match.group("value"))
        except ValueError:
            continue
    return metrics


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="path to audiocpp_cli")
    parser.add_argument("--model", required=True, help="path to the Breeze-TTS-2 GGUF")
    parser.add_argument("--backend", default="hip", help="hip|rocm|cuda|cpu")
    parser.add_argument("--out-dir", required=True, help="directory for the run artifacts")
    parser.add_argument("--case-file", default=str(DEFAULT_CASE_FILE))
    parser.add_argument("--case-name", default="default")
    parser.add_argument("--family", default="breeze_tts")
    parser.add_argument("--task", default="tts")
    parser.add_argument("--warmup", type=int, default=1, help="warmup requests before measuring")
    parser.add_argument("--iterations", type=int, default=5, help="measured requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="max generated acoustic frames (the docs' 100-token case)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="1.0 disables CFG; >1.0 exercises the CFG path")
    parser.add_argument("--instruct", default="Speak clearly and naturally.")
    parser.add_argument("--session-option", action="append", default=[],
                        help="repeatable key=value session option")
    parser.add_argument("--request-option", action="append", default=[],
                        help="repeatable key=value request option")
    parser.add_argument("--label", default="", help="short name for this run")
    parser.add_argument("--per-request-seed", action="store_true",
                        help="give each measured request its own seed (seed + index) so that "
                             "the corpus contains independent samples instead of repeats")
    parser.add_argument("--rocprof", default="", help="rocprofv3 binary for a kernel trace")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    binary = Path(args.binary).resolve()
    model = Path(args.model).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, what in ((binary, "binary"), (model, "model")):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}")

    cases = read_cases(Path(args.case_file), args.case_name)
    if not cases:
        raise SystemExit(f"case {args.case_name!r} has no requests")

    # One process, one session: warmup first, then the measured requests.
    # Batch mode takes the generation options per request, not as CLI flags.
    request_options = {
        "max_tokens": str(args.max_tokens),
        "seed": str(args.seed),
        "temperature": str(args.temperature),
        "top_k": str(args.top_k),
        "top_p": str(args.top_p),
        "guidance_scale": str(args.guidance_scale),
        "instruct": args.instruct,
    }

    requests: list[dict] = []
    for _ in range(max(0, args.warmup)):
        requests.append({"id": f"warmup_{len(requests)}", "text": cases[0],
                         "options": dict(request_options)})
    measured_ids: list[str] = []
    for index in range(args.iterations):
        request_id = f"measure_{index}"
        measured_ids.append(request_id)
        options = dict(request_options)
        if args.per_request_seed:
            options["seed"] = str(args.seed + index)
        requests.append({"id": request_id, "text": cases[index % len(cases)],
                         "options": options})

    sequence_path = out_dir / "request_sequence.json"
    sequence_path.write_text(json.dumps(requests, indent=2), encoding="utf-8")

    log_path = out_dir / "audiocpp.log"
    timing_path = out_dir / "audiocpp_timing.log"
    # The framework appends to the timing log, so a stale file from an earlier
    # run into the same directory would mix two configurations. Clear it, and
    # the previous audio, so every run stands on its own.
    for stale in (timing_path,):
        if stale.exists():
            stale.unlink()
    stale_audio = out_dir / "audio"
    if stale_audio.is_dir():
        for wav in stale_audio.glob("*.wav"):
            wav.unlink()

    command = [
        str(binary),
        "--task", args.task,
        "--family", args.family,
        "--model", str(model),
        "--backend", args.backend,
        "--request-sequence", str(sequence_path),
        "--metrics",
        "--log-file", str(timing_path),
        "--out-dir", str(out_dir / "audio"),
    ]
    for option in args.session_option:
        command += ["--session-option", option]
    for option in args.request_option:
        command += ["--request-option", option]

    gpu_busy_before = wait_for_gpu_idle()
    clock_before = gpu_clock()
    started = time.time()
    if args.rocprof:
        trace_dir = out_dir / "rocprof"
        trace_dir.mkdir(exist_ok=True)
        run_command = [
            args.rocprof, "--kernel-trace", "--stats",
            "--output-format", "csv",
            "--output-directory", str(trace_dir),
            "--", *command,
        ]
    else:
        run_command = command

    env = dict(os.environ)
    env["ENGINE_TIMING_ENABLED"] = "1"
    env["ENGINE_TIMING_FILE"] = str(timing_path)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(run_command) + "\n")
        log.flush()
        result = subprocess.run(
            run_command, cwd=str(repo_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        log.write(result.stdout)
    elapsed = time.time() - started
    clock_after = gpu_clock()

    metrics = parse_metrics(result.stdout)
    if result.returncode != 0:
        print(f"audiocpp_cli failed with exit code {result.returncode}", file=sys.stderr)
        print(result.stdout[-4000:], file=sys.stderr)
        return result.returncode

    measured = []
    for request_id in measured_ids:
        bucket = metrics.get(request_id, {})
        wall_ms = bucket.get("wall_ms")
        duration_ms = bucket.get("audio_duration_ms")
        if wall_ms is None or duration_ms is None or duration_ms <= 0.0:
            continue
        # The WAV hash is the quality gate: a change that is bit-identical in
        # the sampled token stream produces the same file, so two manifests can
        # be compared directly. Any difference means the sampling changed.
        wav_path = out_dir / "audio" / f"{request_id}.wav"
        measured.append({
            "id": request_id,
            "wall_ms": wall_ms,
            "audio_duration_ms": duration_ms,
            "rtf": wall_ms / duration_ms,
            "x_realtime": duration_ms / wall_ms,
            "audio_sha256": sha256_file(wav_path) if wav_path.exists() else None,
        })

    rtf_values = [entry["rtf"] for entry in measured]
    summary = {
        "label": args.label or out_dir.name,
        "backend": args.backend,
        "guidance_scale": args.guidance_scale,
        "case_name": args.case_name,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "session_options": args.session_option,
        "request_options": args.request_option,
        "measured": measured,
        "process_wall_sec": elapsed,
        "rtf_median": statistics.median(rtf_values) if rtf_values else None,
        "rtf_min": min(rtf_values) if rtf_values else None,
        "rtf_max": max(rtf_values) if rtf_values else None,
        "x_realtime_median": (1.0 / statistics.median(rtf_values)) if rtf_values else None,
        "environment": {
            "git_commit": git_commit(repo_root),
            "git_dirty": git_dirty(repo_root),
            "binary_sha256": sha256_file(binary),
            "model_sha256": sha256_file(model),
            "model_path": str(model),
            "case_file_sha256": sha256_file(Path(args.case_file)),
            "prompt_sha256": sha256_text("\n".join(cases)),
            "rocm_version": rocm_version(),
            "gpu_arch": gpu_architecture(),
            "clock_before": clock_before,
            "gpu_busy_before_percent": gpu_busy_before,
            "gpu_busy_after_percent": gpu_busy_percent(),
            "clock_after": clock_after,
            "env": interesting_env(),
            "command": run_command,
        },
    }
    manifest_path = out_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"label            : {summary['label']}")
    print(f"backend          : {args.backend}  guidance_scale={args.guidance_scale}")
    print(f"GPU              : {summary['environment']['gpu_arch']}  ROCm {summary['environment']['rocm_version']}")
    print(f"measured requests: {len(measured)}")
    if measured:
        print(f"RTF median/min/max: {summary['rtf_median']:.4f} / "
              f"{summary['rtf_min']:.4f} / {summary['rtf_max']:.4f}")
        print(f"x_realtime median : {summary['x_realtime_median']:.4f}")
        audio_seconds = sum(entry["audio_duration_ms"] for entry in measured) / 1000.0
        print(f"generated audio   : {audio_seconds:.3f} s")
    print(f"manifest          : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
