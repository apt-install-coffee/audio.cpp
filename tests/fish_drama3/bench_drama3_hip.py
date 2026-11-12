#!/usr/bin/env python3
"""Reproducible ROCm/HIP benchmark for Fish Audio Drama 3 (MoE TTS).

Drives `audiocpp_cli` with a `--request-sequence` so that every measured
request runs inside one process (warm graphs, warm expert cache, warm
allocator) and records, per request, the wall time, the generated audio
duration, the real-time factor and the framework's own stage breakdown.

Drama 3 does not fit in 16 GB of VRAM, so every run also has to say where the
routed experts live. Use `--session-option` for that, for example:

    --session-option fish_drama3.disk_experts=true
    --session-option fish_drama3.expert_cache_size=96
    --session-option fish_drama3.cpu_expert_layers=0

Usage (from the repository root):

    python3 tests/fish_drama3/bench_drama3_hip.py \
        --binary build-hip-gfx1101/bin/audiocpp_cli \
        --model models/Drama-3-GGUF-V2/drama-3-backbone-8bit-experts-Q3_K_S.gguf \
        --out-dir tests/fish_drama3/results/baseline

The script is deliberately driver-only: it does not reimplement the model, it
measures the shipped binary exactly the way a user would run it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vram_guard  # noqa: E402  (sibling module, deliberate late import)


DEFAULT_CASE_FILE = Path(__file__).with_name("drama3_warm_bench_cases.txt")
DEFAULT_REFERENCE_WAV = Path.home() / ".config/maki/scripts/voice.wav"
DEFAULT_REFERENCE_TEXT = Path.home() / ".config/maki/scripts/voice_ref.txt"

METRIC_RE = re.compile(
    r"^metrics\[(?P<id>.+)\]\.(?P<key>[a-z_.0-9]+)=(?P<value>[-+0-9.eE]+)$")
TIMING_RE = re.compile(r"^\[TIMING ts=\S+\] (?P<key>\S+) (?P<value>[-+0-9.eE]+)$")

# Stage keys emitted by the Drama 3 generator, in the order they start.
STAGE_KEYS = (
    "drama3.setup_ms",
    "drama3.embedding_ms",
    "drama3.backbone_prefill_ms",
    "drama3.tower_prefill_ms",
    "drama3.lm_head_ms",
    "drama3.sampling_ms",
    "drama3.tower_ms",
    "drama3.backbone_step_ms",
    "drama3.codec_ms",
)


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


def file_identity(path: Path) -> dict:
    """Cheap identity for very large checkpoints (a full hash costs 30 s)."""
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty(repo_root: Path) -> bool:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
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
            f"unknown case {case_name!r} in {path}; available: {', '.join(sorted(cases))}")
    return cases[case_name]


def gpu_busy_percent() -> int:
    try:
        out = subprocess.run(["rocm-smi", "--showuse"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        return -1
    match = re.search(r"GPU use \(%\)\s*:\s*(\d+)", out)
    if match is None:
        match = re.search(r"GPU\[0\]\s*:\s*(\d+)", out)
    return int(match.group(1)) if match else -1


def wait_for_gpu_idle(max_wait_s: float = 600.0, threshold: int = 20) -> int:
    deadline = time.time() + max_wait_s
    while True:
        busy = gpu_busy_percent()
        if busy < 0 or busy <= threshold or time.time() > deadline:
            return busy
        time.sleep(1.0)


def gpu_architecture() -> str:
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True).stdout
        match = re.search(r"gfx[0-9a-f]+", out)
        return match.group(0) if match else "unknown"
    except FileNotFoundError:
        return "unknown"


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


def parse_stages(timing_log: Path) -> list[dict[str, float]]:
    """Split the append-only timing log into one record per generate() call."""
    records: list[dict[str, float]] = []
    current: dict[str, float] | None = None
    if not timing_log.exists():
        return records
    for line in timing_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.match(line.strip())
        if match is None:
            continue
        key, value = match.group("key"), float(match.group("value"))
        # Model load also emits drama3.* scalars before the first request. A
        # record therefore starts at setup_ms, not at the first drama3 key.
        if key == STAGE_KEYS[0]:
            if current is not None:
                records.append(current)
            current = {}
        if current is None:
            continue
        if key.startswith("drama3.") or key.startswith("disk_experts."):
            current[key] = value
    if current:
        records.append(current)
    return records


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="path to audiocpp_cli")
    parser.add_argument("--model", required=True, help="path to the Drama 3 GGUF")
    parser.add_argument("--backend", default="hip", help="hip|rocm|cuda|cpu")
    parser.add_argument("--out-dir", required=True, help="directory for the run artifacts")
    parser.add_argument("--case-file", default=str(DEFAULT_CASE_FILE))
    parser.add_argument("--case-name", default="default")
    parser.add_argument("--family", default="fish_drama3")
    parser.add_argument("--task", default="tts")
    parser.add_argument("--warmup", type=int, default=1, help="warmup requests before measuring")
    parser.add_argument("--iterations", type=int, default=3, help="measured requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="max generated acoustic frames per request")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=4097)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--reference-wav", default=str(DEFAULT_REFERENCE_WAV),
                        help="voice reference WAV; empty disables the reference")
    parser.add_argument("--reference-text-file", default=str(DEFAULT_REFERENCE_TEXT),
                        help="transcript of the voice reference WAV")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--session-option", action="append", default=[],
                        help="repeatable key=value session option")
    parser.add_argument("--request-option", action="append", default=[],
                        help="repeatable key=value request option")
    parser.add_argument("--label", default="", help="short name for this run")
    parser.add_argument("--per-request-seed", action="store_true",
                        help="give each measured request its own seed (seed + index)")
    parser.add_argument("--min-free-vram-mb", type=int, default=0,
                        help="refuse to start below this much free VRAM; 0 disables the gate")
    parser.add_argument("--kill-vram-mb", type=int, default=0,
                        help="terminate the run at this much used VRAM; 0 disables the watchdog")
    parser.add_argument("--vram-card", default=None,
                        help="DRM card name for the gate, e.g. card1 (default: largest VRAM total)")
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
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for path, what in ((binary, "binary"), (model, "model")):
        if not path.exists():
            raise SystemExit(f"{what} not found: {path}")

    cases = read_cases(Path(args.case_file), args.case_name)
    if not cases:
        raise SystemExit(f"case {args.case_name!r} has no requests")

    reference_wav = Path(args.reference_wav).expanduser().resolve()
    reference_text = ""
    if args.reference_wav:
        if not reference_wav.exists():
            raise SystemExit(f"reference wav not found: {reference_wav}")
        if args.reference_text_file:
            reference_text = Path(args.reference_text_file).expanduser().read_text(
                encoding="utf-8").strip()

    request_options = {
        "max_tokens": str(args.max_tokens),
        "seed": str(args.seed),
        "temperature": str(args.temperature),
        "top_k": str(args.top_k),
        "top_p": str(args.top_p),
    }
    if reference_text:
        request_options["reference_text"] = reference_text
    for option in args.request_option:
        key, _, value = option.partition("=")
        request_options[key] = value

    requests: list[dict] = []
    for _ in range(max(0, args.warmup)):
        entry = {"id": f"warmup_{len(requests)}", "text": cases[0],
                 "options": dict(request_options)}
        if args.reference_wav:
            entry["voice_ref"] = str(reference_wav)
        requests.append(entry)
    measured_ids: list[str] = []
    for index in range(args.iterations):
        request_id = f"measure_{index}"
        measured_ids.append(request_id)
        options = dict(request_options)
        if args.per_request_seed:
            options["seed"] = str(args.seed + index)
        entry = {"id": request_id, "text": cases[index % len(cases)], "options": options}
        if args.reference_wav:
            entry["voice_ref"] = str(reference_wav)
        requests.append(entry)

    sequence_path = out_dir / "request_sequence.json"
    sequence_path.write_text(json.dumps(requests, indent=2), encoding="utf-8")

    log_path = out_dir / "audiocpp.log"
    timing_path = out_dir / "audiocpp_timing.log"
    # The framework appends to the timing log, so a stale file from an earlier
    # run into the same directory would mix two configurations.
    if timing_path.exists():
        timing_path.unlink()
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
        "--threads", str(args.threads),
        "--request-sequence", str(sequence_path),
        "--metrics",
        "--log-file", str(timing_path),
        "--out-dir", str(out_dir / "audio"),
    ]
    for option in args.session_option:
        command += ["--session-option", option]

    vram_start = vram_guard.preflight(args.min_free_vram_mb, args.vram_card)
    gpu_busy_before = wait_for_gpu_idle()
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
    watchdog = vram_guard.Watchdog(args.kill_vram_mb, args.vram_card).start()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(run_command) + "\n")
        log.flush()
        process = subprocess.Popen(
            run_command, cwd=str(repo_root), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        watchdog.attach(process)
        output, _ = process.communicate()
        log.write(output)
    watchdog.stop()
    elapsed = time.time() - started

    metrics = parse_metrics(output)
    if process.returncode != 0:
        print(f"audiocpp_cli failed with exit code {process.returncode}", file=sys.stderr)
        print(output[-4000:], file=sys.stderr)
        return process.returncode

    stages = parse_stages(timing_path)
    measured = []
    for index, request_id in enumerate(measured_ids):
        bucket = metrics.get(request_id, {})
        wall_ms = bucket.get("wall_ms")
        duration_ms = bucket.get("audio_duration_ms")
        if wall_ms is None or duration_ms is None or duration_ms <= 0.0:
            continue
        wav_path = out_dir / "audio" / f"{request_id}.wav"
        stage_record = stages[args.warmup + index] if args.warmup + index < len(stages) else {}
        frames = stage_record.get("drama3.generated_frames", 0.0)
        measured.append({
            "id": request_id,
            "wall_ms": wall_ms,
            "audio_duration_ms": duration_ms,
            "rtf": wall_ms / duration_ms,
            "x_realtime": duration_ms / wall_ms,
            "frames": frames,
            "ms_per_frame": (wall_ms / frames) if frames else None,
            "stages": stage_record,
            "audio_sha256": sha256_file(wav_path) if wav_path.exists() else None,
        })

    rtf_values = [entry["rtf"] for entry in measured]
    stage_totals: dict[str, float] = {}
    for entry in measured:
        for key, value in entry["stages"].items():
            stage_totals[key] = stage_totals.get(key, 0.0) + value
    total_frames = sum(entry["frames"] for entry in measured) or 1.0
    stage_per_frame = {key: value / total_frames for key, value in stage_totals.items()}

    summary = {
        "label": args.label or out_dir.name,
        "backend": args.backend,
        "case_name": args.case_name,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "session_options": args.session_option,
        "request_options": args.request_option,
        "reference_wav": str(reference_wav) if args.reference_wav else None,
        "measured": measured,
        "process_wall_sec": elapsed,
        "rtf_median": statistics.median(rtf_values) if rtf_values else None,
        "rtf_min": min(rtf_values) if rtf_values else None,
        "rtf_max": max(rtf_values) if rtf_values else None,
        "x_realtime_median": (1.0 / statistics.median(rtf_values)) if rtf_values else None,
        "frames_total": total_frames,
        "ms_per_frame_median": (
            statistics.median([e["ms_per_frame"] for e in measured if e["ms_per_frame"]])
            if measured else None),
        "stage_ms_per_frame": stage_per_frame,
        "environment": {
            "git_commit": git_commit(repo_root),
            "git_dirty": git_dirty(repo_root),
            "binary_sha256": sha256_file(binary),
            "model": file_identity(model),
            "case_file_sha256": sha256_file(Path(args.case_file)),
            "prompt_sha256": sha256_text("\n".join(cases)),
            "reference_sha256": sha256_file(reference_wav) if args.reference_wav else None,
            "reference_text_sha256": sha256_text(reference_text),
            "gpu_arch": gpu_architecture(),
            "gpu_busy_before_percent": gpu_busy_before,
            "vram_start": str(vram_start),
            "vram_peak_used_mb": watchdog.peak_mb or None,
            "vram_watchdog_killed": watchdog.killed or None,
            "command": run_command,
        },
    }
    manifest_path = out_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"label            : {summary['label']}")
    print(f"backend          : {args.backend}  case={args.case_name}")
    print(f"measured requests: {len(measured)}   frames total={total_frames:.0f}")
    if measured:
        print(f"RTF median/min/max: {summary['rtf_median']:.4f} / "
              f"{summary['rtf_min']:.4f} / {summary['rtf_max']:.4f}")
        print(f"x_realtime median : {summary['x_realtime_median']:.4f}")
        print(f"ms per frame      : {summary['ms_per_frame_median']:.2f}")
        for key in STAGE_KEYS:
            if key in stage_per_frame:
                print(f"  {key:<26} {stage_per_frame[key]:8.2f} ms/frame")
    print(f"manifest          : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
