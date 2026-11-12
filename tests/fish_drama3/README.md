# Drama 3 benchmark harness (ROCm/HIP)

`bench_drama3_hip.py` drives `audiocpp_cli` with a `--request-sequence` so every
measured request runs inside one process (warm graphs, warm expert cache, warm
allocator) and records, per request, the wall time, the generated audio
duration, the real-time factor and the framework's own stage breakdown
(`drama3.*` scalars from `--log-file`).

Drama 3 does not fit in 16 GB of VRAM, so the session options are part of the
measurement. The configuration used for the numbers in
[`docs/reports/fish_drama3_rocm_gfx1101_optimization_plan.md`](../../docs/reports/fish_drama3_rocm_gfx1101_optimization_plan.md)
is `disk_experts=true`, `expert_cache_size=96`, `cpu_expert_layers=0`.

## Run it

```bash
python3 tests/fish_drama3/bench_drama3_hip.py \
    --binary build-hip-gfx1101/bin/audiocpp_cli \
    --model models/Drama-3-GGUF-V2/drama-3-backbone-8bit-experts-Q3_K_S.gguf \
    --out-dir tests/fish_drama3/results/run \
    --warmup 1 --iterations 4 --max-tokens 150 --per-request-seed \
    --session-option fish_drama3.disk_experts=true \
    --session-option fish_drama3.expert_cache_size=96 \
    --session-option fish_drama3.cpu_expert_layers=0
```

The voice reference defaults to `~/.config/maki/scripts/voice.wav` with its
transcript from `voice_ref.txt`; pass `--reference-wav ""` for a text-only
prompt. `--rocprof <rocprofv3>` records a kernel trace (see the report for why
that fails to attach in the nix ROCm shell).

## Guard rails

`vram_guard.py` keeps a run away from the VRAM ceiling. Drama 3's per-request
graphs are spiky, and exhausting a 16 GB card wedges the GPU ring on gfx1101 —
that kills the desktop compositor and has cost a reboot. Two layers:

* a gate that refuses to start while the card is busy, and
* a watchdog that samples `mem_info_vram_used` every 200 ms and terminates the
  model process before it reaches the wedge zone.

```bash
# what the card looks like now
python3 tests/fish_drama3/vram_guard.py

# gate + watchdog around any command
python3 tests/fish_drama3/vram_guard.py --min-free-mb 13500 --kill-used-mb 15400 -- \
    python3 tests/fish_drama3/bench_drama3_hip.py ...  # same arguments as above
```

`bench_drama3_hip.py` takes `--min-free-vram-mb`, `--kill-vram-mb` and
`--vram-card` and applies both layers itself; the peak usage seen and whether
the watchdog fired land in `benchmark_manifest.json`. The working values for a
16 GB card next to a desktop compositor are a 13 500 MiB gate and a 15 400 MiB
kill threshold: the safe configurations peak at 11.8–12.9 GiB, and
`expert_cache_size=200` blows past 15.4 GiB during prefill graph allocation.

## Compare runs

```bash
python3 tests/fish_drama3/compare_runs.py tests/fish_drama3/results/*
```

Prints one row per run: RTF, ms/frame, and the per-frame stage breakdown
(prefill, tower, backbone step, codec), then the session options that produced
it.

## Quality gate

`tests/breeze_tts/quality_gate.py` is model-agnostic; it reads a run directory's
`request_sequence.json` and transcribes the WAVs with Citrinet:

```bash
python3 tests/breeze_tts/quality_gate.py \
    --binary build-hip-gfx1101/bin/audiocpp_cli \
    --asr-model models/Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf --backend cpu \
    tests/fish_drama3/results/run
```

## Notes

* **Page cache decides the prefill.** The package is 18.9 GiB on a 32 GB machine.
  A cold run reads the expert bank at ~1.7 GB/s; a warm one at ~10 GB/s. Always
  keep the warmup request and compare interleaved runs.
* **Audio is a sampling result.** Comparing two builds by ear or by file hash
  only means something at a fixed seed *and* a fixed graph shape; changes that
  re-tile a kernel (`reference_prefix_cache`, `device_sampling`) are judged with
  the ASR gate instead.
