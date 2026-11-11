# Drama 3 on ROCm/HIP (gfx1101) — profile and optimization plan

This is the Drama 3 counterpart of the Breeze-TTS-2 ROCm work in
[`breeze_tts_rocm_gfx1101_optimization_plan.md`](breeze_tts_rocm_gfx1101_optimization_plan.md).
It measures Fish Audio Drama 3 — a 40-layer hybrid attention/MoE backbone with
256 experts per layer — on a 16 GiB RDNA3 card, finds where a request's time
actually goes, and records what was fixed, what was measured and rejected, and
what is left.

The model does not fit in VRAM (18.9 GiB of weights against 16.4 GiB of VRAM), so
every measurement here is on a **disk-expert** configuration: dense backbone
weights, the audio tower and the codec stay resident, and the 12.9 GiB of routed
expert banks live in the GGUF and are read into a bounded per-layer cache.

---

## 0. Summary

Interleaved A/B, two rounds each, the shipped `hip_perf` defaults against the
pristine baseline binary, one session of 1 warmup + 4 measured requests over the
4-sentence corpus, `max_tokens = 150`, the 58-second reference
(`~/.config/maki/scripts/voice.wav` + its transcript):

| | Baseline | This work | Change |
|---|---:|---:|---:|
| RTF (wall ms ÷ generated audio ms) | 2.249 | **1.172** | **−47.9 %** |
| ms per generated frame | 104.4 | **54.4** | −47.9 % |
| backbone prefill, ms/frame amortised | 56.1 | **7.4** | −86.8 % |
| tower prefill, ms/frame | 2.04 | **0.20** | −90.2 % |
| codebook sampling, ms/frame | 0.88 | **0.12** | −86.4 % |
| audio tower (13 layers), ms/frame | 9.16 | **8.09** | −11.7 % |
| backbone decode step, ms/frame | 27.56 | **26.91** | −2.4 % |
| codec decode, ms/frame | 1.42 | 1.40 | — |
| ASR word-error rate, greedy, 4 samples | 6.82 % | **6.82 %** | equal |
| Output with reference-prefix caching off | — | — | **bit-identical** |

The single biggest item is that **the prompt, not the decode loop, dominates a
Drama 3 request on this hardware**: a 58-second voice reference is 1253 prompt
rows, and pre-fix that prefill was 87 % of the request. Both halves of the fix
are in the prefill: the expert GEMMs were doing 5× more work than the routing
needed (§3.1), and the same reference rows were re-evaluated for every request
(§3.2).

After both, the steady-state request is decode-bound at ~54 ms/frame. The AR
loop itself barely moved, and §4 explains why that is a different problem from
the one Breeze-TTS-2 had.

---

## 1. Environment and the model

| | |
|---|---|
| GPU | AMD Radeon RX 7800 XT, `gfx1101`, 16 368 MiB VRAM, wave size 32, HIP VMM off |
| CPU / RAM | Ryzen 9 5900X (12 cores / 24 threads), 32 GB, NVMe |
| Runtime | ROCm 7.2.3 / HIP 7.2.53211, `nix develop .#rocm` |
| Build | `-DENGINE_ENABLE_HIP=ON -DGPU_TARGETS=gfx1101 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON`, RelWithDebInfo |
| Model | `models/Drama-3-GGUF-V2/drama-3-backbone-8bit-experts-Q3_K_S.gguf` (18.90 GiB, Q3_K experts, Q8_0 dense backbone, F16 tower/codec) |
| Reference | `voice.wav`, 24 kHz mono, 1 393 200 samples (58.0 s) + `voice_ref.txt` |
| Session | `disk_experts=true`, `expert_cache_size=96`, `cpu_expert_layers=0` |

Weight breakdown of the package (measured from the GGUF header):

| Part | GiB |
|---|---:|
| Routed expert banks, 3 projections × 40 layers × 256 experts, Q3_K | 12.89 |
| Dense backbone (embed, lm_head, attention, shared experts, routers) | 2.03 |
| Audio tower (4 shared + 9 residual layers, F16) | 1.89 |
| Codec (encoder, quantizer, decoder, F16) — includes an unused 0.5 GiB `causal_mask` tensor | 0.90 |

Per routed expert one projection is 440 KiB, so an expert is 1.32 MiB and one
cache slot across all 40 layers costs 51.5 MiB.

### 1.1 Why disk experts, and what fits

The 16 GiB card cannot hold 12.9 GiB of experts plus the 4.9 GiB of dense, tower
and codec weights plus the codec encoder's 4.1 GiB workspace. Measured
alternatives on this exact card:

| Configuration | Result |
|---|---|
| `cpu_expert_layers=12` (the guide's 16 GB hint for Q3) | OOM at the codec encode graph (4 122 MiB) |
| `cpu_expert_layers=20` | OOM, same place |
| `cpu_expert_layers=24` | Fits: 7.78 GiB on GPU, 7.92 GiB host experts — but prefill 25.3 s and 42 ms/frame |
| `disk_experts`, cache 96 | Fits: 2.50 dense + 2.17 tower + 0.75 codec + 4.95 experts = 10.4 GiB, peak 14.5 GiB with the encoder graph |
| `disk_experts`, cache 128 | OOM at the codec encode graph |
| `disk_experts`, cache 160 + `codec_encoder_chunk_samples=65536` | Fits, peak 15.4 GiB |

The codec encoder graph is the reason a 12-slot-higher cache does not fit: it
allocates 4.1 GiB for its 16 384-sample convolution window, and it is only alive
while the reference is encoded. `codec_encoder_chunk_samples` shrinks it enough
to buy 64 more expert slots.

---

## 2. Where the time goes

### 2.1 Stage breakdown (baseline, per request, 30 frames)

From the framework's own timing log (`drama3.*` scalars), ms/frame amortised over
the generated frames of the request:

| Stage | ms/frame | share of a request |
|---|---:|---:|
| **backbone prefill (1253 prompt rows)** | **232.1** | **87 %** |
| backbone decode step | 34.8 | 13 % |
| tower prefill | 6.8 | 2.6 % |
| audio tower decode | 10.3 | 3.9 % |
| codebook sampling | 1.0 | 0.4 % |
| codec decode | 1.9 | 0.7 % |

Two things follow immediately:

1. A request that carries a long voice reference is a **prefill** problem. Only
   after §3.1/§3.2 does the decode loop become the largest item.
2. The Breeze-TTS-2 finding ("the AR loop is the whole request, and inside it the
   depth decoder is 84 %") does not transfer: Drama 3's per-frame work is spread
   over 40 sparse MoE layers whose traffic is dominated by *where the experts
   are*, not by how many codebooks are sampled.

### 2.2 Where the prefill time goes

`Drama3OffloadPrefillGraph` evaluates a prompt in per-layer phases: an attention
graph, then one indexed-MoE block per expert group, then a combine graph. It
reports `drama3.offload_prefill_attention_ms`, `..._experts_ms`,
`..._expert_work_rows` and `..._routed_rows`.

Baseline, one 1253-row prompt:

| | value |
|---|---:|
| routed rows (1253 rows × 8 experts × 40 layers) | 471 360 |
| **computed work rows** | **2 343 936** (5.0 ×) |
| attention phase | 0.48 s |
| expert phase | 6.48 s |
| disk reads during the request | 19.8 GB |
| `read_ms` / `upload_ms` | 11.6 s / 0.37 s (cold page cache) |

The 5× figure is the bug that §3.1 fixes: the padded work is
`slots × expert_rows × ceil(busiest_rows / expert_rows)` per group, so an expert
group the size of the cache charges every slot for the busiest expert's row
count.

### 2.3 Where the decode time goes

Per generated frame, with the baseline binary and cache 96:

| | ms/frame |
|---|---:|
| expert staging (`drama3.backbone_step_stage_ms`: reads + uploads + syncs) | 11.6 |
| the rest: GPU work and per-layer host round trips | ~15 |
| audio tower (one cached graph, 13 layers, 1.69 GB of F16 weights/frame) | 9.1 |

The staging split was added for this work
(`drama3.backbone_step_stage_ms`, from `Drama3BackboneDecoderRuntime::expert_stage_ms()`).
It says the decode loop is *still* paying for expert movement: at cache 96 the
session reads ~145 MB per generated frame.

The remaining ~15 ms is GPU work over ~1.6 GB of weight traffic per frame: 40
layers × ~30 MB of dense projections plus 8 experts × 1.32 MiB per layer. At
this card's bandwidth that is ~2.7 ms of pure traffic, so ~12 ms is dispatch and
small-GEMV inefficiency — the same shape of problem Breeze-TTS-2 had, but with
400 small dispatches instead of 2 500 tiny ones. §4.1 covers what is worth doing
about it.

---

## 3. Implemented

### 3.1 `O1` — size the prefill expert group to the group, not to the cache

`src/community_models/fish_drama3/backbone.cpp`, `src/framework/core/disk_expert_weight_store.cpp`

The prefill staged up to `expert_cache_size` experts per group and then ran one
`ggml_mul_mat_id` block per group, sized `expert_cache_size × 32` rows. With 96
slots and 256 experts this computes 32 × 96 rows per block although only a
fraction of those slots hold rows for that block.

Two changes:

* `DiskExpertWeightStore::stage()` takes an optional `slot_limit`. Fresh
  placements are confined to slots `[0, slot_limit)`, so a caller can keep its
  indexed-MoE graph sized to the group while the cache keeps its full capacity.
  An expert already resident above the window is released and re-read (it cannot
  be served from there).
* The prefill graph builds one expert graph per candidate group size (4, 8, 16,
  32, 64, … cache size), and each layer picks the group size whose predicted
  padded work is smallest, with a two-block penalty per extra group so that a
  tiny group cannot win on padding alone.

Measured on the same 1253-row prompt:

| | work rows | expert phase |
|---|---:|---:|
| baseline (group = cache) | 2 343 936 | 6.48 s (cold reads) |
| **this change** | **704 512** | **3.10 s** (same 19.8 GB read, 1.31 s of it) |
| block 16 rows | 593 408 | 8.23 s — more blocks, each launch-bound |
| block 64 rows | 942 080 | 3.37 s — padding grows faster than tiling helps |

So the block stays at 32 rows and the group size adapts. The expert phase is now
1.5× the routed rows (down from 5.0×), and the request-level effect is
prefill 232 → 117 ms/frame, RTF 7.02 → 4.65 on the 30-frame measurement, with
**bit-identical output** (same SHA-256): the arithmetic per row is unchanged, only
which slot holds which expert.

Rejected in the same place: a frequency-aware (LFU) eviction policy instead of
LRU. Routing is not skewed enough at these cache sizes to matter — 5 507 hits
against 5 505 for LRU on the same workload — so it was reverted.

### 3.2 `O2` — keep the reference-prefix state across requests

`src/community_models/fish_drama3/{generator,backbone,tower,prompt}.cpp`, options
`fish_drama3.reference_prefix_cache` (default **true**, disk mode only)

`build_drama3_prompt` places the system prompt, the reference transcript and the
reference VQ codes *before* the user turn, so those rows are identical for every
request that reuses a voice. The generator now:

* asks the prompt builder for the row index where the reference block ends,
* evaluates that prefix and the text tail as two phases on the first request,
  keeping a host copy of the decoder state (KV caches, conv tails, recurrent
  states) and of the tower's speech-timeline caches at the boundary,
* restores that state for a later request whose prompt starts with the same rows
  and only evaluates the text tail.

The tail must go through the **cached decode path**, not the batched prefill
graph: the batched prefill computes attention only among the rows it is given, so
a split tail would silently lose the reference conditioning. That was a real bug
found by the ASR gate (empty take on the second request) and is the reason the
tail is evaluated one step at a time.

Measured (per-request prefill, 150-frame requests):

| | first request | later requests |
|---|---:|---:|
| baseline (no reuse) | 56.1 ms/frame | 56.1 ms/frame |
| **this change** | 60 ms/frame (two phases) | **7.4 ms/frame** |
| disk reads per later request | 19.8 GB | 0.2 GB |

The trade-off is the same one the Breeze-TTS-2 fused-depth path accepted: the
two-segment prompt evaluation re-tiles the attention, so the sampled take for a
fixed seed changes. `fish_drama3.reference_prefix_cache=false` restores the
single-segment evaluation, and was verified to reproduce the pre-change
SHA-256 exactly. The ASR gate (§5) shows identical transcripts with the cache on
and off.

### 3.3 `O3` — HIP gets the performance preset by default

`src/community_models/fish_drama3/generator.cpp`

HIP runs the same ggml-cuda kernels as CUDA, and the `cuda_perf` preset was gated
behind an option name that a HIP user has no reason to look for. There is now a
`hip_perf` preset with identical fields (`fused_moe`, F16 KV, 256-row attention
buckets, device sampling), selected automatically when no `perf_mode` is given on
a HIP backend:

| | baseline | `hip_perf` |
|---|---:|---:|
| tower decode, ms/frame | 10.30 | 9.40 |
| codebook sampling, ms/frame | 0.99 | 0.13 |
| lm_head + sampling round trips | per layer | one graph |

Device sampling folds the 13 tower layers and all nine codebook draws into one
cached graph. Every field remains individually overridable, and `perf_mode=off`
restores the previous defaults.

### 3.4 Instrumentation added for this work

* `drama3.backbone_step_stage_ms` — expert I/O inside the decode loop.
* `drama3.offload_prefill_build_ms` — the prefill graph build (8.8 ms, i.e. the
  two-phase prefill in §3.2 costs almost nothing on the first request).
* `Drama3BackboneDecoderRuntime::expert_stage_ms()`.

---

## 4. Remaining work, re-ranked against the measurements

### 4.1 P0 — the decode step is dispatch- and GEMV-bound (27 ms/frame)

After §3.1–3.3 the AR loop is 54 ms/frame: 27 ms backbone, 8 ms tower, 1.4 ms
codec, 7.4 ms amortised prefill. The backbone step reads ~1.6 GB of weights per
frame and needs ~2.7 ms of that at peak bandwidth, so the gap is kernel
efficiency, not traffic:

* the per-layer projections are batch-1 GEMVs over small matrices
  (`2048×512`, `2048×256`), where the Breeze-TTS-2 work already measured
  75–88 % of peak only on the *large* matrices;
* 400 dispatches per frame at ~5 µs each is 2 ms, and the disk-expert eval
  callback adds 40 host round trips and ~12 ms of staging.

Three directions, in order of expected value:

1. **More expert slots in VRAM.** Cache 160 (with `codec_encoder_chunk_samples=65536`)
   measured 23.9 ms/frame against 26.9 at cache 96, and 192 does not fit. Moving
   the codec to the CPU frees 4.85 GiB (it would allow ~250 slots) but the codec
   decode then costs 244 ms/frame, so the encoder — not the decoder — is what
   should move. A separate `codec_encoder_backend=cpu` option is the next step.
2. **HIP graph capture of the decode step.** The tower's single cached graph runs
   at 210 GB/s; the backbone's scheduler-driven graph does not, and it is the
   graph that is dispatch-bound. Per-layer capture around the expert callback is
   the analogue of the fused depth path, but the callback needs host ids between
   layers, so this needs a double-buffered capture per layer, not one graph.
3. **A fused grouped GEMM for the prefill.** MMQ handles `mul_mat_id` ids on the
   host, one launch per expert: the prefill still pays 40 × 256 × 3 ≈ 30 000
   launches. A device-side expert loop would collapse that (the `expert_rows=16`
   experiment shows how launch-bound it is: 593 k rows took 8.2 s where 704 k
   rows took 2.5 s).

### 4.2 P1 — the pinned-expert path uploads without coalescing

`pin_experts=true` measured 26.9 ms/frame against 29.9 at cache 96 (13.2 GiB of
pinned host RAM), but its uploads are one transfer per expert per projection
(their host addresses are not contiguous), while the staged path issues one
contiguous transfer per slot run. Coalescing pinned uploads through the staging
buffer should recover the difference. It was not pursued here because the machine
has 32 GB of RAM and pinning 13.2 GiB competes with the page cache that makes the
non-pinned path fast.

Also measured and rejected: `disk_expert_read_threads=4` made the decode step
*worse* (26.9 → 36.7 ms/frame) — the OpenMP team is rebuilt per stage call, which
is exactly the cost the store's own comment warns about.

### 4.3 P2 — the codec encoder's 4.1 GiB workspace

It is allocated for one 16 384-sample window, is only used while a reference is
encoded, and is what stops a 128-slot cache from fitting. Running just the
encoder on the CPU (0.3 GiB of its weights) would free 4.85 GiB of VRAM and buy
~90 expert slots, which §4.1 measures as ~3 ms/frame.

### 4.4 P3 — `expert_cache_size` guidance

96 slots is the largest cache that fits with the encoder graph resident; 160 fits
with `codec_encoder_chunk_samples=65536` at a 15.4 GiB peak (too tight to leave a
desktop running). The packaged default stays at 32 for small cards; the docs now
say what to use on 16 GiB.

---

## 5. Quality gates

Two gates, on the same protocol as the Breeze-TTS-2 work.

**Tier A — bit-exactness.** With `reference_prefix_cache=false` the optimized
binary reproduces the baseline WAV byte for byte (SHA-256 `5067ec88e5d54af5…`,
`short` corpus, 30 frames, seed 0). That covers §3.1 (group sizing) and §3.3's
fused routing: expert placement and group size never change arithmetic.

**Tier B — intelligibility.** Citrinet ASR round trip over the 4-sentence corpus,
greedy (`temperature=0`), `max_tokens=150`:

| run | mean WER | per-request transcripts |
|---|---:|---|
| prefix cache off (baseline behaviour) | **6.82 %** | identical |
| prefix cache on | **6.82 %** | identical |

The only error in either run is the ASR spelling "audio c p p" for "audio cpp".
An earlier revision of §3.2 scored 31.82 % with an empty take on the second
request; that is what sent the tail onto the cached decode path. The reuse path
is also deterministic: three identical requests in one session produce the same
WAV hash.

---

## 6. Reproducing

```bash
# build (HIP, gfx1101)
nix develop .#rocm --command bash -c '
  cmake -S . -B build-hip-gfx1101 -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_CUDA=OFF -DGPU_TARGETS=gfx1101 \
    -DAUDIOCPP_MODELS=fish_drama3,citrinet_asr \
    -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON -DENGINE_ENABLE_OPENMP=ON
  make -C build-hip-gfx1101 -j24 audiocpp_cli'

# measure (1 warmup + 4 measured requests, one session, 58 s reference)
python3 tests/fish_drama3/bench_drama3_hip.py \
  --binary build-hip-gfx1101/bin/audiocpp_cli \
  --model models/Drama-3-GGUF-V2/drama-3-backbone-8bit-experts-Q3_K_S.gguf \
  --out-dir tests/fish_drama3/results/run \
  --warmup 1 --iterations 4 --max-tokens 150 --per-request-seed \
  --session-option fish_drama3.disk_experts=true \
  --session-option fish_drama3.expert_cache_size=96 \
  --session-option fish_drama3.cpu_expert_layers=0

# compare runs
python3 tests/fish_drama3/compare_runs.py tests/fish_drama3/results/*

# quality gate
python3 tests/breeze_tts/quality_gate.py \
  --binary build-hip-gfx1101/bin/audiocpp_cli \
  --asr-model models/Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf --backend cpu \
  tests/fish_drama3/results/run
```

The harness records the git commit, binary hash, model size/mtime, GPU
architecture, the exact session options and the per-request stage breakdown in
`benchmark_manifest.json`. Two caveats for anyone repeating this:

* **Page cache matters.** The GGUF is 18.9 GiB and the machine has 32 GB. A run
  with a cold page cache reads the expert bank at ~1.7 GB/s and spends 3–4 s
  more in the prefill than a warm one. Use the warmup request and compare only
  interleaved runs.
* **`rocprofv3` cannot attach to this binary** in the nix ROCm shell — it fails
  with a gflags static/dynamic clash, because `audiocpp_cli` links
  `libgflags.so` (sentencepiece) and the profiler's tracer links it statically.
  The measurements above therefore come from the framework's own scalars plus
  `expert_stage_ms()`, which is why §2.3 reports staging separately from compute
  instead of a kernel table.

---

## Appendix — file index

| File | Change |
|---|---|
| `src/community_models/fish_drama3/backbone.cpp` | adaptive prefill expert groups, per-group expert graphs, `start_row` prefill, state snapshot/restore, `expert_stage_ms()`, build/step instrumentation |
| `src/community_models/fish_drama3/backbone.h`, `tower.h`, `generator.h`, `prompt.h` | state snapshot types, `start_row` parameters, prefix-cache members |
| `src/community_models/fish_drama3/generator.cpp` | `hip_perf` default on HIP, reference-prefix cache, two-phase prefill, tower-time attribution |
| `src/community_models/fish_drama3/tower.cpp` | `start_row` prefill, tower state snapshot/restore |
| `src/community_models/fish_drama3/prompt.cpp` | reports the reference boundary |
| `src/framework/core/disk_expert_weight_store.{h,cpp}` | `stage()` slot window |
| `src/community_models/fish_drama3/loader.cpp`, `model_specs/fish_drama3.json` | new option |
| `tests/fish_drama3/` | benchmark harness, comparison table, case catalog |
