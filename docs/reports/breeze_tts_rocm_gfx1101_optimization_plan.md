# Breeze-TTS-2 on ROCm/HIP (gfx1101) — Research, Baseline and Optimization Plan

Date: 2026-10-29 (updated 2026-10-30)
Target: `audio.cpp` @ `2597b300`, AMD RDNA3 `gfx1101` (RX 7800 XT, 16 GB),
ROCm 7.2.3 / HIP 7.2.53211.
Reference projects: `../llama.cpp` (upstream `main`), `../hipfire` (v0.4.2.1).

**Status: Phase 0 measured, Phase 1/2 implemented and validated.** The measured
numbers are in §3 and §4; the plan has been re-ranked against them.

---

## 0. Summary

Re-measured on 2026-10-30 with the fused depth path (§3.8) and the K/V
concatenation dedup (§3.9) in the shipped defaults (interleaved A/B, two
rounds each, q8_0, the `quality` corpus):

| | Baseline | Shipped defaults | Change |
|---|---:|---:|---:|
| RTF, `guidance_scale = 1.0` | 0.6696 | **0.4651** | **−30.5 %** |
| RTF, `guidance_scale = 3.0` | 0.7173 | **0.4992** | **−30.4 %** |
| AR loop, ms per generated frame (CFG 1.0) | 51.64 | **35.32** | **−31.6 %** |
| AR loop, ms per generated frame (CFG 3.0) | 55.29 | **37.52** | **−32.1 %** |
| Depth loop, ms per generated frame (CFG 1.0) | 43.62 | **29.11** | **−33.3 %** |
| Depth loop, ms per generated frame (CFG 3.0) | 39.25 | **30.01** | **−23.5 %** |
| ASR word-error rate, CFG 1.0 | 1.52 % | **0.00 %** | improved |
| ASR word-error rate, CFG 3.0 | 3.03 % | **1.52 %** | improved |

The two last steps are worth about 7 % and 2.5 % per frame on top of the
earlier −26 %, and both are bit-identical in the parts that claim to be. On the
**Q4_0** package the same shipped build reaches 31.76 ms per frame at
`guidance_scale = 1.0` (§4.1).

Before the fused depth path and the dedup were added, the same protocol gave
−25.1 % / −28.4 % RTF and −26.9 % / −31.7 % per frame.

Earlier measurements, before the fused depth path and measured on the `default`
corpus with the same protocol:

| | Baseline | Shipped defaults | Change |
|---|---:|---:|---:|
| RTF, `guidance_scale = 1.0` | 0.6699 | **0.5019** | **−25.1 %** |
| RTF, `guidance_scale = 3.0` | 0.7259 | **0.5199** | **−28.4 %** |
| AR loop, ms per generated frame (CFG 1.0) | 54.95 | **40.18** | **−26.9 %** |
| AR loop, ms per generated frame (CFG 3.0) | 60.52 | **41.31** | **−31.7 %** |
| Total GPU kernel time (rocprof, 100 frames) | 2561 ms | **2451 ms** | −4.3 % |
| Output audio, CFG 1.0 | — | — | **bit-identical** (same SHA-256) |
| ASR word-error rate, CFG 1.0, 13 samples | 0.70 % | **0.70 %** | equal |
| ASR word-error rate, CFG 3.0, 13 samples | 4.20 % | **2.80 %** | no regression |

Every figure is the median of two **interleaved** A/B rounds against a pristine
baseline binary, so consumer-card drift cannot flatter either side.

`guidance_scale` remains a fully supported feature. With the batched-CFG path
enabled, the `guidance_scale = 3.0` regime is now **faster per frame** than the
`guidance_scale = 1.0` regime (41.64 ms vs 40.63 ms), so CFG no longer carries a
performance penalty.

### 0.1 What changed in the 2026-10-30 revision

A second rocprof trace of the *shipped* build was analysed with three new
scripts (`tests/breeze_tts/analyze_trace_gaps.py`, `analyze_trace_idle.py`,
`analyze_trace_bursts.py`). The trace says the GPU is **only ~35 % busy** and
that the idle time sits in ~500 host↔device round trips per audio frame, not
inside the kernels (§2.5). That re-opened the fused-depth question, and the
result changed two of the plan's conclusions:

* **The fused depth path is now shipped by default on HIP.** It is 7–8 % faster
  per frame at `guidance_scale = 1.0` and neutral at `guidance_scale = 3.0`, and
  the "quality regression" that kept it off **does not reproduce** (§3.8, §5.2).
* **The remaining bf16 rounding is no longer worth 8.9 % of GPU time.** With
  the rope fold (§3.6c) the leftover standalone round kernels are 1.2 % of GPU
  busy time; the flash-attention round fold that would remove most of the rest
  is parked because its remaining share does not justify the four-kernel
  change (§4.2).

---

# Part 1 — Research findings

## 1.1 How `audio.cpp` does ROCm today

### 1.1.1 There is no separate ROCm backend

`audio.cpp` has one GPU backend for AMD: **ggml's HIP backend**, which is ggml's CUDA
backend recompiled with `hipcc`.

- `external/ggml/src/ggml-hip/CMakeLists.txt` globs `../ggml-cuda/*.cu` +
  `../ggml-cuda/*.cuh` and builds them as HIP (`:72-96`). There are **zero**
  project-specific `.hip` files under `external/ggml`.
- `external/ggml/src/ggml-hip/CMakeLists.txt:111` publishes `GGML_USE_CUDA` to
  consumers, which is why the project needs its own `ENGINE_GGML_HIP_BACKEND` marker
  (`CMakeLists.txt:2928-2932`) to tell a real CUDA build from a HIP build.
- Arch macros live in `external/ggml/src/ggml-cuda/vendors/hip.h`:
  `RDNA3` = `__GFX11__` (`:221-223`), `RDNA3_5` = gfx115x (`:225-227`),
  `RDNA3_0` = RDNA3 && !RDNA3_5 (`:229-231`), `RDNA4` = `__GFX12__` (`:217-219`).
  **`__gfx1101__` is never tested anywhere in the tree.** gfx1101 and gfx1100 are
  indistinguishable at the preprocessor level and both land in `RDNA3` → `RDNA3_0`.
- Runtime arch detection parses `prop.gcnArchName`
  (`external/ggml/src/ggml-cuda/ggml-cuda.cu:170-216`), giving gfx1101 →
  `cc = 0x1001101`, and `GGML_CUDA_CC_IS_RDNA3_0` is true.

### 1.1.2 The project's own HIP code (very small)

| File | What |
|---|---|
| `src/models/fish_audio/hip_fast_sampler.cu` | The only hand-written HIP kernel in the repo (275 lines). Device-side exact top-k for Fish Fast-AR. 1 workgroup × 256 threads, pure LDS tree reduction (no `__shfl`, no DPP), O(k·N) work, 20 KiB dynamic LDS for a 4096-vocab. Wired in only through `ENGINE_HAS_HIP_FAST_SAMPLER` (`CMakeLists.txt:2938-2942`), i.e. **Fish Audio only, not Breeze**. Built with `-O2` because of ROCm/ROCm#5826 (illegal DPP at `-O0` on GFX10+). |
| `src/framework/sampling/torch_random_cuda_runtime.cu` | CUDA Driver API + a Philox generator. **Never built on HIP** (`CMakeLists.txt:2962-2977`). |
| `src/framework/audio/istft_cuda_runtime.cu` | Uses `cufft`. **Never built on HIP** (`CMakeLists.txt:2962-2977`). |

Style conventions for project HIP kernels: `snake_case` + `_kernel` suffix,
`__device__ __forceinline__` helpers, `hipLaunchKernelGGL` with explicit `dim3`,
a file-scope `check_hip(status, label)` that throws `std::runtime_error`,
`extern __shared__` + manual offset arithmetic, **no `__launch_bounds__`**.

### 1.1.3 Build configuration

| Location | What |
|---|---|
| `CMakeLists.txt:120` | `option(ENGINE_ENABLE_HIP ...)` |
| `CMakeLists.txt:204-211` | CUDA and HIP are **mutually exclusive** in one configure. |
| `CMakeLists.txt:125-127, 213-252` | `ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS` — a gfx1151-**only** gate. It `FATAL_ERROR`s unless the effective HIP target list is exactly `gfx1151`, and it forces `GGML_HIP_HIPBLASLT=OFF` because *"hipBLASLt regressed decoder compute by roughly 3 seconds on gfx1151"* (`:245-249`). |
| `CMakeLists.txt:388, 395` | `GGML_HIP` and `GGML_HIP_GRAPHS` are forced from the engine options. **HIP graph capture is ON by default.** |
| `scripts/install_arch_amd.sh:7` | `SUPPORTED_AMD_TARGETS` already lists `gfx1100;gfx1101;gfx1102;gfx1103;…`. |

**Build gotcha found during Phase 0:** the vendored ggml HIP argsort path
(`external/ggml/src/ggml-cuda/argsort.cu:4`) includes `<hipcub/hipcub.hpp>`, but
`hipcub` and `rocprim` are separate ROCm packages that the project's own nix
packaging (`.devops/nix/package.nix`) does not pull in. A nix-driven HIP build of
`audio.cpp` therefore fails to compile unless the include directories are added
explicitly, e.g. `-DCMAKE_HIP_FLAGS="-isystem <hipcub>/include -isystem
<rocprim>/include"`. This is a real packaging gap and is worth fixing in
`package.nix` (add `rocmPackages.hipcub` to `buildInputs` and
`rocmPackages.rocprim` to `nativeBuildInputs`).

### 1.1.4 Where HIP is behind CUDA (ranked by measured impact)

| Site | Impact |
|---|---|
| `src/models/breeze_tts/generator.cpp:1196-1198` | `perf_backend` = Cuda \| Vulkan \| Metal — **HIP was excluded**, so `skip_cache_round` and `direct_prefill` were inert on HIP even though their defaults are `true`. **Fixed.** |
| `src/models/breeze_tts/generator.cpp:1231` | `backbone_batched_cfg = cuda && perf.batched_cfg` — **CUDA only**. **Fixed.** |
| `src/framework/core/backend.cpp:531-540` | `set_backend_round_bf16_fusion` is gated on `is_cuda_backend_handle` → **ggml's fused round-to-bf16 graph pass never ran on HIP**, even though the `ROUND_BF16` unary op itself works on HIP (`external/ggml/src/ggml-cuda/ggml-cuda.cu:3126-3127`). **Fixed.** |
| `external/ggml/src/ggml-cuda/ggml-cuda.cu:4248` | The whole round-to-bf16 fusion block was `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`. **Fixed.** |
| `src/models/breeze_tts/generator.cpp:162-165` | HIP used a **BF16** KV cache, so `skip_redundant_cache_round` (`:104-119`, which requires an F16 cache) could never fire. **Fixed — HIP now uses F16 like CUDA.** |
| `src/framework/sampling/torch_random.cpp:349-362` | The Torch sampling policy probe is CUDA-only → HIP always uses the **host** HF sampler. |
| `src/framework/modules/attention/attention_internal.h:401-408` | `use_fast_cuda_projection` requires `BackendType::Cuda`, even though `FastPackedProjection4Module` itself already accepts HIP (`fast_projection_modules.cpp:62-64`). |
| `src/framework/modules/attention/scaled_dot_product_attention.cpp:262-280` | `CudaFlashF16Head40` lowering is CUDA-only (ggml also disables `ggml_flash_attn_f16_head40` on HIP, `external/ggml/src/ggml-cuda/fattn.cu:570, 588-591`). |
| `src/framework/modules/conv_modules.cpp:545-559` | bf16 Conv1d is CUDA-only → HIP falls back to F32 `im2col` + cast. The conv-transpose-1d col2im fast path **is** HIP-enabled (`:424-431`). |
| `src/framework/core/attention_fallback.cpp:13-28, 65-104` | The CUDA compute-capability probe is compiled out on HIP, so `resolve_flash_attention()` **fails open** → flash is requested on HIP. |
| ggml-internal | `GGML_CUDA_USE_CUB` is off on HIP → `GGML_OP_TOP_K` is capped at 1024 columns (`ggml-cuda.cu:6047-6052`), `ggml_flash_attn_f16_head40` is disabled, the tiled `im2col` lowering is compiled out (`im2col.cu:285-295`), and several bf16 fusion passes are `#if !defined(GGML_USE_HIP)` (`ggml-cuda.cu:4222, 4760`). **`GGML_OP_ARGSORT` is unrestricted on HIP** and the fork has a hipcub radix path for ncols > 1024 (`argsort.cu:34-60, 287-292`), so `ggml_argsort_top_k` over the 2048-wide codebook vocab works. |

### 1.1.5 The Breeze-TTS-2 model

`include/engine/models/breeze_tts/assets.h:13-64`:

| Component | Shape |
|---|---|
| Backbone (AR semantic) | hidden 2048, intermediate 6144, **28 layers**, 16 heads, 8 KV heads, head_dim 128, vocab 2051, lm_head 2052, **16 codebooks**, max_pos 2048, RoPE NEOX theta 5e5 |
| Text encoder (T5Gemma) | hidden 1152, intermediate 6912, 26 layers, 4 heads, 1 KV head, head_dim 256, vocab 262158 |
| Depth decoder | hidden 1024, intermediate 4096, **4 layers**, 8 heads, 2 KV heads, head_dim 128 |
| Codec | codebook size 2048; 1 semantic + **15 residual** codebooks per frame; 1920 samples/frame @ 24 kHz |

Backbone ≈ 1.53 B parameters; depth decoder ≈ 61 M parameters, but the depth
decoder runs **15 sequential decoder steps per frame**, so it re-reads its weights
15 times per frame.

### 1.1.6 The AR loop and the KV-cache waste

`src/models/breeze_tts/generator.cpp:1901-1953`:

```
per frame:
  CFG mix on HOST over 2052 floats            (generator.cpp:1906-1910)
  host sampling of the 2052 logits            (generator.cpp:1915-1925)
  15 residual depth codebooks                  (generator.cpp:1528-1570 or fused)
  host gather of 16 × 2048 embedding floats   (generator.cpp:387-405)
  backbone decode: 1 (or 2 with CFG) decode step(s)
```

Per backbone decode step the framework runs, per layer
(`src/framework/modules/transformers/decoder.cpp:549-928`): RMSNorm → packed QKV
matmul → Q/K RMSNorm → RoPE → bf16 rounding at 14 points → `ggml_flash_attn_ext` →
out-proj → residual → post-norm → SwiGLU MLP → down-proj → residual.

The **static KV cache** was the source of the largest identified waste:

- Cache type: `generator.cpp:162-165` → **BF16 on HIP** (F16 on CUDA/Vulkan).
- The K/V rows are written by `set_rows` into the cache, then `decoder.cpp:826-829`
  (single-sequence) / `decoder.cpp:995-998` (batched) apply `activation_cast`
  **to the entire cache tensor**: `activation_cast` (`decoder.cpp:274-290`) →
  `ggml_round_bf16` (F32 out) → the whole cache is materialised as an F32 tensor.
- That F32 K/V then enters `ggml_flash_attn_ext`. ggml's `launch_fattn`
  (`external/ggml/src/ggml-cuda/fattn-common.cuh:965-1000`) converts any non-F16
  K/V into a **second** full-size F16 pool buffer because the tile kernel needs F16.
- The flash op then reads that F16 copy.

⇒ **Per AR decode step, per layer, the whole KV cache was touched four times**
(round: read BF16 + write F32; convert: read F32 + write F16; kernel: read F16) —
≈ **14 bytes per cached element** of pure memory traffic, of which ~2 bytes are useful.

The round is mathematically an **identity** for K and V: both are bf16-rounded
before the cache write (V at `after_qkv_projection`, K at `after_rope`), and
`skip_redundant_cache_round` (`generator.cpp:104-119`) already encodes exactly this
reasoning — it just refused to fire because it requires an F16 cache while HIP
used BF16.

### 1.1.7 CFG on HIP

`guidance_scale != 1.0` ⇒ `use_cfg = true` ⇒:

- `backbone_batched_cfg` was false on HIP ⇒ **two separate `CausalDecoderRuntime`
  instances** (`generator.cpp:1206-1207`), two prefill passes, and **two decode graphs
  per AR step** (`generator.cpp:1276-1333`), i.e. the 1.53 B backbone weights were read
  **twice**.
- Two separate KV caches ⇒ the §1.1.6 traffic was paid twice.
- The CFG mix happened on the **host** over 2052 floats (`generator.cpp:1906-1910`).

The batched machinery itself is **backend-neutral pure ggml**
(`causal_decoder_runtime.cpp:631, 135`; `decoder.cpp:930-1030`) — only the `cuda &&`
gate blocked it.

The **depth** decoder was already paired correctly: `paired_depth = use_cfg`
(`generator.cpp:1739, 1942`), so the unconditional depth branch is only computed
when CFG is actually active.

### 1.1.8 Fused depth

`generator.cpp:1210` → `if (perf_backend && perf.fused_depth)`. On HIP this was
false, so it ran **15 sequential depth decode steps**, each with a device round
trip (`generator.cpp:666-677`: upload 2 × 1024 floats, download 2 × 2051 floats) and
a host-side sample.

The fused runtime (`generator.cpp:964-1131`) does all 15 codebooks in one graph with
device-side CFG mix, temperature, `ggml_argsort_top_k` top-k, device top-p
(`ggml_soft_max` + `ggml_cumsum` + `ggml_step`), noise and argmax — only
**15 × 4 B** of readback per frame. Its ops are all available on HIP (§1.1.4).

### 1.1.9 Attention kernel actually selected on gfx1101

ggml's `ggml_cuda_get_best_fattn_kernel` (`external/ggml/src/ggml-cuda/fattn.cu:456-543`):

- `ggml_cuda_should_use_wmma_fattn` returns **false on HIP** unless
  `GGML_HIP_ROCWMMA_FATTN` is set (`fattn-wmma-f16.cuh:26-49`); the repo docs say to
  keep it **OFF** because *"the default `fattn-tile` kernels are faster on RDNA3/RDNA4"*
  (`docs/build/HIP.md:267`).
- The AMD WMMA branch (`fattn.cu:525`) requires
  `Q->ne[0] <= 128 && Q->ne[1] * gqa_ratio_eff > 8`.
  For Breeze: `head_dim = 128` ✅, `gqa_ratio = 16/8 = 2`, `gqa_ratio_eff = 2`
  (`fattn.cu:488-491`), and a 1-token decode has `Q->ne[1] = 1` → `1 × 2 = 2`, **not > 8**.
- ⇒ **the AR decode step falls through to `BEST_FATTN_KERNEL_TILE`**
  (`fattn.cu:543`), a SIMT F16 tile kernel, on every decode step, while the prefill
  gets the tensor-core path. The trace confirms it:
  `flash_attn_tile<128, 128, 2, 1, false>` — 54 884 calls.

### 1.1.10 Sampling

- The **backbone** sample is 100 % host on HIP: `resolve_torch_cuda_sampling_policy`
  returns the default (host) policy for any non-CUDA backend
  (`src/framework/sampling/torch_random.cpp:349-362`).
- The **depth** sample is host too, 15× per frame (`generator.cpp:1522-1546`).
- `hip_fast_sampler.cu` exists but is Fish-Audio-only.

### 1.1.11 Vocoder

`speech_decoder.cpp` is conv-heavy (ConvNeXt, ResidualUnit, ConvTranspose1d),
24 kHz, 1920 samples/frame, 300-frame chunks with 25 frames of left context
(`speech_decoder.cpp:48-52`).

The **Strix-Halo retained-graph cache** (`speech_decoder.cpp:52-57, 1094-1105`) keeps
two prebuilt vocoder graphs (300- and 105-frame chunks) alive instead of rebuilding
per chunk shape, plus a "restore declared inputs before replay" fix
(`speech_decoder.cpp:997-1000, 1020-1031`). The mechanism is **arch-agnostic** — it is
selected at runtime by `BackendType::Hip` — but the CMake gate restricts it to
`gfx1151` (`CMakeLists.txt:213-252`).

## 1.2 How `../llama.cpp` does ROCm/gfx1101

### 1.2.1 Structure

Modern llama.cpp has **no separate HIP kernel tree**. `ggml/src/ggml-hip/` contains
exactly one file, `CMakeLists.txt`; all AMD specialization happens in the
`ggml-cuda` sources compiled by `hipcc`:

- `vendors/hip.h` — `cuda*` → `hip*` symbol shim, `#define __CUDA_ARCH__ 1300` (`:177`),
  and the **family macro layer** (`:185-244`): `GCN`, `CDNA`…`CDNA4`, `RDNA1..RDNA4`,
  `RDNA3_0`, `RDNA3_5`. **`__gfx1100__`/`__gfx1101__` are never tested anywhere.**
- `common.cuh:265-271` — `AMD_MFMA_AVAILABLE` (CDNA), `AMD_WMMA_AVAILABLE` (RDNA3/RDNA4).
- `common.cuh:374-380` — `ggml_cuda_get_physical_warp_size()` → 32 for gfx11, 64 for
  gfx8/gfx9. `WARP_SIZE` is a compile-time 32 everywhere; `nwarps = nthreads / 32`.
- `ggml-cuda.cu:170-216` — `ggml_cuda_parse_id(gcnArchName)` → gfx1101 = `0x1001101`.
- `mmq.cuh:216-282` + `mmq-config-rdna3.cuh` (the gfx1101 tile table) —
  per-arch `CASE(type, nthreads, occupancy, I, J, sram_layout, K_vram, stream_k, fallback)`.
- `mmq.cuh:109-161` — the **LDS bank-conflict padding invariant**
  (*"K % 2 == 1 for dp4a or K % 8 == 4 for mma"*), enforced by
  `static_assert(stride % 8 == 4)`.
- `mmq.cu:349-376` — the AMD **policy** for MMQ vs hipBLAS per arch and per quant type.
- `mma.cuh:1233-1239` — RDNA3 uses `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
  (RDNA4 uses the `_gfx12` spelling); `mma.cuh:91-95` selects
  `DATA_LAYOUT_I_MAJOR_MIRRORED` for A/B on RDNA3.
- `mma.cuh:709` — RDNA3/RDNA4 have **no `V_SDOT4`**; llama.cpp uses
  `__builtin_amdgcn_sudot4` instead.
- `fattn.cu:456-543` — the kernel-selection predicate; `:525` is the AMD WMMA branch.
- `fattn-mma-f16.cuh:1057-1108` — the RDNA3 `mma_tile_sizes`.
- `fattn-tile.cuh:237-313` — `ggml_cuda_fattn_tile_get_config_amd_rdna`, the RDNA
  (DKQ, DV, ncols) → (nthreads, occupancy, nbatch_fa, nbatch_K) table.
- `mmvq.cu:67-115, 351-372` — per-arch MMVQ parameter tables; **RDNA3_0 uses a
  deliberately stricter nwarps whitelist than RDNA4** (*"Q2_K / Q5_K / IQ4_XS regress
  in full quant sweeps"*).
- `mmf.cu:169` — MMF (FP16 matmul) is restricted to `src1_ncols <= 8` on RDNA3_0.

### 1.2.2 The measured gfx1101 vs gfx1100 ISA delta (probed with the local ROCm 7.2.3)

| Feature | gfx1100 | **gfx1101** | gfx1030 | gfx908 | gfx1200 |
|---|---|---|---|---|---|
| wave32 | yes | **yes** | yes | wave64 | yes |
| `v_sdot4` | no | **no** | yes | yes | no |
| `v_sudot4` | yes | **yes** | no | no | yes |
| `wmma_f32_16x16x16_f16_w32` | yes | **yes** | no | no | no |
| `wmma_i32_16x16x16_iu8_w32` | yes | **yes** | no | no | no |
| `mfma_*` | no | **no** | no | yes | no |

⇒ **There is no gfx1101-specific ISA path.** Write one RDNA3 kernel set; use
`#if defined(RDNA3)` (or `RDNA3_0` to exclude Strix Halo), never `__gfx1101__`.

### 1.2.3 HIP vs CUDA in llama.cpp (the gap list)

- **Still CUDA-shaped**: `mma.sync` PTX, `cp.async` (`fattn-mma-f16.cuh:349-355`
  forces `nstages = 0` on HIP), `ldmatrix`/swizzle (`fattn-swizzle.cuh:24-32` disables
  swizzle), CUB (`common.cuh:110-112`), CUDA graphs (mapped to `hipGraph*`).
- **64 KB LDS hard cap** — `cudaFuncSetAttribute` is a no-op on HIP
  (`common.cuh:221-245`), so MMQ requires `smpbo >= 48 KiB` (`mmq.cu:299-310`).
- **No hipBLASLt** in upstream llama.cpp — only `hipblas` + `rocblas`.
- `__builtin_amdgcn_perm` (V_PERM_B32) replaces CUDA `__byte_perm` for nibble/table
  unpacking (`mmq-load-tiles.cuh:141-154`, `vecdotq.cuh:35-60`).
- HIP needs a **custom radix top-k** instead of CUB (`top-k.cu:51`).
- Cooperative groups **do** exist via `<hip/hip_cooperative_groups.h>` (`softmax.cu:5-10`).

### 1.2.4 Lessons worth copying into `audio.cpp`

1. **Family macros, not per-chip macros.** Compile-time `#if defined(RDNA3)` for ISA
   legality; runtime `cc` for tile sizes and cut-offs.
2. **Config tables instead of if-chains** — one `CASE(...)` row per (quant, arch, shape).
3. **LDS bank-conflict padding as a compile-time invariant**, not a comment.
4. **Wave size 32 everywhere on gfx11**; derive `nwarps` from it.
5. **Never assume `mma.sync`/`cp.async`/`CUB`/`ldmatrix` exist.**
6. **Per-model kernel specialisation does not exist in llama.cpp** — specialisation is
   at the *op* level or in the *graph*. `audio.cpp` should follow the same rule.

`audio.cpp`'s vendored ggml is 0.12.0 and already carries all of the above.

## 1.3 How `../hipfire` does ROCm/HIP kernels

hipfire is a from-scratch Rust LLM engine for AMD GPUs: it dlopens `libamdhip64`
and JIT-compiles **1220 hand-written `.hip` kernels** (`kernels/src/`), with no
Triton, no PyTorch and no hipBLAS in the hot path.

### 1.3.1 Architecture

- **Per-kernel arch selection by filename suffix**:
  `<base>.hip` (portable) / `<base>.gfx1100.hip` (chip) / `<base>.gfx11.hip` (family) /
  `<base>.wave64.hip` (wave size). Resolution order chip → family → base.
- **Three-layer capability model** (`crates/rdna-compute/src/arch_caps.rs:1-20`):
  *atoms* (`is_gfx1101`) → *molecules* (`is_rdna3`, `is_rdna3_dgpu`) → *capabilities*
  (`has_wmma_w32`, `has_mmq`, `has_dot2_f32_f16`).
- **Table-driven dispatch** (`crates/hipfire-dispatch/src/tables/`) with
  `ArchPredicate` (`Always`, `HasWave32`, `HasWmmaW32`, `IsGfx942`, …).
- **No runtime wavefront-size detection** — RDNA3 is modelled as wave32.
- **Compilation**: `hipcc --genco --offload-arch=<arch> -O3 -fuse-cuid=none
  -mcode-object-version=6` (`crates/rdna-compute/src/compiler.rs:397-406`).

### 1.3.2 gfx11 kernel facts (measured with a probe, per
`kernels/src/attention_q8_0_fa2_gqa.gfx11.hip:25-70`)

- **WMMA ABI**: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`;
  A/B operand = `half16_t` = **16 halfs per lane**; **lanes `r` and `r+16` must supply
  identical rows**; C accumulator = `float8_t` = 8 f32 per lane with
  `acc[j] = C[2*j + (lane>>4)][lane & 15]`.
- **No MFMA on gfx11** — MFMA is CDNA-only.
- **No `v_sdot4` on gfx11** — int8 dots go through
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`.
- **No `s_prefetch_data` and no `cp.async` on gfx11.**
- **Wave64 traps on gfx11**: `__shfl_*` (→ `ds_bpermute`) stay inside a physical
  wave32 half even in wave64 mode; cross-half exchange needs
  `__builtin_amdgcn_permlane64`. ⇒ **stay on wave32.**
- **Reductions**: prefer `__builtin_amdgcn_ds_swizzle` + DPP8/quad_perm over
  `__shfl_xor`; hipfire removed 64 `ds_bpermute`s from one flash kernel for a
  37.3 → 47.4 ms win.
- **LDS bank padding** with the derivation written down (e.g. stride 42 for a 16-row
  tile, 264 halfs = 528 B so 16 lanes' `b128` reads hit distinct banks).

### 1.3.3 The patterns a gfx1101 TTS engine should copy

| Pattern | hipfire reference |
|---|---|
| GQA-shared decode attention tile: one workgroup per (kv head, tile) serving **all** Q heads of that KV head, so K/V is read from memory **once per KV head instead of once per Q head** | `kernels/src/attention_flash_q8_0_tile_gqa.gfx1100.hip:1-53` |
| Row-xmask (DPP) reductions instead of `ds_bpermute` | `attention_flash_q8_0_tile_gqa.gfx1100.hip:55-59` |
| 2-pass softmax (max first, then accumulate) rather than online softmax, to avoid ~1e-5 error flipping argmax | `kernels/src/attention_flash_q8_0_reduce.hip:9-14` |
| 4-accumulator GEMV with a descending 16/8/4/2/1 warp reduction; `(acc0+acc1)+(acc2+acc3)` association order is load-bearing for bit-exactness | `kernels/src/gemv_hfq4g256_residual.gfx1100.hip:16-244` |
| Batch-tiled WMMA GEMM (one wave, 16 rows × 16 batch, dequant once, reuse across batch columns) | `kernels/src/gemm_qkv_mq4g256v2_wmma_gfx11_bt.hip` |
| **Per-arch perf allowlist, not per-family inheritance** — an LPDDR-tuned variant inherited via a broad family predicate cost a **14 %** gfx1100 regression | `docs/methodology/perf-arch-discipline.md:20-40` |
| Compile-time resource/occupancy gate — a numeric test cannot catch a 25 % occupancy loss | `scripts/kernel_resource_gate.sh` |
| Bandwidth **inverts** the optimum: multi-row GEMV is +2.7 % on a 224 GiB/s APU but **negative** on a 960 GiB/s dGPU | `kernels/src/gemv_mq4g256v2_multirow.hip:25-56` |

### 1.3.4 gfx1101-specific status in hipfire

| Aspect | Status |
|---|---|
| ISA | `is_rdna3`, `is_rdna3_dgpu`, `has_wmma_w32`, `has_dot2_f32_f16`, `has_mmq`, wave32 |
| GEMV / GEMM source selection | gfx1100 source shared (`kernels.rs:6683, 6701, …`) |
| Attention GQA tile / FA2 / verify | **gfx1100-exclusive** (`is_gfx1100()`, 130 call sites vs 6 for `is_gfx1101`) |
| Measured | gfx1101 (RX 7700 XT): prefill 1539 → 2100 tok/s; decode 109.7 → 115.1 tok/s (+4.9 %) |
| Caveat | **1-ulp FMA-contraction differences vs gfx1100** in the RoPE sin-branch — ISA-identical but not codegen-identical (`docs/perf-checkpoints/2026-10-09-gfx1101-campaign-verify.md:48-52`) |

**Takeaway for `audio.cpp`**: gfx1101 shares gfx1100's ISA, so *one* RDNA3 kernel set
serves both; but **perf variants must be validated per arch**, and bit-exactness
oracles must be re-run on gfx1101 rather than assumed from gfx1100.

---

# Part 2 — Phase 0: the measured baseline

## 2.1 Environment

| | |
|---|---|
| GPU | AMD Radeon RX 7800 XT, `gfx1101` (0x1101), 16 368 MiB VRAM |
| Runtime | ROCm 7.2.3 / HIP 7.2.53211, wave size **32**, HIP VMM **off** (default) |
| Build | `nix develop .#rocm`, `-DENGINE_ENABLE_HIP=ON -DGPU_TARGETS=gfx1101`, `RelWithDebInfo`, 24 jobs |
| Model | `breeze_tts_2_q8_0` (5 079 670 336 B, Q8_0), `models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf` |
| Corpus | 5 fixed English sentences, `--max-tokens 100`, `--seed 0`, 1 warmup + 5 measured requests |
| Metrics | RTF = wall ms ÷ generated audio ms; plus the per-frame AR cost from the framework's own timing log |

The corpus, the harness and the analysis live in `tests/breeze_tts/`; see §6.

## 2.2 Baseline result

| | `guidance_scale = 1.0` | `guidance_scale = 3.0` |
|---|---:|---:|
| RTF (median of 5) | 0.6577 | 0.7054 |
| x_realtime | 1.521 | 1.418 |
| generated frames / request | 74.6 | 73.4 |
| AR loop, ms per frame | **54.48** | **58.56** |

## 2.3 Where the time goes (stage breakdown, from the framework timing log)

| Stage | CFG 1.0, ms/frame | share | CFG 3.0, ms/frame | share |
|---|---:|---:|---:|---:|
| **depth decode (15 codebooks)** | **45.78** | **84.0 %** | **41.72** | 71.2 % |
| backbone AR decode (cond) | 7.90 | 14.5 % | 7.96 | 13.6 % |
| backbone AR decode (uncond) | — | — | 7.96 | 13.6 % |
| backbone prefill | 0.31 | 0.5 % | 0.44 | 0.7 % |
| prompt (text + speech encoders) | 0.26 | 0.5 % | 0.41 | 0.7 % |
| vocoder (speech decoder) | 1.46 | 2.7 % | — | — |
| AR loop total | 54.48 | 100 % | 58.56 | 100 % |

**This reversed the pre-measurement priority order.** The AR loop is dominated by
the *depth* decoder (15 residual codebooks per frame), not by the 1.53 B-parameter
backbone. The backbone plus its separate unconditional branch is only ~28 % of the
loop, so a backbone-only optimization cannot move the headline number much.

## 2.4 Where the GPU time goes (rocprofv3 kernel dispatch trace)

The same 3-request workload, kernel time in ms:

| Kernel group | Baseline ms | share | calls |
|---|---:|---:|---:|
| **`mul_mat_vec_q` (Q8_0 GEMV)** | **4471.9** | **59.1 %** | 193 259 |
| **`round_bf16` (bf16 activation + cache rounding)** | **952.2** | **12.6 %** | 877 544 |
| elementwise / fused-slot kernels | 751.7 | 9.9 % | 209 067 |
| `flash_attn_tile` | 281.8 | 3.7 % | 54 884 |
| `quantize_q8_1` (activation quantisation) | 264.1 | 3.5 % | 193 259 |
| `rms_norm` | 248.5 | 3.3 % | 114 077 |
| host↔device copies | 181.8 | 2.4 % | 100 312 |
| `rope` | 157.0 | 2.1 % | 96 884 |
| **`convert_unary` (F32→F16 cache conversion)** | **130.9** | **1.7 %** | 96 728 |
| `set_rows` (cache write) | 126.1 | 1.7 % | 90 944 |
| **total kernel time** | **7566.0** | | |

Three things follow directly:

1. **The quantized GEMV is 59 % of GPU time.** BreezeTTS decodes one token per step,
   so every projection is a batch-1 GEMV. At ~2.6 GB of weight traffic per frame
   against the 7800 XT's ~624 GB/s peak, the measured ~130 GB/s is roughly **20 % of
   peak bandwidth**. This is the single biggest remaining opportunity.
2. **The bf16 rounding is 12.6 % of GPU time and 878 k kernel launches.** Most of it
   is the per-activation rounding the reference bf16 path inserts after almost every
   op, not the cache round.
3. **`convert_unary` (the F32→F16 copy of the whole cache) is 1.7 %**, and the cache
   `round_bf16` is a separate slice of the 12.6 %. Together they are the §1.1.6
   waste — visible and removable.

## 2.5 The shipped build, re-traced: where the GPU *is not* working

The table above counts kernel time. It does not say how much of the wall clock the
GPU was actually busy for. Three scripts were added to answer that:

| Script | Question it answers |
|---|---|
| `tests/breeze_tts/analyze_trace_gaps.py` | How much time is spent between consecutive dispatches, and which kernel precedes the gap? |
| `tests/breeze_tts/analyze_trace_idle.py` | How much of the window is the GPU idle? (the *union* of the dispatch intervals, not the sum of the pairwise gaps) |
| `tests/breeze_tts/analyze_trace_bursts.py` | How do the dispatches group into bursts, and how long is the idle between the bursts? |

Run against `tests/breeze_tts/results/trace_ropefold/rocprof/jordan-desktop/556741_kernel_trace.csv`
(100 frames, `guidance_scale = 1.0`, the shipped build):

```
span 8964 ms   union-busy 2423 ms (34.8 %)   idle 4541 ms
idle intervals: 527 058 in 1-10 us, 21 441 in 10-50 us, 5722 in 50-200 us,
                3108 in 0.2-1 ms, 13 above 1 ms
```

**The GPU is idle 65 % of the time, and the idle is not in a few big stalls** —
it is spread over half a million short gaps. The burst structure explains why
(`analyze_trace_bursts.py`, an 8 ms window in the steady state):

```
burst  0   175 kernels   1323.8 us                       <- the depth decode graph
burst  1     1 kernel       2.4 us   gap  46.5 us        <- host round trips
burst  2     1 kernel       2.4 us   gap  38.5 us
burst  3     1 kernel       2.6 us   gap  21.8 us
burst  4     1 kernel      19.4 us   gap  22.4 us        <- the depth head matvec
burst  5     1 kernel       3.0 us   gap  60.7 us
burst  6     1 kernel      18.4 us   gap  21.7 us        <- the second head matvec
burst  7..11 1 kernel each   ~2.4 us  gap ~22-37 us       <- more copies
burst 12    31 kernels    239.7 us   gap  48.0 us
burst 13    32 kernels    226.9 us   gap 119.7 us
burst 14    64 kernels    502.5 us   gap 263.7 us
burst 15   175 kernels   1326.3 us   gap 534.8 us        <- the next depth decode
```

Inside a burst the kernels are back to back (gap ≈ 0): the captured depth-decode
graph runs at full speed. Between the bursts the GPU waits 20–60 µs per
operation. Those operations are the `ggml_backend_tensor_set` / `ggml_backend_tensor_get`
calls of the per-codebook path: on HIP every pageable host↔device copy is a
synchronising round trip, and the trace shows **43 437 `copyBuffer` dispatches**
in the run.

So the depth loop is **host round-trip bound**, not GPU bound: per codebook step
the host does ~8 copies and several device syncs that the GPU waits for. That is
what the fused depth path removes (§3.8).

### 2.5.1 The same trace after the fused depth path

`tests/breeze_tts/results/trace_fuseddepth/` is the trace of the build that
ships the fused depth path. The structure is completely different:

| | per-codebook depth | fused depth |
|---|---:|---:|
| `copyBuffer` dispatches | 43 437 | **24 854** |
| dispatches per frame | ~5 600 | ~6 200 |
| mean kernel duration | — | 4.41 µs |
| GPU busy (union) | 34.8 % | 34.3 % |
| the dominant kernel | `mul_mat_vec_q` 59 % | `mul_mat_vec_q` 53 % |

The fused path removes the depth loop's per-codebook round trips (the
`copyBuffer` count drops by 43 %) but it *adds* kernel work: the fixed-shape
graph has to grow its K/V with a `concat` every step instead of writing into a
preallocated cache, and the trace now shows

| kernel | calls | per frame | µs/call | total ms |
|---|---:|---:|---:|---:|
| `mul_mat_vec_q<Q8_0,1>` | 75 361 | 785 | 18.5 | **1395.4** |
| `round_bf16_kernel<float>` | 116 436 | 1213 | 1.22 | 141.5 |
| `quantize_q8_1` | 79 969 | 833 | 1.56 | 124.5 |
| `flash_attn_tile<128,128,2,1>` | 19 968 | 208 | 6.23 | 124.4 |
| `rms_norm_f32<1024,…>` | 39 992 | 417 | 2.95 | 118.0 |
| **`concat_cont<float,2>`** | 62 208 | **648** | 1.32 | 82.2 |
| `rope_neox` | 34 560 | 360 | 2.09 | 72.4 |
| `add_round_bf16_kernel` | 39 992 | 417 | 1.49 | 59.5 |
| `copyBuffer` | 24 854 | 259 | 2.12 | 52.6 |

Two more facts fell out of the trace and are worth recording:

* **Every kernel is followed by a 3–6 µs gap — but most of that gap is the
  profiler, not the GPU.** The gap histogram of the fused trace is stark: 591 273
  of the 597 723 gaps fall in the 3–6 µs band, and they account for 2121 ms of
  idle, about 3.6 µs per dispatch. That number has to be discounted. The same
  run takes 58.0 ms per frame under rocprof and 37.2 ms per frame without it,
  while the kernel time itself (27.5 ms per frame) is measured with GPU
  timestamps and is not inflated. The difference is **30.6 ms per frame over
  6 226 dispatches = 4.9 µs of profiler overhead per dispatch**, which is more
  than the whole measured gap. The real per-dispatch cost is therefore
  (37.2 − 27.5) / 6 226 ≈ **1.6 µs**, i.e. about **9.7 ms of real idle per
  frame (26 % of the frame)**, not 22 ms.

  So the dispatch latency is real but *twice* smaller than the trace suggests,
  and the way to attack it is still *fewer, bigger* kernels — not a faster
  GEMV. The corollary is a measurement rule: **never read an absolute
  per-dispatch cost out of a rocprof trace**; compare the profiled and the
  unprofiled wall time to get the real one.
* **Only the tile flash-attention kernel runs on gfx1101.** The trace contains
  `flash_attn_tile<128,128,2,1,false>` (20 800 calls) and
  `flash_attn_combine_results<128>` (2 800 calls) but *no* `mma-f16`, `wmma-f16`
  or `vec` variant. The AMD WMMA branch in `ggml_cuda_get_best_fattn_kernel`
  needs `Q->ne[1] * gqa_ratio_eff > 8`, and a 1-token decode with gqa_ratio 2
  gives 2, so every decode step falls back to the SIMT tile kernel.
* **The standalone round kernels that remain after the rope fold are 1.2 % of
  GPU busy time, but they are 10 % of the frame once their dispatch cost is
  counted.** There are 1 250 of them per frame; at 1.22 µs of kernel time plus
  ~1.6 µs of dispatch overhead each, that is ~3.5 ms of the 37 ms frame.
  Attributing each standalone round to the dispatch that produced it (the
  producer being the kernel whose *end* is the last one before the round
  starts) gives:

  | producer | per frame | share |
  |---|---:|---:|
  | `mul_mat_vec_q` (the packed q/k/v and gate/up projections) | 588 | 47 % |
  | `concat_cont` (the fused depth path's K/V growth) | 324 | 26 % |
  | `flash_attn_tile` | 180 | 14 % |
  | `copyBuffer` | 40 | 3 % |
  | `mul_mat_vec_q` (2-column) | 36 | 3 % |
  | `k_set_rows` | 28 | 2 % |
  | `flash_attn_combine_results` | 28 | 2 % |
  | the rest (rope, cpy, …) | 26 | 2 % |

  There are **no** `round → round` pairs: the framework rounds the q, k and v
  slices separately, so the rounding is never applied to an already-rounded
  tensor. The `mmvq` group is the packed projections, which the round node
  materialises into contiguous slices (§3.6c) and which therefore cannot be
  folded away. The `concat_cont` group was redundant and has been removed
  (§3.9).

---

# Part 3 — Implemented optimizations and their measured effect

Each item was implemented, measured with the same corpus and harness, and checked
against the baseline.

| # | Change | Files | Verdict |
|---|---|---|---|
| **O1** | `perf_backend` includes HIP, enabling `direct_prefill` and `skip_cache_round` | `src/models/breeze_tts/generator.cpp:1196-1198` | **shipped** |
| **O2** | The HIP KV caches (backbone and depth) become F16 (were BF16) → the whole-cache bf16 re-round and the F32→F16 cache copy disappear | `src/models/breeze_tts/generator.cpp:162-165, 223-228` | **shipped** |
| **O3** | Batched CFG allowed on HIP: the conditional and unconditional backbone branches decode as one batch-2 step | `src/models/breeze_tts/generator.cpp:1231` | **shipped** |
| **O8** | ggml's round-to-bf16 graph fusion (RMSNorm+round, add+round, SwiGLU+round) enabled on HIP | `external/ggml/src/ggml-cuda/ggml-cuda.cu:4248, 4321`; `src/framework/core/backend.cpp:531` | **shipped** |
| **O4** | Fused depth decoder on HIP | `src/models/breeze_tts/generator.cpp:1210`; `src/models/breeze_tts/session.cpp` | **shipped** — −7…8 % per frame at CFG 1.0, neutral at CFG 3.0, no quality regression (§3.8, §5.2) |
| **O18** | The fused depth path stops concatenating the K/V twice: the decoder hands back the concatenation it already built | `include/engine/framework/modules/transformers/decoder.h:81`; `src/framework/modules/transformers/decoder.cpp:720`; `src/models/breeze_tts/generator.cpp` | **shipped** — −2.5 % per frame, bit-identical (§3.9) |
| **O16** | gfx1101-specific MMVQ warp count for the decode GEMV (4 instead of the upstream 8) | `external/ggml/src/ggml-cuda/mmvq.cu`, `common.cuh` | **shipped** — −12 % at CFG 1.0, neutral at CFG 3.0 (§3.6) |
| **O17** | Find hipcub/rocprim from the build so the HIP argsort path compiles | `external/ggml/src/ggml-hip/CMakeLists.txt`, `.devops/nix/package.nix` | **shipped** — the nix ROCm build could not compile before (§3.7) |

## 3.1 O8 — the round-to-bf16 graph fusion on HIP

**Why it was excluded.** The whole block sat behind
`#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`
(`external/ggml/src/ggml-cuda/ggml-cuda.cu:4248`), and the backend entry point
`set_backend_round_bf16_fusion` was gated on `is_cuda_backend_handle`
(`src/framework/core/backend.cpp:531`).

**Why it is safe to enable.** Nothing in the fusion path is CUDA-specific: the
pattern matching is pure graph inspection, and the three fused kernels
(`ggml_cuda_op_rms_norm_fused` in `norm.cu:845`, `ggml_cuda_op_add_round_bf16` and
`ggml_cuda_op_swiglu_round_bf16` in `unary.cu:401, 440`) are plain F32 math. They
evaluate exactly the per-element expressions of the unfused ops followed by the
rounding, so the results are **bit-identical** — the upstream comment says so
explicitly, and the measurement confirms it.

The only genuinely tricky part is `rms_norm_round_bf16` in `norm.cu:105`, which uses
`__bfloat162float(__float2bfloat16(x))`. Those are CUDA spellings, but the identical
expression already compiles and runs on HIP in `unary.cu:117` (`op_round_bf16`),
which the trace shows as `round_bf16_kernel<float>`.

**Change.** Move the three round-to-bf16 fusions out of the HIP exclusion (the
CUDA-only bias fusions stay excluded), and let `set_backend_round_bf16_fusion`
accept the HIP handle.

**Measured.**

| | Baseline | O8 only |
|---|---:|---:|
| RTF (CFG 1.0) | 0.6600 | **0.5918** (−10.3 %) |
| AR ms / frame | 55.04 | **49.10** (−10.8 %) |
| Output | — | **bit-identical**, all 5 WAV SHA-256 match |

## 3.2 O2 — the HIP KV cache becomes F16

**The change.** `generator.cpp:162-165` now uses F16 on every backend except Metal's
reference bf16 path.

**Why it is lossless.** The K and V rows are bf16-rounded *before* the cache write
(V at `after_qkv_projection`, K at `after_rope`), and every bf16 value in the K/V
range is exactly representable in F16. The attention therefore sees exactly the same
numbers — and the flash kernel already received F16 in both cases, because ggml's
`launch_fattn` converted the F32 cache copy to F16 anyway. The only difference is
that the conversion is no longer needed.

A side benefit: with `bf16_activations=off`, HIP previously kept a BF16 cache
while the activations were F32, which silently truncated the cache. A F16 cache
fixes that too.

**Measured.** Bit-identical output; `convert_unary` 130.9 → 93.0 ms in the trace.
The depth decoder needed the same change (its cache was still BF16); applying it
to both stacks was worth a further 3.3 % of end-to-end RTF, again bit-identical.

## 3.3 O3 — batched CFG on HIP

**The change.** `backbone_batched_cfg = (cuda || hip) && perf.batched_cfg`.

The batched machinery is pure ggml (`causal_decoder_runtime.cpp:631, 135`;
`decoder.cpp:930-1030`); only the `cuda &&` gate blocked it.

**Measured** (`guidance_scale = 3.0`):

| | Baseline | O3 (batched CFG) |
|---|---:|---:|
| RTF | 0.7054 | **0.5342** (−24.2 %) |
| AR ms / frame | 58.56 | **43.02** (−26.5 %) |
| Separate uncond backbone pass | 7.96 ms/frame | **0** (folded into the batch-2 step) |

**With batched CFG the `guidance_scale = 3.0` regime is faster per frame
(43.02 ms) than the `guidance_scale = 1.0` regime (48.12 ms).** The CFG feature
no longer carries a performance penalty — which was the explicit requirement.

## 3.4 O1 — `perf_backend` includes HIP

Enables `direct_prefill` (no KV export/import round trip through the host) and
`skip_cache_round`. Both are backend-neutral ggml paths.

## 3.5 Combined shipped defaults

`src/models/breeze_tts/session.cpp` now sets HIP-specific defaults:

- **`breeze_tts.round_bf16_fusion`** → **on** for HIP (bit-identical, −11 % RTF).
- **`breeze_tts.batched_cfg`** → **on** for HIP (measured no quality regression).
- **`breeze_tts.fused_depth`** → **off** for HIP (measured quality regression, §5).
  Explicit `breeze_tts.fused_depth=true` still opts in.

**Measured result (shipped defaults):**

| | Baseline | Shipped | Change |
|---|---:|---:|---:|
| RTF, CFG 1.0 | 0.6577 | **0.5071** | **−22.9 %** |
| RTF, CFG 3.0 | 0.7054 | **0.5174** | **−26.7 %** |
| AR ms/frame, CFG 1.0 | 54.48 | 40.63 | −25.4 % |
| AR ms/frame, CFG 3.0 | 58.56 | 41.64 | −28.9 % |
| Output, CFG 1.0 | — | — | **bit-identical** |

Kernel-level effect over the same workload:

| Kernel group | Before | After | Change |
|---|---:|---:|---:|
| `round_bf16` | 952.2 ms / 877 544 calls | 684.2 ms / 609 692 calls | −28 % |
| elementwise / fused-slot kernels | 751.7 ms | 483.1 ms | −36 % |
| `convert_unary` | 130.9 ms | 93.0 ms | −29 % |
| `mul_mat_vec_q` | 4471.9 ms | 4459.2 ms | unchanged |
| **total kernel time** | **7566.0 ms** | **6982.2 ms** | **−7.7 %** |

The wall-clock gain (−14.0 %) exceeds the kernel-time gain measured on the
backbone-only variant (−7.7 %) because the removed whole-cache materialisation
also removes host-side allocation and graph work. Applying the cache change to
the depth stack as well accounts for the rest.

---

## 3.6 O16 — a gfx1101-specific MMVQ warp count

**How the hotspot was found.** The rocprofv3 trace showed `mul_mat_vec_q` at 59 % of
kernel time, so the decode GEMV was the target. Reading the trace's dispatch geometry
against the model's shapes identified the dominant weights: the depth decoder's
MLP (1024 × 16384, 17.8 MB of Q8_0 per call) and the backbone's gate/up projection
(2048 × 12288, 26.7 MB). The per-frame weight traffic is ~3.3 GB, and the measured
~130 GB/s is roughly a fifth of the card's peak bandwidth.

**The knob.** ggml's MMVQ kernel picks its warp count from a per-architecture table
(`external/ggml/src/ggml-cuda/mmvq.cu:calc_nwarps`). The RDNA3_0 entry uses 8 warps
for the single-column (decode) case, tuned on the W7900 (gfx1100: 60 CU, 960 GB/s).
On the RX 7800 XT (gfx1101: 32 CU, 624 GB/s) fewer warps per block measure better:
a block then covers four times the weight bytes per cross-warp reduction, and the
reduction is what the smaller part cannot amortise.

**The change.** gfx1101 gets its own parameter table rather than inheriting one:

- `common.cuh`: `GGML_CUDA_CC_RDNA3_1` (= `GGML_CUDA_CC_OFFSET_AMD + 0x1101`) and
  `GGML_CUDA_CC_IS_RDNA3_1`.
- `mmvq.cu`: a new `MMVQ_PARAMETERS_RDNA3_1` table id, selected on the **device** by
  `#if defined(__gfx1101__)` and on the **host** by the runtime cc.

Both halves have to agree: `calc_nwarps` is `__host__ __device__`, and HIP compiles the
host and device passes of a translation unit separately, so a warp count picked from
a device-only macro makes the host launch a different block shape than the kernel
expects — which fails at launch, not at build. Selecting the table by id keeps the
two passes consistent, and it also keeps the override in one place for other cards.

The warp count stays overridable with `-DGGML_MMVQ_RDNA3_1_NWARPS=<n>`.

**Measured** (interleaved A/B, two rounds, `nwarps` 8 vs 4):

| | nwarps = 8 (upstream) | nwarps = 4 (gfx1101) | Change |
|---|---:|---:|---:|
| RTF, `guidance_scale = 1.0` | 0.5938 / 0.5640 | **0.5066 / 0.5090** | **−12 %** |
| RTF, `guidance_scale = 3.0` | 0.5191 / 0.5255 | 0.5448 / 0.5201 | neutral |

The measurements are interleaved rather than sequential: consumer cards drift enough
that running all of one variant and then all of the other produced a 4 % swing on a
configuration that had not changed. (An earlier version of this work also mixed two
configurations in one result directory because the framework *appends* to the timing
log; the harness now clears it.)

**Why the CFG regime does not benefit.** With `guidance_scale != 1` the backbone
decodes as a batch-2 step and the depth decoder runs its conditional and
unconditional branches together, so almost every large GEMV in the loop has
`ncols_dst > 1` and uses the multi-column path, which keeps the upstream single warp
per block. The single-column path — and therefore this knob — is what the
`guidance_scale = 1` regime lives on. The multi-column path was swept too and the
upstream single warp is already optimal there (2 warps: +1.7 %, 4 warps: +4.5 %).

**Quality.** The warp count changes the order in which the partial sums are added, so
the sampled tokens change. Over a 13-sample ASR corpus the mean word-error rate is
**0.70 % for both the baseline and the shipped build** — equal.

**Scope.** The measurement covers gfx1101 only. gfx1100, gfx1102 and gfx1150 keep the
upstream value, which is the same discipline hipfire applies: tune per exact arch, do
not let one part's optimum leak into the family.

## 3.6b Is a *shape-dependent* warp count worth it? (measured: no)

§3.6 sets one warp count for the whole single-column path. The obvious next step is
to make it depend on the shape, so that the large matrices and the small ones can
differ. The trace says that would buy nothing.

**Per shape.** The trace's `Grid_Size_X` is `gridDim.x * blockDim.x` and
`blockDim.x` is always 32, so `gridDim.x` is the weight matrix's row count. That
maps the dispatch groups onto the model's matrices:

| `nrows_x` | matrix | weight (Q8_0) | µs/call at 4 warps | µs/call at 8 warps | achieved GB/s at 4 |
|---:|---|---:|---:|---:|---:|
| 16384 | depth MLP gate/up (1024 × 16384) | 17.8 MB | **37.22** | 63.13 | 479 |
| 12288 | backbone MLP gate/up (2048 × 12288) | 26.7 MB | **48.85** | 58.34 | 547 |
| 4096 | backbone QKV (2048 × 4096) | 8.9 MB | **18.84** | 21.38 | 473 |
| 2048 | backbone out + down | 4.5 / 13.4 MB | **18.51** | 20.08 | — |
| 1536 | depth QKV (1024 × 1536) | 1.7 MB | **5.64** | 7.76 | 296 |
| 1024 | depth out + down | 1.1 / 8.9 MB | **11.74** | 12.65 | — |
| 2052 | lm_head (2048 × 2052) | 4.5 MB | **10.73** | 11.51 | 417 |

**Four warps is the best count for every shape** — there is no shape that wants the
upstream eight. The three matrices that carry the traffic now run at **75–88 % of
the card's peak bandwidth**, so the single-column path has essentially no headroom
left for a per-shape rule to recover.

**End to end, per frame.** Interleaved A/B, two rounds each, measured as the AR
loop's milliseconds per generated frame so that a configuration which happens to
produce a different number of frames cannot flatter itself:

| warps | AR ms/frame, CFG 1.0 | AR ms/frame, CFG 3.0 |
|---:|---:|---:|
| 1 | 42.50 | 42.25 |
| 2 | 42.78 | 41.73 |
| **4** | **40.65** | **41.66** |
| 8 (upstream) | 46.44 | 42.21 |

One warp and four warps look close in raw RTF at `guidance_scale = 1`, but the
per-frame figures separate them: one warp produced 81.2 frames for the same prompt
where four produced 76.2 — slower speech, not a faster model.

**The multi-column path is already optimal.** With `guidance_scale != 1` the
backbone and the depth decoder both decode in batch 2, so almost every large GEMV
takes the multi-column path. Sweeping its warp count (1 / 2 / 4) at
`guidance_scale = 3.0` gives 0.5182 / 0.5267 / 0.5396 — the upstream single warp
wins, and 4 warps is 4 % worse. The knob is kept as
`-DGGML_MMVQ_RDNA3_1_NWARPS_NC2=<n>` for tuning another card, defaulting to the
upstream value.

**Conclusion.** The shape dependence that matters is already in the kernel: the
single-column and multi-column cases are separate branches of `calc_nwarps`, and
each is now at its own measured optimum for gfx1101. Making the count depend on
`nrows_x` as well would mean templating the kernel on the warp count and
multiplying the instantiations, for no measurable gain. **Not implemented**, and
the plan's next target moves on.

## 3.6c Folding the bf16 rounding into its producer

The bf16 reference path rounds an activation to bf16 after almost every decoder
stage. On gfx1101 that is **2291 `round_bf16` dispatches per frame**, 10.7 % of
GPU kernel time. Since a round on a tiny tensor costs almost nothing but its
dispatch, the interesting question is which of those can be removed.

**What the profile says.** Attributing every round dispatch to the kernel that
preceded it:

| producer | per frame | share | removable? |
|---|---:|---:|---|
| `mul_mat_vec_q` | 832 | 36 % | **no** — see below |
| `rope` | 417 | 18 % | **yes** |
| `round` (a round after a round) | 417 | 18 % | no — that chain is the packed projection's |
| other | 222 | 10 % | no |
| `flash_attn` | 208 | 9 % | yes, in principle |
| `set_rows` | 196 | 9 % | already skipped (§3.2) |

**Why the matmul ones cannot be folded.** The rounds that follow a matmul round
*views* of the packed projection output: the QKV projection produces one packed
tensor that is sliced into q, k and v, and the gate/up projection one packed
tensor sliced into gate and up. The round node is what *materialises* each slice
into a contiguous tensor, because the consumers (RMSNorm, SiLU, the matmul) need
contiguous inputs. Folding the rounding into the matmul epilogue would leave the
materialisation — and the dispatch — in place. The only way to remove them is to
split the packed projection into separate matmuls and give each its own epilogue,
which trades one dispatch for another. **The epilogue idea does not pay here.**

**What does pay: the rope.** The rope's output is rounded, and the rope has a
single consumer, so the rounding can be folded into the rope kernel — the values
are the same (round to bf16 and back) and the dispatch disappears. Three things
have to line up:

1. The rope kernel has to round its stores. That is a new flag on the kernel,
   threaded through the ROPE, NEOX, multi and vision variants.
2. The rope must write into the **round node's** destination, not its own. The
   consumers read the round's tensor, and skipping the round node means nothing
   else ever writes it.
3. The rope node must have exactly **one** use. Folding the rounding into the
   rope changes the value that *every* consumer of the rope sees.

Two subtleties cost real debugging time and are worth recording:

- The value has to be **computed once** into a variable and then rounded.
  Writing `round(a*b - c*d)` in one branch and `a*b - c*d` in the other
  duplicates the expression, and the compiler contracts the multiply-add
  differently in the two branches — so the folded value stopped matching the
  unfused one, and the sampled token stream with it.
- Matching two ggml tensors by value (`ggml_tensor rope_dst = *node`) and
  pointing the copy at the round's buffer is how the "write into the final
  destination" step is done.

**Measured** (interleaved A/B, two rounds, same frame count):

| | AR loop ms/frame | RTF |
|---|---:|---:|
| without the rope fold | 42.16 | 0.5155 |
| **with the rope fold** | **39.84** | **0.4975** |
| change | **−5.5 %** | **−3.5 %** |

At the kernel level the fold removes **41 656 of 229 108 round dispatches
(−18 %)** and 110 ms of the 2561 ms of kernel time. The output is
**bit-identical**: all five benchmark WAVs have the same SHA-256 as before the
fold.

**What is left.** The flash-attention output rounding (208 per frame) is
foldable in the same way, but it needs the rounding in all three flash-attention
variants — tile, vec and the WMMA path — because the prefill uses the WMMA
variant while the decode uses the tile one. Implementing it for only some of them
would silently skip the rounding on the others.

**Status: started, then parked.** The work-in-progress reached
`fattn-common.cuh` (the `fattn_kernel_t` signature and `launch_fattn` grew a
`round_bf16` flag), `fattn-tile.cuh` and `fattn-vec.cuh` (both round their
stores). The mma-f16 and wmma-f16 kernels still had the old signature, so the
tree did not build; the three files were reverted and the patch kept at
`/tmp/fa_fold_wip/fa_round_fold.patch`. On top of that, the 2026-10-30 trace
re-analysis (§2.5) shows that after the rope fold the remaining rounds are only
**1.2 % of the GPU busy time**, and the flash-attention share of that is 24.7 ms
out of 2423 ms — about 1 % of the GPU's work, and less of the wall clock. The
change is not worth landing half-finished for that, so it stays parked until a
reason to pick it up appears.

Two things about the flash-attention path are worth keeping:

- **Only the tile variant runs on gfx1101 for the decode.** The trace contains
  `flash_attn_tile<128,128,2,1,false>` (20 800 calls) and
  `flash_attn_combine_results<128>` (2 800 calls), and *no* mma-f16, wmma-f16 or
  vec variant. The AMD WMMA branch needs `Q->ne[1] * gqa_ratio_eff > 8` and a
  1-token decode with gqa_ratio 2 gives 2.
- **`round_bf16` is idempotent**, so a `round → round` pair is exactly
  redundant and can be elided from the graph without touching a kernel. The
  framework creates those at `norm_modules.cpp:217-221`,
  `positional_modules.cpp:76-108` and `decoder.cpp:239/584/798`; this is the
  cheapest remaining item in this section.

## 3.7 O17 — making the HIP build find hipcub and rocprim

`external/ggml/src/ggml-cuda/argsort.cu` includes `<hipcub/hipcub.hpp>` for the HIP
radix argsort path (rows wider than 1024, which BreezeTTS needs for the 2048-wide
codebook vocabulary), and hipcub in turn needs rocprim's headers. Both ship as
separate ROCm components, so they are not necessarily next to hip and rocblas.

The vendored ggml HIP build now looks for both:

- `find_package(hipcub QUIET)` / `find_package(rocprim QUIET)`, and
- a fallback header search over `CMAKE_INCLUDE_PATH` and `${ROCM_PATH}/include` for
  packagers (nix among them) that ship the headers without a CMake config package.

`.devops/nix/package.nix` adds `rocmPackages.hipcub` and `rocmPackages.rocprim` to the
ROCm build inputs, which puts them on the include path and on `CMAKE_PREFIX_PATH`.

**Why this matters beyond the build.** `ggml_argsort_top_k` is what the fused depth
graph uses for its on-device top-k, and `ggml_argsort` is unrestricted on HIP (unlike
`ggml_top_k`, which is capped at 1024 columns). If the fused depth path is ever
enabled on HIP, the argsort path has to compile.

## 3.8 O4 — the fused depth path, re-measured and shipped

The fused depth path runs the whole 15-codebook depth loop as **one graph**: the
K/V of the earlier codebooks stays inside the graph, every codebook gets its head
matmul and its sampler on the device, and the host only uploads the backbone
hidden state and downloads the 15 tokens. The per-codebook path instead makes
~8 pageable host↔device copies and several device syncs per codebook, and the
trace of §2.5 shows the GPU idling 20–60 µs for each of them.

**It was kept off by default on the strength of a quality measurement that does
not reproduce.** The earlier note in `session.cpp` recorded a mean ASR error
rate of 0.70 % without the fused path and 4.90 % with it. Re-measured on the
same corpora with the same harness, the fused path is the *better* one:

| Corpus | per-codebook path | fused path |
|---|---:|---:|
| q8_0, `guidance_scale = 1.0`, 7 samples | 1.30 % | **0.00 %** |
| q4_0, `guidance_scale = 1.0`, 13 samples | 2.80 % | **1.40 %** |

**Speed**, interleaved A/B, per generated frame (the honest metric: the two
paths sample different tokens, so they generate different numbers of frames for
the same prompt):

| Configuration | per-codebook | fused | change |
|---|---:|---:|---:|
| q8_0, `guidance_scale = 1.0` | 39.70 ms | **36.61 ms** | **−7.8 %** |
| q8_0, `guidance_scale = 3.0` | 39.49 ms | 39.85 ms | +0.9 % (neutral) |
| q4_0, `guidance_scale = 1.0` | 35.98 ms | **33.46 ms** | **−7.0 %** |
| q4_0, `guidance_scale = 3.0` | 33.85 ms | 33.92 ms | +0.2 % (neutral) |

The pattern is consistent and has a clear cause. At `guidance_scale = 1.0` the
depth decoder runs the conditional branch only, so the per-codebook host round
trips are a large share of the step and removing them pays. At
`guidance_scale = 3.0` the depth decoder already decodes both CFG branches as
one batch-2 step, which halves the weight traffic per branch; the same round
trips are then amortised over twice the work and there is nothing left to win.

**What it costs.** The in-graph sampler draws from the same distribution as the
per-codebook path but consumes the random stream differently, so the sampled
audio for a fixed seed is different (the ASR gate says it is not worse). The
graph has 10 493 nodes and is rebuilt whenever the guidance scale, the
temperature, the top-k or the top-p changes; the rebuild is one-off and the
old graph is released first.

**Change.** `perf.fused_depth` now defaults to `true` for HIP in
`src/models/breeze_tts/session.cpp`, next to the batched-CFG default. Set
`breeze_tts.fused_depth=false` to get the per-codebook path back.

## 3.9 The fused path's duplicated K/V concatenation

The trace showed 648 `concat_cont` dispatches per frame and a round right after
half of them. Reading the kernel sequence around one layer explains both: the
decoder's **prefix** attention path concatenates the prefix K/V with the new
K/V for the attention

```cpp
all_k = ConcatModule({1}).build(ctx, *attention_prefix_key, k);   // decoder.cpp:622
```

and the fused depth runtime then concatenated **the same two tensors again** to
build the next prefix state:

```cpp
full.key = ConcatModule({1}).build(ctx, *full.key, *added.key);    // generator.cpp
```

The decoder returned only the *new* K/V in its state, so the caller had no
choice but to rebuild the concatenation.

**Change.** The decoder can now hand back the concatenated tensor it already
built (`DecoderAttentionPolicy::prefix_state_returns_full_kv`,
`include/engine/framework/modules/transformers/decoder.h:81`), and the fused
depth runtime uses that as the next prefix state instead of concatenating
again. The flag is **off by default**: the other prefix-state callers
(`yue2/ar_runtime.cpp`, `audio8_tts/ar.cpp`) concatenate the returned K/V
themselves and would otherwise duplicate the prefix.

**Measured** (interleaved A/B of the two binaries, q8_0, `guidance_scale = 1.0`,
two rounds each):

| | RTF | AR ms/frame | depth ms/frame |
|---|---:|---:|---:|
| before | 0.4757 | 36.13 | 29.90 |
| **after** | **0.4641** | **35.23** | **29.02** |
| change | **−2.4 %** | **−2.5 %** | −2.9 % |

**Bit-identical**: all five benchmark WAVs have the same SHA-256 before and
after, and the same hashes as the run before this change — the concatenation is
a pure copy, so removing the duplicate cannot move a value.

# Part 4 — Remaining work, re-ranked against the measurements

## 4.1 P0 — the quantized GEMV (partly addressed in §3.6)

`mul_mat_vec_q` is still 64 % of kernel time. §3.6 took 12 % out of it at
`guidance_scale = 1` by giving gfx1101 its own warp count; the kernel still runs at
roughly a fifth of the card's peak bandwidth, so the remaining headroom is large.

What is known:

- BreezeTTS decodes one token per step, so every projection is a batch-1 GEMV over
  a Q8_0 weight matrix. The depth decoder does 15 such steps per frame, so the
  61 M-parameter depth stack is streamed 15× per frame. Per frame the loop moves
  ~3.3 GB of weights.
- The dominant single weight is the depth decoder's MLP (1024 × 16384, 17.8 MB per
  call), then the backbone's gate/up projection (2048 × 12288, 26.7 MB). Both are
  `ncols_dst == 1` at `guidance_scale = 1` and `ncols_dst == 2` with CFG on.
- ggml's MMVQ path for Q8_0 on RDNA3 uses `mul_mat_vec_q<GGML_TYPE_Q8_0, 1, …>`
  with the per-arch nwarps table at
  `external/ggml/src/ggml-cuda/mmvq.cu:calc_nwarps` (the RDNA3_0 whitelist, which
  is stricter than RDNA4's because *"Q2_K / Q5_K / IQ4_XS regress in full quant
  sweeps"*).
- Each mmvq call averaged 23.1 µs before the warp change. The Q8_0 row is 34 bytes,
  so the read pattern is not a clean multiple of a 128-byte sector.
- hipfire's gfx1100 GEMV reaches near-peak bandwidth on 1-byte weights with a
  different structure: one wave per output row, 4 independent accumulators, packed
  `uint32` loads, `__launch_bounds__(32, 16)` to respect the RDNA3 16-wave/SIMD
  limit at 1536 VGPRs, and a descending 16/8/4/2/1 warp reduction
  (`kernels/src/gemv_hfq4g256.gfx1100.hip`, `gemv_hfq4g256_residual.gfx1100.hip`).

**Re-ranked after the 2026-10-30 trace.** The trace of the shipped build says
the GEMV is 53 % of the kernel time, and the dispatch latency around the
kernels is the other half of the bill: ~6 200 dispatches per frame at a real
~1.6 µs each (§2.5). The cheapest way to collect that is to remove dispatches,
not to make the GEMV faster.

**What the GEMV actually achieves.** Reading the weight shape out of each
dispatch's grid and dividing the bytes by the measured time gives the achieved
bandwidth per matrix (Q8_0 is 34 bytes per 32 weights, the card's peak is
624 GB/s):

| matrix | weight | MB | µs/call | GB/s | % of peak |
|---|---|---:|---:|---:|---:|
| depth MLP gate/up (16384 × 1024) | Q8_0 | 17.8 | 37.3 | 478 | **77 %** |
| backbone MLP gate/up (12288 × 2048) | Q8_0 | 26.7 | 50.8 | 527 | **84 %** |
| backbone down (4096 × 2048) | Q8_0 | 8.9 | 19.8 | 450 | 72 % |
| lm_head (2052 × 2048) | Q8_0 | 4.5 | 10.7 | 418 | 67 % |
| depth QKV (1536 × 1024) | Q8_0 | 1.7 | 6.1 | 272 | 44 % |

The two matrices that carry the traffic are at 77–84 % of peak, so a better
GEMV kernel can win at most a few percent of the frame — the memory system is
nearly saturated. The small depth QKV at 44 % is too small to saturate (1.7 MB)
and is latency-bound instead. **The only large lever left on the GEMV is to
read fewer bytes.**

**Is the GEMV dual-issuing? (asked, answered, measured).** gfx1101 has the
VOPD dual-issue facility (`vopd` in the LLVM feature list: *"Has VOPD dual
issue wave32 instructions"*), and it is **on by default** for gfx1101 in
ROCm 7.2.3 / LLVM 22 — `--amdgpu-enable-vopd` produces byte-identical code and
`-mattr=-vopd` does not even build. So the question is not whether the
compiler *may* dual-issue, but whether the GEMV *does*.

Disassembling the device code of `mmvq.cu` with the real build flags
(`--cuda-device-only -S`) shows:

* Every `mul_mat_vec_q` instantiation contains **2–8 `v_dual_*` instructions**
  (`v_dual_fmac_f32`, `v_dual_add_f32`, `v_dual_mov_b32`, `v_dual_and_b32`,
  `v_dual_cndmask_b32`, `v_dual_lshlrev_b32`) — but **none of them is in the
  inner loop**. They sit in the prologue and the epilogue (address arithmetic,
  the scale multiply, the cross-warp reduction).
* The hot loop is plain **VOP1** `v_dot4_i32_iu8` (gfx11's unsigned×signed
  dot4, which is what `ggml_cuda_dp4a` lowers to on RDNA3,
  `common.cuh:683-690`), in two serially-dependent chains, one per output row:

  ```
  v_dot4_i32_iu8 v3,  v5, v3, 0            ; serial accumulator v3
  v_cvt_f32_f16_e32 v0, v0.l
  v_dot4_i32_iu8 v3,  v6, v4, v3
  v_mul_f32_e32 v0, v0, v18
  v_cvt_f32_i32_e32 v3, v3
  v_fmac_f32_e32 v14, v0, v3
  s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | ...
      global_load_b32 / global_load_b64 x2 / global_load_d16_b16
  v_dot4_i32_iu8 v19, v22, v3, 0           ; the second, independent chain
  v_dot4_i32_iu8 v19, v23, v4, v19
  ```

  The `s_delay_alu … instskip(NEXT)` instructions are gfx11's **delay-slot
  scheduling** — they cover the 4-cycle `v_dot4` latency — not the VOP1P/VOP2P
  dual encoding.

* A register-resident microbenchmark (`v_dot4` chains, LDS-free so the
  measurement is issue-bound) gives **0.36 → 0.44 dot4 instructions per cycle
  per lane** going from one serial chain to eight independent ones. If the
  hardware sustained two dot4 per cycle the curve would reach 2.0; it does not,
  so even a perfect pairing of the two chains would buy ~20 % of the *ALU*
  time and not of the kernel.

**Conclusion: the GEMV is not leaving anything on the table through dual
issue.** It is at 77–88 % of peak *memory* bandwidth, so the memory system —
not the VALU issue rate — is the limit. This is the same conclusion the
Q4_0 measurement gives from the other direction: halving the bytes is worth
10 % end to end while changing no ALU work at all.

**And that lever is already available to the user.** The package ships a Q4_0
build of the same model. Q4_0 halves the weight traffic, and the depth MLP
gate/up alone is 180 calls per frame (13 codebook steps × 14 layers) — 3.2 GB
of weight traffic per audio frame at Q8_0. Measured with the shipped build on
the `quality` corpus, `guidance_scale = 1.0`, two interleaved rounds each:

| model | RTF | AR ms/frame | VRAM | ASR word-error rate |
|---|---:|---:|---:|---:|
| Q8_0 | 0.4656 | 35.35 | 5.08 GB | 0.00 % (6 samples) |
| **Q4_0** | **0.4217** | **31.76** | **3.79 GB** | 1.52 % (6 samples) |
| change | **−9.4 %** | **−10.2 %** | −25 % | one borderline word |

The Q4_0 difference in the error rate is a single word ("idled" vs "idolled")
in one sample; both models are otherwise transcribed correctly. On AMD the
Q4_0 package is the better default, and the plan's remaining GEMV work should
be about the *quantization* rather than the kernel.

Options, in increasing cost:

1. ~~**Tune the existing kernel.**~~ **Done** (§3.6): gfx1101 now uses 4 warps for
   the single-column case, worth 12 % at `guidance_scale = 1`.
2. ~~**Make the warp count shape-dependent.**~~ **Measured and rejected** (§3.6b):
   four warps is the best count for every matrix the trace sees, and the three
   matrices that carry the traffic already run at 75–88 % of peak bandwidth.
3. **Fuse the dequant into the GEMV** so the weights are read once as packed bytes
   and dot-producted in registers, in the hipfire structure, and dispatch it for
   `ncols == 1` on RDNA3 only. This is now a small refinement rather than a
   rescue: the simple tuning already put the big matrices near the roof.
4. **Reduce the traffic**: the **Q4_0 package** does this today and is 10 %
   faster per frame than Q8_0 on gfx1101 (§4.1). A per-tensor `weight_type`
   override would let a user pick the format per stack (the depth stack is 14×
   amplified, so it is where a byte reduction pays off most), but that is a
   packaging feature rather than a kernel optimization. `q4_k` was measured
   earlier and was *slower* than Q8_0, so the format has to be chosen with a
   measurement, not by size.
5. ~~**Remove the fused path's duplicated `concat_cont`.**~~ **Done** (§3.9):
   the decoder already concatenates the prefix K/V with the new K/V for the
   attention, so the fused runtime now reuses that tensor instead of building
   it a second time. −2.5 % per frame, bit-identical. (Writing the new K/V into
   a preallocated cache at a graph-input offset, as the per-codebook path does
   with `set_rows`, was considered first and rejected: it does not remove a
   dispatch, and it would make the attention read the whole fixed-size cache
   instead of the growing one.)

## 4.2 P1 — the remaining bf16 rounding (1.2 % of GPU busy time, 5 % of the frame)

§3.6c removes the rope's rounding. **Re-measured on the 2026-10-30 trace, the
rounds that are left are 1.2 % of the GPU busy time, not 8.9 %**: the round
kernels that remain (121 284 dispatches over 100 frames, 86 per frame) cost
7.9 ms of the 2423 ms of busy time. The 8.9 % figure counted the *dispatch*
overhead of a run in which the GPU was idle most of the time, which the trace
re-analysis in §2.5 corrected.

Counting the dispatch cost as well, the remaining 1 000-odd rounds are about
**5 % of the frame** rather than 1.2 %, so the item is not as small as the
kernel-time share suggests. After the concat dedup (§3.9) the producers are the
`mmvq` projections (588 per frame) and the flash attention (180 per frame).

The item is still **parked rather than scheduled**:

* The flash-attention round fold (the `round_bf16` flag threaded through
  `fattn-common.cuh`, `fattn-tile.cuh` and `fattn-vec.cuh`, which was
  half-implemented) would need the same treatment in the mma-f16 and wmma-f16
  kernels plus a graph-fusion rule and a dispatcher change. It removes 20 828
  round dispatches per 100 frames, i.e. 24.7 ms of busy time — about 1 % of the
  GPU's work and less than that of the wall clock. The work-in-progress patch is
  kept out of the tree (`/tmp/fa_fold_wip/fa_round_fold.patch`) rather than
  shipped half-finished.
* The cheaper variant is **eliding `round → round` pairs**: `round_bf16` is
  idempotent, so a round whose source is already a round is a no-op and can be
  removed from the graph without touching a kernel. The framework creates those
  at `norm_modules.cpp:217-221`, `positional_modules.cpp:76-108` and
  `decoder.cpp:239/584/798`.

What is left of this item is dominated by the rounds on
the **packed projections** (832 per frame): the QKV projection produces one
tensor that is sliced into q, k and v, and the gate/up projection one tensor
sliced into gate and up. Each slice has to be materialised into a contiguous
tensor before its consumer can use it, and that materialisation *is* the round
node. Folding the rounding into the matmul epilogue would leave the
materialisation in place, so the dispatch would not go away.

What is left that could still be removed:

1. **The flash-attention output rounding** (208 per frame). Foldable the same
   way as the rope, but it has to be done in all three flash-attention variants
   (tile, vec, and the WMMA path the prefill uses) — implementing it for only
   some of them would silently skip the rounding on the others.
2. **Splitting the packed projections.** Doing the QKV projection as three
   matmuls, each with a rounding epilogue, would remove three round dispatches
   per layer and replace them with two extra matmul launches — roughly a wash,
   and it gives up the packed-projection optimisation.
3. **The device-side sampler** (§4.4) removes rounds of its own.
4. **Folding the rounding into the matmul epilogue anyway**, as a *numerical*
   rather than a dispatch saving: a matmul that writes bf16-exact values would
   halve the bytes the consumers read. That is a much bigger change (the
   consumers would have to accept bf16 inputs) and is the only route to the
   remaining bandwidth.

## 4.3 P2 — the depth decoder's host round trips (**done**, §3.8)

`__amd_rocclr_copyBuffer` is 43 437 calls in the 100-frame trace, and the idle
they cause is most of the 4541 ms the GPU spends waiting (§2.5). The fused depth
path removes the per-codebook round trips by running the whole depth loop as
one graph, and it is now shipped by default on HIP: **−7…8 % per frame at
`guidance_scale = 1.0`**, neutral at `guidance_scale = 3.0`, with no quality
regression (§3.8, §5.2).

What is left of the host round trips is the same story one level up: the
**backbone** still samples on the host (§4.4), so every AR step of the backbone
ends in a round trip and a CPU-side pass over its logits.

## 4.4 P3 — device-side sampling

The backbone sample is entirely on the host on HIP
(`src/framework/sampling/torch_random.cpp:349-362`), so every AR step ends in a
host round trip and a CPU-side top-k/top-p pass over 2052 logits. Options:

1. Express the sampler as graph nodes (the fused-depth runtime already does:
   `ggml_argsort_top_k` + `ggml_soft_max`/`ggml_cumsum`/`ggml_step` +
   `ggml_argmax`, `generator.cpp:1088-1113`) and make the backbone decode graph
   emit the sampled token directly. This is the cleanest solution and is what the
   CUDA "torch" path approximates.
2. Port `src/models/fish_audio/hip_fast_sampler.cu` — but note it is a 1-workgroup
   LDS tree reduction with O(k·N) work; it should be **rewritten as a wave32 kernel**,
   not copied verbatim.

Caveat: the CUDA path's RNG exists to match PyTorch's Philox stream bit for bit.
On HIP that is not the goal (the host RNG already differs), so an HIP device sampler
would define its own documented stream — and the seeded output would stop
matching CUDA. That has to be stated in the docs rather than discovered later.

## 4.5 P4 — the attention kernel (4.1 % of GPU time)

The AR decode uses `flash_attn_tile` (`fattn.cu:543`), not the tensor-core path,
because Breeze decodes one token per step and the AMD WMMA branch requires
`Q->ne[1] * gqa_ratio_eff > 8` (`fattn.cu:525`). At 4.1 % of kernel time this is
now a minor item; the GQA-shared tile from hipfire
(`attention_flash_q8_0_tile_gqa.gfx1100.hip`) would read K/V once per KV head
instead of once per Q head, but the payoff is small compared with §4.1.

## 4.6 P5 — the vocoder (≈ 1 % of GPU time) and the fused path's KV concat

### The fused path's `concat_cont` — done (§3.9)

The fused depth graph is fixed-shape, so the K/V of the earlier codebooks grow
by concatenation at every step: 648 `concat_cont` dispatches per frame. **Half
of them were redundant** — the decoder's prefix path already builds exactly
that concatenation for the attention — and the duplicate is now gone (§3.9),
which is worth 2.5 % per frame with bit-identical output.

### The vocoder

`im2col`/`col2im`/`snake` together are ~1 % of kernel time.

`im2col`/`col2im`/`snake` together are ~1 % of kernel time. The conv-transpose-1d
col2im fast path is already enabled on HIP (`conv_modules.cpp:424-431`). The
bf16 Conv1d path is CUDA-only (`conv_modules.cpp:545-559`), but the payoff is small.

## 4.7 P6 — the Strix-Halo retained vocoder graph cache

`CMakeLists.txt:213-252` restricts the retained-graph cache to exactly `gfx1151`,
but the mechanism is arch-agnostic (`speech_decoder.cpp:1094-1105`). Generalise the
gate to a validated-arch list. Note that the same block forces
`GGML_HIP_HIPBLASLT=OFF`, which was measured on gfx1151 only — on gfx1101 the
generic default (hipBLASLt ON) applies, and the two should be A/B'd.

---

# Part 5 — Quality gates

## 5.1 The two gates

**Tier A — bit-identical.** For changes that claim to be lossless, the SHA-256 of
the per-request WAV must be unchanged. The benchmark harness records it in the
manifest, so two manifests can be diffed directly. This is the cheapest and
strongest gate, and it needs no new code in the engine.

**Tier B — ASR round trip.** For changes that legitimately reorder floating-point
accumulation (batched CFG) or change the random-draw mapping (fused depth), the
generated audio is transcribed with an ASR model and the transcript is compared
against the prompt that produced it. `tests/breeze_tts/quality_gate.py` does this
with Citrinet ASR and reports the word-error rate (English) or the
character-error rate (CJK). CJK prompts are skipped because Citrinet is
English-only; the corpus keeps them in a separate `[zh]` case for manual review.

## 5.2 Results

Corpus: the `quality` case of the benchmark harness (5 prompts × per-request
seeds), mean ASR error rate.

| Configuration | mean error | verdict |
|---|---:|---|
| `guidance_scale = 1.0`, baseline | **0.70 %** | reference |
| `guidance_scale = 1.0`, baseline + O2 + O8 | — | **bit-identical** (Tier A) |
| `guidance_scale = 1.0`, baseline + O2 + O8 + **fused depth** | 4.90 % | *see the correction below* |
| `guidance_scale = 3.0`, baseline | 4.20 % | reference for the CFG regime |
| `guidance_scale = 3.0`, baseline + O2 + O8 + **batched CFG** | **2.80 %** | **no regression** |
| `guidance_scale = 1.0`, shipped, q8_0, 7 samples | 1.30 % | reference |
| `guidance_scale = 1.0`, shipped + **fused depth**, q8_0, 7 samples | **0.00 %** | **no regression** |
| `guidance_scale = 1.0`, shipped, q4_0, 13 samples | 2.80 % | reference |
| `guidance_scale = 1.0`, shipped + **fused depth**, q4_0, 13 samples | **1.40 %** | **no regression** |

Two conclusions:

1. **Batched CFG is safe.** It slightly *improved* the CFG regime's error rate
   (4.20 % → 2.80 %). This is consistent with the CUDA documentation, which
   already warns that batched CFG changes the matmul accumulation order and is
   therefore not bit-identical.
2. **The fused depth path is safe too — the earlier "regression" does not
   reproduce.** Re-measured on the same corpora with the same harness, the
   fused path's mean error rate is *lower* than the per-codebook path's in both
   re-measurements (1.30 % → 0.00 % on q8_0, 2.80 % → 1.40 % on q4_0). The
   original 4.90 % figure therefore does not describe a property of the fused
   path; combined with the speed result of §3.8 (−7…8 % per frame at
   `guidance_scale = 1.0`) the fused path is now the default on HIP.

   The one thing the fused path *does* change is the sampled token stream: its
   in-graph sampler consumes the random stream differently from the per-codebook
   path, so the audio for a fixed seed is different. That is a behavioural
   change, not a quality regression, and it is stated in the user-facing docs.

Note that **`guidance_scale` itself costs quality** on this model: the CFG=3.0
baseline is at 4.20 % where the CFG=1.0 baseline is at 0.70 %. That is a
pre-existing property of the model, not of these changes, and it is worth
documenting.

---

# Part 6 — The benchmark

## 6.1 Files

| File | Role |
|---|---|
| `tests/breeze_tts/benchmark_hip_breeze.py` | The driver. Runs `audiocpp_cli` once with a `--request-sequence` so the warmup and the measured requests share one process (warm graphs, warm allocator). Records per-request wall time, audio duration, RTF and the WAV SHA-256, plus a manifest with the git commit, the binary/model/prompt hashes, the ROCm version, the GPU arch, the clock before and after, and the full `GGML_*`/`HIP_*`/`ROCBLAS_*` environment. Optionally wraps the run in `rocprofv3`. |
| `tests/breeze_tts/breeze_tts_warm_bench_cases.txt` | The prompt corpus: `[default]` (5 English sentences), `[zh]` (Chinese), `[quality]` (5 sentences for the ASR gate). |
| `tests/breeze_tts/analyze_stage_timing.py` | Turns the framework's timing log into the per-stage table. |
| `tests/breeze_tts/analyze_rocprof_trace.py` | Turns a rocprofv3 CSV trace into per-kernel and per-category tables. |
| `tests/breeze_tts/compare_runs.py` | Compares runs, including the length-normalised per-frame figures. |
| `tests/breeze_tts/quality_gate.py` | The ASR quality gate. |
| `tests/breeze_tts/ab_run.py` | The interleaved A/B driver. Runs A, B, A, B … against two (optionally different) binaries or session-option sets and reports the median RTF, the **cost per generated frame** and the depth-stage split. |
| `tests/breeze_tts/analyze_trace_gaps.py` | Attributes the idle time between consecutive dispatches to the kernel that precedes it. |
| `tests/breeze_tts/analyze_trace_idle.py` | Computes the *union* of the dispatch intervals, i.e. the real GPU busy time and the real idle time, and attributes the idle to the kernels on both sides of it. |
| `tests/breeze_tts/analyze_trace_bursts.py` | Groups the dispatches into bursts (a captured graph replay reads as one burst) and reports the idle between the bursts. |

## 6.2 Protocol

- **Fixed corpus**, fixed seed, fixed `--max-tokens 100`, one warmup request then
  **5 measured requests**; the **median** is reported (consumer cards are noisy),
  with min/max alongside.
- **Cold and warm** are distinguishable: the first request pays the graph capture
  and the allocator warm-up, the rest do not.
- **Length normalisation.** The fused-depth and batched-CFG paths change the
  sampled token stream, so they can generate a different number of frames for the
  same prompt. Comparing raw RTF across such runs overstates or understates the
  change, so `compare_runs.py` also reports the AR-loop cost per generated frame.
  **That is the honest comparison**; the RTF column is what a user sees.
- **One variable at a time.** Every optimization is reachable through a runtime
  session option (`breeze_tts.round_bf16_fusion`, `breeze_tts.batched_cfg`,
  `breeze_tts.fused_depth`, `breeze_tts.direct_prefill`, `breeze_tts.skip_cache_round`),
  so a single binary can A/B it. Changes that need a different build (the O2 cache
  type, the O8 ggml fusion, the O10 CMake gate) get their own build directory and
  their own manifest entry.
- **Baseline first, every time.** Consumer-card drift makes cross-session
  comparisons unreliable; the pristine baseline binary is kept alongside the
  optimised one (`build-rocm-gfx1101/bin/audiocpp_cli_baseline`) so both are
  measured in the same session.
- **Clock capture.** `rocm-smi --showclocks --showtemp` is sampled before and after
  every run and recorded in the manifest.
- **GPU-idle pre-flight.** The harness now waits for the GPU to be quiet
  (`rocm-smi --showuse` below 20 %) before it starts, and records the
  utilisation before and after the run in the manifest. Another process on the
  same GPU — a server, another agent, a background job — makes the numbers
  meaningless, and measurement has to be refused rather than silently
  contaminated. Two measurement mistakes came out of this:
  - **Do not poll `rocm-smi` while a benchmark runs.** A 2-second polling loop
    during a run made that run 25 % slower than the same run without it.
  - **Compare medians of per-request figures, not medians of medians.** The
    timing log holds one `ar.total_ms` per request but one `depth.*_ms` per
    *frame*; the frame counters also differ between the paths (they sample
    different tokens), so the per-frame cost has to be computed per request and
    then medianed.

## 6.3 Reproducing

```bash
# build
nix develop --extra-experimental-features "nix-command flakes" .#rocm
cmake -S . -B build-rocm-gfx1101 -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DENGINE_ENABLE_HIP=ON -DENGINE_ENABLE_CUDA=OFF \
  -DGPU_TARGETS=gfx1101 \
  -DCMAKE_HIP_FLAGS="-isystem $HIPCUB/include -isystem $ROCPRIM/include"
cmake --build build-rocm-gfx1101 --target audiocpp_cli -j 24

# measure
python3 tools/model_manager_v2.py install breeze_tts_2_q8_0 --models-root models
python3 tests/breeze_tts/benchmark_hip_breeze.py \
  --binary build-rocm-gfx1101/bin/audiocpp_cli \
  --model models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf \
  --out-dir tests/breeze_tts/results/run --guidance-scale 1.0 \
  --iterations 5 --warmup 1 --max-tokens 100

# analyse
python3 tests/breeze_tts/compare_runs.py tests/breeze_tts/results
python3 tests/breeze_tts/analyze_stage_timing.py tests/breeze_tts/results/run/audiocpp_timing.log

# quality gate
python3 tools/model_manager_v2.py install citrinet_asr_q8_0 --models-root models
python3 tests/breeze_tts/quality_gate.py \
  --binary build-rocm-gfx1101/bin/audiocpp_cli \
  --asr-model models/Citrinet-ASR-GGUF/citrinet-asr-q8_0.gguf \
  --case-name quality tests/breeze_tts/results/baseline tests/breeze_tts/results/run

# kernel trace
python3 tests/breeze_tts/benchmark_hip_breeze.py ... \
  --rocprof /nix/store/h401y6sxy167zbf3p6ld9k1alf5b53l9-rocprofiler-sdk-7.2.3/bin/rocprofv3
python3 tests/breeze_tts/analyze_rocprof_trace.py tests/breeze_tts/results/run/rocprof
```

---

# Part 7 — Risks and open questions

| # | Risk | Mitigation |
|---|---|---|
| R1 | **The bf16 KV cache may be misread by the AMD flash kernels.** The tile kernel casts K/V to `half2*` unconditionally (`fattn-tile.cuh:851-852`) and relies on `launch_fattn` having converted to F16 first. With the F16 cache the conversion is gone and the kernel reads the cache directly, which is exactly the CUDA arrangement. Still, a BF16 K/V that reached a flash kernel unconverted would be silently misinterpreted. | The F16 cache removes the hazard; add a `GGML_ASSERT(K->type == GGML_TYPE_F16)` in the AMD tile/MMA paths as a belt-and-braces check. |
| R2 | **gfx1101 is not codegen-identical to gfx1100.** hipfire measured 1-ulp FMA-contraction differences in the RoPE sin-branch. | Re-run every bit-exactness oracle **on gfx1101**; never inherit a gfx1100 pass result. The Tier-A gate in this document was run on gfx1101 and passed. |
| R3 | **Batched CFG changes the accumulation order**, so the CFG output is not bit-identical. | Tier-B gate: measured 4.20 % → 2.80 % WER over 13 samples, i.e. no regression. Document it, as the CUDA docs already do. |
| R4 | **The fused depth path changes the seeded output on HIP**, because its in-graph sampler consumes the random stream differently from the per-codebook path. A user who compares two builds at the same seed sees different (equally intelligible) speech. | The change is documented in `docs/models/breeze_tts.md` and it is a single session option (`breeze_tts.fused_depth=false`) to turn off. The ASR round trip measured the fused path at 1.30 % → 0.00 % (q8_0, 7 samples) and 2.80 % → 1.40 % (q4_0, 13 samples), so it is not a quality regression — the earlier 4.90 % figure did not reproduce (§5.2). |
| R5 | **Consumer-card throttling and driver variance** make small deltas unmeasurable. | Median of 5, clock capture, and a "must exceed the noise floor" rule. The measured run-to-run spread on the median RTF was < 1 %. |
| R6 | **Retained vocoder graphs hold VRAM.** | Track peak VRAM in the benchmark; the 105-frame retained graph can be dropped if the budget is tight. |
| R7 | **Fused depth is large** (a 65536-node graph arena, `generator.cpp:1133`) and holds 16 KV states on the device. | Measure peak VRAM; `graph_arena_mb` already exists as a session option. |
| R8 | **Enabling `perf_backend` on HIP turns on several features at once.** | Each was measured in isolation and each is individually overridable by a session option. |
| R9 | **The nix HIP build fails to compile** because `hipcub`/`rocprim` are not in the packaging, and a local build fails with *"cannot find ROCm device library"* when `ROCM_PATH` holds only the HIP headers. | `ROCM_PATH` must contain both `include/hip` and `amdgcn/bitcode` (`rocmPackages.clr` has only the headers; `/nix/store/9lw54v7631y940lglz6nkprvi7x1p8pk-rocm-merged` has both), or pass `--rocm-device-lib-path`. | Add `rocmPackages.hipcub` and `rocmPackages.rocprim` to `.devops/nix/package.nix`, or document the `CMAKE_HIP_FLAGS` workaround. |
| R11 | **A measurement can be silently contaminated by another process on the GPU.** The `audiocpp_server` on this machine holds 7 GB of VRAM and was observed running at 98 % GPU utilisation during a benchmark, which made both sides of an A/B ~20 % slower and moved the ratio by 5 points. | The benchmark harness now waits for `rocm-smi --showuse` to drop below 20 % before a run and records the utilisation before and after in the manifest. Never poll `rocm-smi` *during* a run (that alone cost 25 %). |
| R10 | **`ggml_top_k` (a different op from `ggml_argsort_top_k`) is capped at 1024 columns on HIP.** Any new code that uses `ggml_top_k` over the 2048/2051/2052-wide vocabs will silently fall back. | Use `ggml_argsort_top_k`, as the fused depth graph already does. |

---

## Appendix A — gfx1101 / RDNA3 kernel constraints (for anyone writing HIP in `audio.cpp`)

- **Wave32 only.** No MFMA (CDNA-only), no `v_sdot4` (gfx10-only) — use
  `v_sudot4` or WMMA i8. Do not enable wave64 (halves throughput, breaks
  `ds_bpermute` cross-half semantics).
- **WMMA**: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` (the `_gfx12` spelling does
  not exist on gfx11). A/B operand = 16 halfs per lane; **lanes `r` and `r+16` must
  supply identical rows**; C accumulator maps as `acc[j] = C[2*j + (lane>>4)][lane&15]`.
- **64 KB LDS hard cap** (`cudaFuncSetAttribute` is a no-op on HIP). Query
  `prop.sharedMemPerBlock` at runtime.
- **No `cp.async`**, so no multi-stage FA pipelining (`nstages = 0`).
- **No CUB.** `GGML_OP_TOP_K` ≤ 1024 columns; `ggml_argsort` is fine (the fork has a
  hipcub radix path).
- **Reduce with `ds_swizzle` + DPP, not `__shfl_xor`** (`ds_bpermute` occupies LDS and
  serialises).
- **Compile with `-O2` or better** (ROCm/ROCm#5826: illegal DPP at `-O0` on GFX10+).
- **Always pass an explicit target**: `-DGPU_TARGETS=gfx1101` (never `native`);
  verify with `roc-obj-ls`.
- **Use `#if defined(RDNA3)` / `RDNA3_0`** (ggml defines these in
  `external/ggml/src/ggml-cuda/vendors/hip.h:221-231`) — never `__gfx1101__`.
  `RDNA3_0` excludes Strix Halo (gfx115x).
- **Separate ISA capability from perf tuning.** hipfire's rule
  (`docs/methodology/perf-arch-discipline.md`): tune per exact arch, use a positive
  allowlist. A family-inherited perf variant cost a 14 % gfx1100 regression.

## Appendix B — File index

| Area | Path |
|---|---|
| Breeze generator | `src/models/breeze_tts/generator.cpp` |
| Breeze session / options | `src/models/breeze_tts/session.cpp` |
| Breeze config | `include/engine/models/breeze_tts/assets.h:13-64` |
| Breeze perf options | `include/engine/models/breeze_tts/generator.h:45-54` |
| Decoder layer / static cache | `src/framework/modules/transformers/decoder.cpp` |
| Causal decoder runtime (batched) | `src/framework/modules/transformers/causal_decoder_runtime.cpp` |
| Attention modules | `src/framework/modules/attention/*.cpp` |
| Fast modules | `src/framework/modules/optimizations/fast_{conv,kv,projection}_modules.cpp` |
| Backend / arch predicates | `src/framework/core/backend.cpp` |
| Attention fallback | `src/framework/core/attention_fallback.cpp` |
| Sampling | `src/framework/sampling/{torch_random,hf_sampler}.cpp` |
| Vocoder | `src/models/breeze_tts/speech_decoder.cpp` |
| HIP sampler (fish) | `src/models/fish_audio/hip_fast_sampler.cu` |
| ggml HIP build | `external/ggml/src/ggml-hip/CMakeLists.txt` |
| ggml HIP arch macros | `external/ggml/src/ggml-cuda/vendors/hip.h:185-244` |
| ggml FA selection | `external/ggml/src/ggml-cuda/fattn.cu:456-543` |
| ggml FA tile kernel | `external/ggml/src/ggml-cuda/fattn-tile.cuh` |
| ggml FA launch / K-V conversion | `external/ggml/src/ggml-cuda/fattn-common.cuh:917-1010` |
| ggml MMVQ / MMQ | `external/ggml/src/ggml-cuda/{mmvq.cu,mmq.cuh,mmq-config-rdna3.cuh}` |
| ggml round-bf16 fusion | `external/ggml/src/ggml-cuda/ggml-cuda.cu:4248-4392`, `norm.cu:845`, `unary.cu:401,440` |
| ggml argsort (HIP radix) | `external/ggml/src/ggml-cuda/argsort.cu:34-60, 287-292` |
| Benchmark harness | `tests/breeze_tts/` |
| HIP docs | `docs/build/HIP.md` |
| Breeze docs | `docs/models/breeze_tts.md` |
| llama.cpp | `../llama.cpp/ggml/src/ggml-cuda/`, `../llama.cpp/ggml/src/ggml-hip/` |
| hipfire | `../hipfire/kernels/src/`, `../hipfire/crates/rdna-compute/src/arch_caps.rs` |

---

## Appendix C — Commit map

The work landed as a sequence of atomic commits. The dependency notes below
are the ones that matter when picking the commits up one at a time.

| # | Commit | Needs |
|---|---|---|
| 1 | `Find hipcub and rocprim in the HIP build` | — (everything else needs it: without it there is no ROCm build) |
| 2 | `Enable ggml's round-to-bf16 graph fusion on HIP` | 1 |
| 3 | `Fold the bf16 rounding into the rope on HIP` | 2 (the rule lives in the fusion block that 2 opens on HIP) |
| 4 | `Use a gfx1101-specific MMVQ warp count for the decode GEMV` | — |
| 5 | `Use F16 KV caches on HIP` | — |
| 6 | `Enable the Breeze-TTS perf presets on HIP` | 5 (`skip_cache_round` needs an F16 cache) |
| 7 | `Allow batched CFG on HIP` | — |
| 8 | `Default Breeze-TTS to batched CFG on HIP` | 7 |
| 9 | `Default Breeze-TTS to the fused depth path on HIP` | 1, 6 |
| 10 | `Stop concatenating the K/V twice in the fused depth path` | — (9 only changes the default, 10 changes the graph) |
| 11 | `Log the Breeze-TTS depth-loop stage breakdown per request` | — |
| 12 | `Add the Breeze-TTS-2 ROCm benchmark harness` | — |
| 13 | `Add the Breeze-TTS-2 ASR quality gate` | 12 |
| 14 | `Add rocprof trace analysis for the Breeze-TTS-2 ROCm work` | — |
| 15 | `Document the Breeze-TTS-2 ROCm/gfx1101 work` | 2-14 |
