#!/usr/bin/env python3
"""Show the burst structure of a rocprof trace.

Kernels that are dispatched back-to-back form a burst (a graph replay or a
stream of graph nodes).  The idle time between bursts is host overhead.
"""
import argparse
import collections
import csv
import sys

csv.field_size_limit(10 ** 9)


def short(name):
    for prefix, label in (
        ("void flash_attn_tile", "fa_tile"),
        ("void flash_attn_ext_vec", "fa_vec"),
        ("void flash_attn_combine_results", "fa_combine"),
        ("void mul_mat_vec_q", "mmvq"),
        ("void mul_mat_vec_f", "mmvf"),
        ("void mul_mat_q", "mmq"),
        ("void rope_neox", "rope"),
        ("void rope_norm", "rope"),
        ("void rms_norm_f32", "rmsnorm"),
        ("void add_round_bf16_kernel", "add_round"),
        ("round_bf16_kernel", "round"),
        ("void swiglu_round_bf16_kernel", "swiglu_round"),
        ("void quantize_q8_1", "quant_q8_1"),
        ("void quantize_mmq_q8_1", "quant_q8_1_mmq"),
        ("void k_set_rows", "set_rows"),
        ("void k_get_rows", "get_rows"),
        ("void convert_unary", "convert"),
        ("__amd_rocclr_copyBuffer", "copyBuffer"),
        ("__amd_rocclr_fillBufferAligned", "fillBuffer"),
        ("void im2col", "im2col"),
        ("void scale_f32", "scale"),
        ("rocblas_", "rocblas_"),
        ("Cijk_", "rocblas_gemm"),
    ):
        if name.startswith(prefix):
            return label
    return name[:44]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip-ms", type=float, default=2000.0)
    ap.add_argument("--window-ms", type=float, default=4.0)
    ap.add_argument("--burst-us", type=float, default=20.0,
                    help="idle longer than this starts a new burst")
    args = ap.parse_args()

    events = []
    with open(args.trace, newline="") as handle:
        for row in csv.DictReader(handle):
            events.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"]),
                           row["Kernel_Name"], int(row["Stream_Id"])))
    events.sort()
    t0 = events[0][0]
    lo = t0 + int(args.skip_ms * 1e6)
    hi = lo + int(args.window_ms * 1e6)

    # group dispatches into bursts: a burst ends when the union-busy time has a
    # gap larger than burst_us
    bursts = []
    cur_start = None
    cur_end = None
    cur_names = []
    cur_count = 0
    for start, end, name, stream in events:
        if start < lo:
            continue
        if start > hi:
            break
        if cur_end is None or start - cur_end > args.burst_us * 1e3:
            if cur_end is not None:
                bursts.append((cur_start, cur_end, cur_count, cur_names))
            cur_start = start
            cur_names = []
            cur_count = 0
        cur_end = max(cur_end or 0, end)
        cur_count += 1
        cur_names.append(name)
    if cur_end is not None:
        bursts.append((cur_start, cur_end, cur_count, cur_names))

    print(f"window {(hi-lo)/1e6:.1f} ms  bursts: {len(bursts)}")
    total_idle = 0
    for i in range(1, len(bursts)):
        total_idle += bursts[i][0] - bursts[i - 1][1]
    print(f"idle between bursts: {total_idle/1e6:.3f} ms")

    print("\n-- first bursts --")
    for i, (s, e, c, names) in enumerate(bursts):
        gap = (s - bursts[i - 1][1]) / 1e3 if i else 0.0
        kinds = collections.Counter(short(n) for n in names)
        print(f"  burst {i:3d} start={(s-lo)/1e6:8.3f} ms dur={(e-s)/1e3:8.1f} us "
              f"kernels={c:4d} gap_before={gap:9.1f} us  {dict(kinds.most_common(6))}")

    print("\n-- gap histogram between bursts --")
    hist = collections.Counter()
    gsum = 0
    for i in range(1, len(bursts)):
        g = bursts[i][0] - bursts[i - 1][1]
        gsum += g
        hit = 0
        for edge in (2e6, 5e5, 1e5, 2e4, 5e3, 1e3):
            if g >= edge:
                hit = edge
                break
        hist[hit] += 1
    labels = {0: "<1us", 1e3: "1-5us", 5e3: "5-20us", 2e4: "20-100us",
              1e5: "0.1-0.5ms", 5e5: "0.5-2ms", 2e6: ">2ms"}
    for edge in (0, 1e3, 5e3, 2e4, 1e5, 5e5, 2e6):
        if hist[edge]:
            print(f"  {labels[edge]:10s} {hist[edge]:7d}")

    print("\n-- kernels per burst --")
    sizes = collections.Counter(b[2] for b in bursts)
    for size, n in sorted(sizes.items())[:20]:
        print(f"  {size:5d} kernels : {n:6d} bursts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
