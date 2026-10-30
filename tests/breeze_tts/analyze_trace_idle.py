#!/usr/bin/env python3
"""Compute the true GPU idle time of a rocprof trace.

The kernel trace holds one record per dispatch.  Several streams can be in
flight at once, so the idle time is the complement of the *union* of the
dispatch intervals, not the sum of the gaps between consecutive records.
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
    ap.add_argument("--skip-ms", type=float, default=0.0,
                    help="ignore the first N ms (warm-up / weight upload)")
    ap.add_argument("--to-ms", type=float, default=0.0,
                    help="stop at N ms after the first event (0 = until the end)")
    ap.add_argument("--Idle-min-ms", type=float, default=0.02)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    events = []
    with open(args.trace, newline="") as handle:
        for row in csv.DictReader(handle):
            events.append((int(row["Start_Timestamp"]), int(row["End_Timestamp"]),
                           row["Kernel_Name"], int(row["Stream_Id"])))
    events.sort()
    if not events:
        return 1

    t_start = events[0][0]
    t_end = events[-1][1]
    lo = t_start + int(args.skip_ms * 1e6)
    hi = t_start + int(args.to_ms * 1e6) if args.to_ms > 0 else t_end

    # merge intervals
    merged = []
    for start, end, name, stream in events:
        if end <= lo or start >= hi:
            continue
        start = max(start, lo)
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
                merged[-1][2].append(name)
        else:
            merged.append([start, end, [name]])

    total = hi - lo
    busy = sum(e[1] - e[0] for e in merged)
    print(f"window: {(t_end-lo)/1e6:.1f} ms   union-busy: {busy/1e6:.1f} ms "
          f"({100.0*busy/total:.1f}%)   idle: {(total-busy)/1e6:.1f} ms")

    # idle intervals
    idles = []
    cursor = lo
    for start, end, names in merged:
        if start > cursor:
            idles.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < hi:
        idles.append((cursor, hi))

    print(f"idle intervals: {len(idles)}")
    hist = collections.Counter()
    idle_total = 0
    for a, b in idles:
        d = b - a
        idle_total += d
        bucket = 0
        for edge in (1e3, 1e4, 5e4, 2e5, 1e6, 5e6, 2e7):
            if d >= edge:
                bucket += 1
        hist[bucket] += 1
    labels = ["<1us", "1-10us", "10-50us", "50-200us", "0.2-1ms", "1-5ms", "5-20ms", ">20ms"]
    for i, label in enumerate(labels):
        if hist[i]:
            print(f"  {label:10s} {hist[i]:8d}")

    # attribute idle to the kernel that ran right before it
    before = collections.Counter()
    before_n = collections.Counter()
    after = collections.Counter()
    big = []
    import bisect
    starts = [m[0] for m in merged]
    for a, b in idles:
        d = b - a
        i = bisect.bisect_right(starts, a) - 1
        prev = merged[i][2][-1] if i >= 0 else None
        j = bisect.bisect_left(starts, b)
        nxt = merged[j][2][0] if j < len(merged) else None
        key = short(prev or "?")
        before[key] += d
        before_n[key] += 1
        after[short(nxt or "?")] += d
        if d >= args.Idle_min_ms * 1e6:
            big.append((d, key, short(nxt or "?")))

    print(f"\n-- idle attributed to the kernel that ran before it (>={args.Idle_min_ms} ms) --")
    for name, dur in before.most_common(args.top):
        print(f"  {dur/1e6:9.1f} ms  n={before_n[name]:7d}  {name}")

    print("\n-- idle attributed to the kernel that runs after it --")
    for name, dur in after.most_common(args.top):
        print(f"  {dur/1e6:9.1f} ms  {name}")

    print(f"\n-- largest idle intervals --")
    for d, p, n in sorted(big, reverse=True)[:args.top]:
        print(f"  {d/1e6:9.3f} ms  after {p}  before {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
