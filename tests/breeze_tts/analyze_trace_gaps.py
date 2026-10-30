#!/usr/bin/env python3
"""Attribute inter-kernel gaps in a rocprof kernel trace.

The trace only contains GPU-side dispatch records, so "gap" means the idle
time between the end of one dispatch and the start of the next one.  Large gaps
mean the GPU was waiting for the host (graph launch, host->device copy or a
device sync) rather than executing kernels.
"""
import argparse
import collections
import csv
import sys

csv.field_size_limit(10 ** 9)


def load(path):
    events = []
    kinds = collections.Counter()
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kinds[row["Kind"]] += 1
            if row["Kind"] != "KERNEL_DISPATCH":
                continue
            events.append((
                int(row["Start_Timestamp"]),
                int(row["End_Timestamp"]),
                row["Kernel_Name"],
                int(row["Grid_Size_X"]),
                int(row["Grid_Size_Y"]),
                int(row["Grid_Size_Z"]),
                int(row["Stream_Id"]),
                int(row["Correlation_Id"]),
            ))
    events.sort()
    return events, kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-ms", type=float, default=0.15,
                    help="threshold in ms for a gap to count as a stall")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    events, kinds = load(args.trace)
    print(f"kinds: {dict(kinds)}")
    print(f"dispatches: {len(events)}")
    if not events:
        return 1

    span = events[-1][1] - events[0][0]
    busy = sum(e[1] - e[0] for e in events)
    print(f"span: {span/1e6:.3f} s   busy: {busy/1e6:.3f} s   "
          f"busy%: {100.0*busy/span:.1f}")

    # gap accounting: attribute the idle time to the kernel that precedes it
    gap_by_prev = collections.Counter()
    gap_count_by_prev = collections.Counter()
    big_gaps = []
    for i in range(len(events) - 1):
        gap = events[i + 1][0] - events[i][1]
        if gap <= 0:
            continue
        name = events[i][2]
        gap_by_prev[name] += gap
        gap_count_by_prev[name] += 1
        if gap >= args.gap_ms * 1e6:
            big_gaps.append((gap, name, events[i + 1][2]))

    stall = sum(g for g, _, _ in big_gaps)
    print(f"gaps >= {args.gap_ms} ms: {len(big_gaps)}  total {stall/1e6:.3f} s "
          f"({100.0*stall/span:.1f}% of span)")

    print("\n-- stall time attributed to the preceding kernel --")
    for name, gap in gap_by_prev.most_common(args.top):
        if gap < args.gap_ms * 1e6:
            break
        print(f"  {gap/1e6:9.3f} s  n={gap_count_by_prev[name]:7d}  {name}")

    print("\n-- largest single gaps --")
    for gap, prev, nxt in sorted(big_gaps, reverse=True)[:args.top]:
        print(f"  {gap/1e6:9.3f} ms  {prev}  ->  {nxt}")

    # per-second buckets
    print("\n-- per-second buckets --")
    t0 = events[0][0]
    buckets = collections.Counter()
    counts = collections.Counter()
    for start, end, *_ in events:
        buckets[(start - t0) // 1_000_000_000] += end - start
        counts[(start - t0) // 1_000_000_000] += 1
    for sec in sorted(buckets):
        print(f"  t={sec:3d}s  busy={buckets[sec]/1e6:8.1f} ms  "
              f"dispatches={counts[sec]:7d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
