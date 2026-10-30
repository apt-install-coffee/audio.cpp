#!/usr/bin/env python3
"""Print the kernel sequence around a region of a rocprof trace.

Used to see what surrounds the ``__amd_rocclr_copyBuffer`` dispatches that
dominate the stall time of the Breeze-TTS HIP run.
"""
import argparse
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
    return name[:48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--from-ms", type=float, default=3000.0)
    ap.add_argument("--to-ms", type=float, default=3020.0)
    ap.add_argument("--max", type=int, default=400)
    args = ap.parse_args()

    t0 = None
    shown = 0
    with open(args.trace, newline="") as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if t0 is None:
                t0 = start
            rel = (start - t0) / 1e6
            if rel < args.from_ms:
                continue
            if rel > args.to_ms:
                break
            if shown >= args.max:
                print("...")
                break
            gap = 0.0
            if shown:
                gap = (start - last_end) / 1e6
            last_end = end
            print(f"{rel:10.3f} ms  dur={(end-start)/1e3:9.2f} us  gap={gap:8.2f} us  "
                  f"stream={row['Stream_Id']}  grid=({row['Grid_Size_X']},{row['Grid_Size_Y']},{row['Grid_Size_Z']})  "
                  f"{short(row['Kernel_Name'])}")
            shown += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
