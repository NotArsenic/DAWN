#!/usr/bin/env python3
"""Sanity check: oracle grid fitting.

Fit bilateral grids from night_clean to day on a few samples, apply them,
and measure PSNR. This verifies that:
    1. The grid fitting code works correctly.
    2. 16x16x8 grids with 12 coefficients can represent the transform.
    3. Expected oracle PSNR should be 30-40+ dB.

Runs on CPU, no GPU required.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Oracle grid fitting sanity check")
    parser.add_argument("--data-root", type=str, default="data/visdrone", help="VisDrone root")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of samples to test")
    parser.add_argument("--output", type=str, default="outputs/check_oracle", help="Output dir")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Load data, fit grids, apply, compute PSNR. Use the shared compute_guidance,
    # fit_grid, slice_grid and apply_affine: no local re-implementations
    # (docs/conventions.md section 8).
    print("check_oracle.py: not yet implemented")
    print(f"  Data root: {args.data_root}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
