#!/usr/bin/env python3
"""Benchmark inference latency.

Measures wall-clock time and GMACs at various resolutions.
Reports: resolution, GMACs, ms/image, FPS.

Usage:
    python scripts/benchmark_latency.py --checkpoint checkpoints/dawn.pt
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RESOLUTIONS = [
    (640, 480),
    (1280, 720),
    (1920, 1080),
    (2000, 1500),
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark Dawn inference latency")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint (or random weights)")
    parser.add_argument("--nfe", type=int, default=2, help="Number of function evaluations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=50, help="Timed iterations")
    args = parser.parse_args()

    logger.info(f"Benchmarking with {args.nfe} NFE on {args.device}")

    # TODO: Implement benchmarking
    # For each resolution:
    #   1. Create random input
    #   2. Warmup forward passes
    #   3. Time forward passes
    #   4. Report ms/image and FPS
    #   5. Optionally compute GMACs with torch profiler or thop

    logger.warning("Benchmark not yet implemented.")

    print("\n{:<16} {:>8} {:>10} {:>8}".format("Resolution", "GMACs", "ms/img", "FPS"))
    print("-" * 46)
    for h, w in RESOLUTIONS:
        print("{:<16} {:>8} {:>10} {:>8}".format(f"{w}x{h}", "TODO", "TODO", "TODO"))


if __name__ == "__main__":
    main()
