#!/usr/bin/env python3
"""Prepare VisDrone-DET for training: luminance split, manifests and 512-px PNG copies.

Run after scripts/download_visdrone.py. For every image in the requested splits:
    1. Compute the median luma of the sRGB image (0-255).
    2. Classify it with the `prepare` thresholds in configs/data.yaml:
           median <  night_below  -> "night"      real night, held out for evaluation
           median >= day_above    -> "day"        used for training
           otherwise              -> "ambiguous"  unused
    3. Write a PNG copy resized to `visdrone.image_size` on the long side into
       <out>/<split>/images/.
    4. Write one manifest per (split, class): <out>/manifests/<split>_<class>.txt, one path per
       line, relative to <out>. VisDroneDataset reads these.

Usage:
    python scripts/prepare_visdrone.py --root data/visdrone --out data/visdrone_512
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split VisDrone by luminance and make PNG copies")
    parser.add_argument("--root", type=str, default="data/visdrone", help="Raw download directory")
    parser.add_argument("--out", type=str, default="data/visdrone_512", help="Prepared directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to prepare")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--night-below", type=float, default=None,
                        help="Override prepare.night_below")
    parser.add_argument("--day-above", type=float, default=None,
                        help="Override prepare.day_above")
    args = parser.parse_args()

    logger.info("Preparing VisDrone")
    logger.info(f"  Raw: {args.root}")
    logger.info(f"  Out: {args.out}")
    logger.info(f"  Splits: {args.splits}")

    # TODO: Implement
    # 1. Load the data config (omegaconf); apply the CLI threshold overrides
    # 2. Locate the raw images. download_visdrone.py extracts the official zips as they are
    #    (<root>/VisDrone2019-DET-<split>/images/*.jpg) while its docstring says
    #    <root>/<split>/images: reconcile the two
    # 3. Per image: median sRGB luma -> class; resize to visdrone.image_size (long side); save PNG
    # 4. Write the manifests and log the day/night/ambiguous counts per split
    # 5. Log the median-luma histogram so the `day_above` TODO in data.yaml can be calibrated

    logger.warning("Preparation not yet implemented.")


if __name__ == "__main__":
    main()
