#!/usr/bin/env python3
"""Evaluate Dawn on synthetic and real night test sets.

Computes:
    - PSNR, SSIM, LPIPS on synthetic test set
    - Object-crop metrics using VisDrone detection boxes
    - Uncertainty calibration (ECE, calibration curve)
    - Hallucination audit (objects appeared/disappeared via YOLO)

Usage:
    python scripts/evaluate.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Dawn model")
    parser.add_argument("--config", type=str, default="configs/eval.yaml", help="Eval config")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    logger.info("Dawn evaluation")

    # TODO: Implement evaluation
    # 1. Load model
    # 2. Synthetic test set: compute PSNR/SSIM/LPIPS
    # 3. Object-crop metrics
    # 4. Uncertainty calibration
    # 5. Hallucination audit with YOLO
    # 6. Save results table + plots

    logger.warning("Evaluation not yet implemented.")


if __name__ == "__main__":
    main()
