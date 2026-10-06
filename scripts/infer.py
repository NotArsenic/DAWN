#!/usr/bin/env python3
"""Run inference on night images.

Given a trained Dawn model, enhance night images and produce:
    - Enhanced sRGB image
    - Uncertainty map (from analytic propagation)
    - Optional: bilateral grid visualization

Usage:
    python scripts/infer.py --checkpoint checkpoints/dawn.pt --input path/to/night/images
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Dawn inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="outputs/infer", help="Output directory")
    parser.add_argument("--nfe", type=int, default=2, help="Number of function evaluations")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--save-uncertainty", action="store_true", help="Save uncertainty maps")
    parser.add_argument("--save-grid", action="store_true", help="Save bilateral grid coefficients")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Inference with {args.nfe} NFE")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Input: {args.input}")
    logger.info(f"  Output: {output_dir}")

    # TODO: Implement inference pipeline
    # 1. Load checkpoint
    # 2. For each input image:
    #    a. Preprocess (sRGB -> linear, compute guidance)
    #    b. Encode with DINO
    #    c. Sample z_0, run ODE solver -> z_1 (grid coefficients)
    #    d. Unstandardize, slice, apply affine
    #    e. Predict noise params -> compute variance -> uncertainty map
    #    f. Run denoiser with variance as input
    #    g. Save enhanced image + uncertainty map

    logger.warning("Inference pipeline not yet implemented.")


if __name__ == "__main__":
    main()
