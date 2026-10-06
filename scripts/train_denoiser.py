#!/usr/bin/env python3
"""Stage 2: Train the bounded denoiser.

Stage 2a: Train on oracle (fitted) grids.
Stage 2b: Fine-tune on flow-predicted grids.

Usage:
    python scripts/train_denoiser.py --config configs/denoiser.yaml
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train bounded denoiser (Stage 2)")
    parser.add_argument("--config", type=str, default="configs/denoiser.yaml", help="Config file")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--flow-ckpt", type=str, required=True, help="Stage 1 flow checkpoint")
    parser.add_argument("--resume", type=str, default=None, help="Denoiser checkpoint to resume")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info("Stage 2: Bounded denoiser training")
    logger.info(f"  Flow checkpoint: {args.flow_ckpt}")

    # TODO: Implement training loop
    # Stage 2a: Train with oracle-fitted grids + analytic variance
    # Stage 2b: Switch to flow-predicted grids for fine-tuning

    logger.warning("Training loop not yet implemented.")


if __name__ == "__main__":
    main()
