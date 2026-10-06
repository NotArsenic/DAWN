#!/usr/bin/env python3
"""Stage 1: Train the flow matching model.

Trains the DINOv3 encoder + velocity transformer to predict bilateral grid
coefficients via conditional flow matching / rectified flow.

Usage:
    python scripts/train_flow.py --config configs/flow.yaml
    python scripts/train_flow.py --config configs/flow.yaml \
        --override configs/debug.yaml --device cpu     # tiny end-to-end run, no data needed

The loop itself lives in `dawn.flow.trainer`; this file is configuration, logging and
checkpoint plumbing. Raw vs standardized grids: docs/conventions.md section 5.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.flow.dataset import build_night_loaders  # noqa: E402
from dawn.flow.trainer import FlowTrainer  # noqa: E402
from dawn.grid.normalize import compute_grid_stats, load_or_compute_stats  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(config: str, data_config: str, calib_config: str, overrides: list[str]):
    """flow.yaml + data.yaml + every --override, later files winning."""
    cfg = OmegaConf.merge(OmegaConf.load(config), OmegaConf.load(data_config))
    for path in overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    calib = OmegaConf.to_container(OmegaConf.load(calib_config), resolve=True)
    return OmegaConf.to_container(cfg, resolve=True), calib


class WandbRun:
    """wandb if it is installed and `logging.wandb_mode` is not 'disabled'; else a no-op."""

    def __init__(self, cfg):
        self.run = None
        mode = cfg["logging"]["wandb_mode"]
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed; logging to the console only")
            return
        self.run = wandb.init(project=cfg["logging"]["wandb_project"], mode=mode, config=cfg)

    def log(self, values: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def main():
    parser = argparse.ArgumentParser(description="Train flow matching model (Stage 1)")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Config file")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--calib-config", type=str, default="configs/camera_calib.yaml",
                        help="Camera calibration config")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Extra YAML files merged on top, in order (e.g. configs/debug.yaml)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overfit", action="store_true",
                        help="Validate on the TRAINING set and on the live (not EMA) weights: "
                             "the sanity check that render PSNR can reach the oracle at all. "
                             "Run this before any full training.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg, calib = load_config(args.config, args.data_config, args.calib_config, args.override)
    device = torch.device(args.device)

    logger.info("Stage 1: Flow matching training")
    logger.info(f"  Config: {args.config} (+ {', '.join(args.override) or 'no overrides'})")
    logger.info(f"  Device: {device}")

    train_loader, val_loader = build_night_loaders(cfg, calib, overfit=args.overfit)
    if args.overfit:
        logger.info("  Overfit mode: validating on the training images, live weights")
    logger.info("  Train images: %d, val images: %d",
                len(train_loader.dataset), len(val_loader.dataset))

    num_stat_samples = min(cfg["grid"]["stats_num_samples"], len(train_loader.dataset))
    if cfg["loader"]["synthetic_images"]:
        # Never let a debug run poison grid.stats_cache: stats fitted to generated images
        # would silently become the standardization of the next real run.
        logger.info("  Synthetic images: computing grid stats fresh, not touching the cache")
        stats = compute_grid_stats(train_loader.dataset, num_stat_samples)
    else:
        stats = load_or_compute_stats(
            cfg["grid"]["stats_cache"],
            dataset=train_loader.dataset,
            num_samples=num_stat_samples,
            device=torch.device("cpu"),
        )
    trainer = FlowTrainer.from_config(cfg, stats, device)
    for name, (trainable, total) in trainer.param_counts().items():
        logger.info("  %-11s %9d trainable / %9d total", name, trainable, total)

    if args.resume:
        trainer.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False))
        logger.info("  Resumed from %s at step %d", args.resume, trainer.step)

    run = WandbRun(cfg)
    log_cfg = cfg["logging"]
    out_dir = Path(args.out_dir)
    total_steps = cfg["flow"]["num_train_steps"]
    iterator = iter(train_loader)

    try:
        while trainer.step < total_steps:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            step = trainer.step  # the step number this batch trains
            if step == 1 and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)  # skip the one-off warmup allocs
            started = time.perf_counter()
            losses = trainer.train_step(batch)
            losses["sec_per_step"] = time.perf_counter() - started
            if step == 1 and device.type == "cuda":
                logger.info("  Peak VRAM at batch %d x %d timesteps: %.2f GB",
                            cfg["optim"]["batch_size"], cfg["flow"]["timesteps_per_image"],
                            torch.cuda.max_memory_allocated(device) / 2**30)

            if step % log_cfg["log_every"] == 0:
                values = {f"loss/{k}": v for k, v in losses.items()}
                if log_cfg["log_attn_gates"]:
                    gates = trainer.velocity.attn_gates()
                    values.update({f"gate/{k}": v for k, v in gates.items()})
                logger.info(
                    "step %6d | total %.4f | flow %.4f | render %.4f | noise %.4f | align %.4f",
                    step, losses["total"], losses["velocity"], losses["apply"],
                    losses["noise"], losses["align"],
                )
                run.log(values, step=step)

            if (step + 1) % log_cfg["val_every"] == 0 or trainer.step == total_steps:
                nfe = cfg["flow"]["num_inference_steps"]
                # The EMA at 0.999 needs a few thousand steps to be worth looking at, which is
                # exactly what an overfit run does not have.
                with nullcontext() if args.overfit else trainer.ema.applied():
                    scores = trainer.validate(val_loader, nfe=nfe, max_batches=8)
                logger.info("step %6d | %s", step,
                            " | ".join(f"{k} {v:.2f}" for k, v in sorted(scores.items())))
                run.log(scores, step=step)

            if (step + 1) % log_cfg["save_every"] == 0 or trainer.step == total_steps:
                trainer.save_checkpoint(out_dir / f"flow_{trainer.step:06d}.pt")
    finally:
        run.finish()

    trainer.save_checkpoint(out_dir / "flow_last.pt")
    logger.info("Done at step %d.", trainer.step)


if __name__ == "__main__":
    main()
