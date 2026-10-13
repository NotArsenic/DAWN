#!/usr/bin/env python3
"""Train the Stage 1 flow matching model on aligned thermal and RGB image pairs."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.flow.loaders import build_loaders, cycle, load_config, stats_batches  # noqa: E402
from dawn.flow.targets import FitSettings, compute_token_stats, load_or_compute_stats  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import FlowTrainer, budget_summary  # noqa: E402
from dawn.progress import format_duration  # noqa: E402

# Re-exported for backward compatibility with external imports.
from dawn.runlog import CsvLog, JsonlLog, RunLog, WandbRun, write_run_config  # noqa: E402, F401

# How `logging.select_metric` picks the best checkpoint: (validation key, higher is better).
SELECTION = {
    "ciede2000": ("ciede2000/euler", False),
    "psnr_L": ("psnr_L/euler", True),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def token_stats(cfg: dict, layout: TokenLayout, device: torch.device):
    """Compute or load target token mean and standard deviation statistics."""
    fit, size = FitSettings.from_config(cfg), tuple(cfg["encoder"]["input_size"])
    count = int(cfg["grid"]["stats_num_samples"])
    make = lambda: stats_batches(cfg, count, device)  # noqa: E731
    if cfg["dronevehicle"].get("max_pairs"):
        logger.info("  max_pairs is set: computing token stats fresh, not touching the cache")
        return compute_token_stats(make(), layout, fit, device)
    key = {"num_samples": count, "manifest": "train_pairs"}
    return load_or_compute_stats(
        cfg["grid"]["stats_cache"], make, layout, fit, size, key_extra=key, device=device
    )


def main():
    """Execute Stage 1 flow training workflow."""
    parser = argparse.ArgumentParser(description="Train the flow model (Stage 1)")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Config file")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Extra YAML files merged on top, in order (e.g. configs/debug.yaml)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda | mps | cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        metavar="N",
        help="Train AND validate on the first N train pairs, augmentation off, live (not EMA) "
        "weights: the sanity check that the render can reach the oracle at all. Run this "
        "before any full training.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override flow.num_train_steps (the LR schedule follows). For quick probes.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.config, args.data_config, args.override)
    if args.max_steps:
        cfg["flow"]["num_train_steps"] = args.max_steps
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    logger.info("Stage 1: flow matching training")
    logger.info(f"  Config: {args.config} (+ {', '.join(args.override) or 'no overrides'})")
    logger.info(
        f"  Device: {device}, precision {cfg['optim']['precision']} "
        f"(bf16 only takes effect on CUDA)"
    )
    layout = TokenLayout.from_config(cfg)
    logger.info(
        f"  Resolution {cfg['encoder']['input_size']}: {layout.num_tokens} tokens of "
        f"{layout.dim} values, colour map {layout.color_size}, "
        f"brightness correction {'on' if layout.correction else 'off'}"
    )

    loaders = build_loaders(cfg, overfit=args.overfit)
    if args.overfit:
        logger.info(f"  Overfit mode: train and validate on the first {args.overfit} train pairs")
    logger.info(
        "  Train pairs: %d; validation: %s",
        loaders.train_size,
        {k: len(v.dataset) for k, v in loaders.val.items()},
    )

    stats = token_stats(cfg, layout, device)
    trainer = FlowTrainer.from_config(cfg, stats, device)
    for name, (trainable, total) in trainer.param_counts().items():
        logger.info("  %-11s %9d trainable / %9d total", name, trainable, total)

    if args.resume:
        trainer.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False))
        logger.info("  Resumed from %s at step %d", args.resume, trainer.step)

    # Every step-denominated knob, re-read in epochs. The trap this closes: raising
    # optim.batch_size without rescaling the step counts runs a different experiment, and the
    # worst case (a budget shorter than the detach warmup) looks like a healthy run.
    budget = budget_summary(cfg, loaders.train_size)
    logger.info(
        "  Budget: %.0f steps/epoch x %.1f epochs; conditioning detached for %.1f epochs, then "
        "ramped over %d steps; %.0f validation points; %.0f samples/step",
        budget["steps_per_epoch"],
        budget["epochs"],
        budget["detach_epochs"],
        trainer.warmup_ramp_steps,
        budget["val_points"],
        budget["samples_per_step"],
    )
    logger.info("  Wrote %s", write_run_config(cfg, out_dir))

    run = RunLog(cfg, out_dir)
    run.watch(trainer, cfg)
    log_cfg = cfg["logging"]
    total_steps = cfg["flow"]["num_train_steps"]
    headline = cfg["pairs"]["eval_subset"]  # which validation subset selects the best checkpoint
    select_name = str(log_cfg.get("select_metric", "ciede2000"))
    if select_name not in SELECTION:
        raise ValueError(
            f"logging.select_metric must be one of {sorted(SELECTION)}, got {select_name!r}"
        )
    select_key, select_higher = SELECTION[select_name]
    logger.info(
        "  flow_best.pt follows %s on the %s subset (%s is better)",
        select_key,
        headline,
        "higher" if select_higher else "lower",
    )
    best = float("-inf")
    val_loss_batches = log_cfg.get("val_loss_batches")
    batches = cycle(loaders.train)
    step_times: list[float] = []
    started_at = time.perf_counter()

    try:
        while trainer.step < total_steps:
            step = trainer.step  # the step number this batch trains
            if step == 1 and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)  # skip the one-off warmup allocs
            started = time.perf_counter()
            detailed = step % log_cfg["log_every"] == 0  # the extra pass only on logged steps
            losses = trainer.train_step(next(batches), detailed=detailed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            losses["sec_per_step"] = time.perf_counter() - started
            step_times.append(losses["sec_per_step"])
            if device.type == "cuda":
                losses["peak_vram_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            if step == 1 and device.type == "cuda":
                logger.info(
                    "  Peak VRAM at batch %d x %d timesteps: %.2f GB",
                    cfg["optim"]["batch_size"],
                    cfg["flow"]["timesteps_per_image"],
                    losses["peak_vram_gb"],
                )

            if detailed:
                # `grad/...` and `contrib/...` already carry their own panel prefix; everything
                # else is a loss or a schedule value and goes under loss/.
                values = {(k if "/" in k else f"loss/{k}"): v for k, v in losses.items()}
                values["progress/epoch"] = step / budget["steps_per_epoch"]
                values["progress/step"] = float(step)
                if log_cfg["log_attn_gates"]:
                    values.update(
                        {f"gate/{k}": v for k, v in trainer.velocity.attn_gates().items()}
                    )
                segments = " ".join(
                    f"{name[5:8]} {losses[name]:.3f}"
                    for name in sorted(losses)
                    if name.startswith("flow_")
                )
                left = (total_steps - step) * losses["sec_per_step"]
                logger.info(
                    "step %6d/%d (%2.0f%%) | total %.4f | flow %.4f (%s) | render %.4f | "
                    "align %.4f | color %.4f | %.2fs/step | ETA %s",
                    step,
                    total_steps,
                    100 * step / total_steps,
                    losses["total"],
                    losses["flow"],
                    segments,
                    losses["render"],
                    losses["align"],
                    losses["color"],
                    losses["sec_per_step"],
                    format_duration(left),
                )
                run.log("train", step, values)

            if (step + 1) % log_cfg["val_every"] == 0 or trainer.step == total_steps:
                nfe = cfg["flow"]["num_inference_steps"]
                with nullcontext() if args.overfit else trainer.ema.applied():
                    for subset, loader in loaders.val.items():
                        scores = trainer.validate(loader, nfe=nfe)
                        logger.info(
                            "step %6d | val %-11s | %s",
                            trainer.step,
                            subset,
                            " | ".join(f"{k} {v:.3f}" for k, v in sorted(scores.items())),
                        )
                        run.log(
                            f"val_{subset}",
                            trainer.step,
                            {f"val_{subset}/{k}": v for k, v in scores.items()},
                        )
                        if subset != headline:
                            continue
                        value = scores[select_key] * (1.0 if select_higher else -1.0)
                        if value > best:
                            best = value
                            trainer.save_checkpoint(out_dir / "flow_best.pt")

                for subset, loader in loaders.val.items():
                    held = trainer.validate_loss(loader, max_batches=val_loss_batches)
                    logger.info(
                        "step %6d | val %-11s | held-out loss %s",
                        trainer.step,
                        subset,
                        " | ".join(
                            f"{k.split('/')[-1]} {v:.4f}"
                            for k, v in sorted(held.items())
                            if k.startswith("loss/")
                        ),
                    )
                    run.log(
                        f"valloss_{subset}",
                        trainer.step,
                        {f"valloss_{subset}/{k}": v for k, v in held.items()},
                    )

            if (step + 1) % log_cfg["save_every"] == 0 or trainer.step == total_steps:
                trainer.save_checkpoint(out_dir / f"flow_{trainer.step:06d}.pt")
    finally:
        run.close()

    trainer.save_checkpoint(out_dir / "flow_last.pt")
    steady = step_times[5:] or step_times
    summary = (
        f"Done at step {trainer.step}: {time.perf_counter() - started_at:.0f}s wall, "
        f"median {statistics.median(steady):.2f}s/step"
    )
    if device.type == "cuda":
        summary += f", peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GB"
    logger.info(summary + ".")


if __name__ == "__main__":
    main()
