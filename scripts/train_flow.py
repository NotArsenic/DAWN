#!/usr/bin/env python3
"""Stage 1: train the flow model (day thermal -> brightness grid + colour map).

Trains the LoRA-tuned DINOv3 encoder, the velocity transformer and the training-only colour head
by conditional flow matching on the aligned DroneVehicle day pairs (`dawn.data.pairs`).

Usage:
    python scripts/train_flow.py --config configs/flow.yaml                  # the real run (CUDA)
    python scripts/train_flow.py --config configs/flow.yaml \
        --override configs/debug.yaml --device cpu          # tiny end-to-end run on real pairs
    python scripts/train_flow.py --config configs/flow.yaml \
        --override configs/mps.yaml --device mps --overfit 100   # overfit-100 on a Mac, fp32
    python scripts/train_flow.py --config configs/flow.yaml \
        --override configs/gpu_probe.yaml --device cuda --max-steps 50   # VRAM + sec/step probe

The loop itself lives in `dawn.flow.trainer`; this file is configuration, logging and checkpoint
plumbing. Token layout, raw vs standardized and the pixel masks: docs/conventions.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
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

# How `logging.select_metric` picks the best checkpoint: (validation key, higher is better).
SELECTION = {
    "ciede2000": ("ciede2000/euler", False),
    "psnr_L": ("psnr_L/euler", True),
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


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

    SCOPES = {"none": (), "velocity": ("velocity",), "all": ("velocity", "encoder", "color_head")}

    def watch(self, trainer, cfg) -> None:
        """Weight and gradient histograms through `wandb.watch`'s hooks.

        `wandb.watch` takes whole modules, not parameters, so the scope is the unit of choice:

            velocity  the 7M-parameter net that learns the field. The default.
            all       plus the encoder and the colour head. The encoder is 29M parameters of
                      FROZEN DINO backbone around ~0.5M of LoRA, and watching it means hooking
                      all of them; use it to debug LoRA, not for a normal run.
            none      off.

        `watch_freq` throttles the logging, not the hooks: those run on every backward pass
        whatever it says, which is why the scope matters more than the frequency.

        Raises:
            ValueError: `logging.watch_scope` is not one of the three.
        """
        scope = str(cfg["logging"].get("watch_scope", "none"))
        if scope not in self.SCOPES:
            raise ValueError(
                f"logging.watch_scope must be one of {sorted(self.SCOPES)}, got {scope!r}"
            )
        if self.run is None or scope == "none":
            return
        import wandb

        freq = int(cfg["logging"].get("watch_freq", 500))
        wandb.watch(
            [getattr(trainer, name) for name in self.SCOPES[scope]], log="all", log_freq=freq
        )
        logger.info("  wandb.watch: %s, histograms every %d steps", scope, freq)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class JsonlLog:
    """One JSON object per line under the output directory: curves without wandb."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, step: int, kind: str, values: dict) -> None:
        self.handle.write(json.dumps({"step": step, "kind": kind, **values}) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class CsvLog:
    """One CSV per kind of record, rewritten whole on every append.

    W&B owns the live curves; these files exist so every plot can be rebuilt from disk months
    later, by someone without the W&B run, in a spreadsheet. One file per kind keeps the columns
    stable and dense -- a single wide file would be mostly empty, because train rows arrive every
    `log_every` steps and validation rows every `val_every`.

    Rewriting the whole file each time (rather than appending) costs nothing at this scale -- a
    20K-step run writes about 200 train rows and 40 of each validation kind -- and buys two
    things: a column that only appears later (`beta1`, `peak_vram_gb`) lands in the header
    retroactively, and the file on disk is always complete, even if the run is killed.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list[dict]] = {}
        self.columns: dict[str, list[str]] = {}

    def write(self, step: int, kind: str, values: dict) -> None:
        row = {"step": step, **{k: v for k, v in values.items()}}
        rows = self.rows.setdefault(kind, [])
        columns = self.columns.setdefault(kind, [])
        rows.append(row)
        columns.extend(key for key in row if key not in columns)
        path = self.directory / f"{kind}.csv"
        partial = path.with_suffix(".partial")
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)


class RunLog:
    """The one place a metric is recorded: W&B, a JSONL mirror and a CSV per kind.

    Callers pass the final W&B key names, so the three records hold identical keys by
    construction and the local files really can rebuild the W&B plots. Before this existed the
    JSONL held the raw loss dict while W&B got the prefixed one plus the attention gates, so the
    two quietly disagreed about what a run had recorded.
    """

    def __init__(self, cfg, out_dir: Path):
        self.wandb = WandbRun(cfg)
        self.jsonl = JsonlLog(out_dir / "train_log.jsonl")
        self.csv = CsvLog(out_dir / "metrics") if cfg["logging"].get("metrics_csv", True) else None

    def log(self, kind: str, step: int, values: dict) -> None:
        self.wandb.log(values, step=step)
        self.jsonl.write(step, kind, values)
        if self.csv is not None:
            self.csv.write(step, kind, values)

    def watch(self, trainer, cfg) -> None:
        self.wandb.watch(trainer, cfg)

    def close(self) -> None:
        self.wandb.finish()
        self.jsonl.close()


def write_run_config(cfg: dict, out_dir: Path) -> Path:
    """The fully merged config beside the metrics, so a plot can be traced to its settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def token_stats(cfg: dict, layout: TokenLayout, device: torch.device):
    """Per-segment stats of the targets: cached, except when `dronevehicle.max_pairs` is set.

    A debug run on a handful of pairs must never poison `grid.stats_cache` with stats that would
    silently become the standardization of the next real run.
    """
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
                # An ETA on every logged line, because a 20K-step run on a remote box is watched
                # through `tail -f` and "is this finishing tonight?" should not need arithmetic.
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
                # The EMA at 0.999 needs a few thousand steps to be worth looking at, which is
                # exactly what an overfit run does not have.
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

                # The training objective on held-out pairs: a train curve falling while this one
                # turns up is the memorization signal, and nothing else in the run reports it.
                # Deliberately OUTSIDE the EMA context -- the train curve it is read against is
                # on the live weights, and comparing an averaged model with an instantaneous one
                # would put a gap between the two that has nothing to do with overfitting.
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
