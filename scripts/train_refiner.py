#!/usr/bin/env python3
"""Train the Stage 2 local refiner model to sharpen brightness."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.data.seeding import generator_for  # noqa: E402
from dawn.eval.metrics import masked_mean, masked_psnr  # noqa: E402
from dawn.flow.loaders import build_loaders, cycle, load_config  # noqa: E402
from dawn.flow.targets import FitSettings, build_targets, prepare_batch  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import EMA, _plain, budget_summary, cosine_warmup_schedule  # noqa: E402
from dawn.models.refiner import LocalRefiner  # noqa: E402
from dawn.progress import format_duration  # noqa: E402
from dawn.runlog import RunLog, write_run_config  # noqa: E402

WATCH_SCOPES: Mapping[str, tuple[str, ...]] = {"none": (), "refiner": ("refiner",)}
CROP_SALT = "refiner_crop"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def crop_windows(
    size: tuple[int, int],
    crop_size: int,
    image_ids: Sequence[str],
    generator: torch.Generator | None = None,
) -> list[tuple[int, int, int, int]]:
    """Generate spatial crop window coordinates for a batch of images."""
    if int(crop_size) < 1:
        raise ValueError(f"refiner.crop_size must be positive, got {crop_size}")
    height, width = int(size[0]), int(size[1])
    out_h, out_w = min(int(crop_size), height), min(int(crop_size), width)
    windows = []
    for image_id in image_ids:
        rng = generator if generator is not None else generator_for(image_id, CROP_SALT)
        top = int(torch.randint(0, height - out_h + 1, (1,), generator=rng))
        left = int(torch.randint(0, width - out_w + 1, (1,), generator=rng))
        windows.append((top, left, out_h, out_w))
    return windows


def crop_batch(
    tensors: Mapping[str, torch.Tensor], windows: Sequence[tuple[int, int, int, int]]
) -> dict[str, torch.Tensor]:
    """Crop a dictionary of spatial batch tensors using corresponding window coordinates."""
    out = {}
    for name, value in tensors.items():
        if value.shape[0] != len(windows):
            raise ValueError(
                f"{name} has batch {value.shape[0]} but there are {len(windows)} crop windows"
            )
        out[name] = torch.stack(
            [
                value[index, :, top : top + height, left : left + width]
                for index, (top, left, height, width) in enumerate(windows)
            ]
        )
    return out


def improvement_verdict(scores: Mapping[str, float]) -> str:
    """Format a summary string describing refiner performance relative to the baseline grid."""
    if "psnr_L/improvement" not in scores:
        return "no validated images, so no verdict on the refiner"
    gain = scores["psnr_L/improvement"]
    pushed = scores.get("residual/abs_mean_Lstar", float("nan"))
    if gain > 0.0:
        return (
            f"the refiner EARNS its place: {gain:+.3f} dB of L* PSNR over the unrefined grid, "
            f"moving {pushed:.2f} L* on average"
        )
    return (
        f"the refiner does NOT earn its place: {gain:+.3f} dB against the unrefined grid while "
        f"moving {pushed:.2f} L* on average. Stage 2 is the plan's first cut; cut it"
    )


def _param_groups(refiner: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Separate model parameters into decayed weights and non-decayed biases."""
    decayed = [p for p in refiner.parameters() if p.requires_grad and p.ndim > 1]
    plain = [p for p in refiner.parameters() if p.requires_grad and p.ndim <= 1]
    groups = [{"params": decayed, "weight_decay": weight_decay, "name": "weights"}]
    if plain:
        groups.append({"params": plain, "weight_decay": 0.0, "name": "biases"})
    return groups


class RefinerTrainer:
    """Trainer for the Stage 2 local brightness refiner model."""

    def __init__(
        self,
        cfg: Mapping,
        refiner: LocalRefiner,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.refiner = refiner.to(self.device)
        self.layout = TokenLayout.from_config(cfg)
        self.fit = FitSettings.from_config(cfg)
        self.policy = cfg["pairs"]
        self.input_size = tuple(int(v) for v in cfg["encoder"]["input_size"])

        section, optim = cfg["refiner"], cfg["optim"]
        self.crop_size = int(section["crop_size"])
        self.residual_penalty = float(section["residual_penalty"])
        self.num_train_steps = int(section["num_train_steps"])
        self.grad_clip = float(optim["grad_clip"])
        self.optimizer = torch.optim.AdamW(
            _param_groups(self.refiner, float(optim["weight_decay"])),
            lr=float(optim["lr"]),
            betas=tuple(optim["betas"]),
        )
        self.scheduler = cosine_warmup_schedule(
            self.optimizer, int(optim["lr_warmup_steps"]), self.num_train_steps
        )
        self.ema = EMA(
            {f"refiner.{n}": p for n, p in self.refiner.named_parameters() if p.requires_grad},
            decay=float(optim["ema_decay"]),
            warmup=bool(optim.get("ema_warmup", True)),
        )
        self.step = 0
        self.generator = torch.Generator().manual_seed(int(seed))

    @classmethod
    def from_config(
        cls, cfg: Mapping, device: torch.device | str = "cpu", seed: int = 0
    ) -> RefinerTrainer:
        """Build the refiner from config and wrap it in a trainer."""
        return cls(cfg, LocalRefiner.from_config(cfg), device=device, seed=seed)

    def compute_losses(
        self, raw_batch: Mapping[str, Any], generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Compute training losses on cropped patches of a batch."""
        batch = prepare_batch(raw_batch, self.input_size, self.policy, self.device)
        thermal, mask = batch["thermal"], batch["mask"]
        with torch.no_grad():
            targets = build_targets(thermal, batch["rgb"], mask, self.layout, self.fit)
        cropped = crop_batch(
            {
                "thermal": thermal,
                "L_grid": targets.L_grid,
                "L_true": targets.lab[:, :1] / 100.0,
                "mask": mask,
            },
            crop_windows(
                tuple(thermal.shape[-2:]), self.crop_size, batch["image_id"], generator
            ),
        )
        residual = self.refiner.residual(cropped["L_grid"], cropped["thermal"])
        refined = (cropped["L_grid"] + residual).clamp(0.0, 1.0)
        l1 = masked_mean((refined - cropped["L_true"]).abs(), cropped["mask"]).mean()
        residual_abs = masked_mean(residual.abs(), cropped["mask"]).mean()
        return {
            "total": l1 + self.residual_penalty * residual_abs,
            "l1": l1,
            "residual_abs": residual_abs,
        }

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """Execute a single optimization step and update model parameters."""
        self.refiner.train()
        losses = self.compute_losses(batch, generator=self.generator)

        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.refiner.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()
        self.step += 1

        out = {name: float(value.detach()) for name, value in losses.items()}
        out["residual_abs_Lstar"] = 100.0 * out["residual_abs"]
        out["grad_norm"] = float(grad_norm)
        out["lr"] = float(self.optimizer.param_groups[0]["lr"])
        out["ema_decay"] = self.ema.current_decay()
        return out

    @torch.no_grad()
    def validate(
        self, loader: Iterable[Mapping[str, Any]], max_batches: int | None = None
    ) -> dict[str, float]:
        """Compute masked lightness PSNR and improvement metrics across validation pairs."""
        self.refiner.eval()
        totals: dict[str, float] = {}
        count = 0

        def record(key: str, values: torch.Tensor, keep: torch.Tensor) -> None:
            totals[key] = totals.get(key, 0.0) + float(values[keep].sum())

        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = prepare_batch(raw, self.input_size, self.policy, self.device)
            thermal, mask = batch["thermal"], batch["mask"]
            targets = build_targets(thermal, batch["rgb"], mask, self.layout, self.fit)
            keep = mask.sum(dim=(1, 2, 3)) > 0
            if not keep.any():
                continue
            l_true = targets.lab[:, :1]  # L*, the units masked_psnr's peak of 100 is in
            residual = self.refiner.residual(targets.L_grid, thermal)
            refined = (targets.L_grid + residual).clamp(0.0, 1.0)
            record("psnr_L/grid", masked_psnr(100.0 * targets.L_grid, l_true, mask, 100.0), keep)
            record("psnr_L/refined", masked_psnr(100.0 * refined, l_true, mask, 100.0), keep)
            record("residual/abs_mean_Lstar", 100.0 * masked_mean(residual.abs(), mask), keep)
            count += int(keep.sum())

        out = {key: value / max(count, 1) for key, value in totals.items()}
        if "psnr_L/refined" in out:
            out["psnr_L/improvement"] = out["psnr_L/refined"] - out["psnr_L/grid"]
        out["num_images"] = float(count)
        return out

    # --- checkpoints ----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return state dictionary containing model weights, optimizer, and scheduler states."""
        return {
            "step": self.step,
            "config": _plain(self.cfg),
            "refiner": self.refiner.state_dict(),
            "refiner_config": _plain(self.cfg["refiner"]),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore training state from a saved checkpoint state dictionary."""
        saved = dict(state.get("refiner_config") or {})
        bound = saved.get("residual_scale")
        if bound is not None and float(bound) != self.refiner.residual_scale:
            raise ValueError(
                f"the checkpoint was trained with residual_scale {float(bound)} and this run uses "
                f"{self.refiner.residual_scale}: the bound is not a parameter, so loading would "
                "silently change what the weights mean"
            )
        self.step = int(state["step"])
        self.refiner.load_state_dict(state["refiner"])
        self.ema.load_state_dict(state["ema"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

    def save_checkpoint(self, path: str | Path) -> Path:
        """Serialize and save current trainer state dictionary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("saved checkpoint %s (step %d)", path, self.step)
        return path


def main():
    """Execute Stage 2 refiner training workflow."""
    parser = argparse.ArgumentParser(description="Train the local refiner (Stage 2)")
    parser.add_argument(
        "--config", type=str, default="configs/flow.yaml", help="Stage-1 base config"
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument(
        "--refiner-config",
        type=str,
        default="configs/refiner.yaml",
        help="Stage-2 config, merged on top of --config and --data-config",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Extra YAML files merged last, in order (e.g. configs/debug.yaml)",
    )
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", help="cuda | mps | cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (weights and crops)")
    parser.add_argument(
        "--overfit",
        type=int,
        default=None,
        metavar="N",
        help="Train AND validate on the first N train pairs, augmentation off, live (not EMA) "
        "weights: the sanity check that the refiner can beat the grid on data it has seen. If "
        "the improvement is not positive HERE, it will not be positive on held-out pairs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override refiner.num_train_steps (the LR schedule follows). For quick probes, and "
        "the only way to shorten a run with configs/debug.yaml, which cannot set a stage-2 key.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = load_config(args.config, args.data_config, [args.refiner_config, *args.override])
    if "refiner" not in cfg:
        raise ValueError(
            f"{args.refiner_config} has no `refiner:` section: stage 2's own settings (crop size, "
            "residual penalty, step budget) live there and nowhere else"
        )
    if args.max_steps:
        cfg["refiner"]["num_train_steps"] = args.max_steps
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)

    logger.info("Stage 2: local refiner training")
    logger.info(
        f"  Config: {args.config} + {args.refiner_config} "
        f"(+ {', '.join(args.override) or 'no overrides'})"
    )
    logger.info(
        "  Device: %s, fp32 throughout (2,545 parameters: bf16 autocast would buy nothing, and "
        "the targets, the fit and the losses are float32 by convention)",
        device,
    )
    logger.info(
        "  TRAINS ON THE FITTED (oracle) GRID, RUNS ON THE GENERATED ONE: the measured "
        "improvement is an upper bound on the one inference will see (module docstring)"
    )

    loaders = build_loaders(cfg, overfit=args.overfit)
    if args.overfit:
        logger.info(f"  Overfit mode: train and validate on the first {args.overfit} train pairs")
    logger.info(
        "  Train pairs: %d; validation: %s",
        loaders.train_size,
        {k: len(v.dataset) for k, v in loaders.val.items()},
    )

    trainer = RefinerTrainer.from_config(cfg, device, seed=args.seed)
    logger.info(
        "  Refiner: %d parameters, %d x %d receptive field, residual bounded to +/- %.1f L*",
        trainer.refiner.num_params(),
        trainer.refiner.receptive_field,
        trainer.refiner.receptive_field,
        100.0 * trainer.refiner.residual_scale,
    )
    logger.info(
        "  Loss: L1 to the RGB brightness + %.3g * |residual|, masked, on %d x %d crops of "
        "%s frames (the grid is fit on the FULL frame first)",
        trainer.residual_penalty,
        min(trainer.crop_size, trainer.input_size[0]),
        min(trainer.crop_size, trainer.input_size[1]),
        list(trainer.input_size),
    )

    if args.resume:
        trainer.load_state_dict(torch.load(args.resume, map_location=device, weights_only=False))
        logger.info("  Resumed from %s at step %d", args.resume, trainer.step)

    total_steps = int(cfg["refiner"]["num_train_steps"])
    budget = budget_summary(
        {**cfg, "flow": {**cfg["flow"], "num_train_steps": total_steps}}, loaders.train_size
    )
    logger.info(
        "  Budget: %.0f steps/epoch x %.1f epochs (%d steps at batch %d over %d train pairs); "
        "%.0f validation points; %.0f periodic checkpoints",
        budget["steps_per_epoch"],
        budget["epochs"],
        total_steps,
        int(cfg["optim"]["batch_size"]),
        loaders.train_size,
        budget["val_points"],
        budget["save_points"],
    )
    logger.info("  Wrote %s", write_run_config(cfg, out_dir))

    run = RunLog(cfg, out_dir)
    run.watch(trainer, cfg, WATCH_SCOPES)
    log_cfg = cfg["logging"]
    headline = cfg["pairs"]["eval_subset"]  # which validation subset selects the best checkpoint
    logger.info(
        "  refiner_best.pt follows psnr_L/improvement on the %s subset (higher is better); "
        "<= 0 at the end of the run means stage 2 should be cut",
        headline,
    )
    best = float("-inf")
    batches = cycle(loaders.train)
    step_times: list[float] = []
    started_at = time.perf_counter()

    try:
        while trainer.step < total_steps:
            step = trainer.step  # the step number this batch trains
            started = time.perf_counter()
            losses = trainer.train_step(next(batches))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            losses["sec_per_step"] = time.perf_counter() - started
            step_times.append(losses["sec_per_step"])
            eta = format_duration((total_steps - trainer.step) * step_times[-1])

            if step % log_cfg["log_every"] == 0:
                values = {(k if "/" in k else f"loss/{k}"): v for k, v in losses.items()}
                values["progress/epoch"] = step / budget["steps_per_epoch"]
                values["progress/step"] = float(step)
                logger.info(
                    "step %6d/%d (%2.0f%%) | total %.5f | L1 %.5f | |residual| %.3f L* | "
                    "%.2fs/step | ETA %s",
                    step,
                    total_steps,
                    100 * step / total_steps,
                    losses["total"],
                    losses["l1"],
                    losses["residual_abs_Lstar"],
                    losses["sec_per_step"],
                    eta,
                )
                run.log("train", step, values)

            if (step + 1) % log_cfg["val_every"] == 0 or trainer.step == total_steps:
                with nullcontext() if args.overfit else trainer.ema.applied():
                    for subset, loader in loaders.val.items():
                        scores = trainer.validate(loader)
                        logger.info(
                            "step %6d | val %-11s | %s | ETA %s",
                            trainer.step,
                            subset,
                            " | ".join(f"{k} {v:.3f}" for k, v in sorted(scores.items())),
                            eta,
                        )
                        logger.info(
                            "step %6d | val %-11s | %s",
                            trainer.step,
                            subset,
                            improvement_verdict(scores),
                        )
                        run.log(
                            f"val_{subset}",
                            trainer.step,
                            {f"val_{subset}/{k}": v for k, v in scores.items()},
                        )
                        if subset != headline:
                            continue
                        value = scores.get("psnr_L/improvement", float("-inf"))
                        if value > best:
                            best = value
                            trainer.save_checkpoint(out_dir / "refiner_best.pt")

            if (step + 1) % log_cfg["save_every"] == 0 or trainer.step == total_steps:
                trainer.save_checkpoint(out_dir / f"refiner_{trainer.step:06d}.pt")
    finally:
        run.close()

    trainer.save_checkpoint(out_dir / "refiner_last.pt")
    steady = step_times[5:] or step_times
    logger.info(
        "Done at step %d: %.0fs wall, median %.2fs/step. Best psnr_L/improvement on %s: %s.",
        trainer.step,
        time.perf_counter() - started_at,
        statistics.median(steady) if steady else float("nan"),
        headline,
        f"{best:+.3f} dB" if best > float("-inf") else "never validated",
    )


if __name__ == "__main__":
    main()
