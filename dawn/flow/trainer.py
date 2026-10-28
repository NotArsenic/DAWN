"""Training loop, validation routines, and checkpoint management for flow matching."""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dawn.data.color import linear_to_lab, linear_to_srgb
from dawn.data.resample import area_downsample
from dawn.eval.metrics import ciede2000, colorfulness, masked_mean, masked_psnr
from dawn.flow.color_bins import patch_histograms, patch_mask_fraction
from dawn.flow.loss import FlowLoss
from dawn.flow.render import RenderSettings, render
from dawn.flow.sampler import sample_parts
from dawn.flow.targets import FitSettings, build_targets, prepare_batch
from dawn.flow.tokens import TokenLayout
from dawn.grid.normalize import standardize, unstandardize
from dawn.models.color_head import ColorHead
from dawn.models.encoder import DINOEncoder
from dawn.models.velocity import VelocityTransformer

logger = logging.getLogger(__name__)

# Parameters that never get weight decay.
NO_DECAY_SUFFIXES = ("bias", "dino_gate", "pos_embed")


class EMA:
    """Exponential moving average tracker for model parameters."""

    def __init__(
        self, params: Mapping[str, nn.Parameter], decay: float = 0.999, warmup: bool = True
    ):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.params = dict(params)
        self.shadow = {name: p.detach().clone().float() for name, p in self.params.items()}

    def current_decay(self) -> float:
        """Compute effective decay factor for current update step."""
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self) -> None:
        decay = self.current_decay()
        self.num_updates += 1
        for name, param in self.params.items():
            self.shadow[name].lerp_(param.detach().float(), 1.0 - decay)

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Context manager temporarily swapping parameter weights with shadow copies."""
        backup = {name: p.detach().clone() for name, p in self.params.items()}
        try:
            with torch.no_grad():
                for name, param in self.params.items():
                    param.copy_(self.shadow[name].to(param.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, param in self.params.items():
                    param.copy_(backup[name])

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore EMA shadow parameters and update count."""
        self.decay = float(state["decay"])
        self.warmup = bool(state.get("warmup", self.warmup))
        self.num_updates = int(state.get("num_updates", 10**9))
        for name, value in state["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(value.to(self.shadow[name].device))



def split_param_groups(
    encoder: nn.Module,
    velocity: nn.Module,
    color_head: nn.Module,
    lr: float,
    lora_lr: float,
    weight_decay: float,
) -> list[dict]:
    """Split parameters into AdamW parameter groups for LoRA and main modules.

    Separates trainable weights into LoRA and main groups, excluding biases,
    normalization parameters, gates, and positional embeddings from weight decay.
    """
    groups = {
        (scope, decay): {
            "params": [],
            "lr": lora_lr if scope == "lora" else lr,
            "weight_decay": weight_decay if decay else 0.0,
            "name": f"{scope}_{decay}",
        }
        for scope in ("lora", "main")
        for decay in (True, False)
    }
    for scope, module in (("lora", encoder), ("main", velocity), ("main", color_head)):
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            decay = param.ndim > 1 and not name.endswith(NO_DECAY_SUFFIXES)
            groups[(scope, decay)]["params"].append(param)
    return [group for group in groups.values() if group["params"]]


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_factor: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over `warmup_steps`, then cosine decay to `min_factor` at `total_steps`."""

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def one_cycle_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    pct_start: float = 0.2,
    div_factor: float = 25.0,
    final_div_factor: float = 1.0e4,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create a 1-cycle learning rate schedule with cosine warmup and anneal.

    Args:
        optimizer: Optimizer containing parameter groups with target peak learning rates.
        total_steps: Total number of training steps.
        pct_start: Fraction of total steps spent ramping up.
        div_factor: Initial learning rate divisor relative to peak learning rate.
        final_div_factor: Final learning rate divisor relative to peak learning rate.

    Raises:
        ValueError: If pct_start is not in (0, 1) or divisors are non-positive.
    """
    if not 0.0 < pct_start < 1.0:
        raise ValueError(f"one_cycle.pct_start must be in (0, 1), got {pct_start}")
    if div_factor <= 0 or final_div_factor <= 0:
        raise ValueError("one_cycle div factors must be positive")
    warm = max(1, int(round(pct_start * total_steps)))
    start, end = 1.0 / div_factor, 1.0 / final_div_factor

    def factor(step: int) -> float:
        if step < warm:
            progress = step / warm
            return start + (1 - start) * 0.5 * (1 - math.cos(math.pi * progress))
        span = max(1, total_steps - warm)
        progress = min(1.0, (step - warm) / span)
        return end + (1 - end) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class MomentumCycler:
    """Cycles optimizer beta1 inversely with learning rate for 1-cycle scheduling."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        pct_start: float = 0.2,
        max_beta1: float = 0.95,
        min_beta1: float = 0.85,
        enabled: bool = True,
    ):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warm = max(1, int(round(pct_start * total_steps)))
        self.max_beta1, self.min_beta1 = float(max_beta1), float(min_beta1)
        self.enabled = bool(enabled)
        self.step_count = 0
        if self.enabled:
            self._apply(self.max_beta1)

    def _apply(self, beta1: float) -> None:
        for group in self.optimizer.param_groups:
            if "betas" in group:
                group["betas"] = (beta1, group["betas"][1])

    def beta1(self, step: int) -> float:
        if step < self.warm:
            progress = step / self.warm
            span = self.max_beta1 - self.min_beta1
            return self.max_beta1 - span * 0.5 * (1 - math.cos(math.pi * progress))
        span_steps = max(1, self.total_steps - self.warm)
        progress = min(1.0, (step - self.warm) / span_steps)
        span = self.max_beta1 - self.min_beta1
        return self.min_beta1 + span * 0.5 * (1 - math.cos(math.pi * progress))

    def step(self) -> None:
        self.step_count += 1
        if self.enabled:
            self._apply(self.beta1(self.step_count))

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "enabled": self.enabled}

    def load_state_dict(self, state: Mapping) -> None:
        self.step_count = int(state["step_count"])
        self.enabled = bool(state.get("enabled", self.enabled))
        if self.enabled:
            self._apply(self.beta1(self.step_count))


def build_schedule(
    optimizer: torch.optim.Optimizer, cfg: Mapping
) -> tuple[torch.optim.lr_scheduler.LambdaLR, MomentumCycler]:
    """Build the learning rate scheduler and momentum cycler configured by cfg."""
    optim, flow = cfg["optim"], cfg["flow"]
    total = int(flow["num_train_steps"])
    name = str(optim.get("schedule", "cosine"))
    if name == "cosine":
        floor = float((optim.get("cosine") or {}).get("min_factor", 0.0) or 0.0)
        if not 0.0 <= floor < 1.0:
            raise ValueError(f"optim.cosine.min_factor must be in [0, 1), got {floor}")
        scheduler = cosine_warmup_schedule(
            optimizer, int(optim.get("lr_warmup_steps", 0)), total, min_factor=floor
        )
        return scheduler, MomentumCycler(optimizer, total, enabled=False)
    if name == "one_cycle":
        one = dict(optim.get("one_cycle", {}) or {})
        pct_start = float(one.get("pct_start", 0.2))
        scheduler = one_cycle_schedule(
            optimizer,
            total,
            pct_start=pct_start,
            div_factor=float(one.get("div_factor", 25.0)),
            final_div_factor=float(one.get("final_div_factor", 1.0e4)),
        )
        cycler = MomentumCycler(
            optimizer,
            total,
            pct_start=pct_start,
            max_beta1=float(one.get("max_beta1", 0.95)),
            min_beta1=float(one.get("min_beta1", 0.85)),
            enabled=bool(one.get("cycle_momentum", True)),
        )
        return scheduler, cycler
    raise ValueError(f"optim.schedule must be 'cosine' or 'one_cycle', got {name!r}")


def budget_summary(cfg: Mapping, train_size: int) -> dict[str, float]:
    """Compute epoch-denominated training budget metrics for logging.

    Args:
        cfg: Merged training configuration.
        train_size: Number of training pairs in the dataset.

    Returns:
        Dictionary mapping budget metric names to epoch and step counts.
    """
    flow, optim, log = cfg["flow"], cfg["optim"], cfg["logging"]
    batch, total = int(optim["batch_size"]), int(flow["num_train_steps"])
    per_epoch = max(train_size, 1) / max(batch, 1)
    return {
        "train_pairs": float(train_size),
        "steps_per_epoch": per_epoch,
        "epochs": total / per_epoch,
        "detach_epochs": int(flow["warmup_steps"]) / per_epoch,
        "val_points": total / max(int(log["val_every"]), 1),
        "save_points": total / max(int(log["save_every"]), 1),
        "samples_per_step": float(batch * int(flow["timesteps_per_image"])),
    }


@torch.no_grad()
def patch_retrieval(
    adapted: torch.Tensor,
    frozen: torch.Tensor,
    grid: tuple[int, int],
    weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate nearest-neighbor retrieval between adapted and frozen patch tokens.

    Computes top-1 and 1-neighbor patch retrieval accuracy between adapted thermal
    tokens and frozen RGB reference tokens.

    Args:
        adapted: Adapted patch tokens with shape `(B, N, C)`.
        frozen: Frozen reference patch tokens with shape `(B, N, C)`.
        grid: Token grid dimensions `(gy, gx)`.
        weight: Optional token weights with shape `(B, N)`.

    Returns:
        Dictionary with 'top1' and 'within_1' accuracy tensors of shape `(B,)`.
    """
    if adapted.shape != frozen.shape or adapted.ndim != 3:
        raise ValueError(
            f"adapted {tuple(adapted.shape)} and frozen {tuple(frozen.shape)} must be the same "
            "(B, N, C)"
        )
    gy, gx = grid
    num = adapted.shape[1]
    if gy * gx != num:
        raise ValueError(f"grid {grid} does not hold {num} tokens")
    similarity = F.normalize(adapted.float(), dim=-1) @ F.normalize(frozen.float(), dim=-1).mT
    best = similarity.argmax(dim=-1)  # (B, N)
    own = torch.arange(num, device=best.device).unsqueeze(0)
    near = (
        ((best // gx - own // gx).abs() <= 1) & ((best % gx - own % gx).abs() <= 1)
    ).float()
    hit = (best == own).float()
    if weight is None:
        weight = torch.ones_like(hit)
    weight = weight.to(hit.dtype)
    denominator = weight.sum(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
    return {
        "top1": (hit * weight).sum(dim=1) / denominator,
        "within_1": (near * weight).sum(dim=1) / denominator,
    }


def build_models(
    cfg: Mapping, layout: TokenLayout, device: torch.device
) -> tuple[DINOEncoder, VelocityTransformer, ColorHead]:
    """Instantiate the encoder, velocity transformer, and color head modules."""
    encoder = DINOEncoder.from_config(
        cfg["encoder"], detach_attn=cfg["velocity"]["detach_dino_attn"]
    ).to(device)
    if encoder.patch_grid != layout.grid:
        raise ValueError(
            f"the encoder's patch grid {encoder.patch_grid} and the token layout {layout.grid} "
            "disagree: both derive from encoder.input_size"
        )
    velocity = VelocityTransformer.from_config(cfg["velocity"], layout).to(device)
    color_head = ColorHead(dino_dim=encoder.embed_dim).to(device)
    return encoder, velocity, color_head


class FlowTrainer:
    """Coordinates model training, optimization, and evaluation for flow matching."""

    def __init__(
        self,
        cfg: Mapping,
        encoder: DINOEncoder,
        velocity: VelocityTransformer,
        color_head: ColorHead,
        stats: Mapping[str, Mapping[str, torch.Tensor]],
        layout: TokenLayout | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.layout = layout or TokenLayout.from_config(cfg)
        self.encoder = encoder.to(self.device)
        self.velocity = velocity.to(self.device)
        self.color_head = color_head.to(self.device)
        self.set_stats(stats)
        self.criterion = FlowLoss.from_config(cfg, self.layout)
        self.fit = FitSettings.from_config(cfg)
        self.render_settings = RenderSettings.from_config(cfg)
        if self.render_settings.layout != self.layout:
            self.render_settings = RenderSettings(
                self.layout,
                self.render_settings.bgu_grid,
                self.render_settings.bgu_ridge,
                self.render_settings.bgu_refine,
            )
        self.policy = cfg["pairs"]

        flow, optim, color = cfg["flow"], cfg["optim"], cfg["color"]
        self.input_size = tuple(int(v) for v in cfg["encoder"]["input_size"])
        self.render_size = tuple(int(v) for v in flow["render_size"])
        self._check_sizes()
        self.render_patch = (
            self.render_size[0] // self.layout.grid[0],
            self.render_size[1] // self.layout.grid[1],
        )
        self.input_patch = (
            self.input_size[0] // self.layout.grid[0],
            self.input_size[1] // self.layout.grid[1],
        )
        self.val_retrieval = bool(cfg["logging"].get("val_retrieval", True))
        self.soft_k, self.soft_sigma = int(color["soft_k"]), float(color["soft_sigma"])
        self.bin_count_weighting = bool(flow["bin_count_weighting"])
        self.timesteps_per_image = int(flow["timesteps_per_image"])
        self.warmup_steps = int(flow["warmup_steps"])
        ramp = flow.get("warmup_ramp_steps")
        self.warmup_ramp_steps = (
            max(1, self.warmup_steps // 6) if ramp is None else max(0, int(ramp))
        )
        self.num_train_steps = int(flow["num_train_steps"])
        self._warn_if_warmup_eats_the_run()
        self.render_subset = flow.get("render_subset") or None
        self.inference_nfe = int(flow["num_inference_steps"])
        self.grad_clip = float(optim["grad_clip"])
        self.precision = str(optim["precision"])
        if self.precision == "bf16" and self.device.type != "cuda":
            logger.warning(
                "optim.precision is bf16 but the device is %s: running fp32 (bf16 autocast is "
                "CUDA only; MPS/CPU bf16 is patchy)",
                self.device.type,
            )

        self.optimizer = torch.optim.AdamW(
            split_param_groups(
                encoder,
                velocity,
                color_head,
                lr=float(optim["lr"]),
                lora_lr=float(cfg["encoder"]["lora_lr"]),
                weight_decay=float(optim["weight_decay"]),
            ),
            betas=tuple(optim["betas"]),
        )
        self.scheduler, self.momentum = build_schedule(self.optimizer, cfg)
        self.ema = EMA(
            self.trainable_parameters(),
            decay=float(optim["ema_decay"]),
            warmup=bool(optim.get("ema_warmup", True)),
        )
        self.step = 0

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        stats: Mapping[str, Mapping[str, torch.Tensor]],
        device: torch.device | str = "cpu",
    ) -> FlowTrainer:
        device = torch.device(device)
        layout = TokenLayout.from_config(cfg)
        return cls(
            cfg, *build_models(cfg, layout, device), stats=stats, layout=layout, device=device
        )

    def _check_sizes(self) -> None:
        grid, color = self.layout.grid, self.layout.color_size
        render = self.render_size
        if render[0] % grid[0] or render[1] % grid[1]:
            raise ValueError(
                f"flow.render_size {render} must be a multiple of the token grid {grid}: "
                "L_color builds one histogram per patch at render size"
            )
        if render[0] < color[0] or render[1] < color[1]:
            raise ValueError(f"flow.render_size {render} is smaller than the colour map {color}")
        if render[0] > self.input_size[0] or render[1] > self.input_size[1]:
            raise ValueError(
                f"flow.render_size {render} is larger than encoder.input_size {self.input_size}"
            )

    def _warn_if_warmup_eats_the_run(self) -> None:
        """Warn if total training steps are too close to warmup steps."""
        total, warmup = int(self.num_train_steps), int(self.warmup_steps)
        if warmup and total < warmup * 2:
            verdict = (
                "the detach NEVER lifts" if total <= warmup else "under half the run trains jointly"
            )
            logger.warning(
                "flow.num_train_steps (%d) is close to flow.warmup_steps (%d): %s. If you raised "
                "optim.batch_size, rescale the step counts with it.",
                total,
                warmup,
                verdict,
            )

    def set_stats(self, stats: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        """Set per-segment target standardization statistics and vectors."""
        self.stats = {
            name: {key: value.detach().cpu() for key, value in segment.items()}
            for name, segment in stats.items()
        }
        mean, std = self.layout.stats_vector(self.stats)
        self.mean, self.std = mean.to(self.device), std.to(self.device)


    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        """Return all trainable model parameters prefixed by module name."""
        out = {}
        for prefix, module in self.modules().items():
            for name, param in module.named_parameters():
                if param.requires_grad:
                    out[f"{prefix}.{name}"] = param
        return out

    def modules(self) -> dict[str, nn.Module]:
        return {
            "encoder": self.encoder,
            "velocity": self.velocity,
            "color_head": self.color_head,
        }

    def param_counts(self) -> dict[str, tuple[int, int]]:
        """Return trainable and total parameter counts for each module and in total."""
        counts = {
            name: (
                sum(p.numel() for p in module.parameters() if p.requires_grad),
                sum(p.numel() for p in module.parameters()),
            )
            for name, module in self.modules().items()
        }
        counts["total"] = tuple(sum(values) for values in zip(*counts.values()))
        return counts

    def train(self) -> None:
        for module in self.modules().values():
            module.train()

    def eval(self) -> None:
        for module in self.modules().values():
            module.eval()

    def _autocast(self):
        """Context manager for bfloat16 mixed precision on CUDA devices."""
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def conditioning_scale(self) -> float:
        """Compute the gradient scaling factor for conditioning features at current step."""
        if self.step < self.warmup_steps:
            return 0.0
        if self.warmup_ramp_steps <= 0:
            return 1.0
        return min(1.0, (self.step - self.warmup_steps + 1) / self.warmup_ramp_steps)

    def compute_losses(
        self, raw_batch: Mapping[str, Any], generator: torch.Generator | None = None
    ) -> tuple[dict, dict]:
        """Compute all training losses for a single batch.

        Args:
            raw_batch: Batch dictionary from AlignedPairs loader.
            generator: Optional random number generator for time and noise draws.

        Returns:
            Tuple of (losses_dict, info_dict) containing computed loss tensors and conditioning metadata.
        """
        batch = prepare_batch(raw_batch, self.input_size, self.policy, self.device)
        thermal, rgb, mask, fallback = (
            batch["thermal"],
            batch["rgb"],
            batch["mask"],
            batch["fallback"],
        )
        layout, repeats = self.layout, self.timesteps_per_image

        with torch.no_grad():
            targets = build_targets(thermal, rgb, mask, layout, self.fit)
            node_weights = targets.node_weights
            if not self.bin_count_weighting:
                node_weights = torch.ones_like(node_weights)
            z_1 = standardize(layout.pack(targets.parts), self.mean, self.std)
            token_weights = layout.weights(node_weights, targets.fraction)
            thermal_r = area_downsample(thermal, self.render_size)
            lab_r = linear_to_lab(area_downsample(rgb, self.render_size).clamp(0.0, 1.0))
            mask_r = area_downsample(mask, self.render_size)
            histogram = patch_histograms(
                lab_r[:, 1:], mask_r, self.render_patch, self.soft_k, self.soft_sigma
            )
            patch_weight = patch_mask_fraction(mask_r, self.render_patch)
            sample_weight = self.criterion.sample_weight(fallback)

        scale = self.conditioning_scale()
        detach_cond = scale <= 0.0
        with self._autocast():
            cond = self.encoder.encode(thermal, return_attn=True)
            frozen = self.encoder.forward_frozen(rgb).patch_tokens
            logits = self.color_head(cond["patch_tokens"])
        feat_adapted = cond["patch_tokens"]
        if detach_cond:
            patch_tokens = feat_adapted.detach()
        elif scale >= 1.0:
            patch_tokens = feat_adapted
        else:
            patch_tokens = feat_adapted.detach() + scale * (feat_adapted - feat_adapted.detach())

        batch_size = thermal.shape[0]
        features = _repeat(patch_tokens, repeats)
        thumb = _repeat(cond["thumb"], repeats)
        attn_bias = _repeat(cond.get("attn_bias"), repeats)
        z_1_rep = _repeat(z_1, repeats)

        if generator is None:
            t = torch.rand(repeats * batch_size, device=self.device, dtype=torch.float32)
            z_0 = torch.randn_like(z_1_rep)
        else:
            t = torch.rand(repeats * batch_size, generator=generator, dtype=torch.float32)
            z_0 = torch.randn(z_1_rep.shape, generator=generator, dtype=torch.float32)
            t, z_0 = t.to(self.device), z_0.to(self.device)
        t_tok = t.view(-1, 1, 1)
        z_t = (1.0 - t_tok) * z_0 + t_tok * z_1_rep

        with self._autocast():
            v_pred = self.velocity(z_t, t, features, None, thumb, attn_bias)
        v_pred = v_pred.float()

        z1_hat = z_t + (1.0 - t_tok) * v_pred
        index = self._render_index(z1_hat.shape[0])
        parts = layout.unpack(unstandardize(z1_hat[index], self.mean, self.std))
        rendered = render(
            parts, _repeat(thermal_r, repeats)[index], self.render_size, self.render_settings
        )

        losses = self.criterion(
            v_pred=v_pred,
            z_0=z_0,
            z_1=z_1_rep,
            token_weights=_repeat(token_weights, repeats),
            lab_pred=rendered["lab"],
            lab_target=_repeat(lab_r, repeats)[index],
            pixel_weight=_repeat(mask_r, repeats)[index],
            sample_weight=_repeat(sample_weight, repeats)[index],
            feat_adapted=feat_adapted,
            feat_frozen=frozen,
            patch_weight=patch_weight,
            color_logits=logits,
            color_target=histogram,
            color_patch_weight=patch_weight,
        )
        return losses, {"detach_cond": detach_cond, "cond_scale": scale}

    def loss_contributions(self, losses: Mapping[str, torch.Tensor]) -> dict[str, float]:
        """Calculate the proportional contribution of each loss term to the weighted total."""
        criterion, out = self.criterion, {}
        total = float(losses["total"].detach())
        weights = {
            "flow": criterion.velocity_weight,
            "render": criterion.render_weight,
            "align": criterion.align_weight,
            "color": criterion.color_weight,
        }
        for name, weight in weights.items():
            if name in losses:
                out[name] = weight * float(losses[name].detach()) / total if total else 0.0
        return out

    def grad_norms(self) -> dict[str, float]:
        """Compute per-module gradient L2 norms before clipping."""
        out = {}
        for name, module in self.modules().items():
            total = 0.0
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    total += float(param.grad.detach().float().pow(2).sum())
            out[name] = math.sqrt(total)
        return out

    def train_step(self, batch: Mapping[str, Any], detailed: bool = False) -> dict[str, float]:
        """Execute a single training step and return logged metric values.

        Args:
            batch: Training data batch.
            detailed: Whether to compute detailed gradient norms and loss contributions.

        Returns:
            Dictionary of scalar training metrics including losses, norms, and learning rates.
        """
        self.train()
        losses, info = self.compute_losses(batch)

        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        extra = self.grad_norms() if detailed else {}
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]], self.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.momentum.step()
        self.ema.update()
        self.step += 1

        out = {name: float(value.detach()) for name, value in losses.items()}
        out["grad_norm"] = float(grad_norm)
        out["detach_cond"] = float(info["detach_cond"])
        out["cond_scale"] = float(info["cond_scale"])
        out["ema_decay"] = self.ema.current_decay()
        for group in self.optimizer.param_groups:
            out[f"lr_{group['name'].split('_')[0]}"] = float(group["lr"])
        if self.momentum.enabled:
            out["beta1"] = float(self.optimizer.param_groups[0]["betas"][0])
        if detailed:
            out.update({f"grad/{k}": v for k, v in extra.items()})
            out.update({f"contrib/{k}": v for k, v in self.loss_contributions(losses).items()})
        return out

    def _render_index(self, total: int) -> torch.Tensor:
        """Return sample indices selected for rendering loss evaluation."""
        if not self.render_subset or self.render_subset >= total:
            return torch.arange(total, device=self.device)
        return torch.randperm(total, device=self.device)[: int(self.render_subset)]

    # validation
    @torch.no_grad()
    def validate(
        self,
        loader: Iterable[Mapping[str, Any]],
        nfe: int | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Evaluate validation metrics comparing sampled predictions against ground truth and oracle.

        Args:
            loader: Validation data loader.
            nfe: Number of function evaluations for sampling steps.
            max_batches: Optional maximum number of batches to evaluate.

        Returns:
            Dictionary of aggregated validation metrics including psnr_L, ciede2000, and colorfulness.
        """
        self.eval()
        nfe = self.inference_nfe if nfe is None else nfe
        totals: dict[str, float] = {}
        count = 0

        def record(key: str, values: torch.Tensor, keep: torch.Tensor) -> None:
            totals[key] = totals.get(key, 0.0) + float(values[keep].sum())

        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = prepare_batch(raw, self.input_size, self.policy, self.device)
            thermal, rgb, mask = batch["thermal"], batch["rgb"], batch["mask"]
            targets = build_targets(thermal, rgb, mask, self.layout, self.fit)
            keep = mask.sum(dim=(1, 2, 3)) > 0
            if not keep.any():
                continue
            true_lab = targets.lab

            def score(name: str, rendered: Mapping[str, torch.Tensor]) -> None:
                psnr = masked_psnr(rendered["lab"][:, :1], true_lab[:, :1], mask, peak=100.0)
                delta = masked_mean(ciede2000(rendered["lab"], true_lab), mask)
                color = colorfulness(linear_to_srgb(rendered["rgb"]) * 255.0)
                record(f"psnr_L/{name}", psnr, keep)
                record(f"ciede2000/{name}", delta, keep)
                record(f"colorfulness/{name}", color, keep)

            score("oracle", render(targets.parts, thermal, self.input_size, self.render_settings))
            record("colorfulness/gt", colorfulness(linear_to_srgb(rgb) * 255.0), keep)
            with self._autocast():
                cond = self.encoder.encode(thermal, return_attn=True)
                if self.val_retrieval:
                    frozen = self.encoder.forward_frozen(rgb).patch_tokens
                sampled = {
                    method: sample_parts(
                        self.velocity,
                        cond["patch_tokens"].float(),
                        self.layout,
                        self.mean,
                        self.std,
                        attn_bias=cond.get("attn_bias"),
                        thumb=cond["thumb"].float(),
                        method=method,
                        nfe=nfe,
                        image_ids=batch["image_id"],
                    )
                    for method in ("euler", "midpoint")
                }
            for method, parts in sampled.items():
                parts = {name: value[0] for name, value in parts.items()}
                score(method, render(parts, thermal, self.input_size, self.render_settings))
            if self.val_retrieval:
                hits = patch_retrieval(
                    cond["patch_tokens"].float(),
                    frozen.float(),
                    self.layout.grid,
                    patch_mask_fraction(mask, self.input_patch),
                )
                for name, values in hits.items():
                    record(f"retrieval/{name}", values, keep)
            count += int(keep.sum())
        out = {key: value / max(count, 1) for key, value in totals.items()}
        if "psnr_L/oracle" in out:
            out["psnr_L/gap_to_oracle"] = out["psnr_L/oracle"] - out["psnr_L/euler"]
            out["ciede2000/gap_to_oracle"] = out["ciede2000/euler"] - out["ciede2000/oracle"]
            truth = out.get("colorfulness/gt", 0.0)
            if truth > 0:
                for method in ("oracle", "euler", "midpoint"):
                    out[f"colorfulness/ratio_{method}"] = out[f"colorfulness/{method}"] / truth
        if self.val_retrieval:
            out["retrieval/chance"] = 1.0 / self.layout.num_tokens
        out["num_images"] = float(count)
        return out

    @torch.no_grad()
    def validate_loss(
        self,
        loader: Iterable[Mapping[str, Any]],
        max_batches: int | None = None,
        seed: int = 1234,
    ) -> dict[str, float]:
        """Compute training loss terms on a held-out validation set.

        Args:
            loader: Validation data loader.
            max_batches: Optional maximum number of batches to evaluate.
            seed: Random seed for time and noise sampling.

        Returns:
            Dictionary mapping loss names to mean scalar values across validation samples.
        """
        self.eval()
        generator = torch.Generator().manual_seed(int(seed))
        totals: dict[str, float] = {}
        count = 0
        for index, raw in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            size = len(raw["image_id"])
            losses, _ = self.compute_losses(raw, generator=generator)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach()) * size
            count += size
        out = {f"loss/{key}": value / max(count, 1) for key, value in totals.items()}
        out["num_images"] = float(count)
        return out

    # checkpoints
    def state_dict(self) -> dict:
        """Return complete trainer state dictionary for checkpointing.

        Includes trainable model weights, optimizer and scheduler state,
        token layout descriptor, and target standardization statistics.
        """
        return {
            "step": self.step,
            "config": _plain(self.cfg),
            "layout": self.layout.descriptor(),
            "input_size": list(self.input_size),
            "render_size": list(self.render_size),
            "stats": self.stats,
            "velocity": self.velocity.state_dict(),
            "color_head": self.color_head.state_dict(),
            "lora": {k: v for k, v in self.encoder.state_dict().items() if "lora_" in k},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "momentum": self.momentum.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state: Mapping, strict: bool = True) -> None:
        """Restore trainer and model state from a checkpoint dictionary.

        Args:
            state: Saved state dictionary.
            strict: Whether to require all expected LoRA weights to be present.

        Raises:
            ValueError: If checkpoint resolution, layout, or model weights are incompatible.
        """
        self.layout.check(state["layout"])
        if list(state["input_size"]) != list(self.input_size):
            raise ValueError(
                f"resolution mismatch: this run is {list(self.input_size)}, the checkpoint was "
                f"trained at {list(state['input_size'])} (encoder.input_size)"
            )
        self.step = int(state["step"])
        self.set_stats(state["stats"])
        self.velocity.load_state_dict(state["velocity"])
        self.color_head.load_state_dict(state["color_head"])
        missing, unexpected = self.encoder.load_state_dict(state["lora"], strict=False)
        if unexpected:
            raise ValueError(f"checkpoint has unknown encoder keys: {unexpected[:4]}")
        if strict and any("lora_" in name for name in missing):
            raise ValueError("checkpoint is missing LoRA weights this encoder expects")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        if "momentum" in state:
            self.momentum.load_state_dict(state["momentum"])
        self.ema.load_state_dict(state["ema"])

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save trainer state dictionary to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("saved checkpoint %s (step %d)", path, self.step)
        return path


def _repeat(tensor: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    """Repeat tensor along the batch dimension."""
    if tensor is None or repeats == 1:
        return tensor
    return tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))


def _plain(value):
    """Convert OmegaConf containers to plain Python dictionaries and lists."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is a hard dependency
        pass
    return value


__all__ = [
    "EMA",
    "FlowTrainer",
    "MomentumCycler",
    "budget_summary",
    "build_models",
    "build_schedule",
    "cosine_warmup_schedule",
    "one_cycle_schedule",
    "patch_retrieval",
    "split_param_groups",
]
