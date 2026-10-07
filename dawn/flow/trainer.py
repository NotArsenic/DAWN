"""Stage 1 training: one step, one validation pass, one checkpoint.

`scripts/train_flow.py` is a thin CLI over this module. Everything that has to be testable
lives here, so `tests/test_flow_train.py` can run the same step the script runs.

What one step does (docs/conventions.md sections 2 to 6):

    1. A batch from `AlignedPairs` (thermal, linear RGB, masks) is moved to the device and
       area-downsampled to `encoder.input_size` if the stored frames are bigger; the pixel mask
       becomes fractions (`prepare_batch`).
    2. In float32 with no gradient (`build_targets`): guidance, L = L*/100 of the RGB, the
       brightness fit (`grid.refine_steps`), the oracle brightness, the a*b* (and correction)
       targets and the loss weights; then `pack` and `standardize` give z_1.
    3. The encoder runs ONCE per image on the thermal, and once more with the adapters disabled
       on the paired RGB for the L_align target. The colour head reads the adapted patch tokens.
    4. The conditioning is repeated `flow.timesteps_per_image` times; the encoder is not.
    5. Per (image, timestep): z_0 ~ N(0, I), t ~ U(0, 1), z_t = (1-t) z_0 + t z_1, v = net(...),
       z1_hat = z_t + (1-t) v, unstandardize, unpack, `render` at `flow.render_size`.
    6. L_flow + 0.1 L_render + 0.5 L_align + 0.2 L_color, backward, clip, step, EMA.

Precision: bf16 autocast wraps the network forward passes on CUDA only. Target building, the fit,
the render and the losses run in float32, and no module is ever cast with `.to(bf16)` (DINOv3's
RoPE tables must stay fp32). On MPS and CPU the run is fp32 whatever `optim.precision` says.

`flow.warmup_steps` gates the conditioning detach: until then the encoder's patch tokens are cut
off from the velocity net, so LoRA is trained by L_align and L_color alone. After it, everything
trains jointly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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

# Parameters that never get weight decay: 0-dim tensors (the per-block attention gates),
# biases, normalization weights and the learned positional embedding.
NO_DECAY_SUFFIXES = ("bias", "dino_gate", "pos_embed")


class EMA:
    """Exponential moving average of the trainable parameters.

    Tracks the velocity net, the LoRA adapters and the colour head (`optim.ema_decay`, 0.999).
    The shadow is kept in float32 whatever the parameters are, and `applied()` swaps it in for
    validation and inference without touching the optimizer's view of the live weights.
    """

    def __init__(self, params: Mapping[str, nn.Parameter], decay: float = 0.999):
        self.decay = float(decay)
        self.params = dict(params)
        self.shadow = {name: p.detach().clone().float() for name, p in self.params.items()}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.params.items():
            self.shadow[name].lerp_(param.detach().float(), 1.0 - self.decay)

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Temporarily replace the live parameters with the averaged ones."""
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
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping) -> None:
        self.decay = float(state["decay"])
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
    """Four AdamW groups: LoRA vs the rest, each split into decayed and undecayed.

    The encoder's only trainable parameters are the LoRA ones (peft froze everything else), and
    they get `encoder.lora_lr`. Norms, biases, the attention gates and the positional embedding
    are not shrunk toward zero: weight decay on a gate initialized at 0 is a prior that it
    should stay 0, which is exactly the thing being learned.
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
    """Smith's one cycle: cosine up to each group's configured LR, then cosine down to nearly 0.

    The difference from `cosine_warmup_schedule` is not the shape of the tail but the length of
    the ramp (20-30% of the run, against the 5% of `lr_warmup_steps`) and the floor it starts
    from (`lr / div_factor`). A group's configured LR is the **peak**, not the plateau, so
    `optim.lr` and `encoder.lora_lr` keep their ratio and neither is exceeded.

    This is a **fixed-budget** schedule: the anneal lands at `total_steps`, and most of the gain
    is in the last low-LR stretch, so a run stopped early gets a worse model than a shorter cycle
    would have. Decide the budget before starting, do not early-stop a cycle.

    Like `cosine_warmup_schedule` the factor clamps past `total_steps` rather than raising (what
    `torch.optim.lr_scheduler.OneCycleLR` does), so an extra step on resume is not fatal.

    Args:
        optimizer: its groups' `lr` values are the per-group peaks.
        total_steps: the whole budget, `flow.num_train_steps`.
        pct_start: fraction of the budget spent ramping up.
        div_factor: the ramp starts at `peak / div_factor`.
        final_div_factor: the anneal ends at `peak / final_div_factor`.

    Raises:
        ValueError: `pct_start` is not in (0, 1), or a divisor is not positive.
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
    """The other half of one cycle: AdamW's beta1, cycled opposite to the learning rate.

    High momentum while the LR is small, low momentum while it is large, so the two never push
    hard at once. Stepped beside the LR scheduler; a no-op when `enabled` is false, which is the
    default for the plain cosine schedule.
    """

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
    """The LR schedule and its momentum cycler, from `optim.schedule` (`cosine` | `one_cycle`)."""
    optim, flow = cfg["optim"], cfg["flow"]
    total = int(flow["num_train_steps"])
    name = str(optim.get("schedule", "cosine"))
    if name == "cosine":
        scheduler = cosine_warmup_schedule(optimizer, int(optim.get("lr_warmup_steps", 0)), total)
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


def build_models(
    cfg: Mapping, layout: TokenLayout, device: torch.device
) -> tuple[DINOEncoder, VelocityTransformer, ColorHead]:
    """Encoder (+LoRA), velocity transformer and colour head, on `device`."""
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
    """Owns the models, the optimizer and the schedule; runs steps and validations."""

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
        self.soft_k, self.soft_sigma = int(color["soft_k"]), float(color["soft_sigma"])
        self.bin_count_weighting = bool(flow["bin_count_weighting"])
        self.timesteps_per_image = int(flow["timesteps_per_image"])
        self.warmup_steps = int(flow["warmup_steps"])
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
        self.ema = EMA(self.trainable_parameters(), decay=float(optim["ema_decay"]))
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
        """A budget near its own warmup is almost never the run someone meant to start.

        The way in is raising `optim.batch_size` without rescaling the step counts: the 3,662
        train pairs over a batch of 128 are 29 steps an epoch, so a 100-epoch budget is 2,900
        steps, under `flow.warmup_steps`. The conditioning detach would then never lift and the
        velocity net would train on frozen features for the whole run -- a wrong result that
        looks like a healthy one (docs/training_budget_and_changes.md section 1).

        A warning, not an error: short deliberate runs (the overfit and debug overlays) scale
        `warmup_steps` down with the budget and are fine.
        """
        total, warmup = int(self.num_train_steps), int(self.warmup_steps)
        if warmup and total < warmup * 2:
            verdict = (
                "the detach NEVER lifts" if total <= warmup else "under half the run trains jointly"
            )
            logger.warning(
                "flow.num_train_steps (%d) is close to flow.warmup_steps (%d): %s. If you raised "
                "optim.batch_size, rescale the step counts with it "
                "(docs/training_budget_and_changes.md).",
                total,
                warmup,
                verdict,
            )

    def set_stats(self, stats: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        """Install per-segment stats (kept for checkpoints) and their `(D,)` vectors."""
        self.stats = {
            name: {key: value.detach().cpu() for key, value in segment.items()}
            for name, segment in stats.items()
        }
        mean, std = self.layout.stats_vector(self.stats)
        self.mean, self.std = mean.to(self.device), std.to(self.device)

    # --- bookkeeping ----------------------------------------------------------------------

    def trainable_parameters(self) -> dict[str, nn.Parameter]:
        """Every parameter the optimizer and the EMA touch, prefixed by its module."""
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
        """`{module: (trainable, total)}`, plus a 'total' row."""
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
        """bf16 autocast for network forwards; a no-op unless CUDA bf16 was asked for.

        Everything outside these blocks -- target building, the fit, the render, the losses --
        therefore runs in float32, which docs/conventions.md section 1 requires. No module is ever
        cast with `.to(bfloat16)`.
        """
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # --- the step -------------------------------------------------------------------------

    def compute_losses(self, raw_batch: Mapping[str, Any]) -> tuple[dict, dict]:
        """Every loss of one step for `raw_batch` (an `AlignedPairs` batch), with gradients.

        Returns:
            `(losses, info)`: the `FlowLoss` dict (tensors, `total` first to backpropagate) and
            `{"detach_cond": bool}`, whether the encoder was cut off from the velocity net.
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
            # Render-size targets: Lab of the area-downsampled linear RGB, and the colour
            # histograms built from its a*b* (8x8 samples per 16-px patch at the defaults).
            thermal_r = area_downsample(thermal, self.render_size)
            lab_r = linear_to_lab(area_downsample(rgb, self.render_size).clamp(0.0, 1.0))
            mask_r = area_downsample(mask, self.render_size)
            histogram = patch_histograms(
                lab_r[:, 1:], mask_r, self.render_patch, self.soft_k, self.soft_sigma
            )
            patch_weight = patch_mask_fraction(mask_r, self.render_patch)
            sample_weight = self.criterion.sample_weight(fallback)

        detach_cond = self.step < self.warmup_steps
        with self._autocast():
            cond = self.encoder.encode(thermal, return_attn=True)
            frozen = self.encoder.forward_frozen(rgb).patch_tokens
            logits = self.color_head(cond["patch_tokens"])
        feat_adapted = cond["patch_tokens"]
        patch_tokens = feat_adapted.detach() if detach_cond else feat_adapted

        batch_size = thermal.shape[0]
        features = _repeat(patch_tokens, repeats)
        thumb = _repeat(cond["thumb"], repeats)
        attn_bias = _repeat(cond.get("attn_bias"), repeats)
        z_1_rep = _repeat(z_1, repeats)

        t = torch.rand(repeats * batch_size, device=self.device, dtype=torch.float32)
        z_0 = torch.randn_like(z_1_rep)
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
        return losses, {"detach_cond": detach_cond}

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """One optimizer step. Returns every loss separately, plus the learning rates."""
        self.train()
        losses, info = self.compute_losses(batch)

        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
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
        for group in self.optimizer.param_groups:
            out[f"lr_{group['name'].split('_')[0]}"] = float(group["lr"])
        if self.momentum.enabled:
            out["beta1"] = float(self.optimizer.param_groups[0]["betas"][0])
        return out

    def _render_index(self, total: int) -> torch.Tensor:
        """Which of the B*T samples get the render loss (`flow.render_subset`, null = all)."""
        if not self.render_subset or self.render_subset >= total:
            return torch.arange(total, device=self.device)
        return torch.randperm(total, device=self.device)[: int(self.render_subset)]

    # --- validation -----------------------------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        loader: Iterable[Mapping[str, Any]],
        nfe: int | None = None,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Full-resolution metrics at `nfe` for both samplers, next to the fitted-grid oracle.

        Everything is at the working resolution (`encoder.input_size`) on the pixels the losses
        count (the pixel mask; images with no weight are skipped), so the oracle is a real ceiling:
        the fitted target parts rendered by the very same `render`. Noise comes from each image's
        ID, so a validation curve is comparable across runs and batchings.

        Returns:
            Means over images: `psnr_L/{oracle,euler,midpoint}` (L*, peak 100),
            `psnr_L/gap_to_oracle` (oracle - euler), `ciede2000/{...}`, `colorfulness/{...,gt}` and
            `num_images`.
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
            # Only the network forwards run under autocast; the render and the metrics are float32.
            with self._autocast():
                cond = self.encoder.encode(thermal, return_attn=True)
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
            count += int(keep.sum())
        out = {key: value / max(count, 1) for key, value in totals.items()}
        if "psnr_L/oracle" in out:
            out["psnr_L/gap_to_oracle"] = out["psnr_L/oracle"] - out["psnr_L/euler"]
        out["num_images"] = float(count)
        return out

    # --- checkpoints ----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Everything a resume needs, plus the config, the layout and the stats it was trained with.

        The frozen DINO backbone is NOT stored: peft never changes it, and a checkpoint that
        carried 29M frozen floats would be 30x bigger for nothing. A resume rebuilds it from
        `encoder.model_ids` and loads the LoRA weights on top, so the pretrained weights have
        to be the ones the run started from. The token stats and the layout descriptor, by
        contrast, ARE embedded, because inference must never read stats from the cache file
        (docs/conventions.md section 5) and a checkpoint is tied to its resolution.
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
        """Restore a checkpoint.

        Raises:
            ValueError: the checkpoint's token layout or resolution is not this run's, or it
                lacks LoRA weights this encoder expects.
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info("saved checkpoint %s (step %d)", path, self.step)
        return path


def _repeat(tensor: torch.Tensor | None, repeats: int) -> torch.Tensor | None:
    """Stack `repeats` copies along the batch axis, keeping image order inside each copy.

    Copy j holds the whole batch, so sample `j * B + i` is always image `i`: the conditioning,
    the target grid and the night pixels all line up without a second encoder pass.
    """
    if tensor is None or repeats == 1:
        return tensor
    return tensor.repeat(repeats, *([1] * (tensor.ndim - 1)))


def _plain(value):
    """OmegaConf containers -> plain dicts, so a checkpoint loads without omegaconf."""
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
    "build_models",
    "build_schedule",
    "cosine_warmup_schedule",
    "one_cycle_schedule",
    "split_param_groups",
]
