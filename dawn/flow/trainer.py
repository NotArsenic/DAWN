"""Stage 1 training: one step, one validation pass, one checkpoint.

`scripts/train_flow.py` is a thin CLI over this module. Everything that has to be testable
lives here, so `tests/test_flow_train.py` can run the same step the script runs.

What one step does (docs/conventions.md sections 4, 5 and 7):

    1. Workers hand over `night`, `night_clean`, `day` at 256x256, linear, plus `noise_params`.
    2. On the GPU, in float32: guidance = compute_guidance(night_clean), then
       `fit_grid(night_clean, day, guidance)` -> the raw oracle grid and its node weights,
       then `standardize` -> z_1.
    3. The encoder runs ONCE per image on `night`, and once more with the adapters disabled on
       `day` for the L_align target.
    4. The conditioning is repeated `flow.timesteps_per_image` times; the encoder is not.
    5. Per (image, timestep): z_0 ~ N(0, I), t ~ U(0, 1), z_t = (1-t) z_0 + t z_1, v = net(...),
       z1_hat = z_t + (1-t) v, un-standardize, slice, apply to night_clean -> applied_pred.
    6. L_flow + 0.1 L_render + L_noise + 0.5 L_align, backward, clip, step, EMA.

Precision: bf16 autocast wraps the network forward passes only. Grid fitting, slicing,
`apply_affine` and standardization run in float32, and no module is ever cast with `.to(bf16)`
(DINOv3's RoPE tables must stay fp32).

`flow.warmup_steps` gates the conditioning detach: until then the encoder's output is cut off
from the velocity net and the noise head, so LoRA is trained by L_align alone. After it,
everything trains jointly.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import torch
import torch.nn as nn

from dawn.data.color import linear_to_srgb
from dawn.eval.metrics import compute_psnr
from dawn.flow.loss import FlowMatchingLoss
from dawn.flow.sampler import sample_grid
from dawn.grid.fit import fit_grid
from dawn.grid.guidance import compute_guidance
from dawn.grid.normalize import standardize, unstandardize
from dawn.grid.slice import apply_affine, slice_grid
from dawn.models.encoder import DINOEncoder
from dawn.models.noise_head import NoiseHead, log_noise_target
from dawn.models.velocity import VelocityTransformer

logger = logging.getLogger(__name__)

# Parameters that never get weight decay: 0-dim tensors (the per-block attention gates),
# biases, normalization weights and the learned positional embedding.
NO_DECAY_SUFFIXES = ("bias", "dino_gate", "pos_embed")


class EMA:
    """Exponential moving average of the trainable parameters.

    Tracks the velocity net, the LoRA adapters and the noise head (`optim.ema_decay`, 0.999).
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
    noise_head: nn.Module,
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
        (scope, decay): {"params": [], "lr": lora_lr if scope == "lora" else lr,
                         "weight_decay": weight_decay if decay else 0.0, "name": f"{scope}_{decay}"}
        for scope in ("lora", "main")
        for decay in (True, False)
    }
    for scope, module in (("lora", encoder), ("main", velocity), ("main", noise_head)):
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


def build_models(cfg: Mapping, device: torch.device) -> tuple[DINOEncoder, VelocityTransformer,
                                                              NoiseHead]:
    """Encoder (+LoRA), velocity transformer and noise head, on `device`."""
    encoder = DINOEncoder.from_config(
        cfg["encoder"], detach_attn=cfg["velocity"]["detach_dino_attn"]
    ).to(device)
    velocity = VelocityTransformer.from_config(cfg["velocity"], cfg["grid"]).to(device)
    noise_head = NoiseHead(dino_dim=encoder.embed_dim).to(device)
    return encoder, velocity, noise_head


class FlowTrainer:
    """Owns the models, the optimizer and the schedule; runs steps and validations."""

    def __init__(
        self,
        cfg: Mapping,
        encoder: DINOEncoder,
        velocity: VelocityTransformer,
        noise_head: NoiseHead,
        stats: Mapping[str, torch.Tensor],
        device: torch.device | str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.velocity = velocity.to(self.device)
        self.noise_head = noise_head.to(self.device)
        self.mean = stats["mean"].to(self.device)
        self.std = stats["std"].to(self.device)
        self.criterion = FlowMatchingLoss.from_config(cfg["flow"])

        grid = cfg["grid"]
        self.grid_h = self.grid_w = int(grid["spatial_size"])
        self.num_bins = int(grid["num_bins"])
        self.ridge_alpha = float(grid["ridge_alpha"])

        flow, optim = cfg["flow"], cfg["optim"]
        self.timesteps_per_image = int(flow["timesteps_per_image"])
        self.warmup_steps = int(flow["warmup_steps"])
        self.render_subset = flow.get("render_subset") or None
        self.inference_nfe = int(flow["num_inference_steps"])
        self.grad_clip = float(optim["grad_clip"])
        self.precision = str(optim["precision"])

        self.optimizer = torch.optim.AdamW(
            split_param_groups(
                encoder, velocity, noise_head,
                lr=float(optim["lr"]),
                lora_lr=float(cfg["encoder"]["lora_lr"]),
                weight_decay=float(optim["weight_decay"]),
            ),
            betas=tuple(optim["betas"]),
        )
        self.scheduler = cosine_warmup_schedule(
            self.optimizer,
            int(optim.get("lr_warmup_steps", 0)),
            int(flow["num_train_steps"]),
        )
        self.ema = EMA(self.trainable_parameters(), decay=float(optim["ema_decay"]))
        self.step = 0

    @classmethod
    def from_config(
        cls, cfg: Mapping, stats: Mapping[str, torch.Tensor], device: torch.device | str = "cpu"
    ) -> "FlowTrainer":
        device = torch.device(device)
        return cls(cfg, *build_models(cfg, device), stats=stats, device=device)

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
        return {"encoder": self.encoder, "velocity": self.velocity, "noise_head": self.noise_head}

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

        Everything outside these blocks -- grid fitting, standardization, slicing,
        `apply_affine` -- therefore runs in float32, which is what docs/conventions.md
        section 1 requires. No module is ever cast with `.to(bfloat16)`.
        """
        if self.precision == "bf16" and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    # --- the step -------------------------------------------------------------------------

    def fit_targets(
        self, night_clean: torch.Tensor, day: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Oracle grid for one batch: `(z_1 standardized, node_weights, guidance)`, float32."""
        guidance = compute_guidance(night_clean)
        grid_raw, node_weights = fit_grid(
            night_clean,
            day,
            guidance,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            num_bins=self.num_bins,
            ridge_alpha=self.ridge_alpha,
        )
        return standardize(grid_raw, self.mean, self.std), node_weights, guidance

    def render(
        self, grid_std: torch.Tensor, image: torch.Tensor, guidance: torch.Tensor
    ) -> torch.Tensor:
        """Standardized grid -> un-standardize -> slice -> apply. The ONLY render path."""
        coeffs = slice_grid(unstandardize(grid_std.float(), self.mean, self.std), guidance)
        return apply_affine(coeffs, image)

    def train_step(self, batch: Mapping[str, Any]) -> dict[str, float]:
        """One optimizer step. Returns every loss separately, plus the learning rate."""
        self.train()
        night, night_clean, day = (
            batch[key].to(self.device, torch.float32, non_blocking=True)
            for key in ("night", "night_clean", "day")
        )
        noise_gt = log_noise_target(batch["noise_params"]).to(self.device)

        with torch.no_grad():
            z_1, node_weights, guidance = self.fit_targets(night_clean, day)

        detach_cond = self.step < self.warmup_steps
        with self._autocast():
            cond = self.encoder.encode(night, return_attn=True)
            feat_frozen = self.encoder.forward_frozen(day).patch_tokens

        feat_adapted = cond["patch_tokens"]
        patch_tokens = feat_adapted.detach() if detach_cond else feat_adapted
        cls_token = cond["cls"].detach() if detach_cond else cond["cls"]

        repeats = self.timesteps_per_image
        batch_size = night.shape[0]
        features = _repeat(patch_tokens, repeats)
        thumb = _repeat(cond["thumb"], repeats)
        attn_bias = _repeat(cond.get("attn_bias"), repeats)
        z_1_rep = _repeat(z_1, repeats)
        node_weights_rep = _repeat(node_weights, repeats)

        t = torch.rand(repeats * batch_size, device=self.device, dtype=torch.float32)
        z_0 = torch.randn_like(z_1_rep)
        t_grid = t.view(-1, 1, 1, 1, 1)
        z_t = (1.0 - t_grid) * z_0 + t_grid * z_1_rep

        with self._autocast():
            v_pred = self.velocity(z_t, t, features, None, thumb, attn_bias)
            noise_pred = self.noise_head(cls_token)
        v_pred = v_pred.float()

        z1_hat = z_t + (1.0 - t_grid) * v_pred
        index = self._render_index(z1_hat.shape[0])
        applied_pred = self.render(
            z1_hat[index],
            _repeat(night_clean, repeats)[index],
            _repeat(guidance, repeats)[index],
        )

        losses = self.criterion(
            v_pred=v_pred,
            z_0=z_0,
            z_1=z_1_rep,
            t=t,
            noise_pred=noise_pred.float(),
            noise_gt=noise_gt,
            applied_pred=applied_pred,
            day_image=_repeat(day, repeats)[index],
            node_weights=node_weights_rep,
            feat_adapted=feat_adapted,
            feat_frozen=feat_frozen,
        )

        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for group in self.optimizer.param_groups for p in group["params"]], self.grad_clip
        )
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()
        self.step += 1

        out = {name: float(value.detach()) for name, value in losses.items()}
        out["grad_norm"] = float(grad_norm)
        out["detach_cond"] = float(detach_cond)
        for group in self.optimizer.param_groups:
            out[f"lr_{group['name'].split('_')[0]}"] = float(group["lr"])
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
        """Render PSNR at `nfe` for both samplers, next to the fitted-grid oracle.

        PSNR is measured in display (sRGB) units on the same pixels the training render loss
        uses, `night_clean`, so the oracle is a real ceiling for the model. `psnr/euler_night`
        is the honest inference number: the same grid applied to the noisy `night` frame.
        """
        self.eval()
        nfe = self.inference_nfe if nfe is None else nfe
        totals: dict[str, float] = {}
        count = 0
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            night, night_clean, day = (
                batch[key].to(self.device, torch.float32) for key in
                ("night", "night_clean", "day")
            )
            guidance = compute_guidance(night_clean)
            grid_raw, _ = fit_grid(
                night_clean, day, guidance, grid_h=self.grid_h, grid_w=self.grid_w,
                num_bins=self.num_bins, ridge_alpha=self.ridge_alpha,
            )
            scores = {"psnr/oracle": _psnr(apply_affine(slice_grid(grid_raw, guidance),
                                                        night_clean), day)}
            with self._autocast():
                cond = self.encoder.encode(night, return_attn=True)
            for method in ("euler", "midpoint"):
                grids = sample_grid(
                    self.velocity,
                    cond["patch_tokens"].float(),
                    self.mean,
                    self.std,
                    grid_h=self.grid_h,
                    grid_w=self.grid_w,
                    num_bins=self.num_bins,
                    attn_bias=cond.get("attn_bias"),
                    thumb=cond["thumb"].float(),
                    method=method,
                    nfe=nfe,
                )[0]
                coeffs = slice_grid(grids, guidance)
                scores[f"psnr/{method}"] = _psnr(apply_affine(coeffs, night_clean), day)
                if method == "euler":
                    night_guidance = compute_guidance(night)
                    scores["psnr/euler_night"] = _psnr(
                        apply_affine(slice_grid(grids, night_guidance), night), day
                    )
            for key, value in scores.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
        out = {key: value / max(count, 1) for key, value in totals.items()}
        if "psnr/oracle" in out:
            out["psnr/gap_to_oracle"] = out["psnr/oracle"] - out.get("psnr/euler", 0.0)
        return out

    # --- checkpoints ----------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Everything a resume needs, plus the config and the grid stats it was trained with.

        The frozen DINO backbone is NOT stored: peft never changes it, and a checkpoint that
        carried 29M frozen floats would be 30x bigger for nothing. A resume rebuilds it from
        `encoder.model_ids` and loads the LoRA weights on top, so the pretrained weights have
        to be the ones the run started from. The grid stats, by contrast, ARE embedded, because
        inference must never read them from the cache file (docs/conventions.md section 5).
        """
        return {
            "step": self.step,
            "config": _plain(self.cfg),
            "grid_stats": {"mean": self.mean.cpu(), "std": self.std.cpu()},
            "velocity": self.velocity.state_dict(),
            "noise_head": self.noise_head.state_dict(),
            "lora": {k: v for k, v in self.encoder.state_dict().items() if "lora_" in k},
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
        }

    def load_state_dict(self, state: Mapping, strict: bool = True) -> None:
        self.step = int(state["step"])
        self.velocity.load_state_dict(state["velocity"])
        self.noise_head.load_state_dict(state["noise_head"])
        missing, unexpected = self.encoder.load_state_dict(state["lora"], strict=False)
        if unexpected:
            raise ValueError(f"checkpoint has unknown encoder keys: {unexpected[:4]}")
        if strict and any("lora_" in name for name in missing):
            raise ValueError("checkpoint is missing LoRA weights this encoder expects")
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
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


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean PSNR in display units: both images go through `linear_to_srgb` first."""
    return float(compute_psnr(linear_to_srgb(pred), linear_to_srgb(target)).mean())


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
    "build_models",
    "cosine_warmup_schedule",
    "split_param_groups",
]
