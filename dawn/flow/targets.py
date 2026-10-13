"""Target generation, pixel masking, and normalization statistics computation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from dawn.data.color import linear_to_lab
from dawn.data.resample import area_downsample
from dawn.flow.tokens import NUM_AB, TokenLayout
from dawn.grid.fit import fit_affine_grid
from dawn.grid.guidance import compute_guidance
from dawn.grid.normalize import WeightedMoments
from dawn.grid.slice import apply_gain_offset, slice_grid

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitSettings:
    """Settings for fitting bilateral affine grids."""

    ridge_alpha: float = 1e-3
    refine_steps: int = 1

    @classmethod
    def from_config(cls, cfg: Mapping) -> FitSettings:
        """Instantiate FitSettings from configuration mapping."""
        return cls(
            ridge_alpha=float(cfg["grid"]["ridge_alpha"]),
            refine_steps=int(cfg["grid"]["refine_steps"]),
        )


@dataclass
class Targets:
    """Container holding target parts and intermediate tensors."""

    parts: dict[str, torch.Tensor]
    node_weights: torch.Tensor  # (B, K, gy, gx)
    fraction: torch.Tensor  # (B, 1, hc, wc)
    L_grid: torch.Tensor  # (B, 1, H, W)
    lab: torch.Tensor  # (B, 3, H, W)


def pixel_mask(batch: Mapping, policy: Mapping) -> torch.Tensor:
    """Compute binary pixel validity mask based on alignment policy."""
    fallback = batch["fallback"].to(batch["valid"].device).view(-1, 1, 1, 1).bool()
    excluded = torch.where(
        fallback, batch[policy["fallback_pixel_mask"]], batch[policy["box_pixel_mask"]]
    )
    return batch["valid"] & ~excluded


def prepare_batch(
    batch: Mapping,
    size: tuple[int, int],
    policy: Mapping,
    device: torch.device | str | None = None,
) -> dict:
    """Preprocess data loader batch to working resolution.

    Args:
        batch: Raw batch dictionary from dataset loader.
        size: Target spatial dimensions (H, W).
        policy: Pair masking policy configuration.
        device: Optional target device for tensors.

    Returns:
        Prepared batch dictionary with float tensors.

    Raises:
        ValueError: If size exceeds stored image dimensions.
    """
    keep = pixel_mask(batch, policy).to(torch.float32)
    thermal, rgb = batch["thermal"].to(torch.float32), batch["rgb"].to(torch.float32)
    stored = tuple(thermal.shape[-2:])
    size = (int(size[0]), int(size[1]))
    if size[0] > stored[0] or size[1] > stored[1]:
        raise ValueError(f"working size {size} is larger than the stored {stored}")
    if size != stored:
        thermal, rgb, keep = (area_downsample(t, size) for t in (thermal, rgb, keep))
    out = {
        "thermal": thermal,
        "rgb": rgb,
        "mask": keep,
        "fallback": batch["fallback"].to(torch.bool),
        "image_id": list(batch["image_id"]),
    }
    if device is not None:
        out = {k: v.to(device) if torch.is_tensor(v) else v for k, v in out.items()}
    return out


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, size: tuple[int, int]):
    """Compute area-averaged masked mean and mask coverage fraction."""
    fraction = area_downsample(mask, size)
    mean = area_downsample(value * mask, size) / fraction.clamp(min=torch.finfo(value.dtype).tiny)
    return torch.where(fraction > 0, mean, torch.zeros_like(mean)), fraction


@torch.no_grad()
def build_targets(
    thermal: torch.Tensor,
    rgb: torch.Tensor,
    mask: torch.Tensor,
    layout: TokenLayout,
    fit: FitSettings,
) -> Targets:
    """Fit bilateral grid and extract target token components for training batch.

    Args:
        thermal: Thermal image tensor of shape (B, 1, H, W).
        rgb: Linear RGB image tensor of shape (B, 3, H, W).
        mask: Loss mask tensor of shape (B, 1, H, W).
        layout: TokenLayout defining grid and token structure.
        fit: FitSettings specifying grid fitting parameters.

    Returns:
        Targets container with raw part tensors and reference images.
    """
    thermal, rgb, mask = (t.to(torch.float32) for t in (thermal, rgb, mask))
    lab = linear_to_lab(rgb.clamp(0.0, 1.0))
    lightness = lab[:, :1] / 100.0
    guidance = compute_guidance(thermal)
    grid, node_weights = fit_affine_grid(
        thermal,
        lightness,
        guidance,
        mask,
        (layout.num_bins, *layout.grid),
        ridge_alpha=fit.ridge_alpha,
        refine_steps=fit.refine_steps,
    )
    l_grid = apply_gain_offset(slice_grid(grid, guidance), thermal)
    ab, fraction = _masked_mean(lab[:, 1:], mask, layout.color_size)
    parts = {"grid": grid, "ab": ab}
    if layout.correction:
        parts["corr"], _ = _masked_mean(lightness - l_grid, mask, layout.color_size)
    return Targets(parts, node_weights, fraction, l_grid, lab)


@torch.no_grad()
def compute_token_stats(
    batches: Iterable[Mapping],
    layout: TokenLayout,
    fit: FitSettings,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute per-segment mean and standard deviation of target parts across batches."""
    bright = WeightedMoments(2 * layout.num_bins)
    color = WeightedMoments(NUM_AB)
    correction = WeightedMoments(1) if layout.correction else None
    for batch in batches:
        t = build_targets(
            batch["thermal"].to(device),
            batch["rgb"].to(device),
            batch["mask"].to(device),
            layout,
            fit,
        )
        bright.add(t.parts["grid"].permute(0, 3, 4, 1, 2).reshape(-1, 2 * layout.num_bins), None)
        weight = t.fraction.reshape(-1)
        color.add(t.parts["ab"].permute(0, 2, 3, 1).reshape(-1, NUM_AB), weight)
        if correction is not None:
            correction.add(t.parts["corr"].permute(0, 2, 3, 1).reshape(-1, 1), weight)
    stats = {}
    for name, moments in (("brightness", bright), ("color", color), ("correction", correction)):
        if moments is None:
            continue
        mean, std = moments.finalize()
        if name == "brightness":
            mean, std = (v.reshape(2, layout.num_bins) for v in (mean, std))
        stats[name] = {"mean": mean, "std": std}
    return stats


def load_or_compute_stats(
    cache_path: str | Path,
    make_batches: Callable[[], Iterable[Mapping]],
    layout: TokenLayout,
    fit: FitSettings,
    input_size: tuple[int, int],
    key_extra: Mapping | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, torch.Tensor]]:
    """Load cached normalization statistics or compute and cache them if missing."""
    cache_path = Path(cache_path)
    key = {
        "layout": layout.descriptor(),
        "fit": asdict(fit),
        "input_size": [int(v) for v in input_size],
        **dict(key_extra or {}),
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if cached.get("key") == key:
            return cached["stats"]
        logger.info(
            "token stats cache %s is stale (layout or fit changed): recomputing", cache_path
        )
    stats = compute_token_stats(make_batches(), layout, fit, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "stats": stats}, cache_path)
    return stats

