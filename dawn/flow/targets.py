"""Training targets: the pixel mask, the oracle fit and the token statistics.

Everything here runs on the device in float32 with no gradient (docs/conventions.md section 5.3).
From a batch of aligned pairs at the working resolution it builds, with the same code that the
oracle script and validation use, the RAW parts the model is trained to produce:

    grid  brightness (gain, offset) per cell and bin, fit from thermal to L = L*/100 with the
          pixel mask as fit weights, at `grid.refine_steps`
    ab    the masked area average of the true a*b* at the colour-map size
    corr  (only when the correction is on) the masked area average of `L - L_grid`, with `L_grid`
          the FITTED target grid rendered

plus the weights the losses need (node pixel counts, colour-pixel mask fractions). The pixel mask
is `valid & ~moving` for box-aligned pairs and `valid & ~vehicles` for fallback pairs
(`configs/flow.yaml: pairs`). Lab channels are area-averaged as values and masks as fractions
(docs/conventions.md section 4a).
"""

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
    """How the target grid is fit (`grid:` in flow.yaml)."""

    ridge_alpha: float = 1e-3
    refine_steps: int = 1

    @classmethod
    def from_config(cls, cfg: Mapping) -> FitSettings:
        return cls(
            ridge_alpha=float(cfg["grid"]["ridge_alpha"]),
            refine_steps=int(cfg["grid"]["refine_steps"]),
        )


@dataclass
class Targets:
    """What `build_targets` returns. `parts` are RAW; `lab` and `L_grid` are at the working size."""

    parts: dict[str, torch.Tensor]
    node_weights: torch.Tensor  # (B, K, gy, gx)
    fraction: torch.Tensor  # (B, 1, hc, wc) kept share of every colour-map pixel
    L_grid: torch.Tensor  # (B, 1, H, W) the fitted grid rendered (the brightness oracle)
    lab: torch.Tensor  # (B, 3, H, W) the true L*, a*, b* of the RGB


def pixel_mask(batch: Mapping, policy: Mapping) -> torch.Tensor:
    """`valid & ~excluded` as bool `(B, 1, H, W)`, `excluded` by the pair policy.

    Box-aligned pairs exclude `policy["box_pixel_mask"]` (`moving`), fallback pairs
    `policy["fallback_pixel_mask"]` (`vehicles`), per sample.
    """
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
    """An `AlignedPairs` batch at the working resolution `size`, ready for the trainer.

    Thermal and the linear RGB are area-downsampled when `size` is below the stored size; the
    pixel mask becomes the area-averaged **fraction** of kept pixels (used as a weight, never
    thresholded).

    Returns:
        dict: `thermal (B, 1, H, W)`, `rgb (B, 3, H, W)`, `mask (B, 1, H, W)` in [0, 1],
        `fallback (B,)` bool and `image_id` (list of str).

    Raises:
        ValueError: `size` is larger than the stored size.
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
    """`area(value * mask) / area(mask)` at `size`, 0 where `area(mask) == 0`; also `area(mask)`."""
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
    """Raw target parts, node weights and the oracle brightness for a batch.

    Args:
        thermal: (B, 1, H, W) at the working resolution.
        rgb: (B, 3, H, W) linear RGB at the same resolution.
        mask: (B, 1, H, W) pixel mask fractions (`prepare_batch`).
        layout: the token layout (grid and colour-map sizes, correction flag).
        fit: ridge and refine settings of the brightness fit.
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
    """Per-segment mean and std of the RAW targets over prepared batches (`prepare_batch`).

    Brightness stats are `(2, K)` indexed `[coefficient, bin]`, pooled over samples and cells but
    never over bins; colour `(2,)` and correction `(1,)` are pooled over colour pixels, weighted by
    the mask fraction, so neutral-filled masked pixels do not bias them. Only running sums are
    kept, so the memory cost does not grow with the number of samples.

    Raises:
        ValueError: `batches` yielded nothing.
    """
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
        # (B, 2, K, gy, gx) -> one row per (sample, cell), one column per (coefficient, bin).
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
    """Load the cached stats, or compute and save them if the cache is missing or stale.

    The cache is keyed by the token layout, the fit settings, the working resolution and
    `key_extra` (the number of samples, the manifest): any mismatch recomputes and overwrites it.
    Training caches here; checkpoints embed the stats they were trained with, and inference reads
    them from the checkpoint, never from this file (docs/conventions.md section 5.1).

    Args:
        make_batches: Called only on a cache miss; returns prepared batches (`prepare_batch`).
    """
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
