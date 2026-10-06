"""Bilateral guided upsampling (Chen, Adams, Durand et al., SIGGRAPH Asia 2016).

Upsample a low-resolution map `low` to the resolution of a guide image so that its edges follow
the guide's: at low resolution, fit a bilateral grid of `low ~ gain * guide + offset` (per cell and
guide-intensity bin), then slice that grid at the high-resolution guide and apply it to it.

Used twice by the render path (docs/conventions.md section 6): colour a*b* and the optional L
correction, both guided by L. It reuses `fit_affine_grid` and `slice_grid`, so its node weights
come from `slice.trilinear_weights` like everything else (rule 1). The guides are treated as
constants (no gradient through the bin assignment); the result is differentiable with respect to
`low`, which is how L_render reaches the colour tokens.
"""

from __future__ import annotations

import torch

from dawn.grid.fit import fit_affine_grid
from dawn.grid.slice import apply_gain_offset, slice_grid


def bilateral_guided_upsample(
    low: torch.Tensor,
    guide_lo: torch.Tensor,
    guide_hi: torch.Tensor,
    weights_lo: torch.Tensor | None = None,
    *,
    grid_size: tuple[int, int, int] = (8, 16, 20),
    ridge_alpha: float = 1e-3,
    refine_steps: int = 0,
) -> torch.Tensor:
    """Upsample `low` guided by an image: edges in `guide_hi` become edges in the output.

    Args:
        low: (B, C, h, w) the map to upsample (a*b*, or the L correction).
        guide_lo: (B, 1, h, w) the guide at low resolution, in [0, 1]: the area average of
            `guide_hi` over the same footprint (`area_downsample`).
        guide_hi: (B, 1, H, W) the guide at the output resolution, in [0, 1] (the output `L`).
        weights_lo: (B, 1, h, w) fit weights (the area-averaged mask), or None for ones.
        grid_size: (bins, grid_h, grid_w) of the bilateral grid; `color.bgu` in flow.yaml.
        ridge_alpha: Ridge strength of the fit, in units of the mean node weight.
        refine_steps: Extra fit passes (see `fit_affine_grid`).

    Returns:
        (B, C, H, W) the upsampled map, NOT clamped.

    Raises:
        ValueError: the guides do not match `low` and each other.
    """
    if low.ndim != 4:
        raise ValueError(f"low must be (B, C, h, w), got {tuple(low.shape)}")
    if guide_lo.shape != (low.shape[0], 1, *low.shape[2:]):
        raise ValueError(
            f"guide_lo must be (B, 1, h, w) = {(low.shape[0], 1, *low.shape[2:])}, "
            f"got {tuple(guide_lo.shape)}"
        )
    if guide_hi.ndim != 4 or guide_hi.shape[:2] != (low.shape[0], 1):
        raise ValueError(f"guide_hi must be (B, 1, H, W), got {tuple(guide_hi.shape)}")
    guide_lo, guide_hi = guide_lo.detach(), guide_hi.detach()
    grid, _ = fit_affine_grid(
        guide_lo, low, guide_lo, weights_lo, grid_size, ridge_alpha, refine_steps
    )
    return apply_gain_offset(
        slice_grid(grid, guide_hi.to(grid.dtype)), guide_hi.to(grid.dtype), clamp=False
    )
