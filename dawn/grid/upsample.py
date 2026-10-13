"""Spatial upsampling routines for low-resolution feature maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dawn.grid.fit import fit_affine_grid
from dawn.grid.slice import apply_gain_offset, slice_grid

UPSAMPLE_MODES = ("bilinear", "bgu")


def upsample_map(
    low: torch.Tensor,
    guide_lo: torch.Tensor,
    guide_hi: torch.Tensor,
    mode: str = "bilinear",
    weights_lo: torch.Tensor | None = None,
    *,
    grid_size: tuple[int, int, int] = (8, 64, 80),
    ridge_alpha: float = 1e-1,
    refine_steps: int = 2,
) -> torch.Tensor:
    """Upsample low-resolution tensor to match spatial size of guide_hi.

    Args:
        low: (B, C, h, w) low-resolution feature tensor.
        guide_lo: (B, 1, h, w) low-resolution guidance tensor.
        guide_hi: (B, 1, H, W) high-resolution guidance tensor.
        mode: Upsampling mode, either 'bilinear' or 'bgu'.
        weights_lo: Optional (B, 1, h, w) fitting weights for BGU mode.
        grid_size: (num_bins, grid_h, grid_w) dimensions for bilateral grid.
        ridge_alpha: Regularization strength for grid fitting.
        refine_steps: Number of residual refinement passes.

    Returns:
        (B, C, H, W) upsampled tensor.

    Raises:
        ValueError: If mode is not recognized.
    """
    if mode == "bilinear":
        return F.interpolate(low, size=guide_hi.shape[-2:], mode="bilinear", align_corners=False)
    if mode == "bgu":
        return bilateral_guided_upsample(
            low,
            guide_lo,
            guide_hi,
            weights_lo,
            grid_size=grid_size,
            ridge_alpha=ridge_alpha,
            refine_steps=refine_steps,
        )
    raise ValueError(f"upsample mode must be one of {UPSAMPLE_MODES}, got {mode!r}")


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
    """Upsample low-resolution tensor using bilateral guided upsampling (BGU).

    Args:
        low: (B, C, h, w) feature tensor to upsample.
        guide_lo: (B, 1, h, w) guidance tensor at low resolution in [0, 1].
        guide_hi: (B, 1, H, W) guidance tensor at target high resolution in [0, 1].
        weights_lo: Optional (B, 1, h, w) fitting weights.
        grid_size: (num_bins, grid_h, grid_w) dimensions for bilateral grid.
        ridge_alpha: Regularization strength for grid fitting.
        refine_steps: Number of residual refinement passes.

    Returns:
        (B, C, H, W) upsampled feature tensor.

    Raises:
        ValueError: If guidance tensor dimensions do not match expectations.
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
