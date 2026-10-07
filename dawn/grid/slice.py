"""Trilinear grid slicing and gain/offset application.

Given a bilateral grid (B, C, K, gy, gx) and a per-pixel guidance map, interpolate the grid
trilinearly at each pixel's (x, y, guidance) coordinate; `apply_gain_offset` then turns the
resulting gain/offset maps into an image.

Layout, coefficient order and the cell-center coordinate convention are defined in
docs/conventions.md (sections 2 and 3). `trilinear_weights` is the only implementation of the
corner nodes and weights; `fit.py` and `upsample.py` use it too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dawn.grid.layout import NUM_COEFFS


def _axis_weights(
    coord: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clamp a cell-center coordinate, then its two neighbouring nodes and their weights.

    Returns `(i0, i1, w0, w1)`; `i1 == i0` and `w1 == 0` at the outer centers (replicate).
    """
    coord = coord.clamp(0.0, size - 1.0)
    i0 = coord.floor().clamp(0.0, size - 1.0)
    w1 = coord - i0
    i0 = i0.to(torch.int64)
    return i0, (i0 + 1).clamp(max=size - 1), 1.0 - w1, w1


def trilinear_weights(
    guidance: torch.Tensor,
    grid_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Corner nodes and trilinear weights of every pixel (cell-center convention).

    For pixel (r, c) of an H x W image with guidance g:
        yf = (r + 0.5) * gy / H - 0.5
        xf = (c + 0.5) * gx / W - 0.5
        zf = g * K - 0.5
    Each coordinate is clamped to [0, N-1]; then i0 = floor(f), i1 = min(i0 + 1, N-1),
    w1 = f - i0, w0 = 1 - w1. A corner's weight is the product of its three axis weights.

    Args:
        guidance: (B, 1, H, W) in [0, 1], from `compute_guidance`.
        grid_size: (K, gy, gx) = (num_bins, grid_h, grid_w).

    Returns:
        node_index: (B, 8, H, W) int64, flat index k*gy*gx + y*gx + x into a (K, gy, gx) grid.
        weight: (B, 8, H, W) float32, non-negative, sums to 1 over dim 1.
            Corner n = 4*dz + 2*dy + dx, where d* in {0, 1} picks i0 or i1.

    Raises:
        ValueError: `guidance` is not (B, 1, H, W).
    """
    if guidance.ndim != 4 or guidance.shape[1] != 1:
        raise ValueError(f"guidance must be (B, 1, H, W), got {tuple(guidance.shape)}")
    num_bins, grid_h, grid_w = grid_size
    batch, _, height, width = guidance.shape
    device, dtype = guidance.device, guidance.dtype

    rows = torch.arange(height, device=device, dtype=dtype).view(1, height, 1)
    cols = torch.arange(width, device=device, dtype=dtype).view(1, 1, width)
    y0, y1, wy0, wy1 = _axis_weights((rows + 0.5) * grid_h / height - 0.5, grid_h)
    x0, x1, wx0, wx1 = _axis_weights((cols + 0.5) * grid_w / width - 0.5, grid_w)
    z0, z1, wz0, wz1 = _axis_weights(guidance[:, 0] * num_bins - 0.5, num_bins)

    y0, y1 = y0.expand(1, height, width), y1.expand(1, height, width)
    x0, x1 = x0.expand(1, height, width), x1.expand(1, height, width)
    wy0, wy1 = wy0.expand(1, height, width), wy1.expand(1, height, width)
    wx0, wx1 = wx0.expand(1, height, width), wx1.expand(1, height, width)

    plane = grid_h * grid_w
    index, weight = [], []
    for dz, (zi, wz) in enumerate(((z0, wz0), (z1, wz1))):
        for dy, (yi, wy) in enumerate(((y0, wy0), (y1, wy1))):
            for dx, (xi, wx) in enumerate(((x0, wx0), (x1, wx1))):
                assert 4 * dz + 2 * dy + dx == len(index)  # corner order, docs/conventions.md 3
                index.append((zi * plane + yi * grid_w + xi).expand(batch, height, width))
                weight.append((wz * wy * wx).expand(batch, height, width))
    return torch.stack(index, dim=1), torch.stack(weight, dim=1).to(torch.float32)


def slice_grid(
    grid: torch.Tensor,
    guidance: torch.Tensor,
) -> torch.Tensor:
    """Trilinear-interpolate the bilateral grid at each pixel.

    The output resolution is the guidance resolution. Uses `F.grid_sample` with the image's own
    normalized pixel centers and `2g - 1`, which is the same cell-center convention as
    `trilinear_weights` (docs/conventions.md section 3; `tests/test_grid_rules.py` checks that
    the two agree). Works for any channel count.

    Args:
        grid: (B, C, K, gy, gx) raw coefficients (C = 2 for the brightness grid).
        guidance: (B, 1, H, W) in [0, 1], from `compute_guidance` (or a BGU's own guide).

    Returns:
        coeffs: (B, C, H, W) per-pixel coefficients, same channel order as the grid.

    Raises:
        ValueError: `grid` is not 5-D, `guidance` is not (B, 1, H, W), or the batches differ.
    """
    if grid.ndim != 5:
        raise ValueError(f"grid must be (B, C, K, gy, gx), got {tuple(grid.shape)}")
    if guidance.ndim != 4 or guidance.shape[1] != 1:
        raise ValueError(f"guidance must be (B, 1, H, W), got {tuple(guidance.shape)}")
    if grid.shape[0] != guidance.shape[0]:
        raise ValueError(f"batch mismatch: grid {grid.shape[0]}, guidance {guidance.shape[0]}")
    batch, _, height, width = guidance.shape
    device, dtype = grid.device, grid.dtype

    rows = torch.arange(height, device=device, dtype=dtype)
    cols = torch.arange(width, device=device, dtype=dtype)
    ys = ((2 * rows + 1) / height - 1).view(1, height, 1).expand(batch, height, width)
    xs = ((2 * cols + 1) / width - 1).view(1, 1, width).expand(batch, height, width)
    zs = 2 * guidance[:, 0].to(dtype) - 1
    coords = torch.stack([xs, ys, zs], dim=-1).unsqueeze(1)  # (B, 1, H, W, 3)
    sampled = F.grid_sample(
        grid, coords, mode="bilinear", padding_mode="border", align_corners=False
    )
    return sampled.squeeze(2)


def apply_gain_offset(
    coeffs: torch.Tensor,
    x: torch.Tensor,
    clamp: bool = True,
) -> torch.Tensor:
    """Apply per-pixel gain and offset maps: `out_c = gain_c * x + offset_c`.

    Args:
        coeffs: (B, 2C, H, W) per-pixel coefficients, output c at channels 2c (gain) and 2c + 1
            (offset); C = 1 for the brightness grid.
        x: (B, 1, H, W) the regressor (thermal for the brightness grid, the guide for a BGU).
        clamp: Clamp the result to [0, 1]. The brightness render does (L is L*/100); a BGU's
            a*b* output must not.

    Returns:
        out: (B, C, H, W).

    Raises:
        ValueError: Shapes do not match the documented contract.
    """
    if coeffs.ndim != 4 or coeffs.shape[1] % NUM_COEFFS:
        raise ValueError(f"coeffs must be (B, 2C, H, W), got {tuple(coeffs.shape)}")
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"x must be (B, 1, H, W), got {tuple(x.shape)}")
    if coeffs.shape[0] != x.shape[0] or coeffs.shape[2:] != x.shape[2:]:
        raise ValueError(f"coeffs {tuple(coeffs.shape)} and x {tuple(x.shape)} disagree on B, H, W")
    batch, channels, height, width = coeffs.shape
    pairs = coeffs.reshape(batch, channels // NUM_COEFFS, NUM_COEFFS, height, width)
    out = pairs[:, :, 0] * x + pairs[:, :, 1]
    return out.clamp(0.0, 1.0) if clamp else out
