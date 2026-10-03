"""Trilinear grid slicing and affine application.

Given a bilateral grid (B, 12, K, gy, gx) and a per-pixel guidance map, interpolate the grid
trilinearly at each pixel's (x, y, guidance) coordinate, then apply the resulting 3x4 affine
to the pixel.

Layout, coefficient order and the cell-center coordinate convention are defined in
docs/conventions.md (sections 2 and 3). `trilinear_weights` is the only implementation of the
corner nodes and weights; `fit.py` uses it too.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dawn.grid.layout import MATRIX, NUM_COEFFS, OFFSET


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
    the two agree).

    Args:
        grid: (B, 12, K, gy, gx) raw affine coefficients.
        guidance: (B, 1, H, W) in [0, 1], from `compute_guidance`.

    Returns:
        coeffs: (B, 12, H, W) per-pixel affine coefficients, same channel order as the grid.

    Raises:
        ValueError: `grid` is not 5-D with 12 channels, or `guidance` is not (B, 1, H, W).
    """
    if grid.ndim != 5 or grid.shape[1] != NUM_COEFFS:
        raise ValueError(f"grid must be (B, 12, K, gy, gx), got {tuple(grid.shape)}")
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


def apply_affine(
    coeffs: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """Apply per-pixel 3x4 affine transforms to an image.

    out[i] = sum_j(M[i, j] * image[j]) + t[i]   for i in {R, G, B}

    Args:
        coeffs: (B, 12, H, W) per-pixel affine coefficients.
            Channels 0-8: M (3x3 matrix, row-major, row = output channel).
            Channels 9-11: t (offsets).
        image: (B, 3, H, W) linear RGB input.

    Returns:
        out: (B, 3, H, W) transformed linear RGB, clamped to [0, 1].

    Raises:
        ValueError: Shapes do not match the documented contract.
    """
    if coeffs.ndim != 4 or coeffs.shape[1] != NUM_COEFFS:
        raise ValueError(f"coeffs must be (B, 12, H, W), got {tuple(coeffs.shape)}")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must be (B, 3, H, W), got {tuple(image.shape)}")
    if coeffs.shape[0] != image.shape[0] or coeffs.shape[2:] != image.shape[2:]:
        raise ValueError(
            f"coeffs {tuple(coeffs.shape)} and image {tuple(image.shape)} disagree on B, H, W"
        )
    batch, _, height, width = image.shape
    matrix = coeffs[:, MATRIX].reshape(batch, 3, 3, height, width)
    out = (matrix * image.unsqueeze(1)).sum(dim=2) + coeffs[:, OFFSET]
    return out.clamp(0.0, 1.0)
