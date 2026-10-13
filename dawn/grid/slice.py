"""Trilinear bilateral grid slicing and affine parameter application."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from dawn.grid.layout import NUM_COEFFS


def _axis_weights(
    coord: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute 1D cell-center interpolation indices and weights."""
    coord = coord.clamp(0.0, size - 1.0)
    i0 = coord.floor().clamp(0.0, size - 1.0)
    w1 = coord - i0
    i0 = i0.to(torch.int64)
    return i0, (i0 + 1).clamp(max=size - 1), 1.0 - w1, w1


def trilinear_weights(
    guidance: torch.Tensor,
    grid_size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 8-corner node indices and trilinear interpolation weights per pixel.

    Args:
        guidance: (B, 1, H, W) guidance map in [0, 1].
        grid_size: (num_bins, grid_h, grid_w) dimensions of the bilateral grid.

    Returns:
        node_index: (B, 8, H, W) flat integer node indices.
        weight: (B, 8, H, W) float32 corner weights summing to 1 across corner dimension.

    Raises:
        ValueError: If guidance tensor is not (B, 1, H, W).
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
                assert 4 * dz + 2 * dy + dx == len(index)
                index.append((zi * plane + yi * grid_w + xi).expand(batch, height, width))
                weight.append((wz * wy * wx).expand(batch, height, width))
    return torch.stack(index, dim=1), torch.stack(weight, dim=1).to(torch.float32)


def slice_grid(
    grid: torch.Tensor,
    guidance: torch.Tensor,
) -> torch.Tensor:
    """Slice bilateral grid at per-pixel coordinates using trilinear interpolation.

    Args:
        grid: (B, C, K, gy, gx) bilateral grid tensor.
        guidance: (B, 1, H, W) guidance tensor in [0, 1].

    Returns:
        (B, C, H, W) interpolated coefficient maps.

    Raises:
        ValueError: If tensor dimensions or batch sizes do not match.
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
    """Apply affine gain and offset maps to an input tensor.

    Args:
        coeffs: (B, 2C, H, W) coefficient maps containing alternating gain and offset channels.
        x: (B, 1, H, W) input image tensor.
        clamp: Whether to clamp the resulting values to [0, 1].

    Returns:
        (B, C, H, W) transformed image tensor.

    Raises:
        ValueError: If shapes do not match expected dimensions.
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
