"""Area-averaging downsampling utilities."""

from __future__ import annotations

import torch


def fit_long_side(height: int, width: int, long_side: int) -> tuple[int, int]:
    """Compute target (height, width) scaling the long side to long_side while preserving aspect ratio."""
    long = max(height, width)
    if long <= long_side:
        return height, width
    scale = long_side / long
    if height >= width:
        return long_side, max(1, round(width * scale))
    return max(1, round(height * scale)), long_side


def area_downsample(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Downsample the spatial dimensions of an image tensor using area averaging.

    Args:
        img: Floating-point image tensor with shape (..., H, W).
        size: Target (height, width).
    """
    if not img.is_floating_point():
        raise TypeError(f"area_downsample needs a floating-point image, got {img.dtype}")
    (h_in, w_in), (h_out, w_out) = img.shape[-2:], size
    if not (0 < h_out <= h_in and 0 < w_out <= w_in):
        raise ValueError(f"cannot area-downsample {(h_in, w_in)} to {(h_out, w_out)}")
    rows = _area_matrix(h_in, h_out).to(device=img.device, dtype=img.dtype)
    cols = _area_matrix(w_in, w_out).to(device=img.device, dtype=img.dtype)
    return rows @ img @ cols.T


def _area_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """Compute 1D area-averaging projection weights from n_in to n_out."""
    s = n_in / n_out
    start = torch.arange(n_out, dtype=torch.float64).unsqueeze(1) * s
    j = torch.arange(n_in, dtype=torch.float64).unsqueeze(0)
    overlap = torch.minimum(start + s, j + 1) - torch.maximum(start, j)
    return overlap.clamp(min=0) / s
