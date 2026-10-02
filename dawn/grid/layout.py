"""Canonical bilateral-grid layout: constants and token reshapes.

Defined in docs/conventions.md (section 2); this module is its single implementation, so
nothing else re-derives the coefficient order or the token reshape.

    Grid:   (B, 12, K, gy, gx)   coefficient, bin, cell row, cell column
    Tokens: (B, gy*gx, K*12)     token n = y*gx + x, value v = k*12 + c
"""

from __future__ import annotations

import torch
from einops import rearrange

NUM_COEFFS = 12

# Coefficient order: out_i = sum_j M[i, j] * x_j + t_i.
MATRIX = slice(0, 9)  # M[i, j] at 3*i + j (row-major, i = output channel)
OFFSET = slice(9, 12)  # t[i] at 9 + i
IDENTITY_COEFFS = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """Flatten a grid into velocity-transformer tokens.

    Args:
        grid: (B, 12, K, gy, gx), raw or standardized.

    Returns:
        tokens: (B, gy*gx, K*12).
    """
    return rearrange(grid, "b c k y x -> b (y x) (k c)")


def tokens_to_grid(
    tokens: torch.Tensor,
    grid_h: int,
    grid_w: int,
    num_bins: int,
) -> torch.Tensor:
    """Inverse of `grid_to_tokens`.

    Args:
        tokens: (B, gy*gx, K*12).
        grid_h: gy.
        grid_w: gx.
        num_bins: K.

    Returns:
        grid: (B, 12, K, gy, gx).
    """
    return rearrange(
        tokens,
        "b (y x) (k c) -> b c k y x",
        y=grid_h,
        x=grid_w,
        k=num_bins,
        c=NUM_COEFFS,
    )
