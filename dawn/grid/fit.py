"""Weighted least-squares fitting for affine bilateral grids."""

from __future__ import annotations

import torch

from dawn.grid.layout import NUM_COEFFS
from dawn.grid.slice import trilinear_weights

GLOBAL_RIDGE = 1e-6


def _scatter(
    feats: torch.Tensor, node_index: torch.Tensor, weight: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Scatter weighted per-pixel features onto the 8 corner nodes of each pixel.

    Args:
        feats: (B, F, P) per-pixel feature tensor.
        node_index: (B, 8, P) flat integer node indices.
        weight: (B, 8, P) per-corner interpolation and fitting weights.
        num_nodes: Total number of nodes in the grid.

    Returns:
        (B, F, num_nodes) accumulated feature sums.
    """
    batch, num_feats, _ = feats.shape
    acc = feats.new_zeros(batch, num_feats, num_nodes)
    for corner in range(8):
        index = node_index[:, corner].unsqueeze(1).expand(-1, num_feats, -1)
        acc.scatter_add_(2, index, feats * weight[:, corner].unsqueeze(1))
    return acc


def _solve(s0, sx, sxx, sy, sxy, lam, prior_a, prior_b):
    """Solve 2x2 linear normal systems in closed form with ridge regularization."""
    s0, sx, sxx, lam = (t.unsqueeze(1) for t in (s0, sx, sxx, lam))
    a11, a12, a22 = sxx + lam, sx, s0 + lam
    r1, r2 = sxy + lam * prior_a, sy + lam * prior_b
    det = a11 * a22 - a12 * a12
    return (a22 * r1 - a12 * r2) / det, (a11 * r2 - a12 * r1) / det


def _fit_pass(
    x_moments: torch.Tensor,
    y_moments: torch.Tensor,
    shape: tuple[int, int, int],
    ridge_alpha: float,
) -> torch.Tensor:
    """Solve hierarchical image, cell, and node affine fits."""
    num_bins, grid_h, grid_w = shape
    batch, channels = x_moments.shape[0], y_moments.shape[1] // 2
    cells, num_nodes = grid_h * grid_w, num_bins * grid_h * grid_w
    s0, sx, sxx = x_moments.unbind(1)
    sy, sxy = y_moments[:, :channels], y_moments[:, channels:]

    total = s0.sum(dim=1, keepdim=True).clamp(min=1.0)
    lam_node = ridge_alpha * total / num_nodes
    lam_cell = ridge_alpha * total / cells
    lam_image = GLOBAL_RIDGE * total

    zeros = torch.zeros_like(sy.sum(dim=2, keepdim=True))
    mean_y = sy.sum(dim=2, keepdim=True) / s0.sum(dim=1).view(batch, 1, 1).clamp(min=1e-9)
    image = _solve(
        s0.sum(dim=1, keepdim=True),
        sx.sum(dim=1, keepdim=True),
        sxx.sum(dim=1, keepdim=True),
        sy.sum(dim=2, keepdim=True),
        sxy.sum(dim=2, keepdim=True),
        lam_image,
        zeros,
        mean_y,
    )

    def by_cell(t: torch.Tensor) -> torch.Tensor:
        return t.reshape(*t.shape[:-1], num_bins, cells).sum(dim=-2)

    cell = _solve(
        by_cell(s0),
        by_cell(sx),
        by_cell(sxx),
        by_cell(sy),
        by_cell(sxy),
        lam_cell,
        image[0].expand(-1, -1, cells),
        image[1].expand(-1, -1, cells),
    )

    def per_node(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(2).expand(-1, -1, num_bins, -1).reshape(batch, channels, num_nodes)

    a, b = _solve(s0, sx, sxx, sy, sxy, lam_node, per_node(cell[0]), per_node(cell[1]))
    return torch.stack([a, b], dim=2)


def _render(
    ab: torch.Tensor, x: torch.Tensor, node_index: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Evaluate affine coefficients at corner nodes without clamping."""
    batch, channels = ab.shape[:2]
    flat = ab.reshape(batch, channels * 2, -1)
    coeffs = torch.zeros(
        batch, channels * 2, node_index.shape[-1], dtype=ab.dtype, device=ab.device
    )
    for corner in range(8):
        index = node_index[:, corner].unsqueeze(1).expand(-1, channels * 2, -1)
        coeffs += flat.gather(2, index) * weight[:, corner].unsqueeze(1)
    pairs = coeffs.reshape(batch, channels, 2, -1)
    return pairs[:, :, 0] * x + pairs[:, :, 1]


def fit_affine_grid(
    x: torch.Tensor,
    y: torch.Tensor,
    guidance: torch.Tensor,
    weights: torch.Tensor | None = None,
    grid_size: tuple[int, int, int] = (8, 32, 40),
    ridge_alpha: float = 1e-3,
    refine_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit bilateral grid coefficients using weighted least-squares regression.

    Args:
        x: (B, 1, H, W) regressor tensor.
        y: (B, C, H, W) target tensor.
        guidance: (B, 1, H, W) guidance map in [0, 1].
        weights: Optional (B, 1, H, W) per-pixel fitting weights.
        grid_size: (num_bins, grid_h, grid_w) grid dimensions.
        ridge_alpha: Regularization strength relative to mean node weight.
        refine_steps: Number of residual refinement iterations.

    Returns:
        grid: (B, 2C, num_bins, grid_h, grid_w) fitted coefficients.
        node_weights: (B, num_bins, grid_h, grid_w) accumulated node weights.

    Raises:
        ValueError: If input tensor shapes do not match expected dimensions.
    """
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"x must be (B, 1, H, W), got {tuple(x.shape)}")
    if y.ndim != 4 or y.shape[0] != x.shape[0] or y.shape[2:] != x.shape[2:]:
        raise ValueError(
            f"y must be (B, C, H, W) matching x {tuple(x.shape)}, got {tuple(y.shape)}"
        )
    if guidance.shape != x.shape:
        raise ValueError(f"guidance must be {tuple(x.shape)}, got {tuple(guidance.shape)}")
    if weights is not None and weights.shape != x.shape:
        raise ValueError(f"weights must be {tuple(x.shape)}, got {tuple(weights.shape)}")

    batch, channels, height, width = y.shape
    dtype = torch.float64 if y.dtype == torch.float64 else torch.float32
    x, y = x.to(dtype), y.to(dtype)
    pixel_w = torch.ones_like(x) if weights is None else weights.to(dtype)

    node_index, corner_w = trilinear_weights(guidance.detach(), grid_size)
    pixels, num_nodes = height * width, grid_size[0] * grid_size[1] * grid_size[2]
    node_index = node_index.reshape(batch, 8, pixels)
    corner_w = corner_w.to(dtype).reshape(batch, 8, pixels) * pixel_w.reshape(batch, 1, pixels)

    x_flat = x.reshape(batch, 1, pixels)
    x_moments = _scatter(
        torch.cat([torch.ones_like(x_flat), x_flat, x_flat * x_flat], dim=1),
        node_index,
        corner_w,
        num_nodes,
    )

    def y_moments(target: torch.Tensor) -> torch.Tensor:
        flat = target.reshape(batch, channels, pixels)
        return _scatter(torch.cat([flat, flat * x_flat], dim=1), node_index, corner_w, num_nodes)

    ab = _fit_pass(x_moments, y_moments(y), grid_size, ridge_alpha)
    for _ in range(int(refine_steps)):
        residual = y.reshape(batch, channels, pixels) - _render(ab, x_flat, node_index, corner_w)
        ab = ab + _fit_pass(x_moments, y_moments(residual), grid_size, ridge_alpha)

    num_bins, grid_h, grid_w = grid_size
    grid = ab.reshape(batch, channels * NUM_COEFFS, num_bins, grid_h, grid_w)
    return grid.contiguous(), x_moments[:, 0].reshape(batch, num_bins, grid_h, grid_w)
