"""Weighted least-squares fitting of a one-input affine bilateral grid.

For each node of a (K, gy, gx) grid, fit `y_c ~ gain_c * x + offset_c` over the pixels that touch
the node, weighting every pixel by its trilinear slice weight times its own fit weight (the
mask). One function serves both users:

    brightness grid   x = thermal,  y = L = L*/100,     guidance = blurred thermal
    BGU               x = guide L,  y = a*b* or the L correction, guidance = the guide L

The grid is RAW (physical coefficients); standardize the packed tokens before using it as a flow
target. Layout and coefficient order: docs/conventions.md section 2.

Three nested levels, each the ridge prior of the one below:

    image    one affine for the whole image, ridged (almost not at all) toward gain 0 and the
             weighted mean of y, just enough to stay solvable on a constant frame.
    cell     one affine per (y, x) over all bins, ridged toward the image fit.
    node     one affine per (k, y, x), ridged toward its cell's fit.

A node whose total weight is small next to `ridge_alpha * mean_node_weight` is therefore pulled
onto its cell's fit: "no data here" means "do what the rest of this cell does".

The 2x2 normal equations depend only on x and the weights, so every output channel shares them
and they are solved in closed form, batched, with no `linalg.solve` (which not every backend has).
The fit is differentiable with respect to `y`.
"""

from __future__ import annotations

import torch

from dawn.grid.layout import NUM_COEFFS
from dawn.grid.slice import trilinear_weights

# Ridge on the image-level fit, as a fraction of the total weight. Only big enough to keep the
# solve well posed on a degenerate image (a constant x has a rank-1 normal matrix); on any real
# image it moves the fit by far less than float32 noise.
GLOBAL_RIDGE = 1e-6


def _scatter(
    feats: torch.Tensor, node_index: torch.Tensor, weight: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Scatter weighted per-pixel features onto the 8 corner nodes of every pixel.

    Args:
        feats: (B, F, P) per-pixel features.
        node_index: (B, 8, P) int64 flat node index.
        weight: (B, 8, P) corner weight times the pixel's fit weight.

    Returns:
        (B, F, num_nodes) sums.
    """
    batch, num_feats, _ = feats.shape
    acc = feats.new_zeros(batch, num_feats, num_nodes)
    for corner in range(8):
        index = node_index[:, corner].unsqueeze(1).expand(-1, num_feats, -1)
        acc.scatter_add_(2, index, feats * weight[:, corner].unsqueeze(1))
    return acc


def _solve(s0, sx, sxx, sy, sxy, lam, prior_a, prior_b):
    """Ridged 2x2 solves, closed form: `(A + lam I) [a, b] = rhs + lam [prior_a, prior_b]`.

    `A = [[sxx, sx], [sx, s0]]` is shared by every output channel. Shapes: `s0, sx, sxx, lam`
    broadcast against `(B, M)`; `sy, sxy, prior_*` are `(B, C, M)`. Returns `(a, b)` `(B, C, M)`.
    """
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
    """One image -> cell -> node hierarchy. Returns `(a, b)` stacked as (B, C, 2, nodes).

    `x_moments` is (B, 3, nodes) = (sum w, sum w x, sum w x^2); `y_moments` is (B, 2C, nodes) =
    (sum w y_c for each c, then sum w x y_c).
    """
    num_bins, grid_h, grid_w = shape
    batch, channels = x_moments.shape[0], y_moments.shape[1] // 2
    cells, num_nodes = grid_h * grid_w, num_bins * grid_h * grid_w
    s0, sx, sxx = x_moments.unbind(1)
    sy, sxy = y_moments[:, :channels], y_moments[:, channels:]

    total = s0.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1): the fit weight in play
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

    def by_cell(t: torch.Tensor) -> torch.Tensor:  # sum over bins: (..., nodes) -> (..., cells)
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

    def per_node(t: torch.Tensor) -> torch.Tensor:  # (B, C, cells) -> (B, C, nodes), same per bin
        return t.unsqueeze(2).expand(-1, -1, num_bins, -1).reshape(batch, channels, num_nodes)

    a, b = _solve(s0, sx, sxx, sy, sxy, lam_node, per_node(cell[0]), per_node(cell[1]))
    return torch.stack([a, b], dim=2)  # (B, C, 2, nodes)


def _render(
    ab: torch.Tensor, x: torch.Tensor, node_index: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Slice the fitted coefficients at the fit's own nodes and apply them, WITHOUT clamping.

    Used only to form the refinement residual, where clamping would hide the error being
    corrected. Reuses the fit's node indices, so refinement and fitting cannot drift apart.
    `ab` is (B, C, 2, nodes); returns (B, C, P).
    """
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
    """Fit a bilateral grid of `y_c ~ gain_c * x + offset_c` by weighted least squares.

    For each node, solves `min ||W^(1/2) (gain * x + offset - y)||^2 + ridge * ||[gain, offset] -
    prior||^2`, where W are the trilinear interpolation weights times `weights`. The weights come
    from `dawn.grid.slice.trilinear_weights`: this function never computes coordinates or corner
    weights itself (docs/conventions.md section 8, rule 1).

    `ridge_alpha` is in units of the mean weight per node (per cell at the cell level), so the
    same value behaves the same way at any resolution: a node holding `ridge_alpha` times the
    average node's weight is pulled halfway onto its prior.

    Args:
        x: (B, 1, H, W) the regressor (thermal, or a BGU's guide).
        y: (B, C, H, W) the targets, one grid output each. Gradients flow through `y`.
        guidance: (B, 1, H, W) in [0, 1], for the bin axis.
        weights: (B, 1, H, W) per-pixel fit weights (the mask; fractional is fine), or None for
            ones. A pixel with weight 0 has no influence at all.
        grid_size: (K, gy, gx) = (num_bins, grid_h, grid_w).
        ridge_alpha: Regularization strength.
        refine_steps: Extra passes that fit the render's residual and add the correction. 0 is
            the plain splat-and-solve. Each pass costs another fit and buys a fraction of a dB on
            a grid that varies quickly from node to node; see docs/conventions.md section 9.

    Returns:
        grid: (B, 2C, K, gy, gx) raw coefficients, output c at channels 2c (gain), 2c + 1 (offset).
        node_weights: (B, K, gy, gx) sum of fit weight handed to each node ("pixel counts").

    Raises:
        ValueError: Shapes disagree with the documented contract.
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
    # The per-node solve above treats each node as if it alone applied to its pixels, while the
    # renderer blends 8 of them. Re-fitting the render's residual with the very same solve and
    # adding the correction removes that bias (rendering is linear in the coefficients, so the
    # corrections add).
    for _ in range(int(refine_steps)):
        residual = y.reshape(batch, channels, pixels) - _render(ab, x_flat, node_index, corner_w)
        ab = ab + _fit_pass(x_moments, y_moments(residual), grid_size, ridge_alpha)

    num_bins, grid_h, grid_w = grid_size
    grid = ab.reshape(batch, channels * NUM_COEFFS, num_bins, grid_h, grid_w)
    return grid.contiguous(), x_moments[:, 0].reshape(batch, num_bins, grid_h, grid_w)
