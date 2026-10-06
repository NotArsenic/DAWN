"""Weighted least-squares bilateral grid fitting.

For each spatial cell and brightness bin, fits a 3x4 affine color matrix that maps
night_clean pixels to day pixels, weighting every pixel by its trilinear slice weights.

This produces the ground-truth grid targets for flow matching training. The grid is RAW
(physical coefficients); standardize it before using it as a flow target. Layout and
coefficient order: docs/conventions.md section 2.

Three nested levels, each the ridge prior of the one below:

    global   one affine for the whole image, ridged toward identity only enough to stay
             solvable when the image carries no colour information at all.
    cell     one affine per (y, x) over all bins, ridged toward the global fit.
    node     one affine per (k, y, x), ridged toward its cell's fit.

A node whose total trilinear weight is small next to `ridge_alpha * mean_node_weight` is
therefore pulled onto its cell's fit rather than onto identity: "no data here" means "do what
the rest of this cell does", not "don't brighten".
"""

from __future__ import annotations

import torch

from dawn.grid.layout import NUM_COEFFS
from dawn.grid.slice import trilinear_weights

# Ridge on the global fit, as a fraction of the total pixel weight. Only big enough to keep
# `torch.linalg.solve` well posed on a degenerate image (a constant frame has a rank-2 normal
# matrix); on any real image it moves the fit by far less than float32 noise.
GLOBAL_RIDGE = 1e-6
PIXEL_CHUNK = 1 << 20  # pixels accumulated per pass, to bound the (chunk, 4, 4) temporaries


def _identity_solution(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """The 4x3 least-squares solution `X` of the identity affine: `M = X[:3].T`, `t = X[3]`."""
    return torch.cat(
        [
            torch.eye(3, device=device, dtype=dtype),
            torch.zeros(1, 3, device=device, dtype=dtype),
        ]
    )


def _accumulate(
    night_clean: torch.Tensor,
    day: torch.Tensor,
    node_index: torch.Tensor,
    weight: torch.Tensor,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scatter the weighted normal equations of every pixel onto its 8 corner nodes.

    Returns `(ata, atb, node_weight)`: `(B, num_nodes, 4, 4)`, `(B, num_nodes, 4, 3)` and
    `(B, num_nodes)`, where `ata = sum_p w_p phi_p phi_p^T` with `phi_p = [r, g, b, 1]`.
    """
    batch, _, height, width = night_clean.shape
    device, dtype = night_clean.device, night_clean.dtype
    pixels = height * width

    phi = torch.cat([night_clean, torch.ones_like(night_clean[:, :1])], dim=1)
    phi = phi.reshape(batch, 4, pixels).transpose(1, 2)  # (B, P, 4)
    tgt = day.reshape(batch, 3, pixels).transpose(1, 2)  # (B, P, 3)
    index = node_index.reshape(batch, 8, pixels)
    weight = weight.reshape(batch, 8, pixels).to(dtype)

    ata = torch.zeros(batch * num_nodes, 4, 4, device=device, dtype=dtype)
    atb = torch.zeros(batch * num_nodes, 4, 3, device=device, dtype=dtype)
    counts = torch.zeros(batch * num_nodes, device=device, dtype=dtype)
    offsets = (torch.arange(batch, device=device) * num_nodes).view(batch, 1)

    for start in range(0, pixels, PIXEL_CHUNK):
        stop = min(start + PIXEL_CHUNK, pixels)
        phi_c, tgt_c = phi[:, start:stop], tgt[:, start:stop]
        pp = phi_c.unsqueeze(-1) * phi_c.unsqueeze(-2)  # (B, c, 4, 4)
        py = phi_c.unsqueeze(-1) * tgt_c.unsqueeze(-2)  # (B, c, 4, 3)
        for corner in range(8):
            w = weight[:, corner, start:stop]
            flat = (index[:, corner, start:stop] + offsets).reshape(-1)
            scale = w.unsqueeze(-1).unsqueeze(-1)
            ata.index_add_(0, flat, (scale * pp).reshape(-1, 4, 4))
            atb.index_add_(0, flat, (scale * py).reshape(-1, 4, 3))
            counts.index_add_(0, flat, w.reshape(-1))
    return (
        ata.view(batch, num_nodes, 4, 4),
        atb.view(batch, num_nodes, 4, 3),
        counts.view(batch, num_nodes),
    )


def _ridged_solve(ata: torch.Tensor, atb: torch.Tensor, prior: torch.Tensor, lam) -> torch.Tensor:
    """`(A + lam I) X = B + lam * prior`, batched over every leading dimension."""
    eye = torch.eye(4, device=ata.device, dtype=ata.dtype)
    lam = lam if torch.is_tensor(lam) else torch.as_tensor(lam, device=ata.device, dtype=ata.dtype)
    lam = lam.reshape(*lam.shape, 1, 1) if lam.ndim else lam.reshape(1, 1)
    return torch.linalg.solve(ata + lam * eye, atb + lam * prior)


def fit_grid(
    night_clean: torch.Tensor,
    day: torch.Tensor,
    guidance: torch.Tensor,
    grid_h: int = 16,
    grid_w: int = 16,
    num_bins: int = 8,
    ridge_alpha: float = 1e-3,
    refine_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a bilateral affine grid from night_clean to day via weighted least squares.

    For each cell (y, x) and bin k, solves:
        min_A ||W^{1/2} (A @ [night_clean; 1] - day)||^2 + ridge * ||A - A_prior||^2

    where W are the trilinear interpolation weights. They come from
    `dawn.grid.slice.trilinear_weights`: this function never computes coordinates, floors or
    corner weights itself (docs/conventions.md section 8, rule 1).

    Ridge pulls sparse bins toward the cell's all-bin fit, and sparse cells toward the
    whole-image fit. NOT toward identity (identity = "don't brighten"). `ridge_alpha` is in
    units of the mean weight per node (per cell at the cell level), so the same value behaves
    the same way at any resolution: a node holding `ridge_alpha` times the average node's
    pixels is pulled halfway onto its prior.

    Args:
        night_clean: (B, 3, H, W) linear RGB, no noise.
        day: (B, 3, H, W) linear RGB ground truth.
        guidance: (B, 1, H, W) from `compute_guidance(night_clean)`, in [0, 1].
        grid_h: Grid cell rows (gy).
        grid_w: Grid cell columns (gx).
        num_bins: Number of brightness bins (K).
        ridge_alpha: Regularization strength.
        refine_steps: Extra passes that fit the render's residual and add the correction.
            0 (the default) is the plain splat-and-solve. Each pass costs another full fit and
            buys a few dB on a grid that varies quickly from node to node; see
            docs/conventions.md section 9.

    Returns:
        grid: (B, 12, K, gy, gx) raw fitted coefficients, canonical coefficient order.
        node_weights: (B, K, gy, gx) sum of trilinear weights per node ("pixel counts").

    Raises:
        ValueError: Shapes disagree with the documented contract.
    """
    if night_clean.shape != day.shape:
        raise ValueError(f"night_clean {tuple(night_clean.shape)} != day {tuple(day.shape)}")
    if night_clean.ndim != 4 or night_clean.shape[1] != 3:
        raise ValueError(f"images must be (B, 3, H, W), got {tuple(night_clean.shape)}")
    if guidance.shape != (night_clean.shape[0], 1, *night_clean.shape[2:]):
        raise ValueError(f"guidance must be (B, 1, H, W), got {tuple(guidance.shape)}")

    batch = night_clean.shape[0]
    dtype = torch.float32 if night_clean.dtype not in (torch.float64,) else night_clean.dtype
    night_clean, day = night_clean.to(dtype), day.to(dtype)

    node_index, weight = trilinear_weights(guidance, (num_bins, grid_h, grid_w))
    shape = (num_bins, grid_h, grid_w)
    grid, counts = _fit_pass(night_clean, day, node_index, weight, shape, ridge_alpha)

    # Optional: the per-node solve above treats each node as if it alone applied to its pixels,
    # while the renderer blends 8 of them. Re-fitting the render's residual with the very same
    # batched 4x4 solve and adding the correction removes that bias (rendering is linear in
    # the coefficients, so the corrections add). Each step costs one more full fit.
    for _ in range(int(refine_steps)):
        residual = day - _render(grid, night_clean, node_index, weight, shape)
        correction, _ = _fit_pass(
            night_clean, residual, node_index, weight, shape, ridge_alpha, identity_prior=False
        )
        grid = grid + correction

    return grid.contiguous(), counts.view(batch, *shape)


def _render(
    grid: torch.Tensor,
    image: torch.Tensor,
    node_index: torch.Tensor,
    weight: torch.Tensor,
    shape: tuple[int, int, int],
) -> torch.Tensor:
    """Slice `grid` at the fit's own nodes and apply it, WITHOUT the [0, 1] clamp.

    Used only to form the refinement residual, where clamping would hide the error being
    corrected. Reuses the fit's node indices so refinement and fitting cannot drift apart.
    """
    batch, _, height, width = image.shape
    flat = grid.reshape(batch, NUM_COEFFS, shape[0] * shape[1] * shape[2])
    gathered = torch.stack(
        [(flat[b][:, node_index[b]] * weight[b]).sum(dim=1) for b in range(batch)]
    )  # (B, 12, H, W)
    matrix = gathered[:, :9].reshape(batch, 3, 3, height, width)
    return (matrix * image.unsqueeze(1)).sum(dim=2) + gathered[:, 9:12]


def _fit_pass(
    night_clean: torch.Tensor,
    target: torch.Tensor,
    node_index: torch.Tensor,
    weight: torch.Tensor,
    shape: tuple[int, int, int],
    ridge_alpha: float,
    identity_prior: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One global -> cell -> node hierarchy. Returns the raw grid and the flat node weights."""
    num_bins, grid_h, grid_w = shape
    batch, dtype = night_clean.shape[0], night_clean.dtype
    cells, num_nodes = grid_h * grid_w, num_bins * grid_h * grid_w

    ata, atb, counts = _accumulate(night_clean, target, node_index, weight, num_nodes)

    total = counts.sum(dim=1).clamp(min=1.0)  # (B,) == H*W up to float error
    lam_node = ridge_alpha * total / num_nodes
    lam_cell = ridge_alpha * total / cells
    lam_global = GLOBAL_RIDGE * total

    # global: one affine for the image, ridged toward identity purely for conditioning. A
    # refinement pass fits a residual, whose natural zero is the zero affine, not identity.
    if identity_prior:
        prior = _identity_solution(ata.device, dtype).expand(batch, 4, 3)
    else:
        prior = torch.zeros(batch, 4, 3, device=ata.device, dtype=dtype)
    x_global = _ridged_solve(ata.sum(dim=1), atb.sum(dim=1), prior, lam_global)  # (B, 4, 3)

    # cell: sum over bins, ridged toward the global fit.
    ata_cell = ata.view(batch, num_bins, cells, 4, 4).sum(dim=1)
    atb_cell = atb.view(batch, num_bins, cells, 4, 3).sum(dim=1)
    prior_cell = x_global.unsqueeze(1).expand(batch, cells, 4, 3)
    x_cell = _ridged_solve(ata_cell, atb_cell, prior_cell, lam_cell.unsqueeze(1))

    # node: ridged toward its own cell's fit.
    prior_node = x_cell.unsqueeze(1).expand(batch, num_bins, cells, 4, 3).reshape(
        batch, num_nodes, 4, 3
    )
    x_node = _ridged_solve(ata, atb, prior_node, lam_node.unsqueeze(1))

    # X is 4x3: M = X[:3].T (row = output channel), t = X[3]. docs/conventions.md 2.2.
    matrix = x_node[..., :3, :].transpose(-1, -2).reshape(batch, num_nodes, 9)
    coeffs = torch.cat([matrix, x_node[..., 3, :]], dim=-1)  # (B, num_nodes, 12)
    grid = coeffs.transpose(1, 2).reshape(batch, NUM_COEFFS, num_bins, grid_h, grid_w)
    return grid, counts


def _solve_affine_wls(
    src: torch.Tensor,
    tgt: torch.Tensor,
    weights: torch.Tensor,
    ridge: float = 1e-3,
) -> torch.Tensor:
    """Solve weighted least squares for a single cell/bin.

    Solves: X = (Phi^T W Phi + ridge I)^{-1} Phi^T W Y
    where Phi = [src, 1] (homogeneous), Y = tgt. X is 4x3, so M = X[:3].T and t = X[3].

    Reference implementation for one node, used by tests; `fit_grid` does the same arithmetic
    batched over every node at once.

    Args:
        src: (N, 3) source pixels (night_clean).
        tgt: (N, 3) target pixels (day).
        weights: (N,) trilinear interpolation weights.
        ridge: Regularization.

    Returns:
        coeffs: (12,) in canonical order: M row-major (9), then t (3).
    """
    if src.ndim != 2 or src.shape[1] != 3 or tgt.shape != src.shape:
        raise ValueError(f"src and tgt must be (N, 3), got {tuple(src.shape)}, {tuple(tgt.shape)}")
    if weights.shape != src.shape[:1]:
        raise ValueError(f"weights must be (N,), got {tuple(weights.shape)}")
    phi = torch.cat([src, torch.ones_like(src[:, :1])], dim=1)  # (N, 4)
    weighted = phi * weights.unsqueeze(1)
    ata = weighted.T @ phi
    atb = weighted.T @ tgt
    eye = torch.eye(4, device=src.device, dtype=src.dtype)
    x = torch.linalg.solve(ata + ridge * eye, atb)  # (4, 3)
    return torch.cat([x[:3].T.reshape(9), x[3]])
