"""ODE sampler for flow matching inference.

Samples grid coefficients by integrating the velocity field from t=0 to t=1.
Uses Euler or midpoint method with 1-2 function evaluations (NFE).

Works in STANDARDIZED grid space, (B, 12, K, gy, gx): the result must go through
`unstandardize` once before slicing (docs/conventions.md section 5). `sample_grid` does that
and hands back a RAW grid, which is what every pixel-side function expects.

NFE is the number of velocity evaluations, not the number of integration steps: Euler spends
one per step and midpoint two, so "2 NFE" is two Euler steps but a SINGLE midpoint step. Both
cost the same and that is the comparison worth logging.

The encoder runs once, outside the loop: `sample_grid` takes conditioning that is already
computed and keeps it fixed across steps and across the K ensemble members.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch

from dawn.grid.guidance import compute_guidance
from dawn.grid.layout import NUM_COEFFS
from dawn.grid.normalize import unstandardize
from dawn.grid.slice import apply_affine, slice_grid

Method = Literal["euler", "midpoint"]
EVALS_PER_STEP = {"euler": 1, "midpoint": 2}


@torch.no_grad()
def euler_sample(
    velocity_fn,
    z_0: torch.Tensor,
    num_steps: int = 2,
) -> torch.Tensor:
    """Euler ODE integration from z_0 to z_1.

    z_{k+1} = z_k + dt * v(z_k, t_k, c)

    Args:
        velocity_fn: Callable(z_t, t) -> v. Should capture conditioning internally.
        z_0: (B, 12, K, gy, gx) initial noise sample.
        num_steps: Number of Euler steps (NFE = num_steps).

    Returns:
        z_1: (B, 12, K, gy, gx) predicted grid coefficients, standardized.
    """
    dt = 1.0 / num_steps
    z_t = z_0.clone()
    for k in range(num_steps):
        t = torch.full((z_t.shape[0],), k * dt, device=z_t.device, dtype=z_t.dtype)
        v = velocity_fn(z_t, t)
        z_t = z_t + dt * v
    return z_t


@torch.no_grad()
def midpoint_sample(
    velocity_fn,
    z_0: torch.Tensor,
    num_steps: int = 1,
) -> torch.Tensor:
    """Midpoint method ODE integration. NFE = 2 * num_steps.

    Args:
        velocity_fn: Callable(z_t, t) -> v.
        z_0: (B, 12, K, gy, gx) initial noise.
        num_steps: Number of midpoint steps.

    Returns:
        z_1: (B, 12, K, gy, gx) predicted grid coefficients, standardized.
    """
    dt = 1.0 / num_steps
    z_t = z_0.clone()
    for k in range(num_steps):
        t_start = k * dt
        t_mid = t_start + 0.5 * dt

        t_s = torch.full((z_t.shape[0],), t_start, device=z_t.device, dtype=z_t.dtype)
        v_start = velocity_fn(z_t, t_s)

        z_mid = z_t + 0.5 * dt * v_start
        t_m = torch.full((z_t.shape[0],), t_mid, device=z_t.device, dtype=z_t.dtype)
        v_mid = velocity_fn(z_mid, t_m)

        z_t = z_t + dt * v_mid
    return z_t


SAMPLERS: dict[str, Callable] = {"euler": euler_sample, "midpoint": midpoint_sample}


def steps_for_nfe(method: Method, nfe: int) -> int:
    """Integration steps that spend exactly `nfe` velocity evaluations.

    Raises:
        ValueError: Unknown method, `nfe` below one, or an `nfe` the method cannot hit
            exactly (midpoint only lands on even counts).
    """
    if method not in EVALS_PER_STEP:
        raise ValueError(f"method must be one of {sorted(EVALS_PER_STEP)}, got {method!r}")
    per_step = EVALS_PER_STEP[method]
    if nfe < per_step or nfe % per_step:
        raise ValueError(
            f"{method} spends {per_step} evaluation(s) per step, so it cannot run at NFE={nfe}"
        )
    return nfe // per_step


def nfe_for_steps(method: Method, num_steps: int) -> int:
    """Velocity evaluations `num_steps` steps of `method` cost."""
    if method not in EVALS_PER_STEP:
        raise ValueError(f"method must be one of {sorted(EVALS_PER_STEP)}, got {method!r}")
    return EVALS_PER_STEP[method] * num_steps


@torch.no_grad()
def integrate(
    velocity_fn,
    z_0: torch.Tensor,
    method: Method = "euler",
    nfe: int = 2,
) -> torch.Tensor:
    """Integrate 0 -> 1 with `method`, spending `nfe` velocity evaluations.

    Args:
        velocity_fn: Callable(z_t, t) -> v, conditioning captured.
        z_0: (B, 12, K, gy, gx) standardized noise.
        method: "euler" or "midpoint".
        nfe: Velocity evaluations (see the module docstring).

    Returns:
        z_1: (B, 12, K, gy, gx), still standardized.
    """
    return SAMPLERS[method](velocity_fn, z_0, steps_for_nfe(method, nfe))


@torch.no_grad()
def sample_grid(
    velocity_model,
    dino_features: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    grid_h: int = 16,
    grid_w: int = 16,
    num_bins: int = 8,
    dino_attn: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
    thumb: torch.Tensor | None = None,
    method: Method = "euler",
    nfe: int = 2,
    num_samples: int = 1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample RAW bilateral grids for a batch of already-encoded images.

    The encoder has run before this call and its outputs are reused for every step and every
    ensemble member, so the cost is `nfe * num_samples` velocity passes and no DINO passes.
    The K members are stacked into one batch and integrated together.

    Args:
        velocity_model: `VelocityTransformer` (or anything with the same call signature).
        dino_features: (B, N, C) patch tokens.
        mean: (12, K) grid stats, from the checkpoint.
        std: (12, K) grid stats, from the checkpoint.
        grid_h: gy.
        grid_w: gx.
        num_bins: K.
        dino_attn: (B, N, N) head-averaged attention, or None.
        attn_bias: (B, 1, N, N) the same thing already in log space, or None.
        thumb: (B, 3, gy, gx) log thumbnail, or None.
        method: "euler" or "midpoint".
        nfe: Velocity evaluations per sample.
        num_samples: K ensemble members, drawn from independent noise.
        generator: RNG for z_0, so a sample can be reproduced.

    Returns:
        grids: (num_samples, B, 12, K, gy, gx) RAW coefficients, ready for `slice_grid`.
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be at least 1, got {num_samples}")
    batch = dino_features.shape[0]
    device, dtype = dino_features.device, torch.float32

    def expand(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None or num_samples == 1:
            return tensor
        return tensor.repeat(num_samples, *([1] * (tensor.ndim - 1)))

    features, attn, bias, thumbs = (
        expand(dino_features),
        expand(dino_attn),
        expand(attn_bias),
        expand(thumb),
    )
    z_0 = torch.randn(
        num_samples * batch,
        NUM_COEFFS,
        num_bins,
        grid_h,
        grid_w,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    def velocity_fn(z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return velocity_model(z_t, t, features, attn, thumbs, bias).to(z_t.dtype)

    z_1 = integrate(velocity_fn, z_0, method=method, nfe=nfe)
    raw = unstandardize(z_1, mean, std)
    return raw.reshape(num_samples, batch, NUM_COEFFS, num_bins, grid_h, grid_w)


@torch.no_grad()
def render_spread(
    grids: torch.Tensor,
    image: torch.Tensor,
    guidance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply K sampled grids to one image and report the per-pixel mean and spread.

    Flow sample diversity is the model's own statement about colour ambiguity: where the K
    members disagree, the affine map was underdetermined by the night pixels. This is a
    separate quantity from the sensor-noise propagation in `dawn.uncertainty`.

    Args:
        grids: (K, B, 12, bins, gy, gx) RAW grids, from `sample_grid`.
        image: (B, 3, H, W) linear RGB the grids are applied to (the night image).
        guidance: (B, 1, H, W); computed from `image` when omitted.

    Returns:
        mean: (B, 3, H, W) average enhanced image over the K samples.
        spread: (B, 3, H, W) per-pixel standard deviation over the K samples (zero for K = 1).
    """
    if grids.ndim != 6:
        raise ValueError(f"grids must be (K, B, 12, bins, gy, gx), got {tuple(grids.shape)}")
    if guidance is None:
        guidance = compute_guidance(image)
    rendered = torch.stack(
        [apply_affine(slice_grid(grid, guidance), image) for grid in grids.unbind(0)]
    )
    spread = rendered.std(dim=0, unbiased=False) if grids.shape[0] > 1 else torch.zeros_like(image)
    return rendered.mean(dim=0), spread
