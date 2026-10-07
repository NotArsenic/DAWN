"""ODE sampler for flow matching inference.

Samples tokens by integrating the velocity field from t=0 to t=1.
Uses Euler or midpoint method with 1-2 function evaluations (NFE).

Works in STANDARDIZED token space, (B, N, D): the result must go through `unstandardize` once
before rendering (docs/conventions.md section 5). `sample_parts` does that, unpacks the tokens
with the layout and hands back RAW parts, which is what `render` expects.

NFE is the number of velocity evaluations, not the number of integration steps: Euler spends
one per step and midpoint two, so "2 NFE" is two Euler steps but a SINGLE midpoint step. Both
cost the same and that is the comparison worth logging.

The encoder runs once, outside the loop: `sample_parts` takes conditioning that is already
computed and keeps it fixed across steps and across the K ensemble members. The noise of an image
can come from its ID (`noise_for_ids`), so validation curves are comparable across runs and
independent of how the loader batches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Literal

import torch

from dawn.data.seeding import generator_for
from dawn.flow.tokens import TokenLayout
from dawn.grid.normalize import unstandardize

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
        z_0: (B, N, D) initial noise sample.
        num_steps: Number of Euler steps (NFE = num_steps).

    Returns:
        z_1: (B, N, D) predicted tokens, standardized.
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
        z_0: (B, N, D) initial noise.
        num_steps: Number of midpoint steps.

    Returns:
        z_1: (B, N, D) predicted tokens, standardized.
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
        z_0: (B, N, D) standardized noise.
        method: "euler" or "midpoint".
        nfe: Velocity evaluations (see the module docstring).

    Returns:
        z_1: (B, N, D), still standardized.
    """
    return SAMPLERS[method](velocity_fn, z_0, steps_for_nfe(method, nfe))


def noise_for_ids(
    image_ids: Sequence[str],
    num_tokens: int,
    dim: int,
    salt: str = "val_z0",
    sample: int = 0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """`(B, N, D)` standard normal noise, image `i` drawn from `generator_for(id_i, salt)`.

    The same ID gives the same noise on any machine, in any batch composition and order
    (docs/conventions.md section 10.3). Ensemble member `sample > 0` adds its index to the salts;
    member 0 is exactly `generator_for(id, salt)`.
    """
    salts = (salt,) if sample == 0 else (salt, sample)
    noise = [
        torch.randn(num_tokens, dim, generator=generator_for(image_id, *salts))
        for image_id in image_ids
    ]
    return torch.stack(noise).to(device)


@torch.no_grad()
def sample_parts(
    velocity_model,
    dino_features: torch.Tensor,
    layout: TokenLayout,
    mean: torch.Tensor,
    std: torch.Tensor,
    *,
    dino_attn: torch.Tensor | None = None,
    attn_bias: torch.Tensor | None = None,
    thumb: torch.Tensor | None = None,
    method: Method = "euler",
    nfe: int = 2,
    num_samples: int = 1,
    generator: torch.Generator | None = None,
    image_ids: Sequence[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Sample RAW parts for a batch of already-encoded images.

    The encoder has run before this call and its outputs are reused for every step and every
    ensemble member, so the cost is `nfe * num_samples` velocity passes and no DINO passes.
    The K members are stacked into one batch and integrated together; the tokens are
    unstandardized ONCE, at the end.

    Args:
        velocity_model: `VelocityTransformer` (or anything with the same call signature).
        dino_features: (B, N, C) patch tokens.
        layout: The token layout, to unpack the sampled tokens.
        mean: (D,) stats, from the checkpoint.
        std: (D,) stats, from the checkpoint.
        dino_attn: (B, N, N) head-averaged attention, or None.
        attn_bias: (B, 1, N, N) the same thing already in log space, or None.
        thumb: (B, 1, gy, gx) thermal thumbnail, or None.
        method: "euler" or "midpoint".
        nfe: Velocity evaluations per sample.
        num_samples: K ensemble members, drawn from independent noise.
        generator: RNG for z_0, so a sample can be reproduced. Ignored with `image_ids`.
        image_ids: One ID per image: z_0 is then drawn from `noise_for_ids` (member `s` of image
            `i` from its ID), independent of the batch, for comparable validation curves.

    Returns:
        raw parts, each with a leading `(num_samples, B)`: `grid (S, B, 2, K, gy, gx)`,
        `ab (S, B, 2, hc, wc)` and, with the correction, `corr (S, B, 1, hc, wc)`.
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
    shape = (layout.num_tokens, layout.dim)
    if image_ids is not None:
        if len(image_ids) != batch:
            raise ValueError(f"{len(image_ids)} image_ids for a batch of {batch}")
        z_0 = torch.cat(
            [noise_for_ids(image_ids, *shape, sample=s, device=device) for s in range(num_samples)]
        )
    else:
        z_0 = torch.randn(
            num_samples * batch, *shape, device=device, dtype=dtype, generator=generator
        )

    def velocity_fn(z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return velocity_model(z_t, t, features, attn, thumbs, bias).to(z_t.dtype)

    z_1 = integrate(velocity_fn, z_0, method=method, nfe=nfe)
    parts = layout.unpack(unstandardize(z_1, mean, std))
    return {
        name: value.reshape(num_samples, batch, *value.shape[1:]) for name, value in parts.items()
    }
