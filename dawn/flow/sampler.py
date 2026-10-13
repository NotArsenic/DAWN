"""ODE numerical integration samplers for flow matching inference."""

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
    """Integrate velocity field using Euler method.

    Args:
        velocity_fn: Callable computing velocity field v(z_t, t).
        z_0: Initial noise tensor of shape (B, N, D).
        num_steps: Number of integration steps.

    Returns:
        Sampled tokens z_1 of shape (B, N, D).
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
    """Integrate velocity field using midpoint method.

    Args:
        velocity_fn: Callable computing velocity field v(z_t, t).
        z_0: Initial noise tensor of shape (B, N, D).
        num_steps: Number of midpoint steps.

    Returns:
        Sampled tokens z_1 of shape (B, N, D).
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
    """Compute number of integration steps corresponding to given NFE count."""
    if method not in EVALS_PER_STEP:
        raise ValueError(f"method must be one of {sorted(EVALS_PER_STEP)}, got {method!r}")
    per_step = EVALS_PER_STEP[method]
    if nfe < per_step or nfe % per_step:
        raise ValueError(
            f"{method} spends {per_step} evaluation(s) per step, so it cannot run at NFE={nfe}"
        )
    return nfe // per_step


def nfe_for_steps(method: Method, num_steps: int) -> int:
    """Compute number of velocity evaluations required for given step count."""
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
    """Integrate flow trajectory from t=0 to t=1 using specified ODE solver.

    Args:
        velocity_fn: Callable computing velocity field v(z_t, t).
        z_0: Initial noise tensor of shape (B, N, D).
        method: Solver method name ('euler' or 'midpoint').
        nfe: Total number of velocity function evaluations.

    Returns:
        Integrated tokens z_1 of shape (B, N, D).
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
    """Generate reproducible initial noise vectors seeded by sample identifiers."""
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
    """Sample and unpack raw token parts for encoded conditioning inputs.

    Args:
        velocity_model: Velocity transformer model.
        dino_features: Conditioning patch tokens of shape (B, N, C).
        layout: TokenLayout defining token structure.
        mean: Normalization mean vector of shape (D,).
        std: Normalization standard deviation vector of shape (D,).
        dino_attn: Optional patch attention tensor of shape (B, N, N).
        attn_bias: Optional precomputed attention bias of shape (B, 1, N, N).
        thumb: Optional thermal thumbnail tensor of shape (B, 1, gy, gx).
        method: ODE solver method name.
        nfe: Number of velocity function evaluations per sample.
        num_samples: Number of ensemble samples to draw per input image.
        generator: Optional random number generator.
        image_ids: Optional sample identifiers for reproducible seeding.

    Returns:
        Dictionary mapping part names to tensors with leading shape (num_samples, B).
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
