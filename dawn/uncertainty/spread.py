"""Sample spread uncertainty estimation across stochastic flow predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from dawn.data.color import lab_to_linear
from dawn.flow.render import RenderSettings, render
from dawn.flow.sampler import Method, sample_parts
from dawn.flow.tokens import TokenLayout


@dataclass(frozen=True)
class SpreadResult:
    """Ensemble point estimates and uncertainty maps for an image batch.

    Args:
        sample_rgb: (B, 3, H, W) linear RGB tensor for primary sample.
        sample_lab: (B, 3, H, W) Lab tensor for primary sample.
        rgb: (B, 3, H, W) ensemble mean linear RGB tensor.
        lab: (B, 3, H, W) ensemble mean Lab tensor.
        l_std: (B, 1, H, W) lightness standard deviation map.
        ab_std: (B, 1, H, W) chroma RMS standard deviation map.
        num_samples: Number of ensemble draws.
    """

    sample_rgb: torch.Tensor
    sample_lab: torch.Tensor
    rgb: torch.Tensor
    lab: torch.Tensor
    l_std: torch.Tensor
    ab_std: torch.Tensor
    num_samples: int


def spread_from_parts(
    parts: Mapping[str, torch.Tensor],
    thermal: torch.Tensor,
    size: tuple[int, int],
    settings: RenderSettings,
) -> SpreadResult:
    """Render ensemble component parts and compute mean and spread uncertainty maps.

    Args:
        parts: Dictionary of sampled model components with leading sample dimension (S, B, ...).
        thermal: (B, 1, H, W) input thermal tensor in [0, 1].
        size: Output (height, width) spatial dimensions.
        settings: Render settings configuration.

    Returns:
        SpreadResult containing sample outputs, ensemble mean, and spread maps.

    Raises:
        ValueError: If parts mapping is empty or sample counts disagree.
    """
    if not parts:
        raise ValueError("parts is empty: expected the dict sample_parts returns")
    counts = {name: int(value.shape[0]) for name, value in parts.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"parts disagree on the sample count S: {counts}")
    num_samples = next(iter(counts.values()))
    if num_samples < 1:
        raise ValueError(f"parts carry {num_samples} samples, need at least 1")

    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    first: torch.Tensor | None = None
    first_lab: torch.Tensor | None = None
    for index in range(num_samples):
        member = render(
            {name: value[index] for name, value in parts.items()}, thermal, size, settings
        )
        lab = member["lab"].to(torch.float32)
        if first is None:
            first, first_lab = member["rgb"], lab
            mean, m2 = torch.zeros_like(lab), torch.zeros_like(lab)
        delta = lab - mean
        mean = mean + delta / (index + 1)
        m2 = m2 + delta * (lab - mean)

    variance = m2 / (num_samples - 1) if num_samples > 1 else torch.zeros_like(m2)
    return SpreadResult(
        sample_rgb=first,
        sample_lab=first_lab,
        rgb=lab_to_linear(mean).clamp(0.0, 1.0),
        lab=mean,
        l_std=variance[:, 0:1].sqrt(),
        ab_std=(variance[:, 1:2] + variance[:, 2:3]).sqrt(),
        num_samples=num_samples,
    )


@torch.no_grad()
def sample_with_spread(
    velocity_model,
    encoded: Mapping[str, torch.Tensor],
    layout: TokenLayout,
    mean: torch.Tensor,
    std: torch.Tensor,
    thermal: torch.Tensor,
    size: tuple[int, int],
    settings: RenderSettings,
    *,
    num_samples: int = 8,
    method: Method = "euler",
    nfe: int = 2,
    generator: torch.Generator | None = None,
    image_ids: Sequence[str] | None = None,
) -> SpreadResult:
    """Sample multiple predictions from the velocity model and compute spread statistics.

    Args:
        velocity_model: Velocity transformer model.
        encoded: Encoded feature mapping containing patch tokens and thumbnail representations.
        layout: Token layout specification.
        mean: (D,) feature standardization mean vector.
        std: (D,) feature standardization standard deviation vector.
        thermal: (B, 1, H, W) input thermal tensor in [0, 1].
        size: Target (height, width) render resolution.
        settings: Render settings configuration.
        num_samples: Number of stochastic ensemble samples to draw.
        method: ODE integration method ('euler' or 'midpoint').
        nfe: Number of function evaluations per sample.
        generator: Optional random number generator.
        image_ids: Optional sequence of image identifiers for deterministic seeding.

    Returns:
        SpreadResult containing primary sample, ensemble mean, and uncertainty maps.
    """
    parts = sample_parts(
        velocity_model,
        encoded["patch_tokens"],
        layout,
        mean,
        std,
        attn_bias=encoded.get("attn_bias"),
        thumb=encoded.get("thumb"),
        method=method,
        nfe=nfe,
        num_samples=num_samples,
        generator=generator,
        image_ids=image_ids,
    )
    return spread_from_parts(parts, thermal, size, settings)

