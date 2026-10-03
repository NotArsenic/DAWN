"""Scene illumination model: daytime image -> noise-free night image.

Half of the synthetic night degradation (the other half is the camera, camera.py). Two
lighting regimes, picked per image with `moonlight_prob`:
    moonlit    ambient moon/sky light, color temperature from `moonlight_temp_range`
               (3,800-4,300 K; NOT 7,000 K).
    streetlit  local lamp pools plus halos, color temperature from `color_temp_range`
               (sodium to LED).

Everything happens in linear RGB. The output is `night_clean`: linear, float32, [0, 1], with no
sensor noise (docs/conventions.md section 1). It is the source for grid fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class LightingParams:
    """Randomly sampled lighting parameters for one image."""

    regime: Literal["moonlit", "streetlit"]
    exposure: float  # global gain k, LogUniform(exposure_range)
    color_temp: float  # Kelvin of the dominant light
    illum_field: torch.Tensor  # (H, W) smooth spatially-varying gain
    light_positions: torch.Tensor  # (N, 2) normalized (x, y) in [0, 1]; typically empty if moonlit
    light_sigmas: torch.Tensor  # (N,) lamp-pool sigma in pixels
    halo_intensity: float


def sample_lighting_params(
    height: int,
    width: int,
    cfg: dict,
    device: torch.device = torch.device("cpu"),
) -> LightingParams:
    """Sample random lighting parameters from config ranges.

    Args:
        height: Image height.
        width: Image width.
        cfg: The `degradation.lighting` section of data.yaml.
        device: Tensor device.

    Returns:
        LightingParams with all random values filled in.
    """
    raise NotImplementedError


def apply_lighting(
    day_linear: torch.Tensor,
    params: LightingParams,
) -> torch.Tensor:
    """Relight a daytime image as a night scene.

    Args:
        day_linear: (3, H, W) linear RGB image in [0, 1].
        params: Sampled lighting parameters.

    Returns:
        night_clean: (3, H, W) linear RGB in [0, 1], no noise.
    """
    raise NotImplementedError
