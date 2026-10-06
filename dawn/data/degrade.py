"""Synthetic nighttime degradation: one daytime image in, a matched night/clean pair out.

Composes the two physical stages and nothing else:
    1. lighting.py  reflectance * night illumination -> scene radiance (ambient white = 1)
    2. camera.py    sensor, auto exposure, ISP       -> night

What comes back
---------------
    night        what the model sees: noisy, quantized, JPEG'd, linear RGB in [0, 1].
    night_clean  the SAME camera pipeline with the noise draws replaced by their means and the
                 JPEG round trip skipped, so `night_clean == E[night]` up to clipping. It is the
                 target for grid fitting. It is NOT the scene radiance: it already carries the
                 exposure gain, the white balance and the ISP filters.
    noise_a,     per channel, at NATIVE resolution (docs/conventions.md section 6), as is
    noise_b      `quant_var`. At the returned resolution the variance is `(a*y + b + q) / n`.
    n            native pixels area-averaged into one output pixel.
    params       every sampled value, so a render can be reproduced or inspected.

Determinism
-----------
Validation and test renders are seeded from the image ID and the darkness level
(`dawn.data.seeding`), so the same ID gives the same night image on any machine, in any worker,
in any order, at each of the three levels. Training passes no level and no ID and gets a fresh
draw from the global RNG on every call, which is what makes the augmentation an augmentation.

Resolution
----------
The flow path renders a fixed 256x256 image. The daytime frame is area-downsampled to 256x256
FIRST (docs/conventions.md section 4a) and the sensor then simulates pixels that are `n` times
bigger, rather than adding native-strength noise to a small image. The illumination field is
built directly at the output resolution; it is smooth, so this matches downsampling the
radiance to within a fraction of a pixel at a lamp's foot. The native path (denoiser crops)
skips the downsample, uses `n = 1` and switches on the in-camera denoise and sharpening.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dawn.data.camera import CameraParams, CameraResult, apply_camera, sample_camera_params
from dawn.data.lighting import (
    LightingParams,
    Mode,
    apply_lighting,
    choose_mode,
    overcast_score,
    sample_lighting_params,
)
from dawn.data.resample import area_downsample
from dawn.data.seeding import generator_for

DARKNESS_LEVELS = ("easy", "medium", "hard")
FLOW_SIZE = 256


@dataclass
class DegradationParams:
    """Everything sampled for one render."""

    mode: Mode
    level: str | None  # None for a training draw
    lighting: LightingParams
    camera: CameraParams
    n: float  # native pixels per output pixel


@dataclass
class DegradationResult:
    """A matched night / night_clean pair and its exact noise budget."""

    night: torch.Tensor  # (3, H, W) linear RGB, float32, [0, 1]
    night_clean: torch.Tensor  # (3, H, W) linear RGB, float32, [0, 1]
    day: torch.Tensor  # (3, H, W) the daytime image at the SAME resolution
    noise_a: torch.Tensor  # (3,) at native resolution
    noise_b: torch.Tensor  # (3,) at native resolution
    quant_var: float
    n: float
    S: torch.Tensor
    M: torch.Tensor
    params: DegradationParams
    gain: float
    gain_capped: bool
    saturated: float

    @property
    def noise_params(self) -> dict[str, torch.Tensor | float]:
        """The dict the dataloader hands downstream (docs/conventions.md section 6)."""
        return {"a": self.noise_a, "b": self.noise_b, "n": self.n, "quant_var": self.quant_var}

    def variance(self, image: torch.Tensor | None = None) -> torch.Tensor:
        """`Var[night]` per pixel and channel at this resolution."""
        y = self.night_clean if image is None else image
        M_inv = torch.linalg.inv(self.M.to(y.device))
        x = torch.einsum('ij, jhw -> ihw', M_inv, y).clamp(min=0.0)
        a = self.noise_a.view(3, 1, 1).to(y.device)
        b = self.noise_b.view(3, 1, 1).to(y.device)
        v_in = a * x + b + self.quant_var
        return torch.einsum('ij, jhw -> ihw', self.S.to(y.device), v_in) / self.n


def sample_degradation_params(
    height: int,
    width: int,
    cfg: dict,
    calib: dict,
    mode: Mode | None = None,
    level: str | None = None,
    n: float = 1.0,
    day_linear: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device = torch.device("cpu"),
) -> DegradationParams:
    if level is not None and level not in DARKNESS_LEVELS:
        raise ValueError(f"level must be one of {DARKNESS_LEVELS} or None, got {level!r}")
    if mode is None:
        if day_linear is None:
            raise ValueError("pass either `mode` or `day_linear` so a mode can be chosen")
        mode = choose_mode(day_linear, cfg["lighting"], generator)

    lighting = sample_lighting_params(height, width, cfg["lighting"], mode, generator, device)
    camera = sample_camera_params(
        cfg["camera"], calib, mode, level, lighting.ambient_gains, generator, device
    )
    return DegradationParams(mode=mode, level=level, lighting=lighting, camera=camera, n=n)


def apply_degradation(
    day_linear: torch.Tensor,
    params: DegradationParams,
    noise: bool = True,
    native_path: bool = False,
    generator: torch.Generator | None = None,
) -> DegradationResult:
    radiance = apply_lighting(day_linear, params.lighting)
    clean: CameraResult = apply_camera(
        radiance, params.camera, params.n, noise=False, native_path=native_path, jpeg=False
    )
    if noise:
        noisy = apply_camera(
            radiance,
            params.camera,
            params.n,
            noise=True,
            native_path=native_path,
            jpeg=True,
            generator=generator,
        )
    else:
        noisy = clean
    return DegradationResult(
        night=noisy.night,
        night_clean=clean.night,
        day=day_linear.to(torch.float32).clamp(0.0, 1.0),
        noise_a=noisy.noise_a,
        noise_b=noisy.noise_b,
        quant_var=noisy.quant_var,
        n=noisy.n,
        S=noisy.S,
        M=noisy.M,
        params=params,
        gain=noisy.gain,
        gain_capped=noisy.gain_capped,
        saturated=noisy.saturated,
    )


def degrade_image(
    day_linear: torch.Tensor,
    cfg: dict,
    calib: dict,
    image_id: str | None = None,
    level: str | None = None,
    native_size: tuple[int, int] | None = None,
    mode: Mode | None = None,
    native_path: bool = False,
    size: int | None = FLOW_SIZE,
    noise: bool = True,
    deterministic: bool | None = None,
) -> DegradationResult:
    if day_linear.ndim != 3 or day_linear.shape[0] != 3:
        raise ValueError(f"day_linear must be (3, H, W), got {tuple(day_linear.shape)}")
    if deterministic is None:
        deterministic = level is not None
    if deterministic and image_id is None:
        raise ValueError("a deterministic render needs an image_id to seed from")

    day = day_linear.to(torch.float32).clamp(0.0, 1.0)
    if native_path:
        n = 1.0
    else:
        if size is not None and day.shape[1:] != (size, size):
            day = area_downsample(day, (size, size))
        height, width = day.shape[1:]
        if native_size is None:
            n = 1.0
        else:
            native_w, native_h = (int(v) for v in native_size)
            n = max(1.0, (native_w * native_h) / float(height * width))

    generator = generator_for(image_id, level) if deterministic else None
    params = sample_degradation_params(
        day.shape[1],
        day.shape[2],
        cfg,
        calib,
        mode=mode,
        level=level,
        n=n,
        day_linear=day,
        generator=generator,
        device=day.device,
    )
    return apply_degradation(
        day, params, noise=noise, native_path=native_path, generator=generator
    )


__all__ = [
    "DARKNESS_LEVELS",
    "FLOW_SIZE",
    "DegradationParams",
    "DegradationResult",
    "apply_degradation",
    "degrade_image",
    "overcast_score",
    "sample_degradation_params",
]
