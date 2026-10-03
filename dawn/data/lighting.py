"""Scene illumination model: a daytime image becomes a night-time scene radiance.

Half of the synthetic night degradation; the other half is the camera (camera.py). This module
answers "what light was falling on this scene?", never "how dark is the photo?" -- the darkness
and the noise are the camera's job (`n_white` in camera.py).

A daylight image in linear RGB, with its own illumination roughly uniform and white, is treated
as reflectance. Relighting multiplies that reflectance by a night illumination field:

    radiance = reflectance * (ambient_level * illum_field * ambient_rgb + sum_lamps E_i * rgb_i)

Three modes:
    moonlit    ambient only: moon and sky glow, 3,800-4,300 K. Any source image.
    streetlit  the same ambient plus 0-8 lamp pools, each 10-100x the ambient at its foot.
    starlight  ambient only, slightly redder and (via camera.py) far fewer photons.

**Ambient is at relative level 1 by definition.** `illum_field` has mean 1, so a perfectly white
ambient-lit surface reads 1.0. Lamp pools push well above 1; that is deliberate, and the sensor
in camera.py is what clips them. The output is therefore scene radiance, NOT an image in [0, 1]:
it is unbounded above (docs/conventions.md section 1 governs images, not radiance).

Source images (`overcast_score`): a lamp pool or a flat moonlit ambient only looks right on a
source whose own lighting was flat. A sunny frame already carries hard shadows and specular
highlights that no relighting can remove, so streetlit and starlight draw from overcast frames,
measured by a low p95/p05 of linear luminance. moonlit takes any frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from dawn.data.color import color_temp_to_gains, luminance

Mode = Literal["moonlit", "streetlit", "starlight"]
MODES: tuple[Mode, ...] = ("moonlit", "streetlit", "starlight")

AMBIENT_LEVEL = 1.0  # by definition; see the module docstring


@dataclass
class LightingParams:
    """Sampled lighting for one image. Lengths are fractions of the image WIDTH."""

    mode: Mode
    height: int
    width: int
    ambient_temp: float  # Kelvin
    ambient_gains: torch.Tensor  # (3,) relative to 6500 K, green == 1
    illum_field: torch.Tensor  # (H, W) smooth ambient gain, mean 1
    light_xy: torch.Tensor = field(  # (N, 2) lamp foot, in width units
        default_factory=lambda: torch.zeros(0, 2)
    )
    light_height: torch.Tensor = field(default_factory=lambda: torch.zeros(0))  # (N,) h
    light_peak: torch.Tensor = field(default_factory=lambda: torch.zeros(0))  # (N,) E at r = 0
    light_temp: torch.Tensor = field(default_factory=lambda: torch.zeros(0))  # (N,) Kelvin
    light_gains: torch.Tensor = field(default_factory=lambda: torch.zeros(0, 3))  # (N, 3)

    @property
    def num_lights(self) -> int:
        return int(self.light_xy.shape[0])


def _uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    """One float from U[lo, hi) on `generator` (so the draw is reproducible)."""
    return float(torch.rand((), generator=generator) * (hi - lo) + lo)


def _log_uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    """One float from LogUniform[lo, hi): uniform in log space."""
    return float(math.exp(_uniform(math.log(lo), math.log(hi), generator)))


def overcast_score(day_linear: torch.Tensor) -> float:
    """p95 / p05 of the linear luminance: how contrasty the source image's own lighting was.

    Around 3-8 for an overcast frame, 20 and up under direct sun with hard shadows. The floor on
    p05 keeps a frame with crushed blacks from reading as infinitely contrasty.

    Args:
        day_linear: (3, H, W) or (B, 3, H, W) linear RGB in [0, 1].

    Returns:
        The ratio, at least 1.0.
    """
    y = luminance(day_linear.unsqueeze(0) if day_linear.ndim == 3 else day_linear).flatten()
    lo, hi = torch.quantile(y, torch.tensor([0.05, 0.95], dtype=y.dtype, device=y.device))
    return max(1.0, float(hi) / max(float(lo), 1e-4))


def choose_mode(
    day_linear: torch.Tensor,
    cfg: dict,
    generator: torch.Generator | None = None,
) -> Mode:
    """Pick a lighting mode for this source image.

    `cfg["mode_weights"]` gives the base weights. Modes in `cfg["overcast_only"]` are scaled by
    an overcast weight that falls from 1 to 0 (linearly in log ratio) across
    `cfg["overcast_ratio_range"]`, so a sunny frame can only come out moonlit.

    Args:
        day_linear: (3, H, W) linear RGB in [0, 1].
        cfg: The `degradation.lighting` section of data.yaml.
        generator: Seeded RNG; `None` uses the global one (training's fresh draw).

    Returns:
        One of "moonlit", "streetlit", "starlight".
    """
    lo, hi = (float(v) for v in cfg["overcast_ratio_range"])
    ratio = overcast_score(day_linear)
    flatness = (math.log(hi) - math.log(max(ratio, 1e-6))) / (math.log(hi) - math.log(lo))
    flatness = min(1.0, max(0.0, flatness))

    base = cfg["mode_weights"]
    overcast_only = set(cfg.get("overcast_only", ()))
    weights = torch.tensor(
        [float(base[m]) * (flatness if m in overcast_only else 1.0) for m in MODES]
    )
    if float(weights.sum()) <= 0.0:  # a fully sunny frame with no moonlit weight configured
        return "moonlit"
    return MODES[int(torch.multinomial(weights, 1, generator=generator))]


def _illum_field(
    height: int,
    width: int,
    cfg: dict,
    generator: torch.Generator | None,
    device: torch.device,
) -> torch.Tensor:
    """A smooth (H, W) ambient gain with mean exactly 1: uneven sky glow, nothing more."""
    size = int(cfg["illum_grid_size"])
    lo, hi = (float(v) for v in cfg["illum_range"])
    coarse = torch.rand((1, 1, size, size), generator=generator, device=device) * (hi - lo) + lo
    field_ = torch.nn.functional.interpolate(
        coarse, size=(height, width), mode="bilinear", align_corners=False
    )[0, 0]
    return field_ / field_.mean().clamp(min=1e-6)


def sample_lighting_params(
    height: int,
    width: int,
    cfg: dict,
    mode: Mode = "moonlit",
    generator: torch.Generator | None = None,
    device: torch.device = torch.device("cpu"),
) -> LightingParams:
    """Sample lighting for one image of the given size.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        cfg: The `degradation.lighting` section of data.yaml.
        mode: From `choose_mode`. Only "streetlit" places lamps.
        generator: Seeded RNG; `None` uses the global one.
        device: Device for the sampled tensors.

    Returns:
        LightingParams with every random value filled in.

    Raises:
        ValueError: `mode` is not one of MODES.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    temp_key = "starlight_temp_range" if mode == "starlight" else "moonlight_temp_range"
    lo, hi = (float(v) for v in cfg[temp_key])
    ambient_temp = _uniform(lo, hi, generator)

    params = LightingParams(
        mode=mode,
        height=height,
        width=width,
        ambient_temp=ambient_temp,
        ambient_gains=color_temp_to_gains(ambient_temp).to(device),
        illum_field=_illum_field(height, width, cfg, generator, device),
    )
    if mode != "streetlit":
        return params

    n_lo, n_hi = (int(v) for v in cfg["num_lights_range"])
    count = int(torch.randint(n_lo, n_hi + 1, (), generator=generator))
    if count == 0:
        return params

    margin = float(cfg["light_margin"])
    aspect = height / width  # y runs over [0, H/W] so lamp pools stay circular
    xy = torch.rand((count, 2), generator=generator, device=device)
    xy[:, 0] = xy[:, 0] * (1 + 2 * margin) - margin
    xy[:, 1] = xy[:, 1] * (aspect + 2 * margin) - margin

    h_lo, h_hi = (float(v) for v in cfg["light_height_range"])
    p_lo, p_hi = (float(v) for v in cfg["light_peak_range"])
    sodium_prob = float(cfg["sodium_prob"])
    s_lo, s_hi = (float(v) for v in cfg["sodium_temp_range"])
    l_lo, l_hi = (float(v) for v in cfg["led_temp_range"])

    heights, peaks, temps = [], [], []
    for _ in range(count):
        heights.append(_uniform(h_lo, h_hi, generator))
        peaks.append(_log_uniform(p_lo, p_hi, generator))
        sodium = _uniform(0.0, 1.0, generator) < sodium_prob
        temps.append(_uniform(s_lo, s_hi, generator) if sodium else _uniform(l_lo, l_hi, generator))

    params.light_xy = xy
    params.light_height = torch.tensor(heights, dtype=torch.float32, device=device)
    params.light_peak = torch.tensor(peaks, dtype=torch.float32, device=device)
    params.light_temp = torch.tensor(temps, dtype=torch.float32, device=device)
    params.light_gains = torch.stack([color_temp_to_gains(t) for t in temps]).to(device)
    return params


def illumination_field(params: LightingParams) -> torch.Tensor:
    """The (3, H, W) illumination falling on the scene, ambient plus every lamp.

    A lamp at height `h` above the ground, `r` away horizontally, throws

        E(r) = peak * h^3 / (h^2 + r^2)^1.5

    onto a horizontal surface (inverse square times the cosine of incidence), normalized so
    `E(0) = peak`, in units of the ambient. Distances are fractions of the image width, and the
    image plane stands in for the ground plane.

    Returns:
        (3, H, W) float32, strictly positive, mean of the ambient part equal to 1.
    """
    device = params.illum_field.device
    rows = (torch.arange(params.height, device=device, dtype=torch.float32) + 0.5) / params.width
    cols = (torch.arange(params.width, device=device, dtype=torch.float32) + 0.5) / params.width

    ambient = params.illum_field.unsqueeze(0) * params.ambient_gains.view(3, 1, 1)
    total = AMBIENT_LEVEL * ambient
    for i in range(params.num_lights):
        h = params.light_height[i].clamp(min=1e-3)
        dy = rows.view(-1, 1) - params.light_xy[i, 1]
        dx = cols.view(1, -1) - params.light_xy[i, 0]
        falloff = h.pow(3) / (h.pow(2) + dy.pow(2) + dx.pow(2)).pow(1.5)
        total = total + params.light_peak[i] * falloff.unsqueeze(0) * params.light_gains[i].view(
            3, 1, 1
        )
    return total


def apply_lighting(day_linear: torch.Tensor, params: LightingParams) -> torch.Tensor:
    """Relight a daytime image as a night scene.

    Args:
        day_linear: (3, H, W) linear RGB in [0, 1], read as reflectance.
        params: From `sample_lighting_params`, sized for this image.

    Returns:
        radiance: (3, H, W) float32 scene radiance, >= 0 and **unbounded above** (ambient-lit
        white is 1.0, a lamp foot reaches `light_peak`). camera.py turns this into an image.

    Raises:
        ValueError: `day_linear` is not (3, H, W), or its size disagrees with `params`.
    """
    if day_linear.ndim != 3 or day_linear.shape[0] != 3:
        raise ValueError(f"day_linear must be (3, H, W), got {tuple(day_linear.shape)}")
    if day_linear.shape[1:] != (params.height, params.width):
        raise ValueError(
            f"params are for {(params.height, params.width)}, "
            f"image is {tuple(day_linear.shape[1:])}"
        )
    reflectance = day_linear.to(torch.float32).clamp(0.0, 1.0)
    return (reflectance * illumination_field(params)).to(torch.float32)
