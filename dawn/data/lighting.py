"""Scene illumination model: a daytime image becomes a night-time scene radiance.

Half of the synthetic night degradation; the other half is the camera (camera.py). This module
answers "what light was falling on this scene?", never "how dark is the photo?" -- the darkness
and the noise are the camera's job (`n_white` in camera.py).

A daylight image in linear RGB, with its own illumination roughly uniform and white, is treated
as reflectance. Relighting multiplies that reflectance by a night illumination field and adds
scene emitters:

    radiance = reflectance * illumination + emission

where illumination is ambient sky glow plus street lamp pools:

    illumination = ambient_level * illum_field * ambient_rgb + sum_lamps E_i * rgb_i

Three modes:
    moonlit    ambient only (or subtle urban glow), 3,800-4,300 K. Any source image.
    streetlit  the same ambient plus 1-8 lamp pools (20-400x ambient) and urban emitters.
               Accounts for roughly 65-75% of synthetic renders.
    starlight  ambient only, slightly redder and (via camera.py) far fewer photons;
               few or no urban emitters.

**Ambient is at relative level 1 by definition.** `illum_field` has mean 1, so a perfectly white
ambient-lit surface reads 1.0. Lamp pools and emitters push well above 1; that is deliberate,
and the sensor in camera.py is what clips them. The output is therefore scene radiance, NOT an
image in [0, 1]: it is unbounded above (docs/conventions.md section 1 governs images, not radiance).

Emitters (small emissive points):
    Added as an ADDITIVE emission term directly into scene radiance (not reflectance x
    illumination). Emitters are deliberately NOT in the day target (at night the model
    should not recover what is behind a glare). Three types:
        - lit windows: small axis-aligned rectangles in clusters laid out as a grid on a
          facade patch (random rows/cols, per-window on probability), warm 2,700-4,000 K,
          moderate brightness variation.
        - headlights: pairs of small discs, 4,500-6,500 K white, and taillight pairs in
          saturated red.
        - signs: small rectangles/strips in saturated hues (red, green, blue, magenta,
          cyan, amber).
    Each emitter has a bright core (LogUniform 20-1000 relative to ambient white) and a soft
    glow halo (e.g. Gaussian with sigma several times the core size at 1-10% of core level).
    Rendering uses area coverage (analytic 1D overlap for rectangles, subpixel supersampling
    with exact energy normalization for discs) so integrated energy is resolution-independent.

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

SIGN_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.04, 0.02),  # red
    (0.05, 1.0, 0.08),  # green
    (0.04, 0.10, 1.0),  # blue
    (1.0, 0.04, 0.85),  # magenta
    (0.04, 0.95, 1.0),  # cyan
    (1.0, 0.55, 0.03),  # amber
)


@dataclass
class Emitter:
    """One emissive primitive (or element of a cluster/pair).

    Coordinates and sizes are fractions of the image WIDTH.
    """

    shape: Literal["box", "disc"]
    x: float  # center x, in width units
    y: float  # center y, in width units
    w: float  # full width (or diameter for disc), in width units
    h: float  # full height (or diameter for disc), in width units
    radius: float  # radius for disc, 0.0 for box
    color: torch.Tensor  # (3,) linear RGB
    core_peak: float  # core surface brightness relative to ambient white = 1.0
    halo_peak: float  # halo peak brightness relative to ambient white = 1.0
    halo_sigma: float  # standard deviation of Gaussian halo, in width units
    emitter_type: Literal["window", "headlight", "taillight", "sign"] = "window"


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
    emitters: list[Emitter] = field(default_factory=list)
    emitter_counts: dict[str, int] = field(default_factory=dict)

    @property
    def num_lights(self) -> int:
        return int(self.light_xy.shape[0])

    @property
    def num_emitters(self) -> int:
        return len(self.emitters)

    @property
    def num_windows(self) -> int:
        return self.emitter_counts.get("windows", 0)

    @property
    def num_headlights(self) -> int:
        return self.emitter_counts.get("headlights", 0)

    @property
    def num_taillights(self) -> int:
        return self.emitter_counts.get("taillights", 0)

    @property
    def num_signs(self) -> int:
        return self.emitter_counts.get("signs", 0)


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
        mode: From `choose_mode`.
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

    aspect = height / width

    # Sample lamps for streetlit (existing draws preserved in sequence)
    if mode == "streetlit":
        n_lo, n_hi = (int(v) for v in cfg["num_lights_range"])
        count = int(torch.randint(n_lo, n_hi + 1, (), generator=generator))
        if count > 0:
            margin = float(cfg["light_margin"])
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
                temps.append(
                    _uniform(s_lo, s_hi, generator) if sodium else _uniform(l_lo, l_hi, generator)
                )

            params.light_xy = xy
            params.light_height = torch.tensor(heights, dtype=torch.float32, device=device)
            params.light_peak = torch.tensor(peaks, dtype=torch.float32, device=device)
            params.light_temp = torch.tensor(temps, dtype=torch.float32, device=device)
            params.light_gains = torch.stack([color_temp_to_gains(t) for t in temps]).to(device)

    # Sample emitters (draws added AFTER lamps to keep existing random sequence undisturbed)
    counts_cfg = cfg.get("emitter_counts", {}).get(mode, {})
    b_range = cfg.get("emitter_brightness_range", (20.0, 1000.0))
    b_lo, b_hi = float(b_range[0]), float(b_range[1])
    hr_range = cfg.get("emitter_halo_ratio_range", (0.02, 0.08))
    hr_lo, hr_hi = float(hr_range[0]), float(hr_range[1])
    hs_range = cfg.get("emitter_halo_sigma_mult_range", (2.0, 4.0))
    hs_lo, hs_hi = float(hs_range[0]), float(hs_range[1])

    def _count_in_range(key: str) -> int:
        r = counts_cfg.get(key, (0, 0))
        lo_c, hi_c = int(r[0]), int(r[1])
        if hi_c <= lo_c:
            return lo_c
        return int(torch.randint(lo_c, hi_c + 1, (), generator=generator))

    n_windows = _count_in_range("windows")
    n_headlights = _count_in_range("headlights")
    n_taillights = _count_in_range("taillights")
    n_signs = _count_in_range("signs")
    params.emitter_counts = {
        "windows": n_windows,
        "headlights": n_headlights,
        "taillights": n_taillights,
        "signs": n_signs,
    }

    emitters: list[Emitter] = []

    # 1. Lit windows: facade patches with a grid of windows
    for _ in range(n_windows):
        x_f = _uniform(0.05, 0.95, generator)
        y_f = _uniform(0.05, max(0.05, aspect - 0.05), generator)
        rows = int(torch.randint(2, 6, (), generator=generator))
        cols = int(torch.randint(2, 6, (), generator=generator))
        w = _uniform(0.003, 0.008, generator)
        h = _uniform(0.004, 0.012, generator)
        dx = _uniform(1.5, 2.5, generator) * w
        dy = _uniform(1.5, 2.5, generator) * h
        p_on = _uniform(0.3, 0.8, generator)
        base_temp = _uniform(2700.0, 4000.0, generator)
        base_brightness = _log_uniform(b_lo, b_hi, generator)
        halo_mult = _uniform(hs_lo, hs_hi, generator)
        halo_sigma = halo_mult * max(w, h)
        halo_ratio = _uniform(hr_lo, hr_hi, generator)

        candidates: list[tuple[int, int, bool]] = []
        has_lit = False
        for r_idx in range(rows):
            for c_idx in range(cols):
                is_on = _uniform(0.0, 1.0, generator) < p_on
                candidates.append((r_idx, c_idx, is_on))
                if is_on:
                    has_lit = True
        if not has_lit and candidates:
            r_idx, c_idx, _ = candidates[0]
            candidates[0] = (r_idx, c_idx, True)

        for r_idx, c_idx, is_on in candidates:
            if not is_on:
                continue
            win_x = x_f + (c_idx - (cols - 1) / 2.0) * dx
            win_y = y_f + (r_idx - (rows - 1) / 2.0) * dy
            if win_x < -0.05 or win_x > 1.05 or win_y < -0.05 or win_y > aspect + 0.05:
                continue
            t_win = min(4000.0, max(2700.0, base_temp + _uniform(-150.0, 150.0, generator)))
            color = color_temp_to_gains(t_win).to(device)
            b_win = base_brightness * _uniform(0.7, 1.3, generator)
            h_peak = b_win * halo_ratio
            emitters.append(
                Emitter(
                    shape="box",
                    x=win_x,
                    y=win_y,
                    w=w,
                    h=h,
                    radius=0.0,
                    color=color,
                    core_peak=b_win,
                    halo_peak=h_peak,
                    halo_sigma=halo_sigma,
                    emitter_type="window",
                )
            )

    # 2. Headlights: pairs of small discs, 4,500-6,500 K white
    for _ in range(n_headlights):
        x_v = _uniform(0.05, 0.95, generator)
        y_v = _uniform(0.05, max(0.05, aspect - 0.05), generator)
        heading = _uniform(0.0, 2.0 * math.pi, generator)
        d_sep = _uniform(0.010, 0.025, generator)
        r = _uniform(0.002, 0.004, generator)
        t_head = _uniform(4500.0, 6500.0, generator)
        color = color_temp_to_gains(t_head).to(device)
        b_head = _log_uniform(b_lo, b_hi, generator)
        halo_mult = _uniform(hs_lo, hs_hi, generator)
        halo_sigma = halo_mult * r
        halo_ratio = _uniform(hr_lo, hr_hi, generator)
        h_peak = b_head * halo_ratio
        ux, uy = -math.sin(heading), math.cos(heading)
        for sign_dir in (-1.0, 1.0):
            lx = x_v + sign_dir * (d_sep / 2.0) * ux
            ly = y_v + sign_dir * (d_sep / 2.0) * uy
            emitters.append(
                Emitter(
                    shape="disc",
                    x=lx,
                    y=ly,
                    w=2.0 * r,
                    h=2.0 * r,
                    radius=r,
                    color=color,
                    core_peak=b_head,
                    halo_peak=h_peak,
                    halo_sigma=halo_sigma,
                    emitter_type="headlight",
                )
            )

    # 3. Taillights: pairs of small discs, saturated red
    for _ in range(n_taillights):
        x_v = _uniform(0.05, 0.95, generator)
        y_v = _uniform(0.05, max(0.05, aspect - 0.05), generator)
        heading = _uniform(0.0, 2.0 * math.pi, generator)
        d_sep = _uniform(0.010, 0.025, generator)
        r = _uniform(0.0015, 0.0035, generator)
        color = torch.tensor([1.0, 0.03, 0.01], dtype=torch.float32, device=device)
        b_tail = _log_uniform(b_lo * 0.5, b_hi * 0.5, generator)
        halo_mult = _uniform(hs_lo, hs_hi, generator)
        halo_sigma = halo_mult * r
        halo_ratio = _uniform(hr_lo, hr_hi, generator)
        h_peak = b_tail * halo_ratio
        ux, uy = -math.sin(heading), math.cos(heading)
        for sign_dir in (-1.0, 1.0):
            lx = x_v + sign_dir * (d_sep / 2.0) * ux
            ly = y_v + sign_dir * (d_sep / 2.0) * uy
            emitters.append(
                Emitter(
                    shape="disc",
                    x=lx,
                    y=ly,
                    w=2.0 * r,
                    h=2.0 * r,
                    radius=r,
                    color=color,
                    core_peak=b_tail,
                    halo_peak=h_peak,
                    halo_sigma=halo_sigma,
                    emitter_type="taillight",
                )
            )

    # 4. Signs: small rectangles/strips in saturated hues
    for _ in range(n_signs):
        x_s = _uniform(0.05, 0.95, generator)
        y_s = _uniform(0.05, max(0.05, aspect - 0.05), generator)
        w = _uniform(0.006, 0.025, generator)
        h = _uniform(0.003, 0.015, generator)
        c_idx = int(torch.randint(0, len(SIGN_COLORS), (), generator=generator))
        color = torch.tensor(SIGN_COLORS[c_idx], dtype=torch.float32, device=device)
        b_sign = _log_uniform(b_lo, b_hi, generator)
        halo_mult = _uniform(hs_lo, hs_hi, generator)
        halo_sigma = halo_mult * max(w, h)
        halo_ratio = _uniform(hr_lo, hr_hi, generator)
        h_peak = b_sign * halo_ratio
        emitters.append(
            Emitter(
                shape="box",
                x=x_s,
                y=y_s,
                w=w,
                h=h,
                radius=0.0,
                color=color,
                core_peak=b_sign,
                halo_peak=h_peak,
                halo_sigma=halo_sigma,
                emitter_type="sign",
            )
        )

    params.emitters = emitters
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


def emission_field(
    params: LightingParams,
    height: int | None = None,
    width: int | None = None,
) -> torch.Tensor:
    """The (3, H, W) additive emission field from all small emissive points.

    Renders by area coverage so integrated energy is resolution-independent.
    All lengths are fractions of the image WIDTH.

    Args:
        params: LightingParams holding sampled emitters.
        height: Optional height override (defaults to params.height).
        width: Optional width override (defaults to params.width).

    Returns:
        (3, H, W) float32 additive emission field, >= 0.
    """
    h_out = params.height if height is None else height
    w_out = params.width if width is None else width
    device = params.illum_field.device
    emission = torch.zeros(3, h_out, w_out, dtype=torch.float32, device=device)
    if not params.emitters:
        return emission

    w_float = float(w_out)
    for e in params.emitters:
        rad_px = int(math.ceil(4.0 * e.halo_sigma * w_float))
        cx_px = int(e.x * w_float)
        cy_px = int(e.y * w_float)
        c0 = max(0, cx_px - rad_px)
        c1 = min(w_out, cx_px + rad_px + 1)
        r0 = max(0, cy_px - rad_px)
        r1 = min(h_out, cy_px + rad_px + 1)
        if c1 <= c0 or r1 <= r0:
            continue

        cols_idx = torch.arange(c0, c1, device=device, dtype=torch.float32)
        rows_idx = torch.arange(r0, r1, device=device, dtype=torch.float32)

        if e.shape == "box":
            x_min_px = (e.x - e.w / 2.0) * w_float
            x_max_px = (e.x + e.w / 2.0) * w_float
            y_min_px = (e.y - e.h / 2.0) * w_float
            y_max_px = (e.y + e.h / 2.0) * w_float
            cov_x = (
                torch.minimum(cols_idx + 1.0, torch.tensor(x_max_px, device=device))
                - torch.maximum(cols_idx, torch.tensor(x_min_px, device=device))
            ).clamp(0.0, 1.0)
            cov_y = (
                torch.minimum(rows_idx + 1.0, torch.tensor(y_max_px, device=device))
                - torch.maximum(rows_idx, torch.tensor(y_min_px, device=device))
            ).clamp(0.0, 1.0)
            core_cov = cov_y.unsqueeze(1) * cov_x.unsqueeze(0)
        else:
            r_px = e.radius * w_float
            cx_px_f = e.x * w_float
            cy_px_f = e.y * w_float
            if r_px < 1.5:
                # Subpixel 8x8 supersampling with exact energy preservation
                k_sub = 8
                sub = (torch.arange(k_sub, device=device, dtype=torch.float32) + 0.5) / k_sub
                sub_x = (cols_idx.unsqueeze(1) + sub.unsqueeze(0)).flatten()
                sub_y = (rows_idx.unsqueeze(1) + sub.unsqueeze(0)).flatten()
                dx_sub = sub_x.unsqueeze(0) - cx_px_f
                dy_sub = sub_y.unsqueeze(1) - cy_px_f
                inside = ((dx_sub.pow(2) + dy_sub.pow(2)) <= (r_px**2)).float()
                core_cov = inside.view(len(rows_idx), k_sub, len(cols_idx), k_sub).mean(dim=(1, 3))
                cov_sum = core_cov.sum()
                if cov_sum > 0:
                    core_cov = core_cov * (math.pi * (r_px**2) / cov_sum)
            else:
                dx = (cols_idx.unsqueeze(0) + 0.5) - cx_px_f
                dy = (rows_idx.unsqueeze(1) + 0.5) - cy_px_f
                d = torch.sqrt(dx.pow(2) + dy.pow(2))
                core_cov = (r_px - d + 0.5).clamp(0.0, 1.0)

        # Soft glow halo (Gaussian with 4*sigma cutoff)
        cols_norm = (cols_idx + 0.5) / w_float
        rows_norm = (rows_idx + 0.5) / w_float
        dx_h = cols_norm - e.x
        dy_h = rows_norm - e.y
        d2 = dy_h.unsqueeze(1).pow(2) + dx_h.unsqueeze(0).pow(2)
        cutoff_d2 = (4.0 * e.halo_sigma) ** 2
        halo = torch.exp(-0.5 * d2 / (e.halo_sigma**2))
        halo = torch.where(d2 <= cutoff_d2, halo, torch.zeros_like(halo))

        patch = core_cov * e.core_peak + halo * e.halo_peak
        emission[:, r0:r1, c0:c1] += patch.unsqueeze(0) * e.color.view(3, 1, 1)

    return emission


def apply_lighting(day_linear: torch.Tensor, params: LightingParams) -> torch.Tensor:
    """Relight a daytime image as a night scene.

    Scene radiance combines surface reflectance under night illumination with additive scene
    emission from urban emitters:

        radiance = reflectance * illumination + emission

    Args:
        day_linear: (3, H, W) linear RGB in [0, 1], read as reflectance.
        params: From `sample_lighting_params`, sized for this image.

    Returns:
        radiance: (3, H, W) float32 scene radiance, >= 0 and **unbounded above** (ambient-lit
        white is 1.0, a lamp foot reaches `light_peak`, emitters reach `core_peak`). camera.py
        turns this into an image.

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
    illum = illumination_field(params)
    emission = emission_field(params)
    return (reflectance * illum + emission).to(torch.float32)
