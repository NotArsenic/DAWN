"""
Color space conversion utilities.
"""

from __future__ import annotations

import torch

SRGB_BREAKPOINT = 0.04045  # toe/power switch (in sRGB-encoded units)
LINEAR_BREAKPOINT = 0.0031308  # the same switch (in linear units)
REC709_WEIGHTS = (0.2126, 0.7152, 0.0722)  # linear R, G, B -> relative luminance

COLOR_TEMP_MIN = 1700.0
COLOR_TEMP_MAX = 12000.0
REFERENCE_COLOR_TEMP = 6500.0  # gains are relative to this
CHANNEL_FLOOR = 1e-4  # smallest illuminant channel

# XYZ -> linear sRGB
_XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def srgb_to_linear(img: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Convert sRGB [0, 1] to linear RGB [0, 1].
    Applies the inverse sRGB transfer function (IEC 61966-2-1)

    Args:
        img: sRGB values, any shape.
        clamp: Clamp the input to [0, 1] first (default).
    """
    if clamp:
        img = img.clamp(0.0, 1.0)

    high = ((img.clamp(min=SRGB_BREAKPOINT) + 0.055) / 1.055) ** 2.4
    return torch.where(img <= SRGB_BREAKPOINT, img / 12.92, high)


def linear_to_srgb(img: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """
    Convert linear RGB [0, 1] to sRGB [0, 1].
    Applies the sRGB transfer function (IEC 61966-2-1)

    Args:
        img: Linear values, any shape.
        clamp: Clamp the input to [0, 1] first (default).
    """
    if clamp:
        img = img.clamp(0.0, 1.0)

    high = 1.055 * (img.clamp(min=LINEAR_BREAKPOINT) ** (1.0 / 2.4) - 1.0) + 1.0
    return torch.where(img <= LINEAR_BREAKPOINT, img * 12.92, high)


def _planckian_xy(kelvin: float) -> tuple[float, float]:
    """
    CIE 1931 chromaticity (x, y) on the Planckian locus, 1,667-25,000 K.
    Cubic in `1/T` for x, then cubic in x for y, with break points at 4,000 K (x and y) and 2,222 K (y). Agrees with the exact locus to about 5e-4 above 3,000 K and 3e-3 at 1,700 K.
    """
    t = kelvin
    if t <= 4000.0:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def luminance(img_linear: torch.Tensor) -> torch.Tensor:
    """
    Relative luminance of a linear RGB image (Rec. 709 weights).
    Y = 0.2126 R + 0.7152 G + 0.0722 B.

    Args:
        img_linear: (3, H, W) or (B, 3, H, W) linear RGB.

    Returns:
        (1, H, W) or (B, 1, H, W): one channel, with the channel dimension kept.
    """
    if img_linear.ndim not in (3, 4) or img_linear.shape[-3] != 3:
        raise ValueError(f"expected (3, H, W) or (B, 3, H, W), got {tuple(img_linear.shape)}")
    weights = img_linear.new_tensor(REC709_WEIGHTS).view(3, 1, 1)
    return (img_linear * weights).sum(dim=-3, keepdim=True)


def color_temp_to_illuminant_rgb(color_temp: float) -> torch.Tensor:
    """
    Linear sRGB color of a blackbody light at `color_temp` Kelvin, scaled to luminance Y = 1.

    Args:
        color_temp: Kelvin, within [COLOR_TEMP_MIN, COLOR_TEMP_MAX] (1,700 to 12,000).

    Returns:
        (3,) float32 linear RGB, every entry at least 1e-4.

    Raises:
        ValueError: `color_temp` is outside the valid range, or NaN.
    """
    kelvin = float(color_temp)
    if not COLOR_TEMP_MIN <= kelvin <= COLOR_TEMP_MAX:  # also false for NaN
        raise ValueError(
            f"color_temp must be within [{COLOR_TEMP_MIN:g}, {COLOR_TEMP_MAX:g}] K, "
            f"got {color_temp}"
        )
    x, y = _planckian_xy(kelvin)
    xyz = (x / y, 1.0, (1.0 - x - y) / y)  # Y = 1
    rgb = [max(sum(m * c for m, c in zip(row, xyz)), CHANNEL_FLOOR) for row in _XYZ_TO_SRGB]
    return torch.tensor(rgb, dtype=torch.float32)


def color_temp_to_gains(color_temp: float) -> torch.Tensor:
    """
    Per-channel gains that retint a 6,500 K scene (REFERENCE_COLOR_TEMP) to `color_temp`.

    Args:
        color_temp: Kelvin, within [COLOR_TEMP_MIN, COLOR_TEMP_MAX].

    Returns:
        (3,) float32 gains for linear R, G, B; all positive, green exactly 1.

    Raises:
        ValueError: `color_temp` is outside the valid range, or NaN.
    """
    gains = color_temp_to_illuminant_rgb(color_temp) / color_temp_to_illuminant_rgb(
        REFERENCE_COLOR_TEMP
    )
    return gains / gains[1]


def srgb_gamma_slope(linear_val: torch.Tensor) -> torch.Tensor:
    """
    Derivative of the sRGB transfer function at a given linear value.

    Useful for propagating variance from linear space to sRGB display space:
        Var_srgb ≈ (d_srgb/d_linear)^2 * Var_linear

    Args:
        linear_val: Linear RGB values in [0, 1].

    Returns:
        Slope of the sRGB gamma curve at each value.
    """
    threshold = 0.0031308
    low_slope = torch.tensor(12.92, device=linear_val.device)
    high_slope = (1.055 / 2.4) * linear_val.clamp(min=1e-8) ** (1.0 / 2.4 - 1.0)
    return torch.where(linear_val <= threshold, low_slope, high_slope)
