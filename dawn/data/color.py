"""
Color space conversion utilities: sRGB <-> linear, luminance, CIELAB.
"""

from __future__ import annotations

import torch

SRGB_BREAKPOINT = 0.04045  # toe/power switch (in sRGB-encoded units)
LINEAR_BREAKPOINT = 0.0031308  # the same switch (in linear units)
REC709_WEIGHTS = (0.2126, 0.7152, 0.0722)  # linear R, G, B -> relative luminance

# Linear sRGB (D65) -> XYZ. The D65 white point is this matrix applied to (1, 1, 1), so white
# maps to exactly (100, 0, 0) in CIELAB.
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_LAB_DELTA = 6.0 / 29.0  # CIE: the cube root meets its linear segment at t = delta^3
_LAB_DELTA_CUBED = _LAB_DELTA**3


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


def srgb8_to_linear(img: torch.Tensor) -> torch.Tensor:
    """
    Decode 8-bit sRGB codes to linear float32 [0, 1] (the file-input boundary).

    Bit-identical to `srgb_to_linear(img.float() / 255)`, through a 256-entry lookup table.

    Args:
        img: uint8 sRGB codes, any shape.

    Raises:
        TypeError: `img` is not uint8.
    """
    if img.dtype != torch.uint8:
        raise TypeError(f"expected uint8 sRGB codes, got {img.dtype}")
    lut = srgb_to_linear(torch.arange(256, dtype=torch.float32, device=img.device) / 255)
    return lut[img.long()]


def linear_to_srgb8(img: torch.Tensor) -> torch.Tensor:
    """
    Encode linear [0, 1] to 8-bit sRGB codes (the file-output boundary): clamp, gamma, round.

    Args:
        img: Linear values, any shape.

    Returns:
        uint8 tensor of the same shape.
    """
    return (linear_to_srgb(img) * 255).round().to(torch.uint8)


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


def _check_three_channels(img: torch.Tensor) -> None:
    if img.ndim < 3 or img.shape[-3] != 3:
        raise ValueError(f"expected (..., 3, H, W), got {tuple(img.shape)}")


def _xyz_matrices(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`(rgb_to_xyz, xyz_to_rgb, white)` in `img`'s dtype and device, inverted in float64 so the
    round trip does not inherit the rounding of the printed constants."""
    forward = torch.tensor(_SRGB_TO_XYZ, dtype=torch.float64)
    white = forward.sum(dim=1)
    inverse = torch.linalg.inv(forward)
    return tuple(m.to(device=img.device, dtype=img.dtype) for m in (forward, inverse, white))


def linear_to_lab(img_linear: torch.Tensor) -> torch.Tensor:
    """
    Linear sRGB to CIELAB (D65 white), in the pipeline's `(..., 3, H, W)` layout.

    Channels out are L*, a*, b*: L* in [0, 100] for linear values in [0, 1], a* and b* roughly
    within [-100, 100]. The CIE linear segment below `(6/29)^3` keeps gradients finite at black.
    No clamping: callers clamp to [0, 1] first if the input can leave the range.

    Args:
        img_linear: `(..., 3, H, W)` linear RGB (not sRGB-encoded), float32.

    Raises:
        ValueError: The channel dimension (-3) is not 3.
    """
    _check_three_channels(img_linear)
    to_xyz, _, white = _xyz_matrices(img_linear)
    xyz = torch.einsum("ij,...jhw->...ihw", to_xyz, img_linear)
    t = xyz / white.view(3, 1, 1)
    cube_root = t.clamp(min=_LAB_DELTA_CUBED) ** (1.0 / 3.0)  # clamped: no inf gradient at 0
    linear = t / (3.0 * _LAB_DELTA**2) + 4.0 / 29.0
    f = torch.where(t > _LAB_DELTA_CUBED, cube_root, linear)
    fx, fy, fz = f.unbind(dim=-3)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-3)


def lab_to_linear(lab: torch.Tensor) -> torch.Tensor:
    """
    CIELAB (D65 white) to linear sRGB, the inverse of `linear_to_lab`.

    Not clamped: a* and b* that fall outside the sRGB gamut give channels below 0 or above 1, and
    the caller clamps to [0, 1] (docs/conventions.md section 1).

    Args:
        lab: `(..., 3, H, W)` L*, a*, b*, float32.

    Raises:
        ValueError: The channel dimension (-3) is not 3.
    """
    _check_three_channels(lab)
    _, to_rgb, white = _xyz_matrices(lab)
    l_star, a_star, b_star = lab.unbind(dim=-3)
    fy = (l_star + 16.0) / 116.0
    f = torch.stack([fy + a_star / 500.0, fy, fy - b_star / 200.0], dim=-3)
    cubed = f**3
    linear = 3.0 * _LAB_DELTA**2 * (f - 4.0 / 29.0)
    t = torch.where(f > _LAB_DELTA, cubed, linear)
    return torch.einsum("ij,...jhw->...ihw", to_rgb, t * white.view(3, 1, 1))


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
