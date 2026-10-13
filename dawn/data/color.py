"""Color space conversion utilities (sRGB, linear RGB, CIELAB)."""

from __future__ import annotations

import torch

SRGB_BREAKPOINT = 0.04045
LINEAR_BREAKPOINT = 0.0031308
REC709_WEIGHTS = (0.2126, 0.7152, 0.0722)

_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_LAB_DELTA = 6.0 / 29.0
_LAB_DELTA_CUBED = _LAB_DELTA**3


def srgb_to_linear(img: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Convert sRGB values to linear RGB using the inverse sRGB transfer function."""
    if clamp:
        img = img.clamp(0.0, 1.0)

    high = ((img.clamp(min=SRGB_BREAKPOINT) + 0.055) / 1.055) ** 2.4
    return torch.where(img <= SRGB_BREAKPOINT, img / 12.92, high)


def linear_to_srgb(img: torch.Tensor, clamp: bool = True) -> torch.Tensor:
    """Convert linear RGB values to sRGB using the sRGB transfer function."""
    if clamp:
        img = img.clamp(0.0, 1.0)

    high = 1.055 * (img.clamp(min=LINEAR_BREAKPOINT) ** (1.0 / 2.4) - 1.0) + 1.0
    return torch.where(img <= LINEAR_BREAKPOINT, img * 12.92, high)


def srgb8_to_linear(img: torch.Tensor) -> torch.Tensor:
    """Decode 8-bit sRGB uint8 tensor to linear RGB float32 in [0, 1]."""
    if img.dtype != torch.uint8:
        raise TypeError(f"expected uint8 sRGB codes, got {img.dtype}")
    lut = srgb_to_linear(torch.arange(256, dtype=torch.float32, device=img.device) / 255)
    return lut[img.long()]


def linear_to_srgb8(img: torch.Tensor) -> torch.Tensor:
    """Encode linear RGB float tensor in [0, 1] to 8-bit sRGB uint8."""
    return (linear_to_srgb(img) * 255).round().to(torch.uint8)


def luminance(img_linear: torch.Tensor) -> torch.Tensor:
    """Compute relative luminance from linear RGB using Rec. 709 weights."""
    if img_linear.ndim not in (3, 4) or img_linear.shape[-3] != 3:
        raise ValueError(f"expected (3, H, W) or (B, 3, H, W), got {tuple(img_linear.shape)}")
    weights = img_linear.new_tensor(REC709_WEIGHTS).view(3, 1, 1)
    return (img_linear * weights).sum(dim=-3, keepdim=True)


def _check_three_channels(img: torch.Tensor) -> None:
    if img.ndim < 3 or img.shape[-3] != 3:
        raise ValueError(f"expected (..., 3, H, W), got {tuple(img.shape)}")


def _xyz_matrices(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sRGB-to-XYZ, XYZ-to-sRGB, and white point matrices on img's device and dtype."""
    forward = torch.tensor(_SRGB_TO_XYZ, dtype=torch.float64)
    white = forward.sum(dim=1)
    inverse = torch.linalg.inv(forward)
    return tuple(m.to(device=img.device, dtype=img.dtype) for m in (forward, inverse, white))


def linear_to_lab(img_linear: torch.Tensor) -> torch.Tensor:
    """Convert linear RGB to CIELAB under D65 illuminant."""
    _check_three_channels(img_linear)
    to_xyz, _, white = _xyz_matrices(img_linear)
    xyz = torch.einsum("ij,...jhw->...ihw", to_xyz, img_linear)
    t = xyz / white.view(3, 1, 1)
    cube_root = t.clamp(min=_LAB_DELTA_CUBED) ** (1.0 / 3.0)
    linear = t / (3.0 * _LAB_DELTA**2) + 4.0 / 29.0
    f = torch.where(t > _LAB_DELTA_CUBED, cube_root, linear)
    fx, fy, fz = f.unbind(dim=-3)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=-3)


def lab_to_linear(lab: torch.Tensor) -> torch.Tensor:
    """Convert CIELAB under D65 illuminant to linear RGB."""
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
    """Compute the derivative of the sRGB transfer function at given linear values."""
    threshold = 0.0031308
    low_slope = torch.tensor(12.92, device=linear_val.device)
    high_slope = (1.055 / 2.4) * linear_val.clamp(min=1e-8) ** (1.0 / 2.4 - 1.0)
    return torch.where(linear_val <= threshold, low_slope, high_slope)
