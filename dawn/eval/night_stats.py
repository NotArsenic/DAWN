"""Per-image statistics for nighttime drone image enhancement.

Evaluates linear RGB images at native resolution to quantify darkness,
contrast, color fidelity, noise characteristics, and black crushing.
Used to compare real night frames side-by-side with synthetic degradation.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch

from dawn.data.color import linear_to_srgb, linear_to_srgb8, luminance
from dawn.data.prepare import highlight_ratio as compute_highlight_ratio
from dawn.data.prepare import noise_level as compute_noise_level


@dataclass(frozen=True)
class NightStats:
    """Per-image statistics computed on a (3, H, W) linear RGB image."""

    lum_median: float
    lum_p90: float
    sat_mean: float
    sat_pixel_fraction: float
    highlight_ratio: float
    flat_noise_srgb: float
    flat_noise_linear: float
    flat_patch_mean: float
    noise_level: float
    code0_fraction: float

    def as_dict(self) -> dict[str, float]:
        """Return statistics as a dictionary of floats."""
        return asdict(self)


def compute_luminance_stats(img_linear: torch.Tensor) -> tuple[float, float]:
    """Median and 90th percentile of linear Rec. 709 luminance.

    Args:
        img_linear: (3, H, W) linear RGB in [0, 1].

    Returns:
        (lum_median, lum_p90)
    """
    lum = luminance(img_linear).flatten().to(torch.float32)
    median = lum.median().item()
    p90 = torch.quantile(lum, 0.90).item()
    return median, p90


def compute_saturation_stats(
    img_linear: torch.Tensor, luma_threshold: float = 10.0 / 255.0
) -> tuple[float, float]:
    """Mean HSV-style saturation (max-min)/max of sRGB pixels above a luma threshold.

    Restricted to pixels whose sRGB luma is at least `luma_threshold` (default code 10/255)
    so crushed blacks do not dominate.

    Args:
        img_linear: (3, H, W) linear RGB in [0, 1].
        luma_threshold: Minimum sRGB luma to include in the saturation average.

    Returns:
        (sat_mean, sat_pixel_fraction)
    """
    srgb = linear_to_srgb(img_linear.clamp(0.0, 1.0))
    # Rec. 709 luma on non-linear sRGB
    luma = 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2]
    mask = luma >= luma_threshold
    fraction = mask.float().mean().item()
    if not mask.any():
        return float("nan"), fraction

    c_max = srgb.amax(dim=0)
    c_min = srgb.amin(dim=0)
    sat = (c_max - c_min) / c_max.clamp(min=1e-8)
    mean_sat = sat[mask].mean().item()
    return mean_sat, fraction


def compute_code0_fraction(img_linear: torch.Tensor) -> float:
    """Share of pixels whose 8-bit sRGB value is 0 in all channels."""
    srgb8 = linear_to_srgb8(img_linear)
    is_zero = (srgb8 == 0).all(dim=0)
    return is_zero.float().mean().item()


def compute_flat_patch_noise(
    img_linear: torch.Tensor,
    patch_size: int = 8,
    flat_fraction: float = 0.10,
) -> tuple[float, float, float]:
    """Measure residual noise in the flattest patches after removing a plane fit.

    Tiles the image into patch_size x patch_size non-overlapping patches, drops patches
    touching sRGB 8-bit code <= 3 or >= 252 in any channel, detrends the remaining
    patches with a linear plane fit, and selects the flattest `flat_fraction` (e.g. 10%)
    by residual variance.

    Args:
        img_linear: (3, H, W) linear RGB in [0, 1].
        patch_size: Square tile side (default 8).
        flat_fraction: Fraction of surviving patches to keep as flattest (default 0.10).

    Returns:
        (flat_noise_srgb, flat_noise_linear, flat_patch_mean)
        - flat_noise_srgb: Median residual std in 8-bit sRGB code units.
        - flat_noise_linear: Median residual std in linear units.
        - flat_patch_mean: Median patch mean luminance (linear).
        Returns (nan, nan, nan) if no valid patches survive.
    """
    _, height, width = img_linear.shape
    h_p = height // patch_size
    w_p = width // patch_size
    if h_p == 0 or w_p == 0:
        return float("nan"), float("nan"), float("nan")

    srgb8 = linear_to_srgb8(img_linear)
    p_len = patch_size * patch_size

    # Crop to multiple of patch_size
    img_crop = img_linear[:, : h_p * patch_size, : w_p * patch_size]
    srgb8_crop = srgb8[:, : h_p * patch_size, : w_p * patch_size]

    # Tile into (N, 3, patch_size, patch_size)
    patches_lin = (
        img_crop.view(3, h_p, patch_size, w_p, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, 3, patch_size, patch_size)
    )
    patches_srgb8 = (
        srgb8_crop.view(3, h_p, patch_size, w_p, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(-1, 3, patch_size, patch_size)
    )

    # Drop patches touching code <= 3 or >= 252 in any channel
    min_codes = patches_srgb8.amin(dim=(1, 2, 3))
    max_codes = patches_srgb8.amax(dim=(1, 2, 3))
    valid = (min_codes > 3) & (max_codes < 252)
    num_valid = int(valid.sum().item())
    if num_valid == 0:
        return float("nan"), float("nan"), float("nan")

    p_lin_valid = patches_lin[valid]  # (M, 3, patch_size, patch_size)
    p_srgb_valid = patches_srgb8[valid].to(torch.float32)

    # Construct plane projector: removes constant, x slope, y slope (3 DOF)
    coords = torch.linspace(
        -(patch_size - 1) / 2.0,
        (patch_size - 1) / 2.0,
        patch_size,
        device=img_linear.device,
        dtype=torch.float32,
    )
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    basis = torch.stack(
        [
            torch.ones(p_len, device=img_linear.device, dtype=torch.float32),
            ys.reshape(-1),
            xs.reshape(-1),
        ],
        dim=1,
    )  # (p_len, 3)
    pinv = torch.linalg.pinv(basis)
    proj = torch.eye(p_len, device=img_linear.device) - basis @ pinv  # (p_len, p_len)

    # Detrend
    p_lin_flat = p_lin_valid.reshape(num_valid, 3, p_len)
    p_srgb_flat = p_srgb_valid.reshape(num_valid, 3, p_len)
    resid_lin = p_lin_flat @ proj
    resid_srgb = p_srgb_flat @ proj

    # Unbiased residual variance (p_len - 3 degrees of freedom)
    dof = float(p_len - 3)
    var_srgb_c = (resid_srgb**2).sum(dim=-1) / dof  # (M, 3)
    var_lin_c = (resid_lin**2).sum(dim=-1) / dof  # (M, 3)

    std_srgb_patch = var_srgb_c.mean(dim=-1).sqrt()  # (M,)
    std_lin_patch = var_lin_c.mean(dim=-1).sqrt()  # (M,)

    # Rank by texture score (std_srgb_patch) and keep flattest fraction
    k = max(1, math.ceil(num_valid * flat_fraction))
    _, flattest_idx = torch.topk(std_srgb_patch, k=k, largest=False)

    flat_noise_srgb = std_srgb_patch[flattest_idx].median().item()
    flat_noise_linear = std_lin_patch[flattest_idx].median().item()

    # Median patch mean luminance (linear)
    lum_patches = (
        0.2126 * p_lin_valid[:, 0]
        + 0.7152 * p_lin_valid[:, 1]
        + 0.0722 * p_lin_valid[:, 2]
    ).mean(dim=(-1, -2))
    flat_patch_mean = lum_patches[flattest_idx].median().item()

    return flat_noise_srgb, flat_noise_linear, flat_patch_mean


def compute_night_stats(
    img_linear: torch.Tensor,
    luma_threshold: float = 10.0 / 255.0,
    patch_size: int = 8,
    flat_fraction: float = 0.10,
) -> NightStats:
    """Compute full suite of nighttime statistics on a native-resolution linear image.

    Args:
        img_linear: (3, H, W) linear RGB float32 in [0, 1].
        luma_threshold: Minimum sRGB luma for saturation computation.
        patch_size: Tile side for flat-patch noise measurement.
        flat_fraction: Fraction of flat patches to retain.

    Returns:
        NightStats dataclass containing all statistics.
    """
    if img_linear.ndim != 3 or img_linear.shape[0] != 3:
        raise ValueError(f"expected (3, H, W) image, got {tuple(img_linear.shape)}")

    lum_median, lum_p90 = compute_luminance_stats(img_linear)
    sat_mean, sat_pixel_frac = compute_saturation_stats(
        img_linear, luma_threshold=luma_threshold
    )
    code0_frac = compute_code0_fraction(img_linear)
    hl_ratio = compute_highlight_ratio(img_linear)
    noise_lvl = compute_noise_level(img_linear)
    flat_srgb, flat_lin, patch_mean = compute_flat_patch_noise(
        img_linear, patch_size=patch_size, flat_fraction=flat_fraction
    )

    return NightStats(
        lum_median=lum_median,
        lum_p90=lum_p90,
        sat_mean=sat_mean,
        sat_pixel_fraction=sat_pixel_frac,
        highlight_ratio=hl_ratio,
        flat_noise_srgb=flat_srgb,
        flat_noise_linear=flat_lin,
        flat_patch_mean=patch_mean,
        noise_level=noise_lvl,
        code0_fraction=code0_frac,
    )
