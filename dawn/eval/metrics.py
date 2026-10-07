"""Image quality metrics.

Current (flow phase, used by validation and `scripts/check_oracle.py`): `masked_psnr` (L* PSNR with
the peak at 100), `ciede2000` (the colour difference, in torch) and `colorfulness`
(Hasler & Suesstrunk). The rest of this module (SSIM, LPIPS, object-crop metrics) is the
inference/eval phase's and is still a stub.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricResults:
    """Aggregated evaluation metrics."""

    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    # Object-crop metrics (computed on VisDrone detection boxes).
    crop_psnr: float = 0.0
    crop_ssim: float = 0.0
    # Hallucination audit.
    objects_appeared: int = 0  # detected in output but not input
    objects_disappeared: int = 0  # detected in input but not output
    num_images: int = 0


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR between two batches of sRGB images.

    Args:
        pred: (B, 3, H, W) in [0, 1].
        target: (B, 3, H, W) in [0, 1].

    Returns:
        (B,) PSNR values in dB.
    """
    mse = (pred - target).pow(2).mean(dim=[1, 2, 3])
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structural similarity (SSIM) between two batches.

    Uses the standard 11x11 Gaussian window, K1=0.01, K2=0.03.

    Args:
        pred: (B, 3, H, W) in [0, 1].
        target: (B, 3, H, W) in [0, 1].

    Returns:
        (B,) SSIM values.
    """
    raise NotImplementedError


@torch.no_grad()
def evaluate_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    input_img: torch.Tensor | None = None,
    lpips_fn=None,
) -> dict[str, float]:
    """Compute all metrics for a batch.

    Args:
        pred: (B, 3, H, W) predicted sRGB.
        target: (B, 3, H, W) ground-truth sRGB.
        input_img: (B, 3, H, W) original night input (for hallucination audit).
        lpips_fn: Pre-initialized LPIPS loss function (or None to skip).

    Returns:
        dict of metric names to mean values.
    """
    raise NotImplementedError


# --- the flow-phase metrics -----------------------------------------------------------------------


def masked_psnr(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, peak: float = 1.0
) -> torch.Tensor:
    """PSNR over the pixels that count, per image: `10 log10(peak^2 / masked MSE)`.

    For lightness pass L* with `peak=100` (the plan evaluates structure as PSNR on L*).

    Args:
        pred, target: (B, C, H, W), the same units.
        mask: (B, 1, H, W) weights in [0, 1] (the pixel mask; fractional is fine).
        peak: The signal peak.

    Returns:
        (B,) dB. An image with no weight has zero error and so a (large, finite) value; callers
        that average should drop it.
    """
    weight = mask.to(torch.float32)
    channels = pred.shape[1]
    err = ((pred.float() - target.float()).pow(2) * weight).sum(dim=(1, 2, 3))
    mse = err / (weight.sum(dim=(1, 2, 3)) * channels).clamp(min=torch.finfo(torch.float32).tiny)
    return 10.0 * torch.log10(peak**2 / mse.clamp(min=1e-10))


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image mean of `value` over the pixels that count: `sum(value * w) / sum(w)`.

    Args:
        value: (B, H, W) or (B, 1, H, W) per-pixel values (a colour difference, an error norm).
        mask: (B, 1, H, W) weights in [0, 1].

    Returns:
        (B,), 0 for an image with no weight.
    """
    weight = mask.to(torch.float32).flatten(1)
    value = value.to(torch.float32).reshape(value.shape[0], -1)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp(
        min=torch.finfo(torch.float32).tiny
    )


def ciede2000(
    lab_1: torch.Tensor, lab_2: torch.Tensor, k_l: float = 1.0, k_c: float = 1.0, k_h: float = 1.0
) -> torch.Tensor:
    """CIEDE2000 colour difference (Sharma, Wu & Dalal 2005), elementwise.

    Args:
        lab_1, lab_2: `(..., 3, H, W)` L*, a*, b* (the pipeline's layout).

    Returns:
        `(..., H, W)` in the input's dtype. Matches `skimage.color.deltaE_ciede2000`.
    """
    l1, a1, b1 = lab_1.unbind(dim=-3)
    l2, a2, b2 = lab_2.unbind(dim=-3)
    pow7 = 25.0**7
    c_bar = 0.5 * (torch.hypot(a1, b1) + torch.hypot(a2, b2))
    g = 0.5 * (1.0 - torch.sqrt(c_bar**7 / (c_bar**7 + pow7)))
    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p, c2p = torch.hypot(a1p, b1), torch.hypot(a2p, b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0

    d_l, d_c = l2 - l1, c2p - c1p
    zero_chroma = c1p * c2p == 0
    d_h_raw = h2p - h1p
    d_h_raw = torch.where(d_h_raw > 180.0, d_h_raw - 360.0, d_h_raw)
    d_h_raw = torch.where(d_h_raw < -180.0, d_h_raw + 360.0, d_h_raw)
    d_h_raw = torch.where(zero_chroma, torch.zeros_like(d_h_raw), d_h_raw)
    d_hp = 2.0 * torch.sqrt(c1p * c2p) * torch.sin(torch.deg2rad(d_h_raw) / 2.0)

    l_bar, c_bar_p = 0.5 * (l1 + l2), 0.5 * (c1p + c2p)
    h_sum = h1p + h2p
    h_bar = torch.where(
        (h1p - h2p).abs() <= 180.0,
        h_sum / 2.0,
        torch.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
    )
    h_bar = torch.where(zero_chroma, h_sum, h_bar)

    t = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(h_bar - 30.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * h_bar))
        + 0.32 * torch.cos(torch.deg2rad(3.0 * h_bar + 6.0))
        - 0.20 * torch.cos(torch.deg2rad(4.0 * h_bar - 63.0))
    )
    d_theta = 30.0 * torch.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    r_c = 2.0 * torch.sqrt(c_bar_p**7 / (c_bar_p**7 + pow7))
    s_l = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / torch.sqrt(20.0 + (l_bar - 50.0) ** 2)
    s_c = 1.0 + 0.045 * c_bar_p
    s_h = 1.0 + 0.015 * c_bar_p * t
    r_t = -torch.sin(torch.deg2rad(2.0 * d_theta)) * r_c

    term_l, term_c, term_h = d_l / (k_l * s_l), d_c / (k_c * s_c), d_hp / (k_h * s_h)
    return torch.sqrt((term_l**2 + term_c**2 + term_h**2 + r_t * term_c * term_h).clamp(min=0.0))


def colorfulness(srgb: torch.Tensor) -> torch.Tensor:
    """Hasler & Suesstrunk (2003) colorfulness, per image.

    `M = sqrt(std_rg^2 + std_yb^2) + 0.3 sqrt(mean_rg^2 + mean_yb^2)` with `rg = R - G` and
    `yb = (R + G) / 2 - B`. A grey image scores 0.

    Args:
        srgb: (B, 3, H, W) sRGB in 0-255 (display units, not linear).

    Returns:
        (B,) colorfulness.

    Raises:
        ValueError: `srgb` is not (B, 3, H, W).
    """
    if srgb.ndim != 4 or srgb.shape[1] != 3:
        raise ValueError(f"colorfulness needs an RGB image (B, 3, H, W), got {tuple(srgb.shape)}")
    r, g, b = srgb.unbind(dim=1)
    rg, yb = (r - g).flatten(1), (0.5 * (r + g) - b).flatten(1)
    std = torch.sqrt(rg.std(dim=1, unbiased=False) ** 2 + yb.std(dim=1, unbiased=False) ** 2)
    mean = torch.sqrt(rg.mean(dim=1) ** 2 + yb.mean(dim=1) ** 2)
    return std + 0.3 * mean
