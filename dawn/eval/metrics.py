"""Image quality evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MetricResults:
    """Evaluation metrics for a subset of image pairs."""

    psnr: float = 0.0
    ssim: float = 0.0
    ciede2000: float = 0.0
    colorfulness: float = 0.0
    colorfulness_target: float = 0.0
    lpips: float = 0.0
    psnr_vehicles: float = 0.0
    ssim_vehicles: float = 0.0
    ciede2000_vehicles: float = 0.0
    num_images: int = 0


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute peak signal-to-noise ratio between two batches of RGB images.

    Args:
        pred: (B, 3, H, W) images in [0, 1].
        target: (B, 3, H, W) images in [0, 1].

    Returns:
        (B,) PSNR values in dB.
    """
    mse = (pred - target).pow(2).mean(dim=[1, 2, 3])
    return -10.0 * torch.log10(mse.clamp(min=1e-10))


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    """Generate a normalized 2D Gaussian window for SSIM computation."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :]).expand(1, 1, size, size).contiguous()


def masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    peak: float = 100.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """Compute masked structural similarity index (SSIM) per image.

    Args:
        pred: (B, C, H, W) predicted image tensor.
        target: (B, C, H, W) target image tensor.
        mask: (B, 1, H, W) pixel weights in [0, 1].
        peak: Maximum dynamic range of the signal (default: 100.0 for L*).
        window_size: Gaussian window size.
        sigma: Gaussian standard deviation.
        k1: First SSIM stability constant.
        k2: Second SSIM stability constant.

    Returns:
        (B,) SSIM values in [-1, 1].

    Raises:
        ValueError: If tensor shapes do not match or are smaller than window_size.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} must match")
    if min(pred.shape[-2:]) < window_size:
        raise ValueError(
            f"image {tuple(pred.shape[-2:])} is smaller than the {window_size}x{window_size} "
            "SSIM window"
        )
    pred, target = pred.float(), target.float()
    b, c = pred.shape[:2]
    win = _gaussian_window(window_size, sigma, pred.device, pred.dtype).expand(c, 1, -1, -1)
    pad = window_size // 2

    def blur(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(F.pad(x, (pad,) * 4, mode="reflect"), win, groups=c)

    mu_x, mu_y = blur(pred), blur(target)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_xx = blur(pred * pred) - mu_xx
    sigma_yy = blur(target * target) - mu_yy
    sigma_xy = blur(pred * target) - mu_xy

    c1, c2 = (k1 * peak) ** 2, (k2 * peak) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    )
    return masked_mean(ssim_map.mean(dim=1), mask)


def lpips_distance(
    pred_linear: torch.Tensor, target_linear: torch.Tensor, lpips_fn
) -> torch.Tensor:
    """Compute LPIPS perceptual distance between batches of linear RGB images.

    Args:
        pred_linear: (B, 3, H, W) predicted linear RGB in [0, 1].
        target_linear: (B, 3, H, W) target linear RGB in [0, 1].
        lpips_fn: Initialized LPIPS model.

    Returns:
        (B,) perceptual distance values.
    """
    from dawn.data.color import linear_to_srgb

    pred = linear_to_srgb(pred_linear.float().clamp(0.0, 1.0)) * 2.0 - 1.0
    target = linear_to_srgb(target_linear.float().clamp(0.0, 1.0)) * 2.0 - 1.0
    return lpips_fn(pred, target).flatten()


@torch.no_grad()
def evaluate_batch(
    lab_pred: torch.Tensor,
    lab_target: torch.Tensor,
    rgb_pred: torch.Tensor,
    rgb_target: torch.Tensor,
    mask: torch.Tensor,
    vehicle_mask: torch.Tensor | None = None,
    lpips_fn=None,
) -> dict[str, torch.Tensor]:
    """Compute per-image evaluation metrics for a batch.

    Args:
        lab_pred: (B, 3, H, W) predicted Lab image tensor.
        lab_target: (B, 3, H, W) target Lab image tensor.
        rgb_pred: (B, 3, H, W) predicted linear RGB image tensor in [0, 1].
        rgb_target: (B, 3, H, W) target linear RGB image tensor in [0, 1].
        mask: (B, 1, H, W) pixel evaluation mask.
        vehicle_mask: Optional (B, 1, H, W) vehicle region mask.
        lpips_fn: Optional LPIPS model instance.

    Returns:
        Dictionary mapping metric names to (B,) score tensors.

    Raises:
        ValueError: If tensor shapes do not match.
    """
    if lab_pred.shape != lab_target.shape or rgb_pred.shape != rgb_target.shape:
        raise ValueError("pred and target must have the same shape")
    if lab_pred.shape[-2:] != rgb_pred.shape[-2:]:
        raise ValueError(
            f"Lab {tuple(lab_pred.shape[-2:])} and RGB {tuple(rgb_pred.shape[-2:])} "
            "must be the same size: they are the same image in two spaces"
        )
    from dawn.data.color import linear_to_srgb

    lightness = lambda x: x[:, 0:1]  # noqa: E731
    out = {
        "psnr": masked_psnr(lightness(lab_pred), lightness(lab_target), mask, peak=100.0),
        "ssim": masked_ssim(lightness(lab_pred), lightness(lab_target), mask, peak=100.0),
        "ciede2000": masked_mean(ciede2000(lab_pred, lab_target), mask),
        "colorfulness": colorfulness(linear_to_srgb(rgb_pred.clamp(0, 1)) * 255.0),
        "colorfulness_target": colorfulness(linear_to_srgb(rgb_target.clamp(0, 1)) * 255.0),
    }
    if lpips_fn is not None:
        out["lpips"] = lpips_distance(rgb_pred, rgb_target, lpips_fn)
    if vehicle_mask is not None:
        region = vehicle_mask.to(mask.dtype) * mask
        scored = region.sum(dim=(1, 2, 3)) > 0

        def only_where_scored(values: torch.Tensor) -> torch.Tensor:
            return torch.where(scored, values, torch.full_like(values, float("nan")))

        out["psnr_vehicles"] = only_where_scored(
            masked_psnr(lightness(lab_pred), lightness(lab_target), region, peak=100.0)
        )
        out["ssim_vehicles"] = only_where_scored(
            masked_ssim(lightness(lab_pred), lightness(lab_target), region, peak=100.0)
        )
        out["ciede2000_vehicles"] = only_where_scored(
            masked_mean(ciede2000(lab_pred, lab_target), region)
        )
    return out


def masked_psnr(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, peak: float = 1.0
) -> torch.Tensor:
    """Compute masked peak signal-to-noise ratio per image.

    Args:
        pred: (B, C, H, W) predicted image tensor.
        target: (B, C, H, W) target image tensor.
        mask: (B, 1, H, W) pixel weights in [0, 1].
        peak: Maximum dynamic range of the signal.

    Returns:
        (B,) PSNR values in dB.
    """
    weight = mask.to(torch.float32)
    channels = pred.shape[1]
    err = ((pred.float() - target.float()).pow(2) * weight).sum(dim=(1, 2, 3))
    mse = err / (weight.sum(dim=(1, 2, 3)) * channels).clamp(min=torch.finfo(torch.float32).tiny)
    return 10.0 * torch.log10(peak**2 / mse.clamp(min=1e-10))


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude of a single-channel image using 3x3 Sobel filters.

    Args:
        image: (B, 1, H, W) image tensor.

    Returns:
        (B, 1, H, W) gradient magnitude tensor.

    Raises:
        ValueError: If input tensor is not (B, 1, H, W).
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError(f"sobel_magnitude takes (B, 1, H, W), got {tuple(image.shape)}")
    x = image.float()
    kernel = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=x.dtype, device=x.device
    )
    pair = torch.stack([kernel, kernel.T]).unsqueeze(1)  # (2, 1, 3, 3): d/dx, d/dy
    gradient = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), pair)
    return gradient.pow(2).sum(dim=1, keepdim=True).sqrt()


def edge_quantile_mask(
    image: torch.Tensor, mask: torch.Tensor, quantile: float = 0.9
) -> torch.Tensor:
    """Compute a binary edge mask selecting pixels in the top quantile of gradient magnitude.

    Args:
        image: (B, 1, H, W) single-channel image tensor.
        mask: (B, 1, H, W) valid pixel weights in [0, 1].
        quantile: Quantile threshold in [0, 1). Default is 0.9.

    Returns:
        (B, 1, H, W) binary mask tensor.

    Raises:
        ValueError: If quantile is outside [0, 1).
    """
    if not 0.0 <= quantile < 1.0:
        raise ValueError(f"quantile must be in [0, 1), got {quantile}")
    magnitude = sobel_magnitude(image)
    keep = mask.to(torch.float32) > 0
    out = torch.zeros_like(magnitude)
    for index in range(magnitude.shape[0]):
        values = magnitude[index][keep[index]]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values.flatten().float(), quantile)
        selected = (magnitude[index] >= threshold) & (magnitude[index] > 0) & keep[index]
        out[index] = selected.to(out.dtype)
    return out


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute weighted spatial mean per image.

    Args:
        value: (B, H, W) or (B, 1, H, W) per-pixel values.
        mask: (B, 1, H, W) pixel weights in [0, 1].

    Returns:
        (B,) mean values per image.
    """
    weight = mask.to(torch.float32).flatten(1)
    value = value.to(torch.float32).reshape(value.shape[0], -1)
    return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp(
        min=torch.finfo(torch.float32).tiny
    )


def ciede2000(
    lab_1: torch.Tensor, lab_2: torch.Tensor, k_l: float = 1.0, k_c: float = 1.0, k_h: float = 1.0
) -> torch.Tensor:
    """Compute CIEDE2000 color difference elementwise.

    Args:
        lab_1: (..., 3, H, W) first Lab image tensor.
        lab_2: (..., 3, H, W) second Lab image tensor.
        k_l: Lightness parametric factor.
        k_c: Chroma parametric factor.
        k_h: Hue parametric factor.

    Returns:
        (..., H, W) color difference map.
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
    """Compute Hasler and Suesstrunk (2003) colorfulness metric per image.

    Args:
        srgb: (B, 3, H, W) sRGB image tensor in [0, 255].

    Returns:
        (B,) colorfulness metric values.

    Raises:
        ValueError: If srgb is not (B, 3, H, W).
    """
    if srgb.ndim != 4 or srgb.shape[1] != 3:
        raise ValueError(f"colorfulness needs an RGB image (B, 3, H, W), got {tuple(srgb.shape)}")
    r, g, b = srgb.unbind(dim=1)
    rg, yb = (r - g).flatten(1), (0.5 * (r + g) - b).flatten(1)
    std = torch.sqrt(rg.std(dim=1, unbiased=False) ** 2 + yb.std(dim=1, unbiased=False) ** 2)
    mean = torch.sqrt(rg.mean(dim=1) ** 2 + yb.mean(dim=1) ** 2)
    return std + 0.3 * mean
