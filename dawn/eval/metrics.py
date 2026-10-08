"""Image quality metrics.

Current (flow phase, used by validation and `scripts/check_oracle.py`): `masked_psnr` (L* PSNR with
the peak at 100), `ciede2000` (the colour difference, in torch) and `colorfulness`
(Hasler & Suesstrunk). The rest of this module (SSIM, LPIPS, object-crop metrics) is the
inference/eval phase's and is still a stub.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MetricResults:
    """The plan's evaluation table, for one subset of pairs.

    Structure is scored on L* (peak 100), colour as CIEDE2000, vividness as colorfulness and
    perceptual likeness as LPIPS. Every pixel metric is masked; `lpips` is not (see
    `lpips_distance`). `*_vehicles` repeat the pixel metrics over the vehicle region only, which
    is the plan's "do vehicles survive?" row.
    """

    psnr: float = 0.0  # L* PSNR, peak 100, masked
    ssim: float = 0.0  # L* SSIM, masked
    ciede2000: float = 0.0  # mean colour difference, masked
    colorfulness: float = 0.0  # Hasler & Suesstrunk, of the prediction
    colorfulness_target: float = 0.0  # the same for the real RGB, to read it against
    lpips: float = 0.0  # whole-frame, unmasked
    psnr_vehicles: float = 0.0
    ssim_vehicles: float = 0.0
    ciede2000_vehicles: float = 0.0
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


def _gaussian_window(size: int, sigma: float, device, dtype) -> torch.Tensor:
    """`(1, 1, size, size)` normalized separable Gaussian, as SSIM is conventionally defined."""
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
    """SSIM over the pixels that count, per image.

    Matches `skimage.metrics.structural_similarity(gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False)` to 1e-8 **on the interior**. skimage discards the
    `window_size // 2` border; this keeps it (padded by reflection) and lets the mask decide, so
    an all-ones mask differs from skimage by about 6e-4 -- that difference is the border and
    nothing else. Mask the border out and the two agree exactly.


    The plan evaluates structure as PSNR **and SSIM on L\\***, so the default `peak` is 100, not 1.
    Pass the single L* channel, not RGB.

    The SSIM map is computed over the whole image and then averaged with `masked_mean`. The local
    statistics of a window that straddles the mask edge are therefore contaminated by pixels that
    do not count: with `valid` and `moving`/`vehicles` masks, which are large contiguous regions,
    only an 11-px rim is affected. Weighting the local moments by the mask instead would change
    the definition and stop it matching `skimage`, which is the worse trade for a reported number.

    Args:
        pred, target: `(B, C, H, W)`, the same units.
        mask: `(B, 1, H, W)` weights in `[0, 1]`.
        peak: the signal peak (100 for L*, 1 for `[0, 1]` images).
        window_size, sigma, k1, k2: the standard SSIM constants.

    Returns:
        `(B,)` SSIM in `[-1, 1]`. An image with no weight scores 0.

    Raises:
        ValueError: shapes disagree, or the image is smaller than the window.
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
        # 'reflect' matches skimage's default edge handling closely enough that the interior,
        # which is what the mask keeps, agrees to ~1e-6.
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
    """LPIPS between two batches of **linear** RGB, per image.

    LPIPS is an AlexNet/VGG feature distance trained on display-referred images, so the linear RGB
    of the pipeline is converted to sRGB and then to the `[-1, 1]` range the released nets expect
    (docs/conventions.md section 1, output boundary). Passing linear values straight in would
    score a different, darker pair of images than a person would see.

    There is no masked LPIPS: the metric is defined on whole images through a convolutional stack,
    and zeroing pixels would introduce edges the network reads as structure. Report it on the
    frame and read it next to the masked metrics, not instead of them.

    Args:
        pred_linear, target_linear: `(B, 3, H, W)` linear RGB in `[0, 1]`.
        lpips_fn: a constructed `lpips.LPIPS(net=...)`, already on the right device.

    Returns:
        `(B,)` distances; lower is more alike.
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
    """The plan's evaluation row for one batch, **per image** (not yet averaged).

    Per image rather than reduced, so the caller can weight subsets (box-aligned against fallback)
    and report percentiles rather than only a mean. `scripts/evaluate.py` does the aggregating.

    Everything is computed in the units the plan evaluates in: structure on `L*` with the peak at
    100, colour as CIEDE2000 over `L*a*b*`, colorfulness on 0-255 sRGB, LPIPS on sRGB `[-1, 1]`.

    Args:
        lab_pred, lab_target: `(B, 3, H, W)` L*, a*, b*.
        rgb_pred, rgb_target: `(B, 3, H, W)` **linear** RGB in `[0, 1]`, the same images.
        mask: `(B, 1, H, W)` the pixel mask: `valid & ~moving` for box-aligned pairs,
            `valid & ~vehicles` for fallback ones (docs/conventions.md section 5.3).
        vehicle_mask: `(B, 1, H, W)` the vehicle region, for the "do vehicles survive?" row.
            `None` leaves the `*_vehicles` keys out.
        lpips_fn: a constructed `lpips.LPIPS`, or `None` to skip the LPIPS key.

    Returns:
        `{metric: (B,)}`. Keys: `psnr`, `ssim`, `ciede2000`, `colorfulness`,
        `colorfulness_target`, plus `lpips` and the three `*_vehicles` when asked for.

    Raises:
        ValueError: the Lab and RGB batches disagree on shape.
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
        # The vehicle region, intersected with the pixel mask: an invalid pixel never counts,
        # whichever region it is in.
        region = vehicle_mask.to(mask.dtype) * mask
        out["psnr_vehicles"] = masked_psnr(
            lightness(lab_pred), lightness(lab_target), region, peak=100.0
        )
        out["ssim_vehicles"] = masked_ssim(
            lightness(lab_pred), lightness(lab_target), region, peak=100.0
        )
        out["ciede2000_vehicles"] = masked_mean(ciede2000(lab_pred, lab_target), region)
    return out


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


def sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    """Gradient magnitude of a one-channel image by a 3x3 Sobel pair, replicate-padded.

    `(B, 1, H, W)` in, `(B, 1, H, W)` out, in the input's own units per pixel. Used to pick the
    structural edges a region metric is restricted to (`edge_quantile_mask`); it is not a
    resampling step, so the area-averaging rule of docs/conventions.md section 4a does not
    apply. Replicate padding, so the frame border does not read as one long edge.
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
    """The structural edges of each image: pixels above the `quantile` of gradient magnitude.

    A whole-frame mean |delta a*b*| is dominated by the flat asphalt, fields and roofs that make
    up most of an aerial frame, so it cannot answer "does this upsampling bleed across sharp
    boundaries?". Restricting the same metric to the top decile of gradient magnitude can
    (docs/proposals_review.md section 3.B). The threshold is **per image** over the pixels
    `mask` keeps, because frames differ enormously in how much structure they contain.

    Args:
        image: `(B, 1, H, W)` the structure to read edges from -- L* / 100, not a*b*.
        mask: `(B, 1, H, W)` weights in [0, 1]; pixels with no weight never become edges and do
            not enter the threshold.
        quantile: Share of kept pixels treated as flat. 0.9 keeps the top decile.

    Returns:
        `(B, 1, H, W)` float 0/1. An image with no kept pixels, or no gradient at all, gets an
        all-zero mask, so a metric over it has no weight rather than a wrong value.
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
        # `> 0` as well as `>= threshold`: on an image that is flat over more than `quantile` of
        # itself the quantile IS zero, and `>= 0` would mark every pixel -- the opposite of an
        # edge mask. Flat frames then contribute no weight, which is the honest answer.
        selected = (magnitude[index] >= threshold) & (magnitude[index] > 0) & keep[index]
        out[index] = selected.to(out.dtype)
    return out


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
