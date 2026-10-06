"""Image quality and object-level metrics.

Computes PSNR, SSIM, LPIPS on full images and on VisDrone object crops.
Also counts hallucinated / disappeared objects via YOLO.
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
