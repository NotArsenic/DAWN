"""Thermal guidance map computation for brightness grid slicing."""

from __future__ import annotations

import torch
import torch.nn.functional as F

GUIDANCE_BLUR_SIZE = 5


def compute_guidance(
    thermal: torch.Tensor,
    blur_kernel_size: int = GUIDANCE_BLUR_SIZE,
) -> torch.Tensor:
    """Compute guidance map from thermal image using replicate-padded box blur.

    Args:
        thermal: (B, 1, H, W) thermal image tensor in [0, 1].
        blur_kernel_size: Odd box filter kernel size.

    Returns:
        (B, 1, H, W) guidance map clamped to [0, 1].

    Raises:
        ValueError: If thermal is not (B, 1, H, W).
    """
    if thermal.ndim != 4 or thermal.shape[1] != 1:
        raise ValueError(f"thermal must be (B, 1, H, W), got {tuple(thermal.shape)}")
    if blur_kernel_size > 1:
        pad = blur_kernel_size // 2
        thermal = F.pad(thermal, (pad, pad, pad, pad), mode="replicate")
        thermal = F.avg_pool2d(thermal, kernel_size=blur_kernel_size, stride=1)
    return thermal.clamp(0.0, 1.0)
