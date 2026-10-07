"""Thermal guidance for brightness-grid slicing.

The guidance map says which thermal-intensity bin each pixel falls into: the thermal itself,
smoothed. No log (thermal is not photon-starved like a dark photo) and no per-frame stretch (it
measured +0.06 dB because thermal already spans most of [0, 1]).

This is the ONLY guidance implementation: fitting, slicing, rendering, inference and the scripts
all call `compute_guidance` (docs/conventions.md sections 4 and 8).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

GUIDANCE_BLUR_SIZE = 5


def compute_guidance(
    thermal: torch.Tensor,
    blur_kernel_size: int = GUIDANCE_BLUR_SIZE,
) -> torch.Tensor:
    """Guidance from a thermal image: a replicate-padded box blur, clamped to [0, 1].

    Replicate padding matters: zero padding would darken the borders and put the border pixels
    in the wrong bin.

    Args:
        thermal: (B, 1, H, W) thermal, `code / 255`, at the resolution it will be indexed at.
        blur_kernel_size: Odd box filter size. Leave at the default outside of experiments: the
            convention is a 5x5 box.

    Returns:
        guidance: (B, 1, H, W) in [0, 1].

    Raises:
        ValueError: `thermal` is not (B, 1, H, W).
    """
    if thermal.ndim != 4 or thermal.shape[1] != 1:
        raise ValueError(f"thermal must be (B, 1, H, W), got {tuple(thermal.shape)}")
    if blur_kernel_size > 1:
        pad = blur_kernel_size // 2
        thermal = F.pad(thermal, (pad, pad, pad, pad), mode="replicate")
        thermal = F.avg_pool2d(thermal, kernel_size=blur_kernel_size, stride=1)
    return thermal.clamp(0.0, 1.0)
