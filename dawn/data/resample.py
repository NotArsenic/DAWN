"""Area-averaging downsampling: the only way anything in dawn is downsampled.

docs/conventions.md section 4a. Every output pixel is the mean of the input over its footprint,
with input pixels that straddle a footprint edge weighted by their overlap. For an integer factor
this is the plain block mean. It is also what a sensor with bigger pixels would have recorded,
which is why the camera model can treat a downsampled pixel as n binned native pixels.

Downsample LINEAR images (docs/conventions.md section 1). Averaging sRGB values makes bright
lights come out too dim.
"""

from __future__ import annotations

import torch


def fit_long_side(height: int, width: int, long_side: int) -> tuple[int, int]:
    """Output size `(height, width)` with the long side at `long_side` and the aspect preserved.

    The short side is rounded to the nearest pixel (at least 1). Never upsamples: an image whose
    long side is already at most `long_side` keeps its size.
    """
    long = max(height, width)
    if long <= long_side:
        return height, width
    scale = long_side / long
    if height >= width:
        return long_side, max(1, round(width * scale))
    return max(1, round(height * scale)), long_side


def area_downsample(img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Downsample the last two dimensions to `size = (height, width)` by area averaging.

    Exact for any factor, integer or not. Preserves the mean, runs on any device and is
    differentiable. Output dtype and device match the input.

    Args:
        img: `(..., H, W)` floating-point tensor, e.g. `(3, H, W)` or `(B, 3, H, W)`, linear.
        size: Target `(height, width)`, each at most the input size.

    Raises:
        TypeError: `img` is not floating point (decode uint8 files with `srgb8_to_linear` first).
        ValueError: `size` is larger than the input along either axis, or not positive.
    """
    if not img.is_floating_point():
        raise TypeError(f"area_downsample needs a floating-point image, got {img.dtype}")
    (h_in, w_in), (h_out, w_out) = img.shape[-2:], size
    if not (0 < h_out <= h_in and 0 < w_out <= w_in):
        raise ValueError(f"cannot area-downsample {(h_in, w_in)} to {(h_out, w_out)}")
    rows = _area_matrix(h_in, h_out).to(device=img.device, dtype=img.dtype)
    cols = _area_matrix(w_in, w_out).to(device=img.device, dtype=img.dtype)
    return rows @ img @ cols.T


def _area_matrix(n_in: int, n_out: int) -> torch.Tensor:
    """`(n_out, n_in)` float64 weights; row i averages the input over `[i*s, (i+1)*s)`.

    `s = n_in / n_out`. Entry `(i, j)` is the overlap of input pixel `[j, j+1)` with that
    footprint, divided by `s`, so every row sums to 1. Built in float64 on the CPU (MPS has no
    float64), then cast.
    """
    s = n_in / n_out
    start = torch.arange(n_out, dtype=torch.float64).unsqueeze(1) * s
    j = torch.arange(n_in, dtype=torch.float64).unsqueeze(0)
    overlap = torch.minimum(start + s, j + 1) - torch.maximum(start, j)
    return overlap.clamp(min=0) / s
