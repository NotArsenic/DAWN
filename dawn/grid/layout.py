"""Brightness grid coefficient layout and channel indices.

Grid layout: `(B, 2C, K, gy, gx)` for coefficient, bin, cell row, and cell column.
Output channel `c` has gain at channel `2c` and offset at `2c + 1`.
"""

from __future__ import annotations

GAIN = 0
OFFSET = 1
NUM_COEFFS = 2  # coefficients per output channel: out = gain * x + offset


def coeff_channels(output: int) -> tuple[int, int]:
    """Return (gain, offset) channel indices for an output channel."""
    return NUM_COEFFS * output + GAIN, NUM_COEFFS * output + OFFSET
