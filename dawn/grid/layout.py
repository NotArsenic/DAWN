"""Brightness-grid coefficient order.

Defined in docs/conventions.md (section 2); this module is its single implementation, so nothing
else re-derives which channel is the gain. The token packing lives in `dawn/flow/tokens.py`,
which imports these constants; grid code never imports from `flow/`.

    Grid:   (B, 2C, K, gy, gx)   coefficient, bin, cell row, cell column
            output channel c has its gain at channel 2c and its offset at 2c + 1, so the brightness
            grid (one output, C = 1) is (B, 2, K, gy, gx) with channel 0 = gain, 1 = offset.
"""

from __future__ import annotations

GAIN = 0
OFFSET = 1
NUM_COEFFS = 2  # coefficients per output channel: out = gain * x + offset


def coeff_channels(output: int) -> tuple[int, int]:
    """`(gain, offset)` channel indices of output channel `output` in a grid or coefficient map."""
    return NUM_COEFFS * output + GAIN, NUM_COEFFS * output + OFFSET
