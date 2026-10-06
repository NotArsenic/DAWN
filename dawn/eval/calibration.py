"""Uncertainty calibration evaluation.

Checks whether predicted uncertainty correlates with actual error magnitude.
A well-calibrated model should have: E[|error|] proportional to predicted std.

Outputs calibration curves and Expected Calibration Error (ECE).
"""

from __future__ import annotations

import numpy as np
import torch


def calibration_curve(
    pred_std: torch.Tensor,
    actual_error: torch.Tensor,
    num_bins: int = 20,
) -> dict[str, np.ndarray]:
    """Compute calibration curve: bin predicted std vs mean |error|.

    Args:
        pred_std: (N,) predicted standard deviation per pixel.
        actual_error: (N,) absolute error |pred - gt| per pixel.
        num_bins: Number of equal-frequency bins.

    Returns:
        dict with:
            'bin_centers': (num_bins,) mean predicted std per bin.
            'bin_errors': (num_bins,) mean |error| per bin.
            'bin_counts': (num_bins,) number of pixels per bin.
            'ece': scalar Expected Calibration Error.
    """
    raise NotImplementedError


def expected_calibration_error(
    pred_std: torch.Tensor,
    actual_error: torch.Tensor,
    num_bins: int = 20,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_b (n_b / N) * |mean_error_b - mean_std_b|

    Lower is better. Perfect calibration = 0.

    Args:
        pred_std: (N,) predicted standard deviation.
        actual_error: (N,) absolute error.
        num_bins: Number of bins.

    Returns:
        ECE as a float.
    """
    raise NotImplementedError
