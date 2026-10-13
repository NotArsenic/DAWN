"""Uncertainty calibration and rank correlation metrics."""

from __future__ import annotations

import numpy as np
import torch


def _flatten(
    spread: torch.Tensor, error: torch.Tensor, mask: torch.Tensor | None
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten spread and error maps, dropping non-finite entries and unmasked pixels."""
    if spread.shape != error.shape:
        raise ValueError(
            f"spread {tuple(spread.shape)} and error {tuple(error.shape)} must be the same map"
        )
    s = spread.detach().float().flatten().cpu().numpy()
    e = error.detach().float().flatten().cpu().numpy()
    keep = np.isfinite(s) & np.isfinite(e)
    if mask is not None:
        if mask.shape != spread.shape:
            raise ValueError(f"mask {tuple(mask.shape)} must match the maps {tuple(spread.shape)}")
        keep &= mask.detach().float().flatten().cpu().numpy() > 0.5
    return s[keep].astype(np.float64), e[keep].astype(np.float64)


def _subsample(a: np.ndarray, b: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, ...]:
    """Subsample paired arrays up to a maximum limit."""
    if limit <= 0 or a.size <= limit:
        return a, b
    idx = np.random.default_rng(seed).choice(a.size, size=limit, replace=False)
    return a[idx], b[idx]


def rank_correlation(
    spread: torch.Tensor,
    error: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_pixels: int = 100_000,
    seed: int = 0,
) -> float:
    """Compute Spearman rank correlation between uncertainty spread and realized error."""
    s, e = _flatten(spread, error, mask)
    s, e = _subsample(s, e, max_pixels, seed)
    if s.size < 2:
        return 0.0
    from scipy.stats import rankdata

    rs, re = rankdata(s), rankdata(e)
    if rs.std() == 0 or re.std() == 0:
        return 0.0
    return float(np.corrcoef(rs, re)[0, 1])


def fit_scale(spread: torch.Tensor, error: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """Fit a linear scaling factor relating uncertainty spread to absolute error."""
    s, e = _flatten(spread, error, mask)
    denom = float((s * s).sum())
    return float((s * e).sum() / denom) if denom > 0 else 0.0


def calibration_curve(
    spread: torch.Tensor,
    error: torch.Tensor,
    num_bins: int = 20,
    mask: torch.Tensor | None = None,
    max_pixels: int = 1_000_000,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    """Compute quantile calibration curve relating uncertainty spread to realized error."""
    s, e = _flatten(spread, error, mask)
    s, e = _subsample(s, e, max_pixels, seed)
    empty = np.array([], dtype=np.float64)
    if s.size < num_bins:
        return {
            "bin_spread": empty,
            "bin_errors": empty,
            "bin_counts": empty,
            "spearman": 0.0,
            "monotonicity": 0.0,
        }
    edges = np.quantile(s, np.linspace(0.0, 1.0, num_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, num_bins - 1)
    bin_spread, bin_errors, bin_counts = [], [], []
    for b in range(num_bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        bin_spread.append(float(s[sel].mean()))
        bin_errors.append(float(e[sel].mean()))
        bin_counts.append(n)
    errors = np.array(bin_errors)
    rises = float((np.diff(errors) > 0).mean()) if errors.size > 1 else 0.0
    return {
        "bin_spread": np.array(bin_spread),
        "bin_errors": errors,
        "bin_counts": np.array(bin_counts),
        "spearman": rank_correlation(spread, error, mask, max_pixels=max_pixels, seed=seed),
        "monotonicity": rises,
    }


def expected_calibration_error(
    spread: torch.Tensor,
    error: torch.Tensor,
    num_bins: int = 20,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    max_pixels: int = 1_000_000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute expected calibration error between scaled uncertainty spread and error."""
    a = fit_scale(spread, error, mask) if scale is None else float(scale)
    curve = calibration_curve(spread, error, num_bins, mask, max_pixels, seed)
    counts = curve["bin_counts"]
    if counts.size == 0 or counts.sum() == 0:
        return {"ece": 0.0, "scale": a}
    gap = np.abs(curve["bin_errors"] - a * curve["bin_spread"])
    return {"ece": float((counts * gap).sum() / counts.sum()), "scale": a}
