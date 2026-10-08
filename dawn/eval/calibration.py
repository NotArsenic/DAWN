"""Is the uncertainty honest? Ranking the sample spread against the realized error.

The plan's evaluation row is a **rank correlation of each sample-spread map with its actual
error**, and the choice of a rank statistic is the whole point. The spread of K flow samples is a
*disagreement*, not a predictive standard deviation: nothing trains it to come out in L\\* units,
and nothing makes `E[|error|] == spread`. Asking "do the magnitudes match" (a raw ECE) would
therefore report a large number for a perfectly useful map.

The question that can be answered is the ordering one: **where the model disagrees with itself
more, is it in fact more wrong?** Spearman's rho answers exactly that and is invariant to any
monotone rescaling of the spread, so it does not care that the units are arbitrary.

Magnitude calibration is still reachable, but only after fitting the one scalar the spread is
missing: `fit_scale` gives the least-squares `a` in `|error| ~ a * spread`, and
`expected_calibration_error` applies it. Report the ECE *with* its scale, or it means nothing.

Units: pass `l_std` against the `L*` error, or `ab_std` against the a\\*b\\* error distance
(docs/conventions.md section 11). Mixing them compares a lightness spread to a colour error.
"""

from __future__ import annotations

import numpy as np
import torch


def _flatten(
    spread: torch.Tensor, error: torch.Tensor, mask: torch.Tensor | None
) -> tuple[np.ndarray, np.ndarray]:
    """Masked pixels of both maps, as 1-D float64 numpy, with non-finite entries dropped."""
    if spread.shape != error.shape:
        raise ValueError(
            f"spread {tuple(spread.shape)} and error {tuple(error.shape)} must be the same map"
        )
    s = spread.detach().float().flatten().cpu().numpy()
    e = error.detach().float().flatten().cpu().numpy()
    keep = np.isfinite(s) & np.isfinite(e)
    if mask is not None:
        if mask.shape != spread.shape:
            raise ValueError(
                f"mask {tuple(mask.shape)} must match the maps {tuple(spread.shape)}"
            )
        keep &= mask.detach().float().flatten().cpu().numpy() > 0.5
    return s[keep].astype(np.float64), e[keep].astype(np.float64)


def _subsample(a: np.ndarray, b: np.ndarray, limit: int, seed: int) -> tuple[np.ndarray, ...]:
    """At most `limit` pairs, drawn without replacement. Ranking a full frame is 327k points."""
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
    """Spearman's rho between a spread map and an error map, over the masked pixels.

    Invariant to any monotone rescaling of either map, which is what makes it the right
    statistic for a spread that has no calibrated units.

    Args:
        spread: `(..., H, W)` the sample spread (`l_std` or `ab_std`).
        error: `(..., H, W)` the realized error, in that map's own quantity.
        mask: `(..., H, W)` weights; pixels at or below 0.5 are dropped.
        max_pixels: subsample above this many pixels, for speed. 0 uses all of them.
        seed: the subsample draw, so a reported number is reproducible.

    Returns:
        rho in `[-1, 1]`. **0.0 when it is undefined** -- fewer than two pixels survive, or
        either map is constant (no ordering to agree with).
    """
    s, e = _flatten(spread, error, mask)
    s, e = _subsample(s, e, max_pixels, seed)
    if s.size < 2:
        return 0.0
    from scipy.stats import rankdata

    rs, re = rankdata(s), rankdata(e)
    if rs.std() == 0 or re.std() == 0:
        return 0.0
    return float(np.corrcoef(rs, re)[0, 1])


def fit_scale(
    spread: torch.Tensor, error: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    """Least-squares `a` in `|error| ~ a * spread`, the one number the spread is missing.

    Through the origin, because zero disagreement should predict zero error.

    Returns:
        `a >= 0`; 0.0 when the spread is all zeros (a deterministic sampler).
    """
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
    """Bin pixels by spread and report the mean realized error in each bin.

    Equal-frequency bins (quantiles), not equal-width: the spread is heavily skewed, so
    equal-width bins would put almost every pixel in the first one.

    A useful map makes `bin_errors` rise monotonically with `bin_spread`. The curve does not have
    to sit on `y = x`: that is a magnitude claim the spread does not make (see
    `expected_calibration_error`).

    Args:
        spread, error, mask: as `rank_correlation`.
        num_bins: quantile bins. Bins that collapse (tied spread values) are dropped.
        max_pixels, seed: subsampling, as above.

    Returns:
        `bin_spread`, `bin_errors`, `bin_counts` (each `(n_bins,)`), plus `spearman` and
        `monotonicity` -- the share of adjacent bin pairs where the error does rise, which is 1.0
        for a perfectly ordered curve and about 0.5 for noise.
    """
    s, e = _flatten(spread, error, mask)
    s, e = _subsample(s, e, max_pixels, seed)
    empty = np.array([], dtype=np.float64)
    if s.size < num_bins:
        return {
            "bin_spread": empty, "bin_errors": empty, "bin_counts": empty,
            "spearman": 0.0, "monotonicity": 0.0,
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
    """ECE between the **scaled** spread and the realized error: `sum_b (n_b/N) |e_b - a*s_b|`.

    The scale is not optional in spirit. An unscaled spread is in no particular units, so a raw
    ECE measures the arbitrary unit choice far more than it measures calibration. `scale=None`
    fits `a` on this same data (`fit_scale`), which is an optimistic in-sample number -- fit it
    on validation and pass it in to get an honest one.

    Returns:
        `ece` in the error's units, and the `scale` used, so a reported figure carries it.
    """
    a = fit_scale(spread, error, mask) if scale is None else float(scale)
    curve = calibration_curve(spread, error, num_bins, mask, max_pixels, seed)
    counts = curve["bin_counts"]
    if counts.size == 0 or counts.sum() == 0:
        return {"ece": 0.0, "scale": a}
    gap = np.abs(curve["bin_errors"] - a * curve["bin_spread"])
    return {"ece": float((counts * gap).sum() / counts.sum()), "scale": a}
