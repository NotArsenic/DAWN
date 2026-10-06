"""Measure a real camera's noise and in-camera denoising from night frames.

`camera.py` simulates a sensor; this module measures the one it is imitating, so the two agree
on numbers rather than on guesses. Everything here reads real JPEGs and writes
`configs/camera_calib.yaml`. The CLI is `scripts/calibrate_noise.py`.

Two measurements, from the same patches:

1. **Variance against mean, in the flattest patches only** -> the effective `(a, b)` of
   `Var[y | x] = a·x + b` (docs/conventions.md section 6). A patch of sky or road carries noise
   and nothing else, so its residual variance after detrending *is* the noise variance at its
   mean level. A patch of rooftops carries texture that no model should be asked to explain, so
   it is dropped. Regressing the surviving variances on their means separates the part that
   grows with signal (`a`, shot noise) from the part that does not (`b`, read noise).

2. **Spatial autocorrelation of the residual** -> how hard the camera denoised before writing
   the file. Sensor noise is white: neighbouring photosites are independent. Any correlation in
   the delivered image was put there by the ISP's denoise blur, partly undone by its sharpening.
   Fitting `camera.isp_kernel` to the measured correlation recovers both strengths at once.

The two are coupled, which is why they share a pass. The autocorrelation fit has to come
first, because the fitted kernel is what says how much the detrend distorted the variance
(`projected_variance_bias`) and what converts an effective `(a, b)` back into the sensor
constants `camera.py` samples from.

Measuring something by picking the quietest places to look at it is a trap, and most of the
care here goes into not falling into it:

- Selecting patches by their own residual variance keeps the luckiest noise draws and reads
  low. Patches are ranked instead by the *other two channels'* structure, which tracks the
  scene and not this channel's noise (`_structure_score`).
- Ranking within a brightness bin says nothing when the whole bin is texture, so there is also
  an absolute gate, read off each frame (`PatchAccumulator._gate`).
- A bin can only read too high, never too low, so the variance line is fit to the lower
  envelope rather than to least squares (`fit_variance_line`).
- Detrending each patch distorts both measurements, by a few percent for the variance and much
  more for the autocorrelation. Rather than correct for it, the model is pushed through the
  same detrend before being compared (`projected_autocorrelation`).

On synthetic frames with a known sensor and a known ISP, across flat, textured, bright and
near-black scenes, that recovers the effective `a` to within about 8%, `denoise_sigma` to
within about 4%, and the full `(a, b)` to within about 20% once the scene is dark enough for
the read-noise term to be identifiable at all. `tests/test_calibration.py` pins this down.

Honest limits of the method, all recorded in the YAML it writes:

- The frames are JPEGs. The codec quantizes 8x8 DCT blocks, which suppresses high-frequency
  noise and leaves its own correlation footprint. Part of the fitted `denoise_sigma` is really
  the codec, and the fit says so by being a *post-JPEG* equivalent filter, which is the right
  thing for a simulator whose last step is also a JPEG.
- Demosaicing mixes the channels, so the cross-channel ranking is not perfectly independent of
  the noise it selects on, and some of the "quietest places" bias survives.
- Without EXIF there is no way to know what digital gain the camera applied, so the split
  between "dim sensor" and "amplified sensor" is unobservable. The sensor constants are
  reported at a stated `reference_gain` (default 1.0) and scale exactly with it.
- `b` is the intercept of a line. In frames with no near-black region it is not measured at
  all, whatever number comes out; `read_noise_share` reports how much of the fit it carries.
- A flat patch at a given brightness has to exist. `min_bin_patches` refuses to report a bin
  that never got one, rather than fitting a line through texture.
"""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from dawn.data.camera import CameraParams, isp_kernel
from dawn.data.color import linear_to_srgb8, luminance, srgb8_to_linear
from dawn.data.visdrone import read_srgb8

logger = logging.getLogger(__name__)

CHANNELS = 3
PATCH = 16  # px; 253 degrees of freedom after detrending, and a whole number of JPEG blocks
MAX_LAG = 3  # px; the ISP kernel is 9x9, so lag 3 is where its correlation has died out
SURFACE_TERMS = 6  # the detrend removes a constant, two slopes and three curvatures
_EPS = 1e-12


# --- patch statistics ---------------------------------------------------------------------------


def _surface_projector(patch: int, device: torch.device) -> torch.Tensor:
    """`(p*p, p*p)` matrix that removes the best-fit quadratic surface from a flattened patch.

    Vignetting, a sunlit wall and a road at a grazing angle are all smooth over sixteen pixels.
    Subtracting a trend rather than just the mean keeps those patches in the sample instead of
    charging their shading to the read noise.

    Quadratic rather than planar, because the frames that matter most here are the dark ones,
    where the noise is faint and a background's gentle curve is not negligible beside it. Six
    terms out of 256 is a cheap way to stop that curve being counted as noise; the exact cost,
    and the cost to everything measured downstream, is accounted for by `_patch_lag_tables`.
    """
    coords = torch.linspace(-1.0, 1.0, patch, device=device)
    ys, xs = torch.meshgrid(coords, coords, indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    basis = torch.stack([torch.ones_like(ys), ys, xs, ys * ys, ys * xs, xs * xs], dim=1)
    pinv = torch.linalg.pinv(basis)
    return torch.eye(patch * patch, device=device) - basis @ pinv


_STRUCTURE_BLUR = 5  # px; wide enough that near-white noise barely survives it
_STRUCTURE_WHITE = 1.0 / _STRUCTURE_BLUR**2  # a box filter's energy: what it leaves of white noise


def _structure_area(patch: int) -> int:
    """Pixels in the valid region of `_structure_score` for a patch of this size."""
    return max(1, patch - _STRUCTURE_BLUR + 1) ** 2


def _structure_score(resid: torch.Tensor) -> torch.Tensor:
    """How much *structure* a detrended patch still holds, with its noise filtered out.

    Ranking patches by their plain residual variance is the obvious way to find flat ones and
    the wrong one: that statistic IS the quantity being measured, so keeping its smallest tenth
    keeps the tenth of the noise draws that came out quietest, and every number downstream
    reads low. Worse, it also biases the measured autocorrelation, because a patch wins the
    ranking partly by having little low-frequency noise in it.

    Blurring the residual first concentrates the score on structure: texture lives at scales of
    several pixels and survives a 5x5 mean, while sensor noise is close to white at that scale
    and is suppressed by roughly the filter's energy, 1/25.

    That alone is not enough, and on its own it is worse than the naive ranking: squeezing the
    score into the low-frequency band means selecting on it strips the low-frequency noise in
    particular, which is where the positive part of the autocorrelation lives. So the caller
    ranks each channel by the OTHER two channels' scores. Texture is a property of the scene
    and appears in all three channels at once; the noise realization being measured in this
    channel does not appear in the other two. Selection then carries no information about the
    quantity being measured, and the only correction left to make is for the detrend.

    Args:
        resid: `(N, 1, p, p)` detrended patches.

    Returns:
        `(N,)` non-negative scores; lower is flatter.
    """
    kernel = torch.full(
        (1, 1, _STRUCTURE_BLUR, _STRUCTURE_BLUR), 1.0 / (_STRUCTURE_BLUR**2), device=resid.device
    )
    smoothed = torch.nn.functional.conv2d(resid, kernel)  # valid: no padded edge artifacts
    return smoothed.pow(2).sum(dim=(1, 2, 3))


@dataclass
class PatchAccumulator:
    """Running totals over every image, so memory does not grow with the dataset.

    Patch means are binned on a fixed grid in sqrt(mean); only the flattest `flat_quantile` of
    each image's patches in each bin survive into the totals. Autocorrelation sums come from
    the same survivors.
    """

    num_bins: int = 40
    patch: int = PATCH
    flat_quantile: float = 0.15
    gate_quantile: float = 0.25
    gate_scale: float = 1.3
    max_mean: float = 0.90  # above this a patch may be clipping, and clipping is not noise
    min_code: int = 3  # 8-bit sRGB code; patches with any pixel at or below this are crushed

    count: torch.Tensor = field(init=False)
    sum_mean: torch.Tensor = field(init=False)
    sum_var: torch.Tensor = field(init=False)
    corr_num: torch.Tensor = field(init=False)
    corr_den: torch.Tensor = field(init=False)
    num_images: int = 0
    num_patches_seen: int = 0

    def __post_init__(self) -> None:
        shape = (CHANNELS, self.num_bins)
        self.count = torch.zeros(shape, dtype=torch.float64)
        self.sum_mean = torch.zeros(shape, dtype=torch.float64)
        self.sum_var = torch.zeros(shape, dtype=torch.float64)
        self.corr_num = torch.zeros(MAX_LAG + 1, MAX_LAG + 1, dtype=torch.float64)
        self.corr_den = torch.zeros(MAX_LAG + 1, MAX_LAG + 1, dtype=torch.float64)

    @property
    def dof(self) -> int:
        return self.patch * self.patch - SURFACE_TERMS

    def add_image(self, image: torch.Tensor) -> None:
        """Fold one `(3, H, W)` linear image into the totals. Native resolution: never resize."""
        if image.ndim != 3 or image.shape[0] != CHANNELS:
            raise ValueError(f"image must be (3, H, W), got {tuple(image.shape)}")
        patch = self.patch
        height, width = image.shape[1] // patch * patch, image.shape[2] // patch * patch
        if height < patch or width < patch:
            return
        tiles = image[:, :height, :width].unfold(1, patch, patch).unfold(2, patch, patch)
        tiles = tiles.reshape(CHANNELS, -1, patch * patch).to(torch.float32)

        projector = _surface_projector(patch, image.device)
        resid = tiles @ projector  # (3, N, p*p); the trend is gone, the noise is not
        means = tiles.mean(dim=2)
        variances = resid.pow(2).sum(dim=2) / self.dof
        structure = _structure_score(resid.view(-1, 1, patch, patch)).view(CHANNELS, -1)
        # Rank each channel's patches by the OTHER channels' structure. Texture is a property
        # of the scene and shows in all three; the noise draw being measured is not, so this
        # picks flat patches without picking quiet ones. See `_structure_score`.
        flatness = structure.sum(dim=0, keepdim=True) - structure
        others = variances.sum(dim=0, keepdim=True) - variances
        ratio = flatness / (others * _structure_area(patch) * _STRUCTURE_WHITE).clamp(min=_EPS)

        # A patch that touches either rail is not measuring noise: clipping squashes the
        # variance toward zero and would read as a flat patch with impossibly clean pixels.
        # Pixels near code 0 are crushed: pins at code 0 understate the variance.
        # Exclude every patch where any pixel (any channel) is at or below min_code,
        # or at or above clipping (max_mean).
        if self.min_code >= 0:
            min_linear = float(
                srgb8_to_linear(torch.tensor([self.min_code], dtype=torch.uint8)).item()
            )
        else:
            min_linear = -1.0
        uncrushed = (tiles.amin(dim=(0, 2)) > min_linear + 1e-6).unsqueeze(0).expand(CHANNELS, -1)
        usable = (tiles.amax(dim=2) < self.max_mean) & uncrushed
        usable = usable & (ratio <= self._gate(ratio, usable))
        self.num_images += 1
        self.num_patches_seen += int(usable.sum())

        # Bin on sqrt(mean), not mean: read noise is only visible where the signal is
        # faint, and night frames pile up there. Equal-width bins would put the whole
        # dark end, the part that pins `b`, into one or two of them.
        bins = means.clamp(min=0.0).sqrt() * self.num_bins
        bins = bins.floor().clamp(0, self.num_bins - 1).to(torch.int64)
        for channel in range(CHANNELS):
            for index in bins[channel][usable[channel]].unique().tolist():
                in_bin = usable[channel] & (bins[channel] == index)
                keep = self._flattest(flatness[channel], in_bin)
                if keep is None:
                    continue
                self.count[channel, index] += keep.numel()
                self.sum_mean[channel, index] += means[channel][keep].sum(dtype=torch.float64)
                self.sum_var[channel, index] += variances[channel][keep].sum(dtype=torch.float64)
                self._add_correlation(resid[channel][keep].view(-1, patch, patch))

    def _gate(self, ratio: torch.Tensor, usable: torch.Tensor) -> torch.Tensor:
        """Per channel, the largest structure-to-noise ratio a patch may have and still count.

        Keeping the flattest tenth *of each brightness bin* is a relative test, and a relative
        test has no answer when a whole bin is texture: the flattest tenth of a bin that holds
        nothing but rooftops is still rooftops, and it lands in the autocorrelation as a long
        positive tail that reads like heavy denoising. So there is also an absolute test.

        `ratio` is a patch's structure score over the structure score that its own noise would
        produce if that noise were white, which makes it dimensionless and roughly constant
        across brightness: near 1 for white noise, a few for noise the ISP has smeared, tens or
        hundreds for anything with real detail in it. The threshold is read off the image
        itself, as `gate_scale` times the ratio's own lower quartile, so it needs no prior
        knowledge of the camera and adapts to each frame. It assumes a quarter of a night
        aerial frame is featureless, which is a safe thing to assume about roads and fields.
        """
        flat = torch.where(usable, ratio, torch.full_like(ratio, float("inf")))
        quantile = torch.full((CHANNELS, 1), float("inf"))
        for channel in range(CHANNELS):
            finite = flat[channel][torch.isfinite(flat[channel])]
            if finite.numel() >= 8:
                quantile[channel, 0] = torch.quantile(finite, self.gate_quantile)
        return quantile * self.gate_scale

    def _flattest(self, score: torch.Tensor, in_bin: torch.Tensor) -> torch.Tensor | None:
        """Indices of the flattest `flat_quantile` of the patches in one brightness bin."""
        candidates = in_bin.nonzero(as_tuple=True)[0]
        if candidates.numel() == 0:
            return None
        take = max(1, int(round(self.flat_quantile * candidates.numel())))
        order = torch.argsort(score[candidates])
        return candidates[order[:take]]

    def _add_correlation(self, resid: torch.Tensor) -> None:
        """Accumulate `sum(r * r_shifted)` and the matching norms for every lag up to MAX_LAG."""
        for dy in range(MAX_LAG + 1):
            for dx in range(MAX_LAG + 1):
                left = resid[:, : resid.shape[1] - dy, : resid.shape[2] - dx]
                right = resid[:, dy:, dx:]
                self.corr_num[dy, dx] += (left * right).sum(dtype=torch.float64)
                self.corr_den[dy, dx] += torch.sqrt(
                    left.pow(2).sum(dtype=torch.float64) * right.pow(2).sum(dtype=torch.float64)
                )

    def merge(self, other: PatchAccumulator) -> PatchAccumulator:
        """Fold another accumulator's totals into this one. Every total is a plain sum, so
        splitting the images across processes and merging gives the same answer as one pass."""
        if (other.num_bins, other.patch, other.min_code) != (
            self.num_bins,
            self.patch,
            self.min_code,
        ):
            raise ValueError(
                "cannot merge accumulators with different binning, patch size, or min_code"
            )
        self.count += other.count
        self.sum_mean += other.sum_mean
        self.sum_var += other.sum_var
        self.corr_num += other.corr_num
        self.corr_den += other.corr_den
        self.num_images += other.num_images
        self.num_patches_seen += other.num_patches_seen
        return self

    def autocorrelation(self) -> torch.Tensor:
        """`(MAX_LAG+1, MAX_LAG+1)` normalized autocorrelation; `[0, 0]` is 1 by construction."""
        return (self.corr_num / self.corr_den.clamp(min=_EPS)).to(torch.float32)

    def bins(self, min_bin_patches: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per channel and bin: mean signal, mean flat-patch variance, patch count.

        Bins holding fewer than `min_bin_patches` survivors are returned with a zero count and
        must be dropped by the caller: one flat patch in a brightness range is an anecdote.
        """
        count = self.count.clone()
        count[count < min_bin_patches] = 0.0
        safe = count.clamp(min=1.0)
        return (
            (self.sum_mean / safe).to(torch.float32),
            (self.sum_var / safe).to(torch.float32),
            count.to(torch.float32),
        )


# --- modelling what the measurement does to itself ---------------------------------------------


@functools.lru_cache(maxsize=4)
def _patch_lag_tables(patch: int) -> dict[str, torch.Tensor]:
    """Everything needed to predict what `PatchAccumulator` measures, for any noise covariance.

    The accumulator does not see the noise: it sees the noise *after* a smooth trend has been
    subtracted from each patch. Projection throws away the constant and the two ramps, which
    are exactly the modes that carry long-range positive correlation, so a measured
    autocorrelation always reads lower than the truth and a measured variance reads low by a
    factor that depends on how correlated the noise is. Comparing a model against the measured
    numbers therefore means putting the model through the same projection.

    Every statistic the accumulator forms is linear in the patch covariance `C`, and `C` is
    stationary, so each one collapses to a dot product against the kernel's own lag table:
    `sum_ij W_ij (QCQ)_ij = sum_ij (QWQ)_ij C_ij = sum_lag T[lag] c[lag]`. The `T` tables depend
    only on the patch size, so they are built once and cached.

    Returns:
        `{"var": (2p-1, 2p-1), "num": (L, L, 2p-1, 2p-1), "left": ..., "right": ...}` where
        `L = MAX_LAG + 1`. Contract each against a lag-covariance table from
        `_kernel_lag_covariance` to get the statistic the accumulator would report.
    """
    size = patch * patch
    projector = _surface_projector(patch, torch.device("cpu")).double()
    rows, cols = torch.meshgrid(torch.arange(patch), torch.arange(patch), indexing="ij")
    rows, cols = rows.reshape(-1), cols.reshape(-1)
    span = 2 * patch - 1
    lag = (
        (rows.view(-1, 1) - rows.view(1, -1) + patch - 1) * span
        + (cols.view(-1, 1) - cols.view(1, -1) + patch - 1)
    ).reshape(-1)

    def fold(weights: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(span * span, dtype=torch.float64)
        out.index_add_(0, lag, weights.reshape(-1))
        return out.view(span, span)

    def pair(left_idx: torch.Tensor, right_idx: torch.Tensor) -> torch.Tensor:
        # Q W Q for the selector W with a 1 at every (left_idx[m], right_idx[m]).
        return fold(projector[:, left_idx] @ projector[right_idx, :])

    lags = MAX_LAG + 1
    tables = {
        "var": fold(projector),
        "num": torch.zeros(lags, lags, span, span, dtype=torch.float64),
        "left": torch.zeros(lags, lags, span, span, dtype=torch.float64),
        "right": torch.zeros(lags, lags, span, span, dtype=torch.float64),
    }
    flat = torch.arange(size)
    for dy in range(lags):
        for dx in range(lags):
            keep = (rows < patch - dy) & (cols < patch - dx)
            left_idx = flat[keep]
            right_idx = left_idx + dy * patch + dx
            tables["num"][dy, dx] = pair(left_idx, right_idx)
            tables["left"][dy, dx] = pair(left_idx, left_idx)
            tables["right"][dy, dx] = pair(right_idx, right_idx)
    return tables


def _kernel_lag_covariance(kernel: torch.Tensor, patch: int) -> torch.Tensor:
    """`(2p-1, 2p-1)` covariance of `kernel`-filtered unit white noise, indexed by pixel lag.

    Entry `[dy + p - 1, dx + p - 1]` is `sum_q k[q]·k[q + (dy, dx)]`, zero beyond the kernel's
    own reach.
    """
    size = kernel.shape[0]
    full = torch.nn.functional.conv2d(
        kernel.view(1, 1, size, size).double(),
        kernel.flip(0, 1).view(1, 1, size, size).double(),
        padding=size - 1,
    )[0, 0]
    span = 2 * patch - 1
    out = torch.zeros(span, span, dtype=torch.float64)
    reach = min(size - 1, patch - 1)
    center = patch - 1
    out[center - reach : center + reach + 1, center - reach : center + reach + 1] = full[
        size - 1 - reach : size + reach, size - 1 - reach : size + reach
    ]
    return out


def projected_autocorrelation(kernel: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    """What `PatchAccumulator.autocorrelation` reports for noise shaped by `kernel`.

    This is the quantity `fit_isp_strength` matches: the ideal correlation of the filtered
    noise, seen through the per-patch detrend, so model and measurement are the same statistic.
    """
    tables = _patch_lag_tables(patch)
    cov = _kernel_lag_covariance(kernel, patch)
    numerator = (tables["num"] * cov).sum(dim=(2, 3))
    left = (tables["left"] * cov).sum(dim=(2, 3)).clamp(min=_EPS)
    right = (tables["right"] * cov).sum(dim=(2, 3)).clamp(min=_EPS)
    return (numerator / (left * right).sqrt()).to(torch.float32)


def projected_variance_bias(kernel: torch.Tensor, patch: int = PATCH) -> float:
    """Measured patch variance divided by the true per-pixel variance, for `kernel`-shaped noise.

    The detrend removes six modes out of `p*p`, and `PatchAccumulator` already divides by
    `p*p - 6` to allow for that. The division is exactly right for white noise and slightly
    wrong once the noise is correlated, because the removed modes then hold more than their
    share of the variance. This is the leftover factor, a few percent, applied in `calibrate`.
    """
    tables = _patch_lag_tables(patch)
    cov = _kernel_lag_covariance(kernel, patch)
    dof = patch * patch - SURFACE_TERMS
    return float((tables["var"] * cov).sum() / (dof * cov[patch - 1, patch - 1].clamp(min=_EPS)))


def projected_dof(kernel: torch.Tensor, patch: int = PATCH) -> float:
    """Satterthwaite degrees of freedom of a patch variance under `kernel`-correlated noise.

    `trace(QCQ)^2 / trace((QCQ)^2)`: correlated pixels repeat each other, so `p*p` of them
    carry the information of fewer, and the flattest-patch selection in `_truncation_factor`
    bites harder than a white-noise count would suggest.
    """
    projector = _surface_projector(patch, torch.device("cpu")).double()
    span = 2 * patch - 1
    cov_lag = _kernel_lag_covariance(kernel, patch)
    rows, cols = torch.meshgrid(torch.arange(patch), torch.arange(patch), indexing="ij")
    rows, cols = rows.reshape(-1), cols.reshape(-1)
    lag = (rows.view(-1, 1) - rows.view(1, -1) + patch - 1) * span + (
        cols.view(-1, 1) - cols.view(1, -1) + patch - 1
    )
    cov = cov_lag.reshape(-1)[lag]
    projected = projector @ cov @ projector
    trace = float(projected.diagonal().sum())
    return trace * trace / max(float((projected * projected).sum()), _EPS)


# --- the two fits ---------------------------------------------------------------------------------


def fit_variance_line(
    means: torch.Tensor,
    variances: torch.Tensor,
    counts: torch.Tensor,
    trim_ratio: float = 1.6,
    max_iterations: int = 20,
    min_bins: int = 4,
) -> tuple[float, float, float, torch.Tensor]:
    """Weighted non-negative least squares of `Var = a·mean + b`, trimmed to the lower envelope.

    Two things make a plain regression the wrong tool here.

    `a` and `b` are both physically non-negative, so a fit that wants either below zero has run
    out of signal rather than found something. The two-parameter problem is small enough to
    solve exactly: try the unconstrained solution, then each boundary, keep the best feasible.

    And the contamination is one-sided. A bin can only read *too high*, never too low: leftover
    texture adds variance, nothing subtracts it. Whole brightness bins can be texture-only,
    with no flat patch anywhere in the dataset at that level, and least squares through them is
    dragged up by exactly the structure this is meant to exclude. So the line is found in two
    stages, which is the usual shape of a robust fit:

    1. **Descend.** Repeatedly keep only the bins at or below the current line and refit. Each
       pass moves the line down through the cloud; it settles on the lower envelope, where the
       bins that hold real flat patches live. This overshoots, because the last few bins
       standing are the luckiest ones.
    2. **Re-include.** Then grow the set back out to every bin within `trim_ratio` of the line
       and refit, to a fixed point. Noise bins scatter a little either side of the envelope and
       come back; texture bins sit an order of magnitude above it and stay out.

    Both loops are monotone in the kept set, so both terminate.

    Args:
        means: `(n,)` mean signal per bin, linear `[0, 1]`.
        variances: `(n,)` measured flat-patch variance per bin.
        counts: `(n,)` patches behind each bin; zero means "no data", and is the weight otherwise.
        trim_ratio: A bin is dropped when its variance exceeds this multiple of the fitted line.
        max_iterations: Cap on the trim-and-refit loop.
        min_bins: Stop trimming rather than fall below this many bins.

    Returns:
        `(a, b, r_squared, kept)`, `kept` a bool mask over the input bins.

    Raises:
        ValueError: Fewer than two bins have data, or they span no brightness range.
    """
    keep = counts > 0
    if int(keep.sum()) < 2:
        raise ValueError(f"need at least 2 populated brightness bins, got {int(keep.sum())}")

    def solve(mask: torch.Tensor) -> tuple[float, float, float]:
        x, y, w = means[mask].double(), variances[mask].double(), counts[mask].double()
        if float(x.max() - x.min()) < 1e-5:
            raise ValueError(
                f"all surviving bins sit at the same brightness ({float(x.mean()):.4g}): "
                "shot noise and read noise cannot be told apart"
            )

        def loss(a: float, b: float) -> float:
            return float((w * (y - (a * x + b)).pow(2)).sum())

        design = torch.stack([x, torch.ones_like(x)], dim=1)
        weighted = design * w.unsqueeze(1)
        free = torch.linalg.lstsq(design.T @ weighted, weighted.T @ y).solution
        a_free, b_free = float(free[0]), float(free[1])
        options = [(a_free, b_free)] if a_free >= 0 and b_free >= 0 else []
        options.append((0.0, float((w * y).sum() / w.sum())))
        slope_only = float((w * x * y).sum() / (w * x * x).sum().clamp(min=_EPS))
        options.append((max(slope_only, 0.0), 0.0))
        a, b = min(options, key=lambda c: loss(*c))
        mean_y = float((w * y).sum() / w.sum())
        total = float((w * (y - mean_y).pow(2)).sum())
        return a, b, (1.0 - loss(a, b) / total if total > 0 else 1.0)

    populated = keep.clone()
    a, b, r_squared = solve(keep)

    def refit(candidate: torch.Tensor) -> bool:
        """Adopt `candidate` if it is still a usable set of bins. True when the fit moved."""
        nonlocal keep, a, b, r_squared
        if int(candidate.sum()) < max(min_bins, 2) or bool((candidate == keep).all()):
            return False
        if float(means[candidate].max() - means[candidate].min()) < 1e-5:
            return False
        keep = candidate
        a, b, r_squared = solve(keep)
        return True

    for _ in range(max_iterations):  # 1. descend to the lower envelope
        predicted = (a * means + b).clamp(min=_EPS)
        if not refit(keep & (variances <= predicted)):
            break
    for _ in range(max_iterations):  # 2. re-include everything within the band
        predicted = (a * means + b).clamp(min=_EPS)
        if not refit(populated & (variances <= trim_ratio * predicted)):
            break
    return a, b, r_squared, keep


def kernel_autocorrelation(kernel: torch.Tensor, max_lag: int = MAX_LAG) -> torch.Tensor:
    """Normalized autocorrelation that `kernel` imprints on white noise.

    Filtering white noise with `k` gives an output whose covariance at lag `d` is
    `sum_p k[p]·k[p+d]`, so the normalized correlation is that over `sum_p k[p]^2`.
    """
    size = kernel.shape[0]
    out = torch.zeros(max_lag + 1, max_lag + 1)
    energy = max(float(kernel.pow(2).sum()), _EPS)
    for dy in range(max_lag + 1):
        for dx in range(max_lag + 1):
            if dy >= size or dx >= size:
                continue
            overlap = kernel[: size - dy, : size - dx] * kernel[dy:, dx:]
            out[dy, dx] = float(overlap.sum()) / energy
    return out


def _isp_params(denoise_sigma: float, sharpen_amount: float, isp_cfg: dict) -> CameraParams:
    """A CameraParams carrying only the ISP fields, so `camera.isp_kernel` stays the one
    implementation of the composite filter (no second copy to drift out of step)."""
    return CameraParams(
        n_white=1.0,
        read_noise=0.0,
        wb_gains=torch.ones(CHANNELS),
        jpeg_quality=100,
        a_raw=1.0,
        b_raw=0.0,
        ae_target=0.5,
        ae_percentile=0.5,
        gain_min=1.0,
        gain_max=1.0,
        quant_bits=12,
        denoise_sigma=denoise_sigma,
        denoise_radius=int(isp_cfg["denoise_radius"]),
        sharpen_amount=sharpen_amount,
        sharpen_sigma=float(isp_cfg["sharpen_sigma"]),
        sharpen_radius=int(isp_cfg["sharpen_radius"]),
    )


def fit_isp_strength(
    autocorr: torch.Tensor,
    isp_cfg: dict,
    sigma_range: tuple[float, float] = (0.0, 2.5),
    amount_range: tuple[float, float] = (0.0, 2.0),
    steps: int = 41,
    refinements: int = 3,
    patch: int = PATCH,
) -> tuple[float, float, float]:
    """Recover `(denoise_sigma, sharpen_amount)` from the measured noise autocorrelation.

    Blur correlates neighbouring pixels; the sharpening that follows it subtracts a blurred
    copy and pushes the correlation back down, overshooting into negative values at the lags it
    reaches. The pair is identifiable because they act on different lags: the blur sets how far
    the correlation extends, the sharpening sets how deep the dip at the far end goes. So this
    is a two-parameter surface, coarse-to-fine, scored against lags 1 to MAX_LAG.

    Args:
        autocorr: `(MAX_LAG+1, MAX_LAG+1)` measured, from `PatchAccumulator.autocorrelation`.
        isp_cfg: The `isp` section of the calibration config; the radii and the sharpening
            sigma are structural and are held fixed.
        sigma_range: Search bounds on the denoise blur sigma, in native pixels.
        amount_range: Search bounds on the unsharp amount.
        steps: Grid points per axis, per refinement.
        refinements: Coarse-to-fine passes; each shrinks the window around the current best.
        patch: Patch size the measurement used, so the model goes through the same detrend.

    Returns:
        `(denoise_sigma, sharpen_amount, rms_error)` over the fitted lags.
    """
    target = autocorr.clone()
    weight = torch.ones_like(target)
    weight[0, 0] = 0.0  # lag zero is 1 by definition and says nothing about the kernel

    def score(sigma: float, amount: float) -> float:
        kernel = isp_kernel(_isp_params(sigma, amount, isp_cfg))
        model = projected_autocorrelation(kernel, patch)
        return float((weight * (model - target).pow(2)).sum())

    best = (float(sigma_range[0]), float(amount_range[0]))
    lo_s, hi_s = sigma_range
    lo_a, hi_a = amount_range
    for _ in range(refinements):
        sigmas = torch.linspace(lo_s, hi_s, steps).tolist()
        amounts = torch.linspace(lo_a, hi_a, steps).tolist()
        best = min(((s, a) for s in sigmas for a in amounts), key=lambda c: score(*c))
        span_s, span_a = (hi_s - lo_s) / (steps - 1) * 2, (hi_a - lo_a) / (steps - 1) * 2
        lo_s, hi_s = max(sigma_range[0], best[0] - span_s), min(sigma_range[1], best[0] + span_s)
        lo_a, hi_a = max(amount_range[0], best[1] - span_a), min(amount_range[1], best[1] + span_a)

    rms = math.sqrt(score(*best) / max(float(weight.sum()), 1.0))
    return best[0], best[1], rms


# --- the whole measurement --------------------------------------------------------------------


@dataclass
class NoiseCalibration:
    """Everything `scripts/calibrate_noise.py` measured, before it becomes YAML."""

    effective_a: float  # Var = a·x + b as delivered by the camera, at NATIVE resolution
    effective_b: float
    per_channel_a: list[float]
    per_channel_b: list[float]
    r_squared: float
    darkest_bin: float  # mean signal of the darkest bin the line was fit through
    read_noise_share: float  # b / (a·darkest + b): how much of the fit `b` actually carries
    autocorr: list[float]  # lags 1..MAX_LAG along the axes, averaged over x and y
    denoise_sigma: float
    sharpen_amount: float
    autocorr_rms: float
    isp_var_scale: float  # sum of the composite kernel's squared weights
    effective_dof: float
    variance_bias: float
    reference_gain: float
    noise_a: float  # a_raw = 1 / full_well, what camera.py samples from
    noise_b: float  # b_raw = sigma_r^2 · a_raw^2, at the measured read noise
    full_well: float  # electrons
    read_noise_sigma: float  # electrons
    exposure_target: float  # median linear Rec. 709 luminance of the calibration frames
    num_images: int
    num_patches: int
    num_bins_used: int
    exposure_percentile: float = 0.9
    target_log10_quantiles: list[float] = field(default_factory=list)


def calibrate(
    images: Iterable[torch.Tensor],
    isp_cfg: dict,
    flat_quantile: float = 0.15,
    gate_quantile: float = 0.25,
    gate_scale: float = 1.3,
    num_bins: int = 40,
    min_bin_patches: int = 64,
    reference_gain: float = 1.0,
    exposure_percentile: float = 0.9,
    min_code: int = 3,
) -> NoiseCalibration:
    """Measure `(a, b)` and the ISP's denoising from a list of native-resolution night frames.

    Args:
        images: `(3, H, W)` linear float32 frames at NATIVE resolution. Downsampling them first
            would divide the very variance this measures (docs/conventions.md section 6).
        isp_cfg: The `isp` section of the calibration config, for the fixed radii.
        flat_quantile: Share of each brightness bin kept as "flat", per image.
        gate_quantile: Share of each image assumed featureless, which sets the absolute
            flatness gate (`PatchAccumulator._gate`).
        gate_scale: How far above that quantile a patch may sit and still count as flat.
        num_bins: Brightness bins, equally spaced in sqrt(mean) over `[0, 1]`.
        min_bin_patches: Bins with fewer survivors than this are dropped from the line fit.
        reference_gain: Digital gain assumed to have been applied by the real camera. It is not
            observable without EXIF; the sensor constants scale as `1/g` and `1/g^2` with it,
            and it is recorded alongside them.
        exposure_percentile: Percentile of per-frame luminance reported as `exposure.target`.
        min_code: Exclude patches containing pixels at or below this 8-bit sRGB code.

    Returns:
        A NoiseCalibration.

    Raises:
        ValueError: No usable images, or too few populated bins to separate `a` from `b`.
    """
    accumulator = PatchAccumulator(
        num_bins=num_bins,
        flat_quantile=flat_quantile,
        gate_quantile=gate_quantile,
        gate_scale=gate_scale,
        min_code=min_code,
    )
    luminances = []
    for image in images:
        accumulator.add_image(image)
        luminances.append(frame_luminance(image, percentile=exposure_percentile))
    return analyze(
        accumulator,
        luminances,
        isp_cfg,
        min_bin_patches=min_bin_patches,
        reference_gain=reference_gain,
        exposure_percentile=exposure_percentile,
    )


def analyze(
    accumulator: PatchAccumulator,
    luminances: Sequence[float],
    isp_cfg: dict,
    min_bin_patches: int = 64,
    reference_gain: float = 1.0,
    exposure_percentile: float = 0.9,
) -> NoiseCalibration:
    """Turn accumulated patch totals into the two fits and the sensor constants.

    Split out from `calibrate` so the measurement pass can be spread over processes (see
    `calibrate_paths`) and merged before any fitting happens: every total is a sum, so the
    answer does not depend on how the images were divided up.

    Args:
        accumulator: Totals over every calibration frame.
        luminances: One linear luminance statistic per frame, for `exposure.target`.
        isp_cfg: The `isp` section of the calibration config, for the fixed radii.
        min_bin_patches: Bins with fewer survivors than this are dropped from the line fit.
        reference_gain: Digital gain assumed applied by the real camera (see `calibrate`).
        exposure_percentile: Percentile of per-frame luminance reported as `exposure.target`.

    Returns:
        A NoiseCalibration.

    Raises:
        ValueError: No usable frames, or too few populated bins to separate `a` from `b`.
    """
    if accumulator.num_images == 0 or not luminances:
        raise ValueError("no calibration images")
    autocorr_2d = accumulator.autocorrelation()
    denoise_sigma, sharpen_amount, rms = fit_isp_strength(
        autocorr_2d, isp_cfg, patch=accumulator.patch
    )
    kernel = isp_kernel(_isp_params(denoise_sigma, sharpen_amount, isp_cfg))
    isp_var_scale = float(kernel.pow(2).sum())

    # The fitted kernel says how the measurement distorted itself, so the ISP fit has to come
    # first and the variances are de-biased with its answer: `bias` for detrending meeting
    # correlated noise, `factor` for having kept only the luckiest patches.
    dof = projected_dof(kernel, accumulator.patch)
    bias = projected_variance_bias(kernel, accumulator.patch)

    means, variances, counts = accumulator.bins(min_bin_patches)
    variances = variances / bias
    per_channel = [fit_variance_line(means[c], variances[c], counts[c]) for c in range(CHANNELS)]
    pooled_a, pooled_b, pooled_r2, kept = fit_variance_line(
        means.reshape(-1), variances.reshape(-1), counts.reshape(-1)
    )

    # How far down the fit actually reaches. `b` is the intercept of a line, so it is only
    # measured where `a·x` has stopped dominating; in a frame with no near-black region the
    # data say nothing about read noise and the fit will happily return any small number.
    darkest = float(means.reshape(-1)[kept].min()) if bool(kept.any()) else float("nan")
    read_share = pooled_b / max(pooled_a * darkest + pooled_b, _EPS)

    # Undo the ISP and the assumed gain to get the sensor constants camera.py samples from:
    # a_eff = g·a_raw·s and b_eff = g^2·b_raw·s, with s the kernel's variance scale.
    a_raw = pooled_a / max(reference_gain * isp_var_scale, _EPS)
    b_raw = pooled_b / max(reference_gain**2 * isp_var_scale, _EPS)
    full_well = 1.0 / max(a_raw, _EPS)
    read_sigma = math.sqrt(max(b_raw, 0.0)) * full_well

    axis_corr = [float((autocorr_2d[0, d] + autocorr_2d[d, 0]) / 2) for d in range(1, MAX_LAG + 1)]
    log_luminances = [math.log10(max(lum, 1e-12)) for lum in luminances]
    log_tensor = torch.tensor(sorted(log_luminances), dtype=torch.float32)
    q = torch.linspace(0.0, 1.0, 21)
    quantiles = torch.quantile(log_tensor, q)
    target_log10_quantiles = [round(float(v), 3) for v in quantiles]
    target = 10.0 ** float(torch.quantile(log_tensor, torch.tensor(0.5)).item())
    return NoiseCalibration(
        effective_a=pooled_a,
        effective_b=pooled_b,
        per_channel_a=[c[0] for c in per_channel],
        per_channel_b=[c[1] for c in per_channel],
        r_squared=pooled_r2,
        darkest_bin=darkest,
        read_noise_share=read_share,
        autocorr=axis_corr,
        denoise_sigma=denoise_sigma,
        sharpen_amount=sharpen_amount,
        autocorr_rms=rms,
        isp_var_scale=isp_var_scale,
        effective_dof=dof,
        variance_bias=bias,
        reference_gain=reference_gain,
        noise_a=a_raw,
        noise_b=b_raw,
        full_well=full_well,
        read_noise_sigma=read_sigma,
        exposure_target=target,
        num_images=accumulator.num_images,
        num_patches=int(counts.reshape(-1)[kept].sum()),
        num_bins_used=int(kept.sum()),
        exposure_percentile=exposure_percentile,
        target_log10_quantiles=target_log10_quantiles,
    )


def read_native_linear(path: str | Path) -> torch.Tensor:
    """One raw JPEG as `(3, H, W)` linear float32, at native resolution (§1, §4a: no resize)."""
    return srgb8_to_linear(read_srgb8(path))


def frame_luminance(image: torch.Tensor, percentile: float = 0.5) -> float:
    """Percentile of linear Rec. 709 luminance of one `(3, H, W)` frame (docs/conventions.md 4)."""
    lum = luminance(image.unsqueeze(0)).flatten()
    return float(torch.quantile(lum, torch.tensor(percentile, dtype=torch.float32)).item())


def accumulate_path(
    path: str | Path,
    num_bins: int = 40,
    flat_quantile: float = 0.15,
    gate_quantile: float = 0.25,
    gate_scale: float = 1.3,
    min_code: int = 3,
    exposure_percentile: float = 0.9,
) -> tuple[PatchAccumulator, float]:
    """Measure one file. Top level and picklable, so `prepare.run_parallel` can spread it out."""
    image = read_native_linear(path)
    accumulator = PatchAccumulator(
        num_bins=num_bins,
        flat_quantile=flat_quantile,
        gate_quantile=gate_quantile,
        gate_scale=gate_scale,
        min_code=min_code,
    )
    accumulator.add_image(image)
    return accumulator, frame_luminance(image, percentile=exposure_percentile)


def calibrate_paths(
    paths: Sequence[str | Path],
    isp_cfg: dict,
    workers: int = 1,
    flat_quantile: float = 0.15,
    gate_quantile: float = 0.25,
    gate_scale: float = 1.3,
    num_bins: int = 40,
    min_bin_patches: int = 64,
    reference_gain: float = 1.0,
    exposure_percentile: float = 0.9,
    exposure_paths: Iterable[str | Path] | None = None,
    min_code: int = 3,
) -> NoiseCalibration:
    """`calibrate`, reading from disk and measuring the frames in parallel.

    Args:
        exposure_paths: Subset of `paths` whose luminances set `exposure.target`; all of them
            by default. The two jobs pull in opposite directions: `b` is only identifiable
            where something is nearly black, so the calibration set is deliberately stacked
            with the darkest frames available, and a brightness target read off that set would
            aim the simulated auto-exposure far below where real night frames sit.

    Raises:
        ValueError: `paths` is empty, or nothing in it could be measured.
    """
    from dawn.data.prepare import run_parallel

    if not paths:
        raise ValueError("no calibration images")
    jobs = [
        (path, num_bins, flat_quantile, gate_quantile, gate_scale, min_code, exposure_percentile)
        for path in paths
    ]
    results = run_parallel(accumulate_path, jobs, workers, desc="calibrate")
    wanted = None if exposure_paths is None else {str(Path(p).resolve()) for p in exposure_paths}
    total = PatchAccumulator(
        num_bins=num_bins,
        flat_quantile=flat_quantile,
        gate_quantile=gate_quantile,
        gate_scale=gate_scale,
        min_code=min_code,
    )
    luminances, subset = [], []
    for path, (accumulator, p_val) in zip(paths, results, strict=True):
        total.merge(accumulator)
        luminances.append(p_val)
        if wanted is not None and str(Path(path).resolve()) in wanted:
            subset.append(p_val)
    return analyze(
        total,
        subset or luminances,
        isp_cfg,
        min_bin_patches=min_bin_patches,
        reference_gain=reference_gain,
        exposure_percentile=exposure_percentile,
    )


# --- synthetic ground truth ----------------------------------------------------------------


def _random_field(
    height: int, width: int, scale: int, generator: torch.Generator, channels: int = CHANNELS
) -> torch.Tensor:
    """A smooth random image: white noise at 1/`scale` of the size, bilinearly upsampled."""
    coarse = torch.rand(
        1, channels, max(2, height // scale), max(2, width // scale), generator=generator
    )
    return F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)[0]


def synthetic_frame(
    height: int,
    width: int,
    generator: torch.Generator,
    kernel: torch.Tensor,
    level: float = 0.06,
    a_raw: float = 2.0e-4,
    b_raw: float = 1.6e-7,
    texture_fraction: float = 0.5,
    region: int = 128,
    texture_strength: float = 0.7,
    crush_blacks: bool = False,
) -> torch.Tensor:
    """One frame with a known sensor and a known ISP, to check the estimator against truth.

    There is no laboratory flat field here, so the only way to know whether a measurement is
    right is to measure something whose answer is already known. Shared by the CLI's
    `--self-test` and by `tests/test_calibration.py` so both check the same thing.

    The scene is deliberately awkward in the two ways that broke earlier versions of the
    estimator. The background is smooth but never flat, so a patch that looks featureless
    still carries a gentle curve the detrend has to remove. The textured regions are far
    larger than one patch, so there is no setting of the patch size at which texture averages
    away: a patch either lands in detail or it does not, which is exactly the choice the
    flatness ranking has to get right.

    Args:
        height: Rows. A few hundred at least, or there are too few patches per brightness bin.
        width: Columns.
        generator: Seeded RNG. Same seed, same frame.
        kernel: `(k, k)` ISP filter, from `camera.isp_kernel`. The returned frame has
            effective `a = a_raw * kernel.pow(2).sum()` and likewise for `b`.
        level: Roughly the brightest the clean scene gets, in linear [0, 1].
        a_raw: Shot-noise gain of the simulated sensor, before the ISP.
        b_raw: Read-noise variance of the simulated sensor, before the ISP.
        texture_fraction: Share of the frame covered by detail.
        region: Side of one textured region, in pixels.
        texture_strength: How hard the texture modulates the background.
        crush_blacks: Clip at 0 and round to 8-bit sRGB codes before returning.

    Returns:
        `(3, height, width)` linear float32 in [0, 1], as a camera would deliver it.
    """
    scene = _random_field(height, width, max(1, height // 8), generator) * level + 0.002
    blocks = torch.rand(1, height // region, width // region, generator=generator)
    mask = (blocks < texture_fraction).float()
    mask = mask.repeat_interleave(region, 1).repeat_interleave(region, 2)
    detail = sum(
        _random_field(height, width, scale, generator, channels=1) / step
        for step, scale in enumerate([4, 8, 16], start=1)
    )
    detail = ((detail - detail.mean()) / detail.std()).expand(CHANNELS, height, width)
    scene = (scene * (1 + texture_strength * detail * mask)).clamp(1e-4, 1.0)

    noise = torch.randn(scene.shape, generator=generator) * (a_raw * scene + b_raw).sqrt()
    radius = kernel.shape[0] // 2
    weight = kernel.view(1, 1, *kernel.shape).expand(CHANNELS, 1, *kernel.shape)
    padded = F.pad((scene + noise).unsqueeze(0), (radius,) * 4, mode="replicate")
    out = F.conv2d(padded, weight, groups=CHANNELS)[0].clamp(0.0, 1.0)
    if crush_blacks:
        out = srgb8_to_linear(linear_to_srgb8(out))
    return out
