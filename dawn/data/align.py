"""Aligning day thermal to day RGB with the vehicle boxes (plan steps 3-10).

The thermal and RGB cameras are mounted a few centimetres apart and expose at slightly
different times, so a pair is a similarity warp (plus moving vehicles) apart, not identical. The
labels give correspondences without any cross-modal feature matching: each vehicle is boxed
separately in both modalities, and the box centres are the points.

Pipeline for one pair (`align_pair` does the I/O, `align_boxes` the geometry):

1. `match_boxes`: copied boxes out, then same-class **mutual** nearest-neighbour centres within a
   gate.
2. `ransac_similarity`: scale + rotation + translation, thermal -> RGB. With at most
   `ransac_enumerate_max` matches every 2-match minimal set is tried, so no RNG is needed; above
   that the sets are drawn from `generator_for(image_id, "ransac")` (docs/conventions.md 10.3).
3. A homography (`ransac_homography`) only if the similarity fails on the residual alone.
4. `failed_checks`: the acceptance thresholds, in plan order, each with its own reason code.
5. `edge_check`: an independent test on the pixels, not the boxes.
6. `warp_thermal`, `moving_mask`: the products `write` stores.

Conventions (docs/conventions.md section 10)
--------------------------------------------
- **Pixel-index coordinates**: x is the column and y the row, and the centre of the pixel in row
  r and column c is (c, r). XML boxes (after the border crop) and every transform here use them,
  so an integer shift of `(dx, dy)` moves pixel (c, r) to (c + dx, r + dy) exactly.
- **Transforms map thermal onto RGB**, as 3x3 matrices acting on `(x, y, 1)`. The RGB image is
  never resampled; the thermal image is, by inverse mapping (`warp_thermal`).
- **Shift** is the displacement of the frame centre `(W / 2, H / 2)` = (320, 256) under the
  transform, not the raw translation, which depends on the origin of the coordinates.
- `warp_thermal` is a bilinear resample, not a downsample: the area-averaging rule of
  docs/conventions.md section 4a does not apply to it.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from dawn.data import dronevehicle as dv
from dawn.data.color import linear_to_srgb, luminance, srgb8_to_linear
from dawn.data.seeding import generator_for

logger = logging.getLogger(__name__)

_EPS = 1e-9  # slack on inclusive thresholds, so 1.05 passes a "within 5%" check in floating point
_NAN = float("nan")
_MIN_EDGE_PIXELS = 1000  # fewer valid pixels than this and the edge check is not meaningful


@dataclass(frozen=True)
class AlignConfig:
    """Every alignment threshold (the `align` section of data.yaml; defaults are the same)."""

    max_match_dist_px: float = 40.0
    copy_tol_px: float = 0.5
    exclude_copied_boxes: bool = True
    ransac_inlier_px: float = 4.0
    ransac_enumerate_max: int = 40
    ransac_iterations: int = 500
    homography_enabled: bool = True
    homography_min_matches: int = 10
    homography_iterations: int = 1000
    min_matches: int = 6
    min_inlier_ratio: float = 0.6
    max_median_residual_px: float = 2.0
    max_scale_deviation: float = 0.05
    max_rotation_deg: float = 2.0
    max_shift_px: float = 30.0
    grad_enabled: bool = True
    grad_blur_sigma: float = 1.0
    grad_min_gain: float = -0.2
    grad_search_px: int = 4
    grad_max_best_shift_px: int = 3
    moving_min_residual_px: float = 4.0
    moving_dilate_px: int = 3

    @classmethod
    def from_config(cls, cfg: Mapping) -> AlignConfig:
        """Build from the `align` section of data.yaml; missing keys keep their defaults."""
        homography = cfg.get("homography") or {}
        grad = cfg.get("grad_check") or {}
        moving = cfg.get("moving") or {}
        flat = {
            k: cfg[k]
            for k in (
                "max_match_dist_px",
                "copy_tol_px",
                "exclude_copied_boxes",
                "ransac_inlier_px",
                "ransac_enumerate_max",
                "ransac_iterations",
                "min_matches",
                "min_inlier_ratio",
                "max_median_residual_px",
                "max_scale_deviation",
                "max_rotation_deg",
                "max_shift_px",
            )
            if k in cfg
        }
        nested = {
            "homography_enabled": homography.get("enabled"),
            "homography_min_matches": homography.get("min_matches"),
            "homography_iterations": homography.get("ransac_iterations"),
            "grad_enabled": grad.get("enabled"),
            "grad_blur_sigma": grad.get("blur_sigma"),
            "grad_min_gain": grad.get("min_gain"),
            "grad_search_px": grad.get("search_px"),
            "grad_max_best_shift_px": grad.get("max_best_shift_px"),
            "moving_min_residual_px": moving.get("min_residual_px"),
            "moving_dilate_px": moving.get("dilate_px"),
        }
        return cls(**flat, **{k: v for k, v in nested.items() if v is not None})


# --- copied boxes and matching -------------------------------------------------------------------


def _stack(boxes: Sequence[dv.Box]) -> np.ndarray:
    return np.stack([b.vertices for b in boxes]) if boxes else np.zeros((0, 4, 2))


def copied_flags(
    thermal_boxes: Sequence[dv.Box], rgb_boxes: Sequence[dv.Box], tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Which boxes are copies of a box in the other modality.

    About 17% of pairs have identical RGB and thermal label sets, and in the rest a median of 25%
    of RGB boxes are exact copies of a thermal box. A copy has zero residual by construction, so
    it is not alignment evidence. A thermal box is a copy of an RGB box when the (normalized)
    class matches and every coordinate of all four vertices, in order, is within `tol` px.

    Returns:
        `(thermal_copied (nT,), rgb_copied (nR,))` bool arrays.
    """
    t_flags = np.zeros(len(thermal_boxes), dtype=bool)
    r_flags = np.zeros(len(rgb_boxes), dtype=bool)
    if not len(thermal_boxes) or not len(rgb_boxes):
        return t_flags, r_flags
    gap = np.abs(_stack(thermal_boxes)[:, None] - _stack(rgb_boxes)[None]).max(axis=(2, 3))
    same_class = np.array([[t.category == r.category for r in rgb_boxes] for t in thermal_boxes])
    pairs = same_class & (gap <= tol)
    return pairs.any(axis=1), pairs.any(axis=0)


def labels_identical(
    thermal_boxes: Sequence[dv.Box], rgb_boxes: Sequence[dv.Box], tol: float
) -> bool:
    """True when the two label sets are the same: equal, non-zero counts, every box copied."""
    if len(thermal_boxes) != len(rgb_boxes) or not thermal_boxes:
        return False
    t_flags, r_flags = copied_flags(thermal_boxes, rgb_boxes, tol)
    return bool(t_flags.all() and r_flags.all())


@dataclass(frozen=True)
class Matches:
    """Matched box centres, thermal to RGB, in the cropped frame's pixel-index coordinates."""

    thermal_idx: np.ndarray  # (n,) indices into the thermal box list
    rgb_idx: np.ndarray  # (n,) indices into the RGB box list
    thermal_xy: np.ndarray  # (n, 2)
    rgb_xy: np.ndarray  # (n, 2)
    n_thermal: int  # boxes on each side before any exclusion
    n_rgb: int
    n_copied: int  # thermal boxes that are copies (excluded when `exclude_copied_boxes`)
    labels_identical: bool
    n_unmatched_thermal: int  # candidate boxes without a match (copies are not candidates)
    n_unmatched_rgb: int

    @property
    def n(self) -> int:
        return len(self.thermal_idx)


def match_boxes(
    thermal_boxes: Sequence[dv.Box], rgb_boxes: Sequence[dv.Box], cfg: AlignConfig
) -> Matches:
    """Mutual nearest neighbours between thermal and RGB box centres.

    Boxes are compared per class by the distance between their centres (the mean of the four
    vertices). A pair matches when each is the other's nearest neighbour and they are at most
    `cfg.max_match_dist_px` apart (a gate the plan does not state). Copied boxes are not candidates
    on either side when `cfg.exclude_copied_boxes`.
    """
    t_copied, r_copied = copied_flags(thermal_boxes, rgb_boxes, cfg.copy_tol_px)
    identical = labels_identical(thermal_boxes, rgb_boxes, cfg.copy_tol_px)
    t_cand = (
        np.flatnonzero(~t_copied) if cfg.exclude_copied_boxes else np.arange(len(thermal_boxes))
    )
    r_cand = np.flatnonzero(~r_copied) if cfg.exclude_copied_boxes else np.arange(len(rgb_boxes))
    t_pick, r_pick = [], []
    for category in sorted({thermal_boxes[i].category for i in t_cand}):
        ti = np.array([i for i in t_cand if thermal_boxes[i].category == category])
        ri = np.array([j for j in r_cand if rgb_boxes[j].category == category], dtype=int)
        if not len(ri):
            continue
        t_xy = np.stack([thermal_boxes[i].center for i in ti])
        r_xy = np.stack([rgb_boxes[j].center for j in ri])
        dist = np.linalg.norm(t_xy[:, None] - r_xy[None], axis=2)
        nearest_r, nearest_t = dist.argmin(axis=1), dist.argmin(axis=0)
        for a, b in enumerate(nearest_r):
            if nearest_t[b] == a and dist[a, b] <= cfg.max_match_dist_px:
                t_pick.append(ti[a])
                r_pick.append(ri[b])
    order = np.argsort(t_pick, kind="stable")
    t_idx = np.array(t_pick, dtype=int)[order]
    r_idx = np.array(r_pick, dtype=int)[order]
    return Matches(
        thermal_idx=t_idx,
        rgb_idx=r_idx,
        thermal_xy=np.array([thermal_boxes[i].center for i in t_idx]).reshape(-1, 2),
        rgb_xy=np.array([rgb_boxes[j].center for j in r_idx]).reshape(-1, 2),
        n_thermal=len(thermal_boxes),
        n_rgb=len(rgb_boxes),
        n_copied=int(t_copied.sum()),
        labels_identical=identical,
        n_unmatched_thermal=len(t_cand) - len(t_idx),
        n_unmatched_rgb=len(r_cand) - len(r_idx),
    )


# --- geometry ------------------------------------------------------------------------------------


def apply_transform(T: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply a 3x3 transform (similarity or homography) to `(n, 2)` points `(x, y)`."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = xy @ T[2, :2] + T[2, 2]
        return (xy @ T[:2, :2].T + T[:2, 2]) / w[:, None]


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity `dst ~ s R src + t` (no reflection), as a 3x3 matrix.

    In complex form `z' = a z + b`, with `a = s e^{i theta}`; exact for two points (the minimal
    set of the RANSAC) and the Umeyama solution for more.

    Raises:
        ValueError: Fewer than two points, or the source points are not distinct.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) < 2:
        raise ValueError("a similarity needs at least two points")
    zs, zd = src[:, 0] + 1j * src[:, 1], dst[:, 0] + 1j * dst[:, 1]
    xs, xd = zs - zs.mean(), zd - zd.mean()
    denom = float((np.abs(xs) ** 2).sum())
    if denom < 1e-12:
        raise ValueError("source points are not distinct")
    a = (np.conj(xs) * xd).sum() / denom
    b = zd.mean() - a * zs.mean()
    return np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag], [0.0, 0.0, 1.0]])


def _normalizer(points: np.ndarray) -> np.ndarray:
    """Hartley normalization: centroid to the origin, mean distance sqrt(2)."""
    centre = points.mean(axis=0)
    spread = np.linalg.norm(points - centre, axis=1).mean()
    if spread < 1e-12:
        raise ValueError("points are not distinct")
    s = math.sqrt(2) / spread
    return np.array([[s, 0, -s * centre[0]], [0, s, -s * centre[1]], [0, 0, 1.0]])


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Homography `dst ~ H src` by the normalized direct linear transform, `H[2, 2] = 1`.

    Raises:
        ValueError: Fewer than four points, or a degenerate (collinear / coincident) configuration.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    if len(src) < 4:
        raise ValueError("a homography needs at least four points")
    ts, td = _normalizer(src), _normalizer(dst)
    ps = src @ ts[:2, :2].T + ts[:2, 2]
    pd = dst @ td[:2, :2].T + td[:2, 2]
    rows = []
    for (x, y), (u, v) in zip(ps, pd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, singular, vt = np.linalg.svd(np.array(rows))
    if singular[-2] < 1e-9 * singular[0]:  # null space is more than one-dimensional
        raise ValueError("degenerate point configuration")
    h = np.linalg.inv(td) @ vt[-1].reshape(3, 3) @ ts
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography")
    return h / h[2, 2]


def transform_summary(T: np.ndarray, size: tuple[int, int]) -> tuple[float, float, float]:
    """`(scale, rotation_deg, shift_px)` of a 3x3 transform on a frame of `size = (H, W)`.

    The shift is the displacement of the frame centre `(W / 2, H / 2)`. For a similarity, scale and
    rotation are exact; for a homography they are those of the local Jacobian at the frame
    centre (the nearest similarity there): `scale = sqrt(|det J|)`.
    """
    centre = np.array([size[1] / 2.0, size[0] / 2.0])
    mapped = apply_transform(T, centre[None])[0]
    w = float(centre @ T[2, :2] + T[2, 2])
    jac = (T[:2, :2] - np.outer(mapped, T[2, :2])) / w
    scale = math.sqrt(abs(np.linalg.det(jac)))
    rotation = math.degrees(math.atan2(jac[1, 0] - jac[0, 1], jac[0, 0] + jac[1, 1]))
    return scale, rotation, float(np.linalg.norm(mapped - centre))


def _residuals(T: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    res = np.linalg.norm(apply_transform(T, src) - dst, axis=1)
    return np.where(np.isfinite(res), res, np.inf)


def _best_hypothesis(residual: np.ndarray, valid: np.ndarray, inlier_px: float) -> int:
    """Index of the hypothesis with the most inliers, then the smallest truncated error sum, then
    the lowest index. `residual` is `(P, n)`."""
    inliers = (residual <= inlier_px) & valid[:, None]
    count = np.where(valid, inliers.sum(axis=1), -1)
    error = np.minimum(residual, inlier_px).sum(axis=1)
    return int(np.lexsort((np.arange(len(count)), error, -count))[0])


def _refine(fit, T: np.ndarray, src: np.ndarray, dst: np.ndarray, inlier_px: float, min_pts: int):
    """Refit on the inliers of `T` until the inlier set stops changing (at most 5 rounds)."""
    mask = _residuals(T, src, dst) <= inlier_px
    for _ in range(5):
        if mask.sum() < min_pts:
            break
        try:
            refit = fit(src[mask], dst[mask])
        except ValueError:
            break
        new_mask = _residuals(refit, src, dst) <= inlier_px
        if new_mask.sum() < min_pts:
            break
        T = refit
        if (new_mask == mask).all():
            mask = new_mask
            break
        mask = new_mask
    return T, _residuals(T, src, dst) <= inlier_px


def ransac_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_px: float,
    enumerate_max: int,
    iterations: int,
    generator: torch.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC similarity fit, thermal `src` -> RGB `dst`, refit on the inliers.

    Minimal sets are two matches. With `len(src) <= enumerate_max` every pair is tried, which is
    deterministic and needs no generator; above that, `iterations` pairs are drawn from
    `generator` (pass `generator_for(image_id, "ransac")`). The best hypothesis has the most
    inliers (residual <= `inlier_px`), ties broken by the smaller truncated error, then by order.

    Returns:
        `(T (3, 3), inliers (n,) bool)`, the inliers measured under the refit `T`.

    Raises:
        ValueError: Fewer than two matches, or more than `enumerate_max` and no generator.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    n = len(src)
    if n < 2:
        raise ValueError("a similarity needs at least two matches")
    if n <= enumerate_max:
        i, j = np.triu_indices(n, 1)
    else:
        if generator is None:
            raise ValueError(f"{n} matches exceed enumerate_max={enumerate_max}: pass a generator")
        draw = torch.randint(0, n, (iterations, 2), generator=generator).numpy()
        draw = draw[draw[:, 0] != draw[:, 1]]
        i, j = draw[:, 0], draw[:, 1]
    zs, zd = src[:, 0] + 1j * src[:, 1], dst[:, 0] + 1j * dst[:, 1]
    dz = zs[i] - zs[j]
    valid = np.abs(dz) > 1e-9
    a = np.where(valid, (zd[i] - zd[j]) / np.where(valid, dz, 1.0), 0)
    b = zd[i] - a * zs[i]
    residual = np.abs(a[:, None] * zs[None, :] + b[:, None] - zd[None, :])
    best = _best_hypothesis(residual, valid, inlier_px)
    T0 = np.array(
        [
            [a[best].real, -a[best].imag, b[best].real],
            [a[best].imag, a[best].real, b[best].imag],
            [0.0, 0.0, 1.0],
        ]
    )
    return _refine(fit_similarity, T0, src, dst, inlier_px, 2)


def ransac_homography(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_px: float,
    iterations: int,
    generator: torch.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC homography fit with 4-match minimal sets drawn from `generator`, refit on inliers.

    Returns:
        `(H (3, 3), inliers (n,) bool)`.

    Raises:
        ValueError: Fewer than four matches, or no non-degenerate minimal set was found.
    """
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    n = len(src)
    if n < 4:
        raise ValueError("a homography needs at least four matches")
    best, best_key = None, None
    for _ in range(iterations if n > 4 else 1):
        pick = torch.randperm(n, generator=generator)[:4].numpy()
        try:
            candidate = fit_homography(src[pick], dst[pick])
        except ValueError:
            continue
        res = _residuals(candidate, src, dst)
        key = (int((res <= inlier_px).sum()), -float(np.minimum(res, inlier_px).sum()))
        if best_key is None or key > best_key:
            best, best_key = candidate, key
    if best is None:
        raise ValueError("no non-degenerate minimal set")
    return _refine(fit_homography, best, src, dst, inlier_px, 4)


# --- acceptance ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryMetrics:
    """What the acceptance thresholds look at, for one fitted model."""

    n_matches: int
    labels_identical: bool
    n_inliers: int
    inlier_ratio: float
    median_residual: float  # px, over the inliers: the number the threshold applies to
    median_residual_all: float  # px, over all matches (logged only)
    scale: float
    rotation_deg: float
    shift_px: float


def failed_checks(m: GeometryMetrics, cfg: AlignConfig) -> list[str]:
    """Every acceptance check the fit fails, in plan order (empty = accepted).

    `few_matches` (`labels_copied` when the two label sets are identical), `inlier_ratio`,
    `residual`, `scale`, `rotation`, `shift`. Thresholds: matches >= min, ratio >= min, median
    inlier residual <= max, |scale - 1| <= max, |rotation| < max, shift < max. NaN fails.
    """
    failed = []
    if not m.n_matches >= cfg.min_matches:
        failed.append("labels_copied" if m.labels_identical else "few_matches")
    if not m.inlier_ratio >= cfg.min_inlier_ratio - _EPS:
        failed.append("inlier_ratio")
    if not m.median_residual <= cfg.max_median_residual_px + _EPS:
        failed.append("residual")
    if not abs(m.scale - 1.0) <= cfg.max_scale_deviation + _EPS:
        failed.append("scale")
    if not abs(m.rotation_deg) < cfg.max_rotation_deg:
        failed.append("rotation")
    if not m.shift_px < cfg.max_shift_px:
        failed.append("shift")
    return failed


def first_reason(m: GeometryMetrics, cfg: AlignConfig) -> str:
    """`ok`, or the first failed check in plan order."""
    failed = failed_checks(m, cfg)
    return failed[0] if failed else "ok"


def _metrics(
    matches: Matches, T: np.ndarray, inliers: np.ndarray, size: tuple[int, int]
) -> GeometryMetrics:
    res = _residuals(T, matches.thermal_xy, matches.rgb_xy)
    scale, rotation, shift = transform_summary(T, size)
    return GeometryMetrics(
        n_matches=matches.n,
        labels_identical=matches.labels_identical,
        n_inliers=int(inliers.sum()),
        inlier_ratio=float(inliers.sum()) / matches.n,
        median_residual=float(np.median(res[inliers])),
        median_residual_all=float(np.median(res)),
        scale=scale,
        rotation_deg=rotation,
        shift_px=shift,
    )


@dataclass(frozen=True)
class AlignResult:
    """The geometry of one pair: matches, the model used, its transform and the verdict."""

    matches: Matches
    model: str  # "similarity", "homography" or "none" (fewer than two matches)
    T: np.ndarray | None  # 3x3, thermal -> RGB; None when `model == "none"`
    metrics: GeometryMetrics  # of the model used (what the acceptance applied to)
    sim: GeometryMetrics | None  # the similarity attempt
    hom: GeometryMetrics | None  # the homography attempt, when it was tried
    reason: str  # "ok" or the first failed check
    n_moving: int
    n_unmatched_thermal: int
    n_unmatched_rgb: int


def moving_flags(matches: Matches, T: np.ndarray | None, min_residual_px: float) -> np.ndarray:
    """`(n,)` bool: matched boxes whose own residual under `T` exceeds `min_residual_px`."""
    if T is None or not matches.n:
        return np.zeros(matches.n, dtype=bool)
    return _residuals(T, matches.thermal_xy, matches.rgb_xy) > min_residual_px


def align_boxes(
    thermal_boxes: Sequence[dv.Box],
    rgb_boxes: Sequence[dv.Box],
    cfg: AlignConfig,
    image_id: str,
    size: tuple[int, int] = (512, 640),
) -> AlignResult:
    """Match, fit and judge the geometry of one pair (steps 3-5 of the plan, no pixels).

    The similarity is fitted whenever there are two matches, even when there are too few to accept
    (so the audit can see what a looser `min_matches` would have done). The homography is tried
    only when the similarity fails on the residual alone and there are at least
    `cfg.homography_min_matches` matches; it is used when it passes every check with the same
    thresholds, and logged either way.

    Args:
        size: Frame `(H, W)` after the crop, for the shift definition.
    """
    matches = match_boxes(thermal_boxes, rgb_boxes, cfg)
    nan = GeometryMetrics(
        matches.n, matches.labels_identical, 0, _NAN, _NAN, _NAN, _NAN, _NAN, _NAN
    )
    if matches.n < 2:
        reason = first_reason(nan, cfg)
        return AlignResult(
            matches,
            "none",
            None,
            nan,
            None,
            None,
            reason,
            0,
            matches.n_unmatched_thermal,
            matches.n_unmatched_rgb,
        )
    generator = generator_for(image_id, "ransac")
    T, mask = ransac_similarity(
        matches.thermal_xy,
        matches.rgb_xy,
        cfg.ransac_inlier_px,
        cfg.ransac_enumerate_max,
        cfg.ransac_iterations,
        generator if matches.n > cfg.ransac_enumerate_max else None,
    )
    sim = _metrics(matches, T, mask, size)
    chosen, model, hom = sim, "similarity", None
    if (
        cfg.homography_enabled
        and matches.n >= cfg.homography_min_matches
        and failed_checks(sim, cfg) == ["residual"]
    ):
        try:
            h_fit, h_mask = ransac_homography(
                matches.thermal_xy,
                matches.rgb_xy,
                cfg.ransac_inlier_px,
                cfg.homography_iterations,
                generator,
            )
        except ValueError:
            pass
        else:
            hom = _metrics(matches, h_fit, h_mask, size)
            if not failed_checks(hom, cfg):
                chosen, model, T = hom, "homography", h_fit
    n_moving = int(moving_flags(matches, T, cfg.moving_min_residual_px).sum())
    return AlignResult(
        matches,
        model,
        T,
        chosen,
        sim,
        hom,
        first_reason(chosen, cfg),
        n_moving,
        matches.n_unmatched_thermal,
        matches.n_unmatched_rgb,
    )


# --- warp ----------------------------------------------------------------------------------------


def warp_thermal(
    thermal: torch.Tensor, T: np.ndarray, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample thermal onto the RGB frame by bilinear inverse mapping.

    RGB pixel `(x, y)` takes the thermal value at `T^-1 (x, y)` (pixel-index coordinates, x =
    column, y = row; module docstring). The RGB image itself is never resampled. This is an
    interpolation at arbitrary positions, not a downsample, so docs/conventions.md section 4a
    (area averaging) does not apply. Computed in float64, so the identity and integer shifts
    reproduce the input exactly.

    The validity mask is False wherever the source position falls outside the thermal frame
    (beyond the extent of its pixels, `[-0.5, W_t - 0.5]` and likewise in y), then eroded by one
    pixel (8-neighbourhood; the outside of the RGB frame counts as valid) for the bilinear edge.

    Args:
        thermal: `(1, H_t, W_t)` or `(H_t, W_t)` float, codes / 255.
        T: 3x3, thermal -> RGB.
        size: Output `(H, W)`, the RGB frame.

    Returns:
        `(warped (1, H, W) float32, valid (1, H, W) bool)`. Where `valid` is False the value is
        a clamped edge sample and must not be used.
    """
    plane = thermal.detach().cpu().numpy().astype(np.float64)
    plane = plane[0] if plane.ndim == 3 else plane
    h_t, w_t = plane.shape
    h, w = size
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    src = np.linalg.inv(T) @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
    with np.errstate(divide="ignore", invalid="ignore"):
        sx, sy = (src[0] / src[2]).reshape(h, w), (src[1] / src[2]).reshape(h, w)
    inside = (sx >= -0.5) & (sx <= w_t - 0.5) & (sy >= -0.5) & (sy <= h_t - 0.5)
    inside &= np.isfinite(sx) & np.isfinite(sy)
    cx = np.clip(np.nan_to_num(sx), 0, w_t - 1)
    cy = np.clip(np.nan_to_num(sy), 0, h_t - 1)
    x0, y0 = np.floor(cx).astype(int), np.floor(cy).astype(int)
    x1, y1 = np.minimum(x0 + 1, w_t - 1), np.minimum(y0 + 1, h_t - 1)
    fx, fy = cx - x0, cy - y0
    out = (
        plane[y0, x0] * (1 - fx) * (1 - fy)
        + plane[y0, x1] * fx * (1 - fy)
        + plane[y1, x0] * (1 - fx) * fy
        + plane[y1, x1] * fx * fy
    )
    valid = ndimage.binary_erosion(inside, structure=np.ones((3, 3), bool), border_value=1)
    return torch.from_numpy(out.astype(np.float32))[None], torch.from_numpy(valid)[None]


# --- moving-vehicle mask -------------------------------------------------------------------------


def rasterize(polygons: Sequence[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """`(H, W)` bool mask of the union of `(k, 2)` polygons in pixel-index coordinates. Edges are
    included: a box with vertices at x = 45 and x = 55 covers columns 45 to 55."""
    canvas = Image.new("L", (size[1], size[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        draw.polygon([tuple(map(float, p)) for p in poly], fill=1, outline=1)
    return np.asarray(canvas, dtype=bool)


def moving_mask(
    thermal_boxes: Sequence[dv.Box],
    rgb_boxes: Sequence[dv.Box],
    matches: Matches,
    T: np.ndarray | None,
    size: tuple[int, int],
    cfg: AlignConfig,
) -> tuple[torch.Tensor, int]:
    """Mask of the vehicles that moved between the two exposures (plan step 10).

    A matched box whose own residual under `T` exceeds `cfg.moving_min_residual_px` is moving.
    Its mask is the union of its RGB polygon and its thermal polygon pushed through `T` (where
    the thermal pixels of that vehicle end up), dilated by `cfg.moving_dilate_px` as a Euclidean
    disk. Unmatched and copied boxes are not masked.

    Returns:
        `(mask (1, H, W) bool, number of moving vehicles)`.
    """
    flags = moving_flags(matches, T, cfg.moving_min_residual_px)
    polygons = []
    for k in np.flatnonzero(flags):
        polygons.append(rgb_boxes[matches.rgb_idx[k]].vertices)
        polygons.append(apply_transform(T, thermal_boxes[matches.thermal_idx[k]].vertices))
    mask = rasterize(polygons, size)
    if mask.any() and cfg.moving_dilate_px > 0:
        mask = ndimage.distance_transform_edt(~mask) <= cfg.moving_dilate_px
    return torch.from_numpy(mask)[None], int(flags.sum())


# --- independent edge check ----------------------------------------------------------------------


def rgb_edge_luma(rgb_linear: torch.Tensor) -> np.ndarray:
    """`(H, W)` float32 sRGB-coded luma of a linear `(3, H, W)` RGB image, for `edge_check`.

    Edges are compared in a display-like space, like the thermal codes (docs/conventions.md
    section 1), so a shadowed edge counts as much as a sunlit one.
    """
    return linear_to_srgb(luminance(rgb_linear))[0].numpy().astype(np.float32)


def gradient_magnitude(img: np.ndarray, sigma: float) -> np.ndarray:
    """Sobel gradient magnitude after a Gaussian blur of `sigma` px (edge-replicated)."""
    smooth = ndimage.gaussian_filter(img.astype(np.float32), sigma, mode="nearest")
    gy = ndimage.sobel(smooth, axis=0, mode="nearest")
    gx = ndimage.sobel(smooth, axis=1, mode="nearest")
    return np.hypot(gx, gy)


def _pearson(a: np.ndarray, b_centered: np.ndarray, b_norm: float) -> float:
    a = a - a.mean()
    denom = float(np.sqrt((a * a).sum())) * b_norm
    return float((a * b_centered).sum() / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class EdgeResult:
    """Outcome of `edge_check`. `best_dx` / `best_dy` is the integer shift (columns, rows) to
    apply to the warped thermal, within `+-search_px`, that best lines its edges up with RGB."""

    before: float  # gradient-magnitude correlation with the unwarped thermal
    after: float  # ... with the warped thermal, same pixels
    gain: float  # after - before
    best_dx: int
    best_dy: int
    passed: bool


def edge_check(
    thermal: torch.Tensor, rgb_luma: np.ndarray, T: np.ndarray, cfg: AlignConfig
) -> EdgeResult:
    """Independent check on the pixels: do the thermal edges line up with the RGB edges?

    Normalized (Pearson) correlation of gradient magnitudes (light blur, then Sobel) between the
    thermal and the RGB luma, over the pixels where the warp is valid (and a margin the blur and
    Sobel reach into), before the warp (identity) and after. With near-identity warps "must
    improve" is noisy, so the gain is logged, and the warped pair is also shifted by every
    integer offset within `+-grad_search_px` to find where its edges fit best. Accepted when
    `gain >= grad_min_gain` and the best shift is at most `grad_max_best_shift_px` on each axis.

    Args:
        thermal: `(1, H, W)` float, codes / 255, the unwarped thermal crop.
        rgb_luma: `(H, W)` float32, from `rgb_edge_luma`.
        T: 3x3 thermal -> RGB.
    """
    plane = thermal.detach().cpu().numpy()[0]
    if plane.shape != rgb_luma.shape:
        raise ValueError(f"thermal {plane.shape} and RGB {rgb_luma.shape} differ in size")
    warped, valid = warp_thermal(thermal, T, rgb_luma.shape)
    sigma = cfg.grad_blur_sigma
    margin = int(math.ceil(3 * sigma)) + 1  # what the blur and the Sobel kernel reach
    keep = ndimage.binary_erosion(
        valid[0].numpy(), structure=np.ones((2 * margin + 1,) * 2, bool), border_value=1
    )
    g_rgb = gradient_magnitude(rgb_luma, sigma)
    g_before = gradient_magnitude(plane, sigma)
    g_after = gradient_magnitude(warped[0].numpy(), sigma)
    if int(keep.sum()) < _MIN_EDGE_PIXELS:
        return EdgeResult(_NAN, _NAN, _NAN, 0, 0, False)

    def correlation(a: np.ndarray, b: np.ndarray, where: np.ndarray) -> float:
        bs = b[where]
        bc = bs - bs.mean()
        return _pearson(a[where], bc, float(np.sqrt((bc * bc).sum())))

    before, after = correlation(g_before, g_rgb, keep), correlation(g_after, g_rgb, keep)

    s = cfg.grad_search_px
    best_dx = best_dy = 0
    if s > 0:
        h, w = g_rgb.shape
        ref, where = g_rgb[s : h - s, s : w - s], keep[s : h - s, s : w - s]
        ref_sel = ref[where]
        ref_c = ref_sel - ref_sel.mean()
        ref_norm = float(np.sqrt((ref_c * ref_c).sum()))
        offsets = sorted(
            ((dy, dx) for dy in range(-s, s + 1) for dx in range(-s, s + 1)),
            key=lambda o: (max(abs(o[0]), abs(o[1])), o[0] ** 2 + o[1] ** 2, o),
        )
        best = -math.inf
        for dy, dx in offsets:  # moves the warped content by (+dx, +dy)
            cand = g_after[s - dy : h - s - dy, s - dx : w - s - dx]
            score = _pearson(cand[where], ref_c, ref_norm)
            if score > best + 1e-12:
                best, best_dx, best_dy = score, dx, dy
    gain = after - before
    passed = bool(
        gain >= cfg.grad_min_gain - _EPS
        and max(abs(best_dx), abs(best_dy)) <= cfg.grad_max_best_shift_px
    )
    return EdgeResult(before, after, gain, best_dx, best_dy, passed)


# --- per-pair verdict, log and I/O ---------------------------------------------------------------


@dataclass(frozen=True)
class AlignRow:
    """One row of the align log: every metric, the fitted transform and the verdict.

    The top-level metrics (`n_inliers` ... `shift_px`) belong to the model used (`model`), which is
    the similarity unless the homography fallback passed. `hom_*` are the homography attempt when
    one was made (`hom_tried`), else NaN. `edge_*` belong to `transform`, NaN when the edge check
    did not run. `transform` is 6 numbers (the first two rows of a similarity) or 9 (a homography),
    so `write` never refits. `reason` is `ok` or the first failed check; `accepted` is
    `reason == ok`.
    """

    image_id: str
    n_thermal: int
    n_rgb: int
    n_copied: int
    labels_identical: bool
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    median_residual: float
    median_residual_all: float
    model: str
    scale: float
    rotation_deg: float
    shift_px: float
    transform: str
    hom_tried: bool
    hom_inlier_ratio: float
    hom_median_residual: float
    hom_scale: float
    hom_rotation_deg: float
    hom_shift_px: float
    n_moving: int
    n_unmatched_thermal: int
    n_unmatched_rgb: int
    edge_before: float
    edge_after: float
    edge_gain: float
    edge_dx: int
    edge_dy: int
    accepted: bool
    reason: str


def format_transform(model: str, T: np.ndarray | None) -> str:
    """Space-separated numbers, 10 significant digits: 6 for a similarity, 9 for a homography."""
    if model == "none" or T is None:
        return ""
    values = T[:2].ravel() if model == "similarity" else T.ravel()
    return " ".join(f"{v:.10g}" for v in values)


def parse_transform(model: str, text: str) -> np.ndarray | None:
    """Inverse of `format_transform`."""
    if model == "none" or not text.strip():
        return None
    values = np.array([float(v) for v in text.split()])
    if model == "similarity":
        return np.vstack([values.reshape(2, 3), [0.0, 0.0, 1.0]])
    return values.reshape(3, 3)


def transform_of(row: AlignRow) -> np.ndarray | None:
    """The 3x3 thermal -> RGB transform of a log row, or None when there was no fit."""
    return parse_transform(row.model, row.transform)


def final_reason(result: AlignResult, edge: EdgeResult | None, cfg: AlignConfig) -> str:
    """The pair's verdict: the geometry's reason, else `edge_check` if the independent check is on
    and failed (or could not run), else `ok`."""
    if result.reason != "ok":
        return result.reason
    if cfg.grad_enabled and (edge is None or not edge.passed):
        return "edge_check"
    return "ok"


def row_from_result(
    image_id: str, result: AlignResult, edge: EdgeResult | None, cfg: AlignConfig
) -> AlignRow:
    """Flatten an `AlignResult` and its `EdgeResult` into a log row."""
    m, hom = result.metrics, result.hom
    reason = final_reason(result, edge, cfg)
    return AlignRow(
        image_id=image_id,
        n_thermal=result.matches.n_thermal,
        n_rgb=result.matches.n_rgb,
        n_copied=result.matches.n_copied,
        labels_identical=result.matches.labels_identical,
        n_matches=m.n_matches,
        n_inliers=m.n_inliers,
        inlier_ratio=m.inlier_ratio,
        median_residual=m.median_residual,
        median_residual_all=m.median_residual_all,
        model=result.model,
        scale=m.scale,
        rotation_deg=m.rotation_deg,
        shift_px=m.shift_px,
        transform=format_transform(result.model, result.T),
        hom_tried=hom is not None,
        hom_inlier_ratio=hom.inlier_ratio if hom else _NAN,
        hom_median_residual=hom.median_residual if hom else _NAN,
        hom_scale=hom.scale if hom else _NAN,
        hom_rotation_deg=hom.rotation_deg if hom else _NAN,
        hom_shift_px=hom.shift_px if hom else _NAN,
        n_moving=result.n_moving,
        n_unmatched_thermal=result.n_unmatched_thermal,
        n_unmatched_rgb=result.n_unmatched_rgb,
        edge_before=edge.before if edge else _NAN,
        edge_after=edge.after if edge else _NAN,
        edge_gain=edge.gain if edge else _NAN,
        edge_dx=edge.best_dx if edge else 0,
        edge_dy=edge.best_dy if edge else 0,
        accepted=reason == "ok",
        reason=reason,
    )


def _failed_row(image_id: str, reason: str) -> AlignRow:
    return AlignRow(
        image_id,
        0,
        0,
        0,
        False,
        0,
        0,
        _NAN,
        _NAN,
        _NAN,
        "none",
        _NAN,
        _NAN,
        _NAN,
        "",
        False,
        _NAN,
        _NAN,
        _NAN,
        _NAN,
        _NAN,
        0,
        0,
        0,
        _NAN,
        _NAN,
        _NAN,
        0,
        0,
        False,
        reason,
    )


def align_pair(layout: dv.DroneVehicleLayout, cfg: AlignConfig, split: str, stem: str) -> AlignRow:
    """Align one pair from the raw files: parse both label files, fit, and run the edge check.

    The edge check runs for every pair that has a fitted transform, accepted or not, so the audit
    and the sensitivity table can see how the rejected pairs would have done. A pair that cannot
    be read gets the reason `read_error` instead of raising (a `run_parallel` job).
    """
    image_id = dv.image_id(split, stem)
    try:
        thermal_ann = dv.parse_annotation(
            layout.thermal_label(split, stem), layout.class_map, layout.border
        )
        rgb_ann = dv.parse_annotation(
            layout.rgb_label(split, stem), layout.class_map, layout.border
        )
    except Exception as exc:
        logger.warning(f"{image_id}: labels: {exc}")
        return _failed_row(image_id, "read_error")
    result = align_boxes(thermal_ann.boxes, rgb_ann.boxes, cfg, image_id, layout.crop_size)
    edge = None
    if cfg.grad_enabled and result.T is not None:
        try:
            thermal = dv.crop_border(
                dv.read_thermal8(layout.thermal_image(split, stem)),
                layout.border,
                layout.frame_size,
            )
            rgb = dv.crop_border(
                dv.read_rgb8(layout.rgb_image(split, stem)), layout.border, layout.frame_size
            )
        except Exception as exc:
            logger.warning(f"{image_id}: images: {exc}")
            return _failed_row(image_id, "read_error")
        edge = edge_check(thermal.float() / 255, rgb_edge_luma(srgb8_to_linear(rgb)), result.T, cfg)
    return row_from_result(image_id, result, edge, cfg)


_TYPES = {
    "str": str,
    "int": int,
    "float": lambda v: float(v) if v != "" else _NAN,
    "bool": lambda v: v == "1",
}


def write_align_log(path: str | Path, rows: Sequence[AlignRow]) -> None:
    """Write the committed align log, one row per pair in the order given: floats to 3 decimals
    (NaN empty), the transform to 10 significant digits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def cell(value: object) -> object:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.3f}"
        return value

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([field.name for field in fields(AlignRow)])
        for row in rows:
            writer.writerow([cell(v) for v in astuple(row)])


def read_align_log(path: str | Path) -> dict[str, AlignRow]:
    """Read the align log into `{image_id: AlignRow}`, in file order."""
    types = {field.name: _TYPES[field.type] for field in fields(AlignRow)}
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = [
            AlignRow(**{name: types[name](value) for name, value in record.items()})
            for record in csv.DictReader(f)
        ]
    return {row.image_id: row for row in rows}


def _metrics_of(row: AlignRow) -> GeometryMetrics:
    return GeometryMetrics(
        n_matches=row.n_matches,
        labels_identical=row.labels_identical,
        n_inliers=row.n_inliers,
        inlier_ratio=row.inlier_ratio,
        median_residual=row.median_residual,
        median_residual_all=row.median_residual_all,
        scale=row.scale,
        rotation_deg=row.rotation_deg,
        shift_px=row.shift_px,
    )


def _hom_metrics_of(row: AlignRow) -> GeometryMetrics:
    return GeometryMetrics(
        n_matches=row.n_matches,
        labels_identical=row.labels_identical,
        n_inliers=round(row.hom_inlier_ratio * row.n_matches) if row.hom_tried else 0,
        inlier_ratio=row.hom_inlier_ratio,
        median_residual=row.hom_median_residual,
        median_residual_all=_NAN,
        scale=row.hom_scale,
        rotation_deg=row.hom_rotation_deg,
        shift_px=row.hom_shift_px,
    )


def reevaluate(row: AlignRow, cfg: AlignConfig) -> str:
    """The verdict a logged pair would get under other thresholds, without refitting.

    Applies `cfg` to the logged metrics: the model used first, then (when a homography was tried)
    the homography's, then the logged edge check against `cfg`'s edge thresholds. It is exact for
    thresholds that only move the acceptance of the logged fits. It cannot know about a homography
    that a tighter residual threshold would have triggered but the run never tried, nor re-match
    boxes (`exclude_copied_boxes`, `max_match_dist_px`) or re-fit (RANSAC thresholds); those need
    a new `align` run.
    """
    failed = failed_checks(_metrics_of(row), cfg)
    if failed and row.hom_tried and not failed_checks(_hom_metrics_of(row), cfg):
        failed = []
    if failed:
        return failed[0]
    if cfg.grad_enabled:
        shift = max(abs(row.edge_dx), abs(row.edge_dy))
        if not (row.edge_gain >= cfg.grad_min_gain - _EPS and shift <= cfg.grad_max_best_shift_px):
            return "edge_check"
    return "ok"


def config_fingerprint(cfg: AlignConfig) -> str:
    """A short stable hash of every alignment threshold, to notice a changed config that would
    make an existing align log stale."""
    return hashlib.sha1(repr(cfg).encode("utf-8")).hexdigest()[:12]
