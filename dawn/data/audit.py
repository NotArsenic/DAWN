"""The alignment audit (plan step 11): contact sheets with thermal edges drawn over the RGB, and
histograms of the fit metrics.

An aligned pair is judged by eye: the edges of the **warped** thermal, drawn in magenta over the
RGB crop, should sit on the RGB edges of the same objects. The sheets show accepted pairs and
rejected ones, the rejected stratified by reason so a rare reason is not drowned by `few_matches`.
Display only: the thumbnails average sRGB codes, which is fine for looking at and is never used
for training data.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from dawn.data import align
from dawn.data import dronevehicle as dv
from dawn.data.resample import area_downsample
from dawn.data.sheets import run_parallel, tile_sheet

logger = logging.getLogger(__name__)

TILE_BOX = (256, 320)  # audit thumbnails (height, width): half the 640x512 crop
EDGE_COLOR = (255, 0, 255)
# Chart palette: the dataviz reference instance (surface, inks, categorical slots 1 and 2).
_SURFACE, _GRID, _INK, _INK_2 = "#fcfcfb", "#e4e3de", "#0b0b0b", "#52514e"
_ACCEPTED, _REJECTED = "#2a78d6", "#eb6834"


def edge_overlay(
    rgb8: torch.Tensor,
    thermal: torch.Tensor,
    valid: torch.Tensor | None = None,
    keep_fraction: float = 0.04,
    blur_sigma: float = 1.0,
    grow: bool = False,
) -> torch.Tensor:
    """Draw the strongest thermal edges over an RGB image.

    The edge pixels are the `keep_fraction` of valid pixels with the largest blurred Sobel
    gradient magnitude of the thermal; one pixel wide unless `grow` (then grown by one pixel).
    Drawn at full resolution, so after the tile is area-averaged down the lines stay thin and
    blend into the RGB instead of hiding what is under them.

    Args:
        rgb8: `(3, H, W)` uint8 sRGB.
        thermal: `(1, H, W)` float, the warped thermal (code / 255).
        valid: `(1, H, W)` bool; edges are drawn only where True. Default: everywhere.

    Returns:
        `(3, H, W)` uint8 copy of `rgb8` with the edge pixels set to magenta.
    """
    gradient = align.gradient_magnitude(thermal[0].numpy(), blur_sigma)
    keep = np.ones_like(gradient, dtype=bool) if valid is None else valid[0].numpy()
    if not keep.any():
        return rgb8.clone()
    threshold = np.quantile(gradient[keep], 1.0 - keep_fraction)
    edges = (gradient >= threshold) & keep
    if grow:
        edges = ndimage.binary_dilation(edges, structure=np.ones((3, 3), bool)) & keep
    out = rgb8.clone()
    out[:, torch.from_numpy(edges)] = torch.tensor(EDGE_COLOR, dtype=torch.uint8)[:, None]
    return out


def _thumb8(img8: torch.Tensor, box: tuple[int, int]) -> torch.Tensor:
    """Area-averaged uint8 thumbnail fitting `box = (height, width)` (display only)."""
    h, w = img8.shape[-2:]
    scale = min(1.0, box[0] / h, box[1] / w)
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    return area_downsample(img8.float(), size).round().clamp(0, 255).to(torch.uint8)


def audit_tile(
    layout: dv.DroneVehicleLayout, row: align.AlignRow, box: tuple[int, int] = TILE_BOX
) -> torch.Tensor:
    """`(3, h, w)` uint8 audit tile of one logged pair: the thermal is warped with the logged
    transform (the unwarped thermal when the pair had no fit) and its edges drawn over the RGB."""
    split, stem = dv.parse_image_id(row.image_id)
    rgb8 = dv.crop_border(
        dv.read_rgb8(layout.rgb_image(split, stem)), layout.border, layout.frame_size
    )
    thermal8 = dv.crop_border(
        dv.read_thermal8(layout.thermal_image(split, stem)), layout.border, layout.frame_size
    )
    transform = align.transform_of(row)
    thermal = thermal8.float() / 255
    if transform is None:
        valid = torch.ones(1, *layout.crop_size, dtype=torch.bool)
    else:
        thermal, valid = align.warp_thermal(thermal, transform, layout.crop_size)
    return _thumb8(edge_overlay(rgb8, thermal, valid), box)


def _caption(row: align.AlignRow) -> str:
    gain = "" if math.isnan(row.edge_gain) else f" g{row.edge_gain:+.2f}"
    residual = "" if math.isnan(row.median_residual) else f" r{row.median_residual:.1f}"
    return f"{row.image_id} {row.reason}{residual}{gain}"


def save_audit_sheets(
    layout: dv.DroneVehicleLayout,
    accepted: Sequence[align.AlignRow],
    rejected: Sequence[align.AlignRow],
    out_dir: str | Path,
    per_sheet: int = 20,
    workers: int = 1,
) -> list[Path]:
    """Contact sheets `accepted_NN.jpg` and `rejected_NN.jpg`, `per_sheet` tiles each, 4 per row.

    Each caption is `image_id reason r<median residual px> g<edge gain>`. The rows are drawn in the
    order given, so pass them already sampled (`report.stratified_sample`).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rows in (("accepted", accepted), ("rejected", rejected)):
        for start in range(0, len(rows), per_sheet):
            page = list(rows[start : start + per_sheet])
            tiles = run_parallel(audit_tile, [(layout, row) for row in page], workers)
            title = (
                f"{name} {start + 1}-{start + len(page)} of {len(rows)}: thermal edges (magenta) "
                "over the RGB; r = median inlier residual px, g = edge-correlation gain"
            )
            path = out_dir / f"{name}_{start // per_sheet + 1:02d}.jpg"
            tile_sheet(tiles, [_caption(r) for r in page], title, columns=4, box=TILE_BOX).save(
                path, quality=90
            )
            written.append(path)
    return written


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=8)


def plot_fit_histograms(
    rows: Sequence[align.AlignRow], cfg: align.AlignConfig, out_path: str | Path
) -> None:
    """Histograms of the fit metrics (median inlier residual, scale, rotation, shift, edge-check
    gain) for accepted and rejected pairs that have a fit, with the acceptance thresholds marked.

    The rejected pairs are cut off at the plot range, so a few wild fits do not squash the rest;
    the count outside the range is in each panel's title.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fitted = [r for r in rows if r.model != "none"]
    panels = [
        ("median_residual", "median inlier residual (px)", (0, 6), [cfg.max_median_residual_px]),
        ("scale", "scale", (0.9, 1.1), [1 - cfg.max_scale_deviation, 1 + cfg.max_scale_deviation]),
        ("rotation_deg", "rotation (deg)", (-4, 4), [-cfg.max_rotation_deg, cfg.max_rotation_deg]),
        ("shift_px", "shift of the frame centre (px)", (0, 60), [cfg.max_shift_px]),
        ("edge_gain", "edge-correlation gain", (-0.3, 0.8), [cfg.grad_min_gain]),
    ]
    fig = Figure(figsize=(11, 6.2), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3).ravel()
    for ax, (field, label, (lo, hi), limits) in zip(axes, panels):
        _style(ax)
        bins = np.linspace(lo, hi, 41)
        outside = 0
        for accepted, colour, name in (
            (True, _ACCEPTED, "accepted"),
            (False, _REJECTED, "rejected"),
        ):
            values = np.array([getattr(r, field) for r in fitted if r.accepted == accepted], float)
            values = values[np.isfinite(values)]
            outside += int(((values < lo) | (values > hi)).sum())
            ax.hist(
                values,
                bins=bins,
                histtype="stepfilled",
                alpha=0.45,
                color=colour,
                label=f"{name} ({len(values)})",
            )
        for limit in limits:
            ax.axvline(limit, color=_INK_2, linewidth=1.0, linestyle="--")
        ax.set_title(
            f"{label}" + (f"  ({outside} outside range)" if outside else ""),
            loc="left",
            color=_INK,
            fontsize=9,
        )
        ax.set_ylabel("pairs", color=_INK_2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[5].axis("off")
    axes[5].legend(handles, labels, loc="center", frameon=False, labelcolor=_INK, fontsize=10)
    axes[5].text(
        0.5,
        0.2,
        "dashed lines: acceptance thresholds",
        ha="center",
        color=_INK_2,
        fontsize=9,
        transform=axes[5].transAxes,
    )
    fig.suptitle("Alignment fit metrics", color=_INK, fontsize=11, x=0.01, ha="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)


# --- dedup examples ------------------------------------------------------------------------------


def pair_tiles(
    layout: dv.DroneVehicleLayout, image_id: str, matched_id: str, box: tuple[int, int] = TILE_BOX
) -> list[torch.Tensor]:
    """Four `(3, h, w)` uint8 tiles for one cross-split match: the candidate's RGB, the matched
    train pair's RGB, the candidate's thermal and the matched pair's thermal (raw crops)."""
    tiles = []
    for kind in ("rgb", "thermal"):
        for pair_id in (image_id, matched_id):
            split, stem = dv.parse_image_id(pair_id)
            if kind == "rgb":
                img = dv.read_rgb8(layout.rgb_image(split, stem))
                img = dv.crop_border(img, layout.border, layout.frame_size)
            else:
                img = dv.read_thermal8(layout.thermal_image(split, stem))
                img = dv.crop_border(img, layout.border, layout.frame_size).expand(3, -1, -1)
            tiles.append(_thumb8(img.contiguous(), box))
    return tiles


def pick_dedup_examples(rows: Sequence, max_hamming: int, count: int) -> list:
    """Cross-split matches just under and just over the threshold, half each when available.

    Just under: distance in `[max_hamming - 2, max_hamming]`, the closest to the threshold first.
    Just over: distance in `(max_hamming, max_hamming + 3]`, the closest first. When fewer than
    `count` fall in those windows (as when no val / test pair comes anywhere near a train pair),
    the sheet is filled up with the closest remaining matches, so it always shows the nearest
    cross-split pairs. `rows` are `dawn.data.dedup.DedupRow`s.
    """
    matched = [r for r in rows if r.matched_id]
    under = sorted(
        (r for r in matched if max_hamming - 2 <= r.distance <= max_hamming),
        key=lambda r: (-r.distance, r.image_id),
    )
    over = sorted(
        (r for r in matched if max_hamming < r.distance <= max_hamming + 3),
        key=lambda r: (r.distance, r.image_id),
    )
    half = count // 2
    picked = (under[: max(half, count - len(over))] + over[: max(half, count - len(under))])[:count]
    if len(picked) < count:
        chosen = {r.image_id for r in picked}
        rest = sorted(
            (r for r in matched if r.image_id not in chosen), key=lambda r: (r.distance, r.image_id)
        )
        picked += rest[: count - len(picked)]
    return picked


def save_dedup_sheet(
    layout: dv.DroneVehicleLayout,
    rows: Sequence,
    max_hamming: int,
    out_path: str | Path,
    count: int = 24,
    workers: int = 1,
) -> Path | None:
    """One row per example: candidate RGB | matched train RGB | candidate thermal | matched
    thermal, with the Hamming distances in the captions. Shows what sits just under and just over
    `max_hamming`, topped up with the closest matches overall (`pick_dedup_examples`). Returns None
    only when there is no match at all."""
    picked = pick_dedup_examples(rows, max_hamming, count)
    if not picked:
        return None
    jobs = [(layout, r.image_id, r.matched_id) for r in picked]
    tiles = [t for group in run_parallel(pair_tiles, jobs, workers) for t in group]
    labels = []
    for r in picked:
        verdict = "DUPLICATE" if r.distance <= max_hamming else "kept"
        labels += [
            f"{r.image_id} d={r.distance} {verdict}",
            f"train: {r.matched_id}",
            f"rgb {r.d_rgb} bits",
            f"thermal {r.d_thermal} bits",
        ]
    title = (
        "closest cross-split matches; d = worse modality in bits, "
        f"dropped when d <= {max_hamming}; "
        "columns: candidate RGB, train RGB, candidate thermal, train thermal"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_sheet(tiles, labels, title, columns=4, box=TILE_BOX).save(out_path, quality=90)
    return out_path


# --- the usable pairs ----------------------------------------------------------------------


def plot_usable_luminance(
    stats: Sequence,
    day_log: Mapping[str, tuple[bool, str]],
    usable_ids: Iterable[str],
    min_lum_median: float,
    out_path: str | Path,
    bin_log10: float = 0.05,
) -> dict[str, tuple[int, int]]:
    """Histogram of log10(median RGB luminance) of the day pairs, one panel per split, stacked:
    the usable pairs (in the final manifests) and the day pairs that were rejected later.

    It shows whether the filters are luminance-blind or lose the dark end disproportionately, and
    where the cutoff sits relative to the pairs the model will see.

    Args:
        stats: `dawn.data.day_filter.PairStats` rows.
        day_log: `{image_id: (is_day, reason)}` from the day log.
        usable_ids: Image IDs in the final manifests.
        min_lum_median: The day cutoff, drawn as a line.

    Returns:
        `{split: (usable, day pairs)}`.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    usable = set(usable_ids)
    day = [r for r in stats if day_log.get(r.image_id, (False, ""))[0]]
    splits = [s for s in dv.SPLITS if any(r.split == s for r in day)]
    log10 = lambda r: math.log10(max(r.lum_median, 1e-4))  # noqa: E731
    cut = math.log10(min_lum_median)
    lo = math.floor(min(log10(r) for r in day) / bin_log10) * bin_log10
    hi = math.ceil(max(log10(r) for r in day) / bin_log10) * bin_log10 + bin_log10
    bins = np.arange(lo, hi + 1e-9, bin_log10)
    fig = Figure(figsize=(9, 0.9 + 2.3 * len(splits)), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(splits), 1, sharex=True, squeeze=False)[:, 0]
    counts = {}
    for ax, split in zip(axes, splits):
        rows = [r for r in day if r.split == split]
        kept = [log10(r) for r in rows if r.image_id in usable]
        lost = [log10(r) for r in rows if r.image_id not in usable]
        _style(ax)
        ax.hist(
            [kept, lost],
            bins=bins,
            stacked=True,
            color=[_ACCEPTED, _REJECTED],
            label=[f"usable ({len(kept)})", f"day pair rejected later ({len(lost)})"],
            edgecolor=_SURFACE,
            linewidth=0.6,
        )
        ax.axvline(cut, color=_INK_2, linewidth=1.0, linestyle="--")
        ax.set_ylabel("pairs", color=_INK_2, fontsize=9)
        ax.set_title(
            f"{split}: {len(kept)} usable of {len(rows)} day pairs",
            loc="left",
            color=_INK,
            fontsize=10,
        )
        ax.legend(frameon=False, fontsize=8, labelcolor=_INK, loc="upper left")
        counts[split] = (len(kept), len(rows))
    axes[-1].set_xlabel(
        f"log10(median linear luminance of the cropped RGB); dashed: cutoff {min_lum_median:g}",
        color=_INK_2,
        fontsize=9,
    )
    fig.suptitle("Luminance of the usable pairs", color=_INK, fontsize=11, x=0.01, ha="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)
    return counts


def pick_usable(lum_by_id: Mapping[str, float], how: str, count: int, seed: int = 0) -> list[str]:
    """IDs of usable pairs to look at: `how="darkest"` is the `count` lowest median luminance
    (where dusk and night hide), `"random"` a seeded sample, sorted darkest first."""
    if how == "darkest":
        return sorted(lum_by_id, key=lambda i: (lum_by_id[i], i))[:count]
    if how == "random":
        ids = sorted(lum_by_id)
        chosen = random.Random(f"{seed}/usable").sample(ids, min(count, len(ids)))
        return sorted(chosen, key=lambda i: (lum_by_id[i], i))
    raise ValueError(f"how must be 'darkest' or 'random', got {how!r}")


def usable_tiles(
    layout: dv.DroneVehicleLayout, image_id: str, box: tuple[int, int] = (192, 240)
) -> list[torch.Tensor]:
    """`[rgb, thermal]` `(3, h, w)` uint8 thumbnails of one written pair (the aligned PNGs, so
    what the model will actually be given)."""
    from PIL import Image

    with Image.open(layout.aligned_path(image_id, "rgb")) as img:
        rgb = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1)
    with Image.open(layout.aligned_path(image_id, "thermal")) as img:
        thermal = torch.from_numpy(np.asarray(img.convert("L")).copy())[None].expand(3, -1, -1)
    return [_thumb8(rgb.contiguous(), box), _thumb8(thermal.contiguous(), box)]


def save_usable_sheet(
    layout: dv.DroneVehicleLayout,
    image_ids: Sequence[str],
    lum_by_id: Mapping[str, float],
    out_path: str | Path,
    title: str,
    workers: int = 1,
) -> Path | None:
    """A sheet of usable pairs as RGB | thermal, two pairs per row, captioned with the ID and
    log10 luminance. Returns None when `image_ids` is empty."""
    if not image_ids:
        return None
    pairs = run_parallel(usable_tiles, [(layout, i) for i in image_ids], workers)
    tiles = [tile for pair in pairs for tile in pair]
    labels = []
    for i in image_ids:
        lum = math.log10(max(lum_by_id.get(i, float("nan")), 1e-4))
        labels += [f"{i} log10 L {lum:+.2f}", "thermal (warped)"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_sheet(tiles, labels, title, columns=4, box=(192, 240)).save(out_path, quality=90)
    return out_path
