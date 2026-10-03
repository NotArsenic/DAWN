"""The three offline phases behind scripts/prepare_visdrone.py.

    1. stats   slow, once: per raw image, native size, luminance percentiles, box count, noise
               level and highlight ratio -> the stats CSV.
    2. split   instant, rerun freely: stats CSV + the hand labels -> manifests, a luminance
               histogram, a noise-vs-highlight scatter and one contact sheet per manifest.
               Before labelling, `save_sequence_sheets` and `write_labels_template` give one row
               per video sequence to label.
    3. resize  512-px copies (long side), area-averaged in linear space, saved as 8-bit sRGB PNG.

Phases 1 and 3 do not depend on the labels, so they never need rerunning when those change.
Every downsample is `area_downsample` on linear values (docs/conventions.md section 4a).
"""

from __future__ import annotations

import csv
import logging
import math
import multiprocessing
import random
import statistics
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from dawn.data.color import linear_to_srgb8, luminance, srgb8_to_linear
from dawn.data.resample import area_downsample, fit_long_side
from dawn.data.visdrone import (
    LABELS,
    SUBSETS,
    BoxFilter,
    ImageStats,
    VisDroneLayout,
    is_still_collection,
    manifest_path,
    read_annotations,
    read_srgb8,
    sequence_id,
    split_rank,
    write_manifest,
)

logger = logging.getLogger(__name__)

STATS_LONG_SIDE = 256  # phase 1 measures luminance at about the DINO input scale
LOWRES_LONG_SIDE = 512  # phase 3
THUMB_BOX = (144, 192)  # contact-sheet thumbnails fit inside (height, width)
LOG_LUMINANCE_FLOOR = 1e-4  # same floor as the guidance map (docs/conventions.md section 4)
# Noise is measured where the local mean luminance is in this range (about sRGB codes 3 to 249):
# crushed blacks and clipped highlights have no noise left.
NOISE_MEASURABLE = (1e-3, 0.95)
UNLABELLED = "unlabelled"

# Chart colors: the dataviz reference palette (light surface, text inks, categorical slots 1-3
# for day / night / dusk, neutral grays for exclude / unlabelled).
_SURFACE, _GRID, _INK, _INK_2, _SERIES = "#fcfcfb", "#e4e3de", "#0b0b0b", "#52514e", "#2a78d6"
_LABEL_COLORS = {"day": "#2a78d6", "night": "#eb6834", "dusk": "#1baf7a", "exclude": "#52514e"}
_UNLABELLED_COLOR = "#c3c2b7"


def _read_linear(path: Path, long_side: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Decode, linearize and area-downsample to `long_side`. Returns the image and native (W, H)."""
    srgb8 = read_srgb8(path)
    _, height, width = srgb8.shape
    small = area_downsample(srgb8_to_linear(srgb8), fit_long_side(height, width, long_side))
    return small, (width, height)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)  # scalable font, Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def _save_srgb8(img: torch.Tensor, path: Path, **save_kwargs) -> None:
    Image.fromarray(img.permute(1, 2, 0).numpy()).save(path, **save_kwargs)


def noise_level(img_linear: torch.Tensor) -> float:
    """Sensor-gain proxy: robust spread of a high-pass luminance residual, relative to brightness.

    Residual = luminance minus its 3x3 mean, scaled to unit gain for white noise, divided by the
    square root of the local mean, then 1.4826 * MAD. For Var = a*x + b that is about sqrt(a) at
    any exposure (plus b/x in the dark), so a night shot that auto-exposure made bright still
    reads as high gain. Crushed blacks and clipped highlights carry no noise and are skipped.

    Args:
        img_linear: `(3, H, W)` linear RGB at native resolution (downsampling averages noise away).

    Returns:
        The estimate, or NaN when fewer than 1,000 pixels are measurable.
    """
    y = luminance(img_linear).unsqueeze(0)  # (1, 1, H, W)
    local = F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    measurable = (local > NOISE_MEASURABLE[0]) & (local < NOISE_MEASURABLE[1])
    if int(measurable.sum()) < 1000:
        return float("nan")
    residual = (y - local) / math.sqrt(8 / 9)  # y - 3x3 mean has 8/9 of white noise's variance
    z = residual[measurable] / local[measurable].sqrt()
    return 1.4826 * (z - z.median()).abs().median().item()


def highlight_ratio(img_linear: torch.Tensor) -> float:
    """99.5th percentile over median luminance (median floored at 1e-4).

    Streetlights and headlights put a few very bright points on a dark median at night.

    Args:
        img_linear: `(3, H, W)` linear RGB at native resolution.
    """
    y = luminance(img_linear).flatten()
    high = y.kthvalue(max(1, math.ceil(0.995 * y.numel()))).values
    return (high / y.median().clamp(min=LOG_LUMINANCE_FLOOR)).item()


def compute_image_stats(
    layout: VisDroneLayout,
    split: str,
    image_id: str,
    box_filter: BoxFilter | None = None,
) -> ImageStats:
    """Phase 1 for one image: decode, linearize, area-downsample to 256 px, luminance.

    `lum_*` are the median, mean, 5th and 95th percentile (linear interpolation) of the
    luminance of the downsampled image. `num_boxes` counts boxes kept by `box_filter`'s label
    rules. `noise` and `highlight_ratio` are measured on the native image.
    """
    box_filter = box_filter or BoxFilter()
    native = srgb8_to_linear(read_srgb8(layout.raw_image(split, image_id)))
    _, height, width = native.shape
    small = area_downsample(native, fit_long_side(height, width, STATS_LONG_SIDE))
    lum = luminance(small).flatten()
    p05, median, p95 = torch.quantile(lum, torch.tensor([0.05, 0.5, 0.95])).tolist()
    annotations = read_annotations(layout.raw_annotation(split, image_id))
    return ImageStats(
        image_id=image_id,
        split=split,
        native_width=width,
        native_height=height,
        lum_median=median,
        lum_mean=lum.mean().item(),
        lum_p05=p05,
        lum_p95=p95,
        num_boxes=int(box_filter.keep_labels(annotations).sum()),
        noise=noise_level(native),
        highlight_ratio=highlight_ratio(native),
    )


def run_parallel(fn: Callable, jobs: Sequence, workers: int, desc: str = "") -> list:
    """`[fn(*job) for job in jobs]` on `workers` processes (in-process when `workers <= 1`).

    Results come back in job order. Each worker runs torch single-threaded, so `workers` equal
    to the number of cores uses every core without oversubscribing them. Logs progress.
    """
    jobs = list(jobs)
    total, results = len(jobs), []
    every = max(1, total // 20)

    def progress(done: int) -> None:
        if desc and (done % every == 0 or done == total):
            logger.info(f"{desc}: {done}/{total}")

    if workers <= 1 or total <= 1:
        for done, job in enumerate(jobs, 1):
            results.append(fn(*job))
            progress(done)
        return results
    # spawn, not fork: forking a process that has already started torch's thread pools can hang.
    context = multiprocessing.get_context("spawn")
    chunksize = max(1, min(32, total // (workers * 8)))
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_one_thread) as pool:
        for done, result in enumerate(pool.map(fn, *zip(*jobs), chunksize=chunksize), 1):
            results.append(result)
            progress(done)
    return results


def _one_thread() -> None:
    torch.set_num_threads(1)


def resolve_labels(
    stats: Iterable[ImageStats], labels: Mapping[str, str]
) -> dict[str, str | None]:
    """Hand labels (`visdrone.read_labels`) -> `{image_id: label or None}` for every image.

    An image key wins over its sequence key, and a blank label is as if the row were absent (an
    image row left blank falls back to its sequence). An image with no label is unlabelled (None)
    and goes into no manifest: being left out is the safe side.

    Raises:
        ValueError: A key that is no sequence or image in `stats` (a typo would otherwise silently
            label nothing), or a label outside `LABELS`.
    """
    rows = list(stats)
    image_ids = {row.image_id for row in rows}
    sequences = {sequence_id(image_id) for image_id in image_ids}
    for key, label in labels.items():
        if key not in image_ids and key not in sequences:
            raise ValueError(f"label key {key!r} is no sequence or image in the stats CSV")
        if label and label not in LABELS:
            raise ValueError(f"{key}: unknown label {label!r}, expected one of {LABELS}")
    return {
        row.image_id: labels.get(row.image_id) or labels.get(sequence_id(row.image_id)) or None
        for row in rows
    }


def write_manifests(
    stats: Iterable[ImageStats],
    image_labels: Mapping[str, str | None],
    manifest_dir: str | Path,
) -> dict[str, dict[str, int]]:
    """Phase 2: write `<manifest_dir>/<split>/<subset>.txt` for every split present in `stats`.

    One manifest per subset (day, night, dusk), written for every split even when empty, sorted
    by image ID. Excluded and unlabelled images are in none of them.

    Returns:
        Counts, `{split: {day, night, dusk, exclude, unlabelled: n}}`.
    """
    counts = {}
    for split, groups in _by_split_and_label(stats, image_labels).items():
        for subset in SUBSETS:
            ids = [r.image_id for r in groups[subset]]
            write_manifest(manifest_path(manifest_dir, split, subset), ids)
        counts[split] = {name: len(rows) for name, rows in groups.items()}
    return counts


def _log_luminance(lum: float) -> float:
    return math.log10(max(lum, LOG_LUMINANCE_FLOOR))


def _by_split_and_label(
    stats: Iterable[ImageStats], image_labels: Mapping[str, str | None]
) -> dict[str, dict[str, list[ImageStats]]]:
    """`{split: {label or "unlabelled": rows sorted by image_id}}`, every label present."""
    groups: dict[str, dict[str, list[ImageStats]]] = {}
    for row in sorted(stats, key=lambda r: (split_rank(r.split), r.split, r.image_id)):
        by_label = groups.setdefault(row.split, {name: [] for name in (*LABELS, UNLABELLED)})
        by_label[image_labels.get(row.image_id) or UNLABELLED].append(row)
    return groups


def _video_sequences(stats: Iterable[ImageStats]) -> list[tuple[str, list[ImageStats]]]:
    """Video sequences (still collections left out), darkest first by median lum_median, each with
    its frames sorted by lum_median."""
    sequences: dict[str, list[ImageStats]] = {}
    for row in stats:
        sequences.setdefault(sequence_id(row.image_id), []).append(row)
    videos = [
        (seq, sorted(rows, key=lambda r: (r.lum_median, r.image_id)))
        for seq, rows in sequences.items()
        if not is_still_collection(seq)
    ]
    return sorted(
        videos, key=lambda item: (statistics.median(r.lum_median for r in item[1]), item[0])
    )


def summarize_frames(rows: Sequence[ImageStats]) -> dict[str, str]:
    """Display strings for a group of frames: split(s), frame count, median log10 brightness,
    median noise level and median highlight ratio."""
    splits = sorted({r.split for r in rows}, key=split_rank)
    noise = [r.noise for r in rows if not math.isnan(r.noise)]
    return {
        "split": "+".join(splits),
        "frames": str(len(rows)),
        "log10_lum": f"{_log_luminance(statistics.median(r.lum_median for r in rows)):+.2f}",
        "noise": f"{statistics.median(noise):.3f}" if noise else "nan",
        "highlight_ratio": f"{statistics.median(r.highlight_ratio for r in rows):.0f}",
    }


def write_labels_template(stats: Iterable[ImageStats], path: str | Path) -> None:
    """Start the hand-label CSV: one row per video sequence (still collections are left out),
    darkest first, label blank, plus helper columns (sheet row, split, frames, brightness, noise,
    highlight ratio) in the same order as `save_sequence_sheets`.

    Raises:
        FileExistsError: `path` exists. Hand labels are never overwritten.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["key", "label", "note", "sheet_row", "split", "frames", "log10_lum", "noise",
               "highlight_ratio"]
    try:
        f = path.open("x", newline="", encoding="utf-8")  # exclusive create: never overwrites
    except FileExistsError:
        raise FileExistsError(f"{path} exists; hand labels are never overwritten") from None
    with f:
        writer = csv.DictWriter(f, columns, lineterminator="\n")
        writer.writeheader()
        for number, (seq, rows) in enumerate(_video_sequences(stats), 1):
            writer.writerow(
                {"key": seq, "label": "", "note": "", "sheet_row": number}
                | summarize_frames(rows)
            )


def save_sequence_sheets(
    layout: VisDroneLayout,
    stats: Sequence[ImageStats],
    out_dir: str | Path,
    per_page: int = 50,
    workers: int = 1,
) -> list[Path]:
    """Labelling sheets: one row per video sequence, darkest first, `per_page` rows per page.

    Each row shows the sequence's darkest, median and brightest frame (by lum_median) and prints
    the row number, sequence ID, split, frame count and median brightness, noise and highlight
    ratio. Row numbers match the `sheet_row` column of `write_labels_template`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = _video_sequences(stats)
    pages = []
    for start in range(0, len(videos), per_page):
        page = videos[start : start + per_page]
        picks = [pick_three(rows) for _, rows in page]
        jobs = [(layout, r.split, r.image_id) for frames in picks for r in frames]
        thumbs = iter(run_parallel(thumbnail, jobs, workers))
        title = (
            f"Video sequences {start + 1}-{start + len(page)} of {len(videos)}, darkest first. "
            "Each row: darkest, median and brightest frame. Label: day / night / dusk / exclude"
        )
        rows = [
            (start + number, seq, summarize_frames(frames), [next(thumbs) for _ in pick])
            for number, ((seq, frames), pick) in enumerate(zip(page, picks), 1)
        ]
        path = out_dir / f"sequences_{len(pages) + 1:02d}.jpg"
        _sequence_page(rows, title).save(path, quality=88)
        pages.append(path)
    return pages


def frames_for_key(stats: Iterable[ImageStats], key: str) -> list[ImageStats]:
    """The frames a label key covers, darkest first: every frame of a sequence, or one image.

    Raises:
        ValueError: `key` is no sequence or image in `stats`.
    """
    rows = [r for r in stats if r.image_id == key or sequence_id(r.image_id) == key]
    if not rows:
        raise ValueError(f"{key} is no sequence or image in the stats CSV")
    return sorted(rows, key=lambda r: (r.lum_median, r.image_id))


def preview_image(
    layout: VisDroneLayout, frames: Sequence[ImageStats], box: tuple[int, int] = (300, 400)
) -> Image.Image:
    """The frames side by side on a dark background, each fitted into `box` = (height, width)."""
    (box_h, box_w), pad = box, 8
    canvas = Image.new("RGB", (len(frames) * (box_w + pad) + pad, box_h + 2 * pad), (26, 26, 25))
    for index, row in enumerate(frames):
        thumb = thumbnail(layout, row.split, row.image_id, box)
        _, h, w = thumb.shape
        tile = Image.fromarray(thumb.permute(1, 2, 0).numpy())
        canvas.paste(tile, (pad + index * (box_w + pad) + (box_w - w) // 2, pad + (box_h - h) // 2))
    return canvas


def pick_three(rows: Sequence[ImageStats]) -> list[ImageStats]:
    """Darkest, median and brightest frame (rows sorted by lum_median), without repeats."""
    picks = [rows[0], rows[len(rows) // 2], rows[-1]]
    return [r for i, r in enumerate(picks) if r not in picks[:i]]


def _sequence_page(rows: Sequence[tuple], title: str) -> Image.Image:
    (cell_h, cell_w), text_w, title_h, pad = THUMB_BOX, 300, 34, 6
    width = text_w + 3 * (cell_w + pad) + pad
    height = title_h + len(rows) * (cell_h + pad) + pad
    sheet = Image.new("RGB", (width, height), (26, 26, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill=(235, 235, 230), font=_font(16))
    big, small = _font(22), _font(14)
    for index, (number, seq, summary, thumbs) in enumerate(rows):
        y0 = title_h + index * (cell_h + pad)
        draw.line([(0, y0 - pad // 2), (width, y0 - pad // 2)], fill=(60, 60, 58))
        draw.text((pad + 4, y0 + 6), f"#{number:03d}  {seq}", fill=(245, 245, 240), font=big)
        frames = f"{summary['frames']} frame{'' if summary['frames'] == '1' else 's'}"
        lines = [
            f"{summary['split']}, {frames}",
            f"log10 L {summary['log10_lum']}",
            f"noise {summary['noise']}   highlights {summary['highlight_ratio']}x",
        ]
        for line_index, line in enumerate(lines):
            draw.text((pad + 4, y0 + 40 + 22 * line_index), line, fill=(195, 194, 183), font=small)
        for column, thumb in enumerate(thumbs):
            _, h, w = thumb.shape
            tile = Image.fromarray(thumb.permute(1, 2, 0).numpy())
            sheet.paste(tile, (text_w + pad + column * (cell_w + pad) + (cell_w - w) // 2, y0))
    return sheet


def plot_log_luminance_histogram(
    stats: Sequence[ImageStats],
    image_labels: Mapping[str, str | None],
    out_path: str | Path,
) -> None:
    """Histogram of log10(lum_median), one panel per split, with label counts in the titles.

    Log scale on purpose: in linear units every night image piles up against zero.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    groups = _by_split_and_label(stats, image_labels)
    values = {
        split: [_log_luminance(r.lum_median) for rows in by_label.values() for r in rows]
        for split, by_label in groups.items()
    }
    everything = [v for vs in values.values() for v in vs]
    bins = np.arange(
        math.floor(min(everything) * 20) / 20, math.ceil(max(everything) * 20) / 20 + 0.05, 0.05
    )
    fig = Figure(figsize=(9, 0.6 + 2.2 * len(groups)), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(groups), 1, sharex=True, squeeze=False)[:, 0]
    for ax, (split, by_label) in zip(axes, groups.items()):
        _style(ax)
        ax.hist(values[split], bins=bins, color=_SERIES, edgecolor=_SURFACE, linewidth=0.8)
        ax.set_ylabel("images", color=_INK_2, fontsize=9)
        counts = " · ".join(f"{name} {len(rows)}" for name, rows in by_label.items())
        ax.set_title(f"{split}: {len(values[split])} images · {counts}", loc="left",
                     color=_INK, fontsize=10)
    axes[-1].set_xlabel(
        "log10(median linear luminance), 256-px linearized image", color=_INK_2, fontsize=9
    )
    fig.suptitle("VisDrone-DET median luminance", color=_INK, fontsize=11, x=0.01, ha="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=8)


def plot_feature_scatter(
    stats: Sequence[ImageStats],
    image_labels: Mapping[str, str | None],
    out_path: str | Path,
) -> None:
    """Noise level against highlight ratio, one point per image, both axes logarithmic.

    Coloured by hand label once any exist (unlabelled points in gray), by log10 brightness before
    that. Night means high gain and bright points on a dark median, so it should separate from
    dim day better than brightness alone.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = [r for r in stats if not math.isnan(r.noise)]
    fig = Figure(figsize=(9, 6.5), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    _style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    labelled = any(image_labels.get(r.image_id) for r in rows)
    if labelled:
        for name in (UNLABELLED, "exclude", "day", "dusk", "night"):  # gray first, underneath
            group = [r for r in rows if (image_labels.get(r.image_id) or UNLABELLED) == name]
            if group:
                ax.scatter(
                    [r.highlight_ratio for r in group], [r.noise for r in group],
                    s=9, alpha=0.75 if name != UNLABELLED else 0.4, linewidths=0.3,
                    color=_LABEL_COLORS.get(name, _UNLABELLED_COLOR), edgecolors=_SURFACE,
                    marker="x" if name == "exclude" else "o", label=f"{name} ({len(group)})",
                )
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK, markerscale=2)
        colouring = "coloured by hand label"
    else:
        rows = sorted(rows, key=lambda r: -r.lum_median)  # dark points on top
        points = ax.scatter(
            [r.highlight_ratio for r in rows], [r.noise for r in rows],
            c=[_log_luminance(r.lum_median) for r in rows], cmap="Blues_r", s=9, alpha=0.8,
            linewidths=0.3, edgecolors=_SURFACE,
        )
        bar = fig.colorbar(points, ax=ax, shrink=0.8)
        bar.set_label("log10 median luminance", color=_INK_2, fontsize=9)
        bar.ax.tick_params(colors=_INK_2, labelsize=8)
        colouring = "coloured by brightness (no hand labels yet)"
    ax.set_xlabel("highlight ratio: 99.5th percentile / median luminance", color=_INK_2, fontsize=9)
    ax.set_ylabel("noise level (about sqrt of sensor gain)", color=_INK_2, fontsize=9)
    ax.set_title(f"VisDrone-DET, {len(rows)} images, {colouring}", loc="left", color=_INK,
                 fontsize=11)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)


def save_contact_sheets(
    layout: VisDroneLayout,
    stats: Sequence[ImageStats],
    image_labels: Mapping[str, str | None],
    out_dir: str | Path,
    per_sheet: int = 50,
    seed: int = 0,
    workers: int = 1,
) -> list[Path]:
    """One contact sheet per non-empty manifest (split, subset): up to `per_sheet` random
    thumbnails, darkest first, each labelled with its log10(lum_median). Same seed, same sheet.
    Returns the written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split, by_label in _by_split_and_label(stats, image_labels).items():
        for subset in SUBSETS:
            rows = by_label[subset]
            if not rows:
                continue
            picked = random.Random(f"{seed}/{split}/{subset}").sample(
                rows, min(per_sheet, len(rows))
            )
            picked.sort(key=lambda r: (r.lum_median, r.image_id))
            jobs = [(layout, r.split, r.image_id) for r in picked]
            thumbs = run_parallel(thumbnail, jobs, workers)
            labels = [f"{_log_luminance(r.lum_median):+.2f}" for r in picked]
            title = (
                f"{split} / {subset}: {len(picked)} of {len(rows)}, darkest first; "
                "label = log10 median luminance"
            )
            path = out_dir / f"{split}_{subset}.jpg"
            _tile(thumbs, labels, title).save(path, quality=90)
            written.append(path)
    return written


def save_auto_day_sheet(
    layout: VisDroneLayout,
    stats: Sequence[ImageStats],
    image_ids: Sequence[str],
    out_dir: str | Path,
    count: int = 200,
    seed: int = 0,
    workers: int = 1,
) -> Path | None:
    """Contact sheet of `count` random auto-accepted day stills, for spot-checking by eye.

    Nothing hand-labelled the 99999xx collections, so the auto-day cutoffs
    (`dawn.data.splitting`) are the only thing standing between a night still and the daytime
    training pool. This sheet is how a person checks that: darkest first, each thumbnail
    labelled with its log10 median luminance, so a night frame that slipped through shows up at
    the top left rather than hiding in the middle. Same seed, same sheet.

    Returns:
        The written path, or None when `image_ids` is empty.
    """
    if not image_ids:
        return None
    by_id = {row.image_id: row for row in stats}
    rows = [by_id[image_id] for image_id in image_ids if image_id in by_id]
    picked = random.Random(f"{seed}/auto_day").sample(rows, min(count, len(rows)))
    picked.sort(key=lambda r: (r.lum_median, r.image_id))
    thumbs = run_parallel(
        thumbnail, [(layout, r.split, r.image_id) for r in picked], workers, desc="auto-day sheet"
    )
    labels = [f"{_log_luminance(r.lum_median):+.2f}" for r in picked]
    title = (
        f"auto-day stills: {len(picked)} of {len(rows)} accepted, darkest first; "
        "label = log10 median luminance. Any night frame here is a false accept."
    )
    path = Path(out_dir) / "auto_day.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    _tile(thumbs, labels, title).save(path, quality=90)
    return path


def thumbnail(
    layout: VisDroneLayout, split: str, image_id: str, box: tuple[int, int] = THUMB_BOX
) -> torch.Tensor:
    """`(3, h, w)` uint8 sRGB thumbnail of a raw image, area-averaged in linear space to fit `box`
    = (height, width). Never upsamples."""
    img = srgb8_to_linear(read_srgb8(layout.raw_image(split, image_id)))
    _, height, width = img.shape
    scale = min(1.0, box[0] / height, box[1] / width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    return linear_to_srgb8(area_downsample(img, size))


def _tile(thumbs: Sequence[torch.Tensor], labels: Sequence[str], title: str) -> Image.Image:
    (cell_h, cell_w), columns, label_h, title_h, pad = THUMB_BOX, 10, 20, 34, 6
    rows = math.ceil(len(thumbs) / columns)
    width = columns * (cell_w + pad) + pad
    height = title_h + rows * (cell_h + label_h) + pad
    sheet = Image.new("RGB", (width, height), (26, 26, 25))
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill=(235, 235, 230), font=_font(18))
    label_font = _font(13)
    for index, (thumb, label) in enumerate(zip(thumbs, labels)):
        row, column = divmod(index, columns)
        x0 = pad + column * (cell_w + pad)
        y0 = title_h + row * (cell_h + label_h)
        _, h, w = thumb.shape
        tile = Image.fromarray(thumb.permute(1, 2, 0).numpy())
        sheet.paste(tile, (x0 + (cell_w - w) // 2, y0))
        draw.text((x0 + 2, y0 + h + 3), label, fill=(195, 194, 183), font=label_font)
    return sheet


def make_lowres_copy(
    layout: VisDroneLayout,
    split: str,
    image_id: str,
    long_side: int = LOWRES_LONG_SIDE,
    overwrite: bool = False,
) -> Path:
    """Phase 3 for one image: raw -> linear -> area-downsample to `long_side` -> 8-bit sRGB PNG.

    Averaging happens in linear space: averaging sRGB codes makes bright lights too dim. PNG,
    not JPEG, so the training targets get no second round of compression artifacts. An existing
    output is kept unless `overwrite`.
    """
    path = layout.lowres_image(split, image_id)
    if path.exists() and not overwrite:
        return path
    small, _ = _read_linear(layout.raw_image(split, image_id), long_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.png")  # an interrupted run leaves no PNG
    _save_srgb8(linear_to_srgb8(small), partial)
    partial.replace(path)
    return path
