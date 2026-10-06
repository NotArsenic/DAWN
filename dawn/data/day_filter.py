"""Per-pair RGB statistics and the day filter (plan steps 1-2).

`compute_pair_stats` measures one pair on the **cropped** 640x512 frame: the RGB luminance
percentiles (linear Rec. 709 luminance, docs/conventions.md section 1), the existence, frame,
border and thermal-channel checks, the folder and file names from the XML, and a 64-bit
perceptual hash of each modality for the cross-split dedup. Rows go to the stats CSV.

The day decision is one pure comparison, `lum_median >= day_filter.min_lum_median`. The
distribution has a bright day mode and a continuum of dusk below it with no clean gap, so the
cutoff is a human decision; the histogram, the contact sheets and the per-folder table here are
what that decision is made from. Defects (a missing file, a bad border, differing thermal
channels) are reason codes of their own so a pair is never silently dropped.
"""

from __future__ import annotations

import csv
import logging
import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from dawn.data import dedup
from dawn.data import dronevehicle as dv
from dawn.data.color import luminance, srgb8_to_linear
from dawn.data.sheets import run_parallel, thumbnail, tile_sheet

logger = logging.getLogger(__name__)

LOG_FLOOR = 1e-4  # luminance floor before a log10, as in the old guidance map
SHEET_BOX = (180, 240)  # contact-sheet thumbnails fit inside (height, width)
# Chart palette: the dataviz reference instance (light surface, ink, categorical slots 1 and 2).
_SURFACE, _GRID, _INK, _INK_2 = "#fcfcfb", "#e4e3de", "#0b0b0b", "#52514e"
_SERIES, _CUTOFF = "#2a78d6", "#eb6834"
_REC709 = (0.2126, 0.7152, 0.0722)


@dataclass(frozen=True)
class PairStats:
    """One row of the stats CSV: one stem of one split, paired or not.

    `status` is `ok` when both images and both label files exist and decode; otherwise the first
    thing that failed (`missing_thermal`, `missing_rgb`, `missing_label`, `bad_frame`,
    `read_error`) and every measurement is NaN / empty. `border_*` is "" for a clean border,
    else `dark` or `wide` (`dronevehicle.check_border`). Luminance is linear Rec. 709 of the
    cropped RGB frame, in [0, 1].
    """

    image_id: str
    split: str
    stem: str
    status: str
    folder: str  # <folder> of the RGB XML
    rgb_filename: str  # <filename> of the RGB XML
    thermal_filename: str  # <filename> of the thermal XML
    border_rgb: str
    border_thermal: str
    thermal_channels_ok: bool  # the three channels of a 3-channel thermal JPEG are identical
    lum_median: float
    lum_p05: float
    lum_p95: float
    lum_mean: float
    rgb_hash: str  # 16 hex characters, dronevehicle crop, "" when unmeasured
    thermal_hash: str


_NAN = float("nan")
_FIELD_TYPES = {"str": str, "float": float, "bool": lambda v: v == "1"}


def is_day(lum_median: float, min_lum_median: float) -> bool:
    """The day decision: median linear luminance at or above the cutoff (NaN is never day)."""
    return lum_median >= min_lum_median


def day_reason(row: PairStats, min_lum_median: float) -> str:
    """`ok`, or the first reason this pair is not a day pair, in this order: the stats `status`,
    `bad_border`, `thermal_channels`, `not_day`."""
    if row.status != "ok":
        return row.status
    if row.border_rgb or row.border_thermal:
        return "bad_border"
    if not row.thermal_channels_ok:
        return "thermal_channels"
    return "ok" if is_day(row.lum_median, min_lum_median) else "not_day"


def _failed(image_id: str, split: str, stem: str, status: str, **fields_: object) -> PairStats:
    base = dict(
        image_id=image_id,
        split=split,
        stem=stem,
        status=status,
        folder="",
        rgb_filename="",
        thermal_filename="",
        border_rgb="",
        border_thermal="",
        thermal_channels_ok=True,
        lum_median=_NAN,
        lum_p05=_NAN,
        lum_p95=_NAN,
        lum_mean=_NAN,
        rgb_hash="",
        thermal_hash="",
    )
    return PairStats(**{**base, **fields_})


def compute_pair_stats(layout: dv.DroneVehicleLayout, split: str, stem: str) -> PairStats:
    """Everything the stats stage records about one stem. Never raises for bad data: a failure
    becomes the row's `status`."""
    iid = dv.image_id(split, stem)
    paths = {
        "missing_thermal": layout.thermal_image(split, stem),
        "missing_rgb": layout.rgb_image(split, stem),
    }
    for status, path in paths.items():
        if not path.exists():
            return _failed(iid, split, stem, status)
    names = {}
    for key, path in (
        ("rgb", layout.rgb_label(split, stem)),
        ("thermal", layout.thermal_label(split, stem)),
    ):
        if not path.exists():
            return _failed(iid, split, stem, "missing_label")
        try:
            names[key] = dv.parse_annotation(path, layout.class_map, layout.border)
        except Exception as exc:  # malformed XML
            logger.warning(f"{iid}: {path.name}: {exc}")
            return _failed(iid, split, stem, "read_error")
    meta = dict(
        folder=names["rgb"].folder,
        rgb_filename=names["rgb"].filename,
        thermal_filename=names["thermal"].filename,
    )
    try:
        rgb = dv.read_rgb8(layout.rgb_image(split, stem))
        dv.crop_border(rgb, layout.border, layout.frame_size)
    except dv.FrameError:
        return _failed(iid, split, stem, "bad_frame", **meta)
    except Exception as exc:  # truncated JPEG
        logger.warning(f"{iid}: RGB: {exc}")
        return _failed(iid, split, stem, "read_error", **meta)
    border_rgb = dv.check_border(rgb, layout.border_check, layout.border) or ""

    channels_ok, border_thermal, thermal_hash = True, "", ""
    try:
        thermal = dv.read_thermal8(layout.thermal_image(split, stem))
        dv.crop_border(thermal, layout.border, layout.frame_size)
    except dv.ThermalChannelError:
        channels_ok, thermal = False, None
    except dv.FrameError:
        return _failed(iid, split, stem, "bad_frame", **meta)
    except Exception as exc:
        logger.warning(f"{iid}: thermal: {exc}")
        return _failed(iid, split, stem, "read_error", **meta)
    if thermal is not None:
        border_thermal = dv.check_border(thermal, layout.border_check, layout.border) or ""
        crop_t = dv.crop_border(thermal, layout.border, layout.frame_size)
        thermal_hash = dedup.to_hex(dedup.phash(crop_t.float() / 255))

    crop = dv.crop_border(rgb, layout.border, layout.frame_size)
    lum = luminance(srgb8_to_linear(crop)).flatten()
    p05, median, p95 = torch.quantile(lum, torch.tensor([0.05, 0.5, 0.95])).tolist()
    weights = torch.tensor(_REC709).view(3, 1, 1)
    codes_luma = (crop.float() / 255 * weights).sum(dim=0, keepdim=True)  # sRGB-coded luma
    return PairStats(
        image_id=iid,
        split=split,
        stem=stem,
        status="ok",
        border_rgb=border_rgb,
        border_thermal=border_thermal,
        thermal_channels_ok=channels_ok,
        lum_median=median,
        lum_p05=p05,
        lum_p95=p95,
        lum_mean=lum.mean().item(),
        rgb_hash=dedup.to_hex(dedup.phash(codes_luma)),
        thermal_hash=thermal_hash,
        **meta,
    )


# --- CSV -----------------------------------------------------------------------------------------


def _split_rank(split: str) -> int:
    return dv.SPLITS.index(split) if split in dv.SPLITS else len(dv.SPLITS)


def write_stats_csv(path: str | Path, rows: Iterable[PairStats]) -> None:
    """Write the stats CSV, sorted by (split in `SPLITS` order, image ID). Full precision."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: (_split_rank(r.split), r.image_id))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([field.name for field in fields(PairStats)])
        for row in ordered:
            writer.writerow(
                [
                    int(v) if isinstance(v, bool) else repr(v) if isinstance(v, float) else v
                    for v in astuple(row)
                ]
            )


def read_stats_csv(path: str | Path) -> list[PairStats]:
    """Read the stats CSV back, in file order."""
    types = {field.name: _FIELD_TYPES[field.type] for field in fields(PairStats)}
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [
            PairStats(**{name: types[name](value) for name, value in record.items()})
            for record in csv.DictReader(f)
        ]


def _log10(value: float) -> float:
    return math.log10(max(value, LOG_FLOOR)) if not math.isnan(value) else _NAN


def write_day_log(path: str | Path, rows: Iterable[PairStats], min_lum_median: float) -> None:
    """The committed per-pair day log: ID, folder, log10 luminance percentiles (3 decimals), the
    decision and its reason. Sorted like the stats CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = lambda v: "" if math.isnan(v) else f"{v:.3f}"  # noqa: E731
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "image_id",
                "folder",
                "log10_lum_median",
                "log10_lum_p05",
                "log10_lum_p95",
                "day",
                "reason",
            ]
        )
        for row in sorted(rows, key=lambda r: (_split_rank(r.split), r.image_id)):
            reason = day_reason(row, min_lum_median)
            writer.writerow(
                [
                    row.image_id,
                    row.folder,
                    fmt(_log10(row.lum_median)),
                    fmt(_log10(row.lum_p05)),
                    fmt(_log10(row.lum_p95)),
                    int(reason == "ok"),
                    reason,
                ]
            )


def read_day_log(path: str | Path) -> dict[str, tuple[bool, str]]:
    """`{image_id: (is_day, reason)}` from `write_day_log`."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {r["image_id"]: (r["day"] == "1", r["reason"]) for r in csv.DictReader(f)}


# --- tables for choosing the cutoff --------------------------------------------------------------


def counts_at_cutoffs(
    rows: Iterable[PairStats], cutoffs: Sequence[float]
) -> dict[float, dict[str, int]]:
    """Day pairs per split at each cutoff (defects excluded, as in `day_reason`)."""
    rows = list(rows)
    splits = sorted({r.split for r in rows}, key=_split_rank)
    return {
        cut: {s: sum(r.split == s and day_reason(r, cut) == "ok" for r in rows) for s in splits}
        for cut in cutoffs
    }


def folder_table(
    rows: Iterable[PairStats], min_lum_median: float, mixed_only: bool = False
) -> list[dict]:
    """Per XML `<folder>`: frames, day frames, day share, median log10 luminance and splits, the
    largest folders first. Folders with a mixed day share are where dusk hides (`mixed_only`
    keeps just those). Only rows with `status == "ok"` count."""
    groups: dict[str, list[PairStats]] = {}
    for row in rows:
        if row.status == "ok":
            groups.setdefault(row.folder, []).append(row)
    table = []
    for folder, members in groups.items():
        n_day = sum(day_reason(r, min_lum_median) == "ok" for r in members)
        share = n_day / len(members)
        if mixed_only and not 0 < share < 1:
            continue
        table.append(
            dict(
                folder=folder,
                frames=len(members),
                day=n_day,
                day_share=share,
                median_log10_lum=statistics.median(_log10(r.lum_median) for r in members),
                splits="+".join(sorted({r.split for r in members}, key=_split_rank)),
            )
        )
    return sorted(table, key=lambda t: (-t["frames"], t["folder"]))


# --- histogram and contact sheets ----------------------------------------------------------------


def _style(ax) -> None:
    ax.set_facecolor(_SURFACE)
    ax.grid(color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=8)


def plot_log_luminance_histogram(
    rows: Sequence[PairStats], min_lum_median: float, out_path: str | Path, bin_log10: float = 0.05
) -> None:
    """Histogram of log10(median linear luminance), one panel per split, with the cutoff marked.

    Log scale on purpose: in linear units every dusk and night frame piles up against zero. Only
    rows with `status == "ok"` are drawn; the panel titles give the counts and the day share at
    the cutoff.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    usable = [r for r in rows if r.status == "ok" and not math.isnan(r.lum_median)]
    splits = sorted({r.split for r in usable}, key=_split_rank) or ["none"]
    values = {s: [_log10(r.lum_median) for r in usable if r.split == s] for s in splits}
    everything = [v for vs in values.values() for v in vs] or [0.0]
    lo = math.floor(min(everything) / bin_log10) * bin_log10
    hi = math.ceil(max(everything) / bin_log10) * bin_log10 + bin_log10
    bins = np.arange(lo, hi + 1e-9, bin_log10)
    cut = _log10(min_lum_median)
    fig = Figure(figsize=(9, 0.8 + 2.2 * len(splits)), facecolor=_SURFACE, layout="constrained")
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(splits), 1, sharex=True, squeeze=False)[:, 0]
    for ax, split in zip(axes, splits):
        _style(ax)
        ax.hist(values[split], bins=bins, color=_SERIES, edgecolor=_SURFACE, linewidth=0.8)
        ax.axvline(cut, color=_CUTOFF, linewidth=1.6)
        ax.set_ylabel("pairs", color=_INK_2, fontsize=9)
        n_day = sum(day_reason(r, min_lum_median) == "ok" for r in usable if r.split == split)
        ax.set_title(
            f"{split}: {len(values[split])} readable pairs, {n_day} day pairs at the cutoff",
            loc="left",
            color=_INK,
            fontsize=10,
        )
    axes[0].annotate(
        f"cutoff {min_lum_median:g} (log10 {cut:.2f})",
        (cut, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(-5, -12),
        ha="right",
        textcoords="offset points",
        color=_CUTOFF,
        fontsize=9,
    )
    axes[-1].set_xlabel(
        "log10(median linear luminance of the cropped RGB frame)", color=_INK_2, fontsize=9
    )
    fig.suptitle("DroneVehicle RGB median luminance", color=_INK, fontsize=11, x=0.01, ha="left")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE)


def rgb_thumbnail(
    layout: dv.DroneVehicleLayout, split: str, stem: str, box: tuple[int, int] = SHEET_BOX
) -> torch.Tensor:
    """`(3, h, w)` uint8 sRGB thumbnail of the cropped RGB frame (a `run_parallel` job)."""
    crop = dv.crop_border(
        dv.read_rgb8(layout.rgb_image(split, stem)), layout.border, layout.frame_size
    )
    return thumbnail(srgb8_to_linear(crop), box)


def _sheet(
    layout: dv.DroneVehicleLayout, picked: Sequence[PairStats], title: str, path: Path, workers: int
) -> Path:
    thumbs = run_parallel(rgb_thumbnail, [(layout, r.split, r.stem) for r in picked], workers)
    labels = [f"{_log10(r.lum_median):+.2f} {r.split[:2]}{r.stem}" for r in picked]
    path.parent.mkdir(parents=True, exist_ok=True)
    tile_sheet(thumbs, labels, title, columns=8, box=SHEET_BOX).save(path, quality=90)
    return path


def save_day_sheets(
    layout: dv.DroneVehicleLayout,
    rows: Sequence[PairStats],
    min_lum_median: float,
    out_dir: str | Path,
    count: int = 100,
    band_log10: float = 0.15,
    seed: int = 0,
    workers: int = 1,
) -> list[Path]:
    """Contact sheets for choosing and checking the cutoff; each caption is `log10 L split+stem`.

    - `day_random.jpg`: `count` random accepted frames, darkest first.
    - `band_above.jpg`: accepted frames within `band_log10` above the cutoff (the borderline
      frames that just made it).
    - `band_below.jpg`: rejected frames within `band_log10` below it (what was just missed).
    - `night_folders.jpg`: every accepted frame from a folder whose name contains "night" (there
      should be none). Always written, empty when there are none, so a rerun replaces a stale one.

    Only pairs with no defect are drawn. Same seed, same sheets.
    """
    out_dir = Path(out_dir)
    usable = [r for r in rows if r.status == "ok" and not r.border_rgb and not r.border_thermal]
    cut = _log10(min_lum_median)
    day = [r for r in usable if is_day(r.lum_median, min_lum_median)]
    night = [r for r in usable if not is_day(r.lum_median, min_lum_median)]
    above = [r for r in day if _log10(r.lum_median) < cut + band_log10]
    below = [r for r in night if _log10(r.lum_median) >= cut - band_log10]
    from_night_folder = [r for r in day if "night" in r.folder.lower()]

    def pick(group: Sequence[PairStats], name: str, limit: int) -> list[PairStats]:
        chosen = random.Random(f"{seed}/{name}").sample(list(group), min(limit, len(group)))
        return sorted(chosen, key=lambda r: (r.lum_median, r.image_id))

    plan = [
        ("day_random", pick(day, "random", count), f"{count} random accepted of {len(day)}"),
        (
            "band_above",
            pick(above, "above", count),
            f"accepted within {band_log10} dex above the cutoff",
        ),
        (
            "band_below",
            pick(below, "below", count),
            f"rejected within {band_log10} dex below the cutoff",
        ),
        (
            "night_folders",
            sorted(from_night_folder, key=lambda r: r.image_id),
            "accepted frames from folders named night",
        ),
    ]
    written = []
    for name, picked, what in plan:
        if name == "night_folders":
            what = f"{len(picked)} {what}" + (" (expected none)" if not picked else "")
        title = f"{what}; cutoff {min_lum_median:g}; label = log10 L, split+stem; darkest first"
        written.append(_sheet(layout, picked, title, out_dir / f"{name}.jpg", workers))
    return written
