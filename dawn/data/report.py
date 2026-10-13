"""Reporting and summary generation for the dataset pipeline."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from dawn.data.align import AlignRow, VehicleBranches
from dawn.data.day_filter import PairStats
from dawn.data.dedup import DedupRow

SPLITS = ("train", "val", "test")
_BRANCH_NAMES = ("matched", "thermal_only", "static", "moved", "unsure")
_UNPAIRED = ("missing_thermal", "missing_rgb")
_UNREADABLE = (
    *_UNPAIRED,
    "missing_label",
    "bad_frame",
    "read_error",
    "bad_border",
    "thermal_channels",
)


def _split(image_id: str) -> str:
    return image_id.split("_", 1)[0]


def final_reasons(
    stats: Sequence[PairStats],
    day_log: Mapping[str, tuple[bool, str]],
    align_rows: Mapping[str, AlignRow],
    dedup_rows: Mapping[str, DedupRow],
) -> dict[str, str]:
    """Map each image ID to its terminal pipeline status reason."""
    final = {}
    for row in stats:
        iid = row.image_id
        is_day, day_reason = day_log.get(iid, (False, "not_in_day_log"))
        if not is_day:
            final[iid] = day_reason
        elif iid not in align_rows:
            final[iid] = "not_aligned"
        elif not align_rows[iid].accepted:
            final[iid] = align_rows[iid].reason
        elif iid in dedup_rows and dedup_rows[iid].near_duplicate:
            final[iid] = "near_duplicate"
        else:
            final[iid] = "ok"
    return final


def fallback_ids(final: Mapping[str, str], align_rows: Mapping[str, AlignRow]) -> list[str]:
    """Return sorted image IDs aligned via edge correlation fallback."""
    return sorted(
        i
        for i, reason in final.items()
        if reason == "ok" and i in align_rows and align_rows[i].model == "edge_translation"
    )


def stage_counts(
    stats: Sequence[PairStats],
    day_log: Mapping[str, tuple[bool, str]],
    align_rows: Mapping[str, AlignRow],
    dedup_rows: Mapping[str, DedupRow],
) -> dict[str, dict[str, int]]:
    """Compute surviving pair counts across pipeline stages per split."""
    final = final_reasons(stats, day_log, align_rows, dedup_rows)
    stages = {
        name: Counter() for name in ("stems", "paired", "readable", "day", "aligned", "final")
    }
    for row in stats:
        split = row.split
        reason = day_log.get(row.image_id, (False, ""))[1]
        stages["stems"][split] += 1
        stages["paired"][split] += row.status not in _UNPAIRED
        stages["readable"][split] += reason not in _UNREADABLE and row.status == "ok"
        stages["day"][split] += day_log.get(row.image_id, (False, ""))[0]
        stages["aligned"][split] += row.image_id in align_rows and align_rows[row.image_id].accepted
        stages["final"][split] += final[row.image_id] == "ok"
    return {
        name: {s: int(counter.get(s, 0)) for s in _splits_in(stats)}
        for name, counter in stages.items()
    }


def _splits_in(stats: Sequence[PairStats]) -> list[str]:
    present = {r.split for r in stats}
    return [s for s in SPLITS if s in present] + sorted(present - set(SPLITS))


_STAGE_LABELS = {
    "stems": "stems in the image folders",
    "paired": "paired (both images present)",
    "readable": "readable (labels, frame, border, thermal channels ok)",
    "day": "day (RGB median luminance at or above the cutoff)",
    "aligned": "aligned (geometry and edge check)",
    "final": "final pairs (manifests)",
}


def _table(header: str, rows: list[tuple[str, dict[str, int]]], splits: Sequence[str]) -> list[str]:
    lines = [f"| {header} | " + " | ".join(splits) + " | total |"]
    lines.append("|" + "---|" * (len(splits) + 2))
    for label, counts in rows:
        cells = [str(counts.get(s, 0)) for s in splits]
        lines.append(
            f"| {label} | " + " | ".join(cells) + f" | {sum(counts.get(s, 0) for s in splits)} |"
        )
    return lines


def _flat(prefix: str, section: Mapping) -> list[str]:
    lines = []
    for key, value in section.items():
        if isinstance(value, Mapping):
            lines += _flat(f"{prefix}{key}.", value)
        else:
            lines.append(f"- `{prefix}{key}: {value}`")
    return lines


def write_vehicle_log(
    path: str | Path,
    fractions: Mapping[str, float],
    max_fraction: float,
    branches: Mapping[str, VehicleBranches] | None = None,
) -> list[str]:
    """Write vehicle mask coverage statistics to a CSV log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged = []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            ["image_id", "vehicles_fraction", "dense"] + [f"n_{name}" for name in _BRANCH_NAMES]
        )
        for image_id in sorted(fractions):
            dense = fractions[image_id] > max_fraction
            counts = (
                [getattr(branches[image_id], name) for name in _BRANCH_NAMES]
                if branches is not None
                else [""] * len(_BRANCH_NAMES)
            )
            writer.writerow([image_id, f"{fractions[image_id]:.3f}", int(dense), *counts])
            if dense:
                flagged.append(image_id)
    return flagged


def build_summary(
    stats: Sequence[PairStats],
    day_log: Mapping[str, tuple[bool, str]],
    align_rows: Mapping[str, AlignRow],
    dedup_rows: Mapping[str, DedupRow],
    cfg: Mapping,
    vehicle_fractions: Mapping[str, float] | None = None,
    max_vehicle_fraction: float = 0.30,
    vehicle_branches: Mapping[str, VehicleBranches] | None = None,
) -> str:
    """Generate Markdown summary report from pipeline logs."""
    splits = _splits_in(stats)
    counts = stage_counts(stats, day_log, align_rows, dedup_rows)
    final = final_reasons(stats, day_log, align_rows, dedup_rows)
    by_reason: dict[str, Counter] = {}
    for iid, reason in final.items():
        by_reason.setdefault(reason, Counter())[_split(iid)] += 1
    ordered = sorted(
        (r for r in by_reason if r != "ok"), key=lambda r: (-sum(by_reason[r].values()), r)
    )
    reason_rows = [(r, dict(by_reason[r])) for r in ordered + (["ok"] if "ok" in by_reason else [])]

    rows = list(align_rows.values())
    diagnostics = [
        f"- homography fallback used: {sum(r.model == 'homography' for r in rows)} "
        f"(tried: {sum(r.hom_tried for r in rows)})",
        "- pairs with a moving vehicle (accepted): "
        f"{sum(r.n_moving > 0 for r in rows if r.accepted)}",
        "- aligned by the edge-correlation fallback: "
        f"{sum(r.model == 'edge_translation' for r in rows)} "
        f"(tried: {sum(r.fb_tried for r in rows)})",
        f"- aligned pairs with copied boxes excluded: {sum(r.n_copied > 0 for r in rows)}",
        f"- pairs whose label sets are identical (copied): {sum(r.labels_identical for r in rows)}",
    ]
    out = [
        "# DroneVehicle data summary",
        "",
        "Written by `scripts/prepare_dronevehicle.py write`. Each pair is counted once, at the "
        "first stage that rejects it. Counts are of pairs (a stem present in both image folders).",
        "",
        "## Pairs per stage",
        "",
        *_table("stage", [(_STAGE_LABELS[k], counts[k]) for k in _STAGE_LABELS], splits),
        "",
        "## Verdict per pair (first failing stage)",
        "",
        *_table("reason", reason_rows, splits),
        "",
        "## Alignment diagnostics",
        "",
        *diagnostics,
        "",
        *_vehicles_section(
            cfg, align_rows, vehicle_fractions, max_vehicle_fraction, vehicle_branches
        ),
        "## Thresholds in force",
        "",
    ]
    thresholds = []
    for section in ("day_filter", "align", "dedup", "masks"):
        if cfg.get(section):
            thresholds += _flat(f"{section}.", cfg[section])
    out += thresholds or ["- (no configuration given)"]
    return "\n".join(out) + "\n"


def _vehicles_section(
    cfg: Mapping,
    align_rows: Mapping[str, AlignRow],
    fractions: Mapping[str, float] | None,
    max_fraction: float,
    branches: Mapping[str, VehicleBranches] | None,
) -> list[str]:
    if not fractions:
        return []
    section = (cfg.get("masks") or {}).get("vehicles") or {}
    dilate, extend = section.get("dilate_px", "?"), section.get("extend_px", "?")
    dense = sum(f > max_fraction for f in fractions.values())
    mean = 100 * sum(fractions.values()) / len(fractions)
    lines = [
        "## Vehicles mask",
        "",
        f"- margin {dilate} px round every vehicle; vehicles that moved or may have are extended "
        f"{extend} px each way along their long axis (the p95 of moving-vehicle displacement)",
        f"- mean masked fraction {mean:.1f}% of a frame",
        f"- frames masked over {100 * max_fraction:.0f}%: {dense} of {len(fractions)} "
        "(flagged in `vehicles.csv`)",
    ]
    if branches:
        order = ("matched", "static", "moved", "unsure", "thermal_only")
        totals = {n: sum(getattr(b, n) for b in branches.values()) for n in order}
        lines.append(
            "- vehicles per branch (all pairs): "
            + ", ".join(f"{n.replace('_', '-')} {totals[n]}" for n in order)
        )
    fallback = {
        i for i in fractions if i in align_rows and align_rows[i].model == "edge_translation"
    }
    for label, ids in (
        ("fallback pairs", sorted(fallback)),
        ("box-aligned pairs", sorted(set(fractions) - fallback)),
    ):
        if ids:
            share = 100 * sum(fractions[i] for i in ids) / len(ids)
            over = sum(fractions[i] > max_fraction for i in ids)
            lines.append(
                f"- {label} ({len(ids)}): mean masked fraction {share:.1f}%, over "
                f"{100 * max_fraction:.0f}%: {over}"
            )
    return [*lines, ""]


def stratified_sample(
    groups: Mapping[str, Sequence], n: int, seed: int
) -> list[tuple[str, object]]:
    """Draw a stratified sample of n items proportionally across groups."""
    sizes = {g: len(items) for g, items in groups.items() if len(items)}
    total = sum(sizes.values())
    names = sorted(sizes)
    if n >= total:
        return [(g, item) for g in names for item in groups[g]]
    order = sorted(names, key=lambda g: (-sizes[g], g))
    quota = {g: 0 for g in names}
    if n <= len(order):
        for g in order[:n]:
            quota[g] = 1
    else:
        ideal = {g: n * sizes[g] / total for g in names}
        quota = {g: min(sizes[g], max(1, math.floor(ideal[g]))) for g in names}
        while sum(quota.values()) > n:
            g = max((g for g in names if quota[g] > 1), key=lambda g: (quota[g] - ideal[g], g))
            quota[g] -= 1
        while sum(quota.values()) < n:
            g = max(
                (g for g in names if quota[g] < sizes[g]), key=lambda g: (ideal[g] - quota[g], g)
            )
            quota[g] += 1
    picked = []
    for g in names:
        if quota[g]:
            chosen = random.Random(f"{seed}/{g}").sample(list(groups[g]), quota[g])
            picked += [(g, item) for item in chosen]
    return picked
