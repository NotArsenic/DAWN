"""Counts, verdicts and samples for reporting on the data pipeline (pure functions over the logs).

Every pair ends the pipeline with one verdict, `final_reasons`: the reason of the first stage that
rejected it (`day` -> `align` -> `dedup`), or `ok`. `build_summary` renders those, and the count at
each stage, as the committed `summary.md`.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

from dawn.data.align import AlignRow
from dawn.data.day_filter import PairStats
from dawn.data.dedup import DedupRow

SPLITS = ("train", "val", "test")
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
    """`{image_id: reason}`: the first stage that rejected the pair, or `ok`.

    A day pair with no align row is `not_aligned` (the `align` stage has not run for it), never
    silently `ok`.
    """
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


def stage_counts(
    stats: Sequence[PairStats],
    day_log: Mapping[str, tuple[bool, str]],
    align_rows: Mapping[str, AlignRow],
    dedup_rows: Mapping[str, DedupRow],
) -> dict[str, dict[str, int]]:
    """Pairs surviving each stage, per split: `stems` (every stem in either image folder),
    `paired` (both images present), `readable` (labels, frame, border and thermal channels fine),
    `day`, `aligned` (geometry and edge check) and `final` (after dedup: the manifests)."""
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


def build_summary(
    stats: Sequence[PairStats],
    day_log: Mapping[str, tuple[bool, str]],
    align_rows: Mapping[str, AlignRow],
    dedup_rows: Mapping[str, DedupRow],
    cfg: Mapping,
) -> str:
    """The text of `summary.md`: counts per stage and split, each reason code's count, the
    alignment diagnostics and the thresholds in force (day, align, dedup sections of `cfg`)."""
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
        "## Thresholds in force",
        "",
    ]
    thresholds = []
    for section in ("day_filter", "align", "dedup"):
        if cfg.get(section):
            thresholds += _flat(f"{section}.", cfg[section])
    out += thresholds or ["- (no configuration given)"]
    return "\n".join(out) + "\n"


# --- stratified sample ---------------------------------------------------------------------------


def stratified_sample(
    groups: Mapping[str, Sequence], n: int, seed: int
) -> list[tuple[str, object]]:
    """`n` items spread over the groups, proportionally to group size but with every group
    represented (largest groups first when there are fewer slots than groups). Seeded.

    A group never gives more than it has; the surplus goes to the others. Asking for at least as
    many as exist returns everything.

    Returns:
        `[(group, item), ...]`, grouped by group name.
    """
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
        while sum(quota.values()) > n:  # over by the floor of one: take from the most over-served
            g = max((g for g in names if quota[g] > 1), key=lambda g: (quota[g] - ideal[g], g))
            quota[g] -= 1
        while sum(quota.values()) < n:  # under by rounding: give to the most under-served
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
