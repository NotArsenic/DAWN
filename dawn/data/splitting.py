"""Hand labels and image statistics -> the four manifests training reads.

    train_day   clean daytime sources for the degradation pipeline
    val_day     the same, held out by GROUP
    test_day    the official test-dev split, daytime only
    hard_test   real night frames, never a source: the thing the model is judged on

Three things happen here, in this order.

1. **Vocabulary.** The hand labels say day / night / dusk / exclude; training wants
   day / hard_test / dropped. `configs/data.yaml: split.label_map` does the translation, so
   `data/labels/visdrone_labels.csv` is never edited to suit the code. Dusk is dropped: it is
   neither a clean daytime source nor a real night test.

2. **Auto-labelling the still collections.** The 49 `99999xx` groups hold 6,497 unrelated
   stills, day and night interleaved frame by frame, and nobody hand-labelled them. Two cutoffs
   on measurements phase 1 already made -- median luminance and highlight ratio -- accept the
   bright, flat ones as day. The cutoffs are fitted on the hand-labelled VIDEO frames under one
   hard constraint: **zero** non-day frames may pass. Recall is whatever is left over. Being
   wrong here means feeding a night frame to the relighting pipeline as if it were daylight,
   which poisons a training pair; missing a real day frame only costs data, and VisDrone has
   plenty.

3. **Re-splitting by group.** VisDrone's own train/val boundary cuts across nothing in
   particular, and consecutive frames of one video are near-duplicates, so a random image-level
   split leaks. Train and val are redrawn so that a whole group (a video sequence, or one still
   collection) lands entirely on one side. test-dev is left exactly as the benchmark defines
   it, and `assert_groups_disjoint` checks it shares no group with train or val.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dawn.data.visdrone import ImageStats, is_still_collection, sequence_id, write_manifest

logger = logging.getLogger(__name__)

FINAL_SUBSETS = ("train_day", "val_day", "test_day", "hard_test")
DAY = "day"
HARD_TEST = "hard_test"
LOG_FLOOR = 1e-4  # same floor as the guidance map (docs/conventions.md section 4)
_EPS = 1e-9


# --- 1. label vocabulary ---------------------------------------------------------------------


def map_labels(
    image_labels: Mapping[str, str | None],
    label_map: Mapping[str, str | None],
) -> dict[str, str]:
    """Translate hand labels into training labels, dropping anything mapped to nothing.

    Args:
        image_labels: `{image_id: hand label or None}` from `prepare.resolve_labels`.
        label_map: `split.label_map` from data.yaml, e.g. `{day: day, night: hard_test,
            dusk: null, exclude: null}`.

    Returns:
        `{image_id: training label}`, holding only the images that kept one.

    Raises:
        ValueError: A hand label has no entry in `label_map` (silently dropping it would be a
            data loss nobody notices).
    """
    out = {}
    for image_id, label in image_labels.items():
        if label is None:
            continue
        if label not in label_map:
            raise ValueError(f"{image_id}: hand label {label!r} is missing from split.label_map")
        target = label_map[label]
        if target:
            out[image_id] = target
    return out


# --- 2. auto-labelling the still collections --------------------------------------------------


@dataclass(frozen=True)
class AutoDayCutoffs:
    """Two one-sided cutoffs that accept an image as daytime, plus how they were chosen.

    An image is accepted when `lum_median >= min_luminance` AND
    `highlight_ratio <= max_highlight_ratio`.
    """

    min_luminance: float
    max_highlight_ratio: float
    recall: float  # share of hand-labelled day frames these cutoffs would keep
    num_day: int  # hand-labelled day frames the fit saw
    num_other: int  # hand-labelled non-day frames the fit had to exclude
    errors: int  # non-day frames that still pass. Must be 0.
    margin: float

    def accepts(self, row: ImageStats) -> bool:
        return (
            row.lum_median >= self.min_luminance
            and row.highlight_ratio <= self.max_highlight_ratio
        )


def _log(value: float) -> float:
    return math.log10(max(float(value), LOG_FLOOR))


def fit_auto_day_cutoffs(
    stats: Iterable[ImageStats],
    training_labels: Mapping[str, str],
    margin: float = 0.10,
    min_recall: float = 0.20,
) -> AutoDayCutoffs:
    """Fit the two cutoffs on the hand-labelled video frames, making zero errors on them.

    Both features are heavy-tailed, so the search runs in log10. It walks every distinct
    luminance a non-day frame takes; for each, the highlight ceiling is pinned just below the
    lowest highlight ratio among the non-day frames that the luminance floor failed to reject,
    which is the most permissive ceiling that still makes no error. The pair that keeps the most
    day frames wins; ties go to the stricter luminance floor.

    `margin` then moves both cutoffs a fraction of the way from the boundary toward the day
    median, in log units, so the decision does not rest on a single image sitting on the line.

    Args:
        stats: Rows from the stats CSV. Still collections are ignored: they have no hand label.
        training_labels: `{image_id: "day" | "hard_test"}` from `map_labels`.
        margin: Fraction of the gap to the day median to pull each cutoff in by.
        min_recall: Fail if the fitted cutoffs keep less than this share of hand-labelled day
            frames, which would mean the two features cannot separate the classes at all.

    Returns:
        AutoDayCutoffs.

    Raises:
        ValueError: No labelled day or no labelled non-day video frames to fit on, or the
            resulting recall is below `min_recall`.
    """
    day: list[tuple[float, float]] = []
    other: list[tuple[float, float]] = []
    for row in stats:
        if is_still_collection(sequence_id(row.image_id)):
            continue
        label = training_labels.get(row.image_id)
        if label is None:
            continue
        point = (_log(row.lum_median), _log(row.highlight_ratio))
        (day if label == DAY else other).append(point)

    if not day or not other:
        raise ValueError(
            f"need hand-labelled day AND non-day video frames to fit cutoffs; "
            f"got {len(day)} day, {len(other)} non-day"
        )

    other_sorted = sorted(other, key=lambda p: -p[0])
    # candidates[k] = (luminance floor, highlight ceiling) making zero errors
    candidates: list[tuple[float, float]] = []
    running_min_hl = math.inf
    index = 0
    while index < len(other_sorted):
        value = other_sorted[index][0]
        # Floor just above `value` rejects every non-day frame seen so far by luminance alone.
        candidates.append((value + _EPS, running_min_hl - _EPS))
        while index < len(other_sorted) and other_sorted[index][0] == value:
            running_min_hl = min(running_min_hl, other_sorted[index][1])
            index += 1
    candidates.append((-math.inf, running_min_hl - _EPS))  # luminance unconstrained

    best = max(
        candidates,
        key=lambda c: (sum(1 for lum, hl in day if lum >= c[0] and hl <= c[1]), c[0]),
    )
    lum_cut, hl_cut = best

    # Pull both cutoffs toward the day median so the boundary is not one image wide.
    day_lum = sorted(p[0] for p in day)
    day_hl = sorted(p[1] for p in day)
    median_lum = day_lum[len(day_lum) // 2]
    median_hl = day_hl[len(day_hl) // 2]
    # Only ever tighten. If a cutoff already sits past the day median the fit was barely
    # separable, and loosening it back to the median would undo the zero-error guarantee.
    if math.isfinite(lum_cut):
        lum_cut += margin * max(0.0, median_lum - lum_cut)
    if math.isfinite(hl_cut):
        hl_cut -= margin * max(0.0, hl_cut - median_hl)

    kept = sum(1 for lum, hl in day if lum >= lum_cut and hl <= hl_cut)
    errors = sum(1 for lum, hl in other if lum >= lum_cut and hl <= hl_cut)
    recall = kept / len(day)
    if recall < min_recall:
        raise ValueError(
            f"zero-error cutoffs keep only {recall:.1%} of hand-labelled day frames "
            f"(min_recall={min_recall:.0%}); luminance and highlight ratio do not separate "
            f"day from night on this data"
        )
    return AutoDayCutoffs(
        min_luminance=10**lum_cut if math.isfinite(lum_cut) else 0.0,
        max_highlight_ratio=10**hl_cut if math.isfinite(hl_cut) else math.inf,
        recall=recall,
        num_day=len(day),
        num_other=len(other),
        errors=errors,
        margin=margin,
    )


def auto_day_ids(stats: Iterable[ImageStats], cutoffs: AutoDayCutoffs) -> list[str]:
    """Still-collection image IDs the cutoffs accept as daytime, sorted."""
    return sorted(
        row.image_id
        for row in stats
        if is_still_collection(sequence_id(row.image_id)) and cutoffs.accepts(row)
    )


# --- 3. re-splitting by group ------------------------------------------------------------------


def group_id(image_id: str) -> str:
    """The group an image may not be split away from: its video sequence or still collection."""
    return sequence_id(image_id)


def regroup_train_val(
    image_ids: Iterable[str],
    val_fraction: float = 0.10,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    """Split groups (not images) into train and validation.

    Consecutive frames of a VisDrone video are near-duplicates, so an image-level split puts
    almost the same picture on both sides and validation stops measuring generalization. Whole
    groups move together. The shuffle is seeded off a sorted group list, so the same seed gives
    the same split on any machine.

    Args:
        image_ids: Every daytime image eligible for train or val.
        val_fraction: Target share of GROUPS held out, not of images.
        seed: Shuffle seed.

    Returns:
        `(train_groups, val_groups)`, disjoint sets of group IDs.

    Raises:
        ValueError: `val_fraction` is not in (0, 1), or there are too few groups to split.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    groups = sorted({group_id(image_id) for image_id in image_ids})
    if len(groups) < 2:
        raise ValueError(f"need at least 2 groups to hold one out, got {len(groups)}")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    num_val = max(1, min(len(groups) - 1, round(val_fraction * len(groups))))
    val = set(shuffled[:num_val])
    return set(groups) - val, val


def assert_groups_disjoint(manifests: Mapping[str, Sequence[str]]) -> dict[str, set[str]]:
    """Check that no group ID appears in two of the given manifests.

    The point of a group split is that nothing recognizable is shared. This is the check that
    says so out loud, including test-dev against train and val.

    Args:
        manifests: `{name: image_ids}`.

    Returns:
        `{name: group IDs}`.

    Raises:
        ValueError: Two manifests share a group.
    """
    groups = {name: {group_id(i) for i in ids} for name, ids in manifests.items()}
    names = sorted(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = groups[left] & groups[right]
            if shared:
                sample = ", ".join(sorted(shared)[:5])
                raise ValueError(
                    f"{left} and {right} share {len(shared)} group(s): {sample}"
                    + (" ..." if len(shared) > 5 else "")
                )
    return groups


# --- putting it together ------------------------------------------------------------------------


def build_final_manifests(
    stats: Sequence[ImageStats],
    image_labels: Mapping[str, str | None],
    cfg: dict,
) -> tuple[dict[str, list[str]], AutoDayCutoffs | None]:
    """Hand labels + stats -> `{train_day, val_day, test_day, hard_test: [image_id]}`.

    Args:
        stats: Every row of the stats CSV.
        image_labels: `{image_id: hand label or None}` from `prepare.resolve_labels`.
        cfg: The `split` section of data.yaml.

    Returns:
        `(manifests, cutoffs)`. `cutoffs` is None when auto-day is switched off.

    Raises:
        ValueError: A manifest would share a group with another (see `assert_groups_disjoint`).
    """
    by_id = {row.image_id: row for row in stats}
    training_labels = map_labels(image_labels, cfg["label_map"])

    auto = cfg.get("auto_day", {})
    cutoffs = None
    if auto.get("enabled", False):
        cutoffs = fit_auto_day_cutoffs(
            stats,
            training_labels,
            margin=float(auto.get("margin", 0.10)),
            min_recall=float(auto.get("min_recall", 0.20)),
        )
        for image_id in auto_day_ids(stats, cutoffs):
            training_labels.setdefault(image_id, DAY)
        logger.info(
            "auto-day: %d still images accepted, lum >= %.4f and highlight <= %.1f, "
            "recall %.1f%% on %d hand-labelled day frames, %d errors",
            sum(1 for i in training_labels if is_still_collection(group_id(i))),
            cutoffs.min_luminance,
            cutoffs.max_highlight_ratio,
            cutoffs.recall * 100,
            cutoffs.num_day,
            cutoffs.errors,
        )

    regroup = cfg.get("regroup", {})
    test_split = regroup.get("test_split", "test-dev")

    day_pool, test_day, hard_test = [], [], []
    for image_id, label in sorted(training_labels.items()):
        row = by_id.get(image_id)
        if row is None:
            continue
        if label == HARD_TEST:
            hard_test.append(image_id)
        elif label == DAY:
            (test_day if row.split == test_split else day_pool).append(image_id)

    manifests: dict[str, list[str]] = {name: [] for name in FINAL_SUBSETS}
    manifests["test_day"] = sorted(test_day)
    manifests["hard_test"] = sorted(hard_test)
    if day_pool:
        train_groups, _ = regroup_train_val(
            day_pool,
            val_fraction=float(regroup.get("val_fraction", 0.10)),
            seed=int(regroup.get("seed", 0)),
        )
        for image_id in day_pool:
            key = "train_day" if group_id(image_id) in train_groups else "val_day"
            manifests[key].append(image_id)
        manifests["train_day"].sort()
        manifests["val_day"].sort()

    assert_groups_disjoint({k: v for k, v in manifests.items() if k != "hard_test"})
    return manifests, cutoffs


def write_final_manifests(
    manifests: Mapping[str, Sequence[str]], manifest_dir: str | Path
) -> dict[str, int]:
    """Write `<manifest_dir>/<name>.txt` for each of FINAL_SUBSETS. Returns the counts."""
    out = Path(manifest_dir)
    counts = {}
    for name in FINAL_SUBSETS:
        ids = list(manifests.get(name, ()))
        write_manifest(out / f"{name}.txt", ids)
        counts[name] = len(ids)
    return counts
