"""dawn/data/report.py: final verdicts, the summary table and the stratified audit sample."""

import dataclasses

import pytest

pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data import align, dedup, report  # noqa: E402
from dawn.data.day_filter import PairStats  # noqa: E402


def _stats(image_id, status="ok", lum=0.3, border="", channels=True):
    split, stem = image_id.split("_")
    return PairStats(
        image_id=image_id,
        split=split,
        stem=stem,
        status=status,
        folder="f",
        rgb_filename="a",
        thermal_filename="b",
        border_rgb=border,
        border_thermal="",
        thermal_channels_ok=channels,
        lum_median=lum,
        lum_p05=0.1,
        lum_p95=0.5,
        lum_mean=0.3,
        rgb_hash="0" * 16,
        thermal_hash="0" * 16,
    )


def _align(image_id, reason, **kw):
    row = dataclasses.replace(align._failed_row(image_id, reason), accepted=reason == "ok")
    return dataclasses.replace(row, **kw)


def _dedup(image_id, near):
    return dedup.DedupRow(image_id, "train_1", 2 if near else 30, 2, 1, near)


@pytest.fixture()
def world():
    stats = [
        _stats("train_1"),
        _stats("train_2"),
        _stats("train_3"),
        _stats("train_4", lum=0.01),
        _stats("train_5", status="missing_thermal"),
        _stats("train_6", border="dark"),
        _stats("val_1"),
        _stats("val_2"),
        _stats("val_3"),
        _stats("test_1"),
        _stats("test_2", lum=0.01),
    ]
    day_log = {
        r.image_id: (
            r.lum_median >= 0.1 and r.status == "ok" and not r.border_rgb,
            "ok"
            if r.lum_median >= 0.1 and r.status == "ok" and not r.border_rgb
            else (
                "missing_thermal"
                if r.status != "ok"
                else "bad_border"
                if r.border_rgb
                else "not_day"
            ),
        )
        for r in stats
    }
    aligned = {
        "train_1": _align("train_1", "ok", model="similarity"),
        "train_2": _align("train_2", "scale"),
        "train_3": _align("train_3", "labels_copied", n_copied=5),
        "val_1": _align("val_1", "ok", model="homography"),
        "val_2": _align("val_2", "ok", n_moving=2),
        "val_3": _align("val_3", "edge_check"),
        "test_1": _align("test_1", "ok"),
    }
    dedups = {
        "val_1": _dedup("val_1", True),
        "val_2": _dedup("val_2", False),
        "test_1": _dedup("test_1", False),
    }
    return stats, day_log, aligned, dedups


def test_final_reason_is_the_first_stage_that_rejected_the_pair(world):
    stats, day_log, aligned, dedups = world
    final = report.final_reasons(stats, day_log, aligned, dedups)
    assert final == {
        "train_1": "ok",
        "train_2": "scale",
        "train_3": "labels_copied",
        "train_4": "not_day",
        "train_5": "missing_thermal",
        "train_6": "bad_border",
        "val_1": "near_duplicate",
        "val_2": "ok",
        "val_3": "edge_check",
        "test_1": "ok",
        "test_2": "not_day",
    }


def test_a_day_pair_that_was_never_aligned_is_not_silently_ok(world):
    stats, day_log, aligned, dedups = world
    del aligned["train_1"]
    assert report.final_reasons(stats, day_log, aligned, dedups)["train_1"] == "not_aligned"


def test_stage_counts_per_split(world):
    stats, day_log, aligned, dedups = world
    counts = report.stage_counts(stats, day_log, aligned, dedups)
    assert counts["stems"] == {"train": 6, "val": 3, "test": 2}
    assert counts["paired"] == {"train": 5, "val": 3, "test": 2}
    assert counts["readable"] == {"train": 4, "val": 3, "test": 2}
    assert counts["day"] == {"train": 3, "val": 3, "test": 1}
    assert counts["aligned"] == {"train": 1, "val": 2, "test": 1}
    assert counts["final"] == {"train": 1, "val": 1, "test": 1}


def test_summary_markdown_has_the_tables_the_thresholds_and_the_match_gate(world):
    stats, day_log, aligned, dedups = world
    cfg = {
        "day_filter": {"min_lum_median": 0.1},
        "align": {"max_match_dist_px": 40, "min_matches": 6, "homography": {"min_matches": 10}},
        "dedup": {"max_hamming": 6},
    }
    text = report.build_summary(stats, day_log, aligned, dedups, cfg)
    assert text.startswith("# DroneVehicle data summary")
    assert "| stage | train | val | test | total |" in text
    assert "| final pairs (manifests) | 1 | 1 | 1 | 3 |" in text
    assert "| scale | 1 | 0 | 0 | 1 |" in text
    assert "| near_duplicate | 0 | 1 | 0 | 1 |" in text
    assert "max_match_dist_px: 40" in text  # the gate the plan does not state
    assert "min_lum_median: 0.1" in text and "max_hamming: 6" in text
    assert "homography" in text.lower()
    assert "\r" not in text


def test_summary_reports_alignment_diagnostics(world):
    stats, day_log, aligned, dedups = world
    text = report.build_summary(stats, day_log, aligned, dedups, {})
    assert "homography" in text and "moving" in text and "copied" in text


# --- stratified sampling -------------------------------------------------------------------------


def test_stratified_sample_covers_every_group_and_totals_n():
    groups = {"a": list(range(100)), "b": list(range(10)), "c": list(range(3))}
    picked = report.stratified_sample(groups, 20, seed=0)
    assert len(picked) == 20
    assert {g for g, _ in picked} == {"a", "b", "c"}
    assert len({(g, i) for g, i in picked}) == 20  # no repeats


def test_stratified_sample_is_proportional_beyond_the_floor_of_one():
    groups = {"big": list(range(1000)), "small": list(range(100))}
    counts = {
        g: sum(1 for k, _ in report.stratified_sample(groups, 22, 0) if k == g) for g in groups
    }
    assert counts == {"big": 20, "small": 2}


def test_a_small_group_gives_all_it_has_and_the_surplus_goes_elsewhere():
    groups = {"a": list(range(50)), "b": [0]}
    picked = report.stratified_sample(groups, 10, seed=1)
    assert sum(1 for g, _ in picked if g == "b") == 1
    assert len(picked) == 10


def test_stratified_sample_is_seeded():
    groups = {"a": list(range(100)), "b": list(range(100))}
    assert report.stratified_sample(groups, 12, 5) == report.stratified_sample(groups, 12, 5)
    assert report.stratified_sample(groups, 12, 5) != report.stratified_sample(groups, 12, 6)


def test_asking_for_more_than_exists_returns_everything():
    groups = {"a": [1, 2], "b": [3]}
    assert len(report.stratified_sample(groups, 50, 0)) == 3


def test_fewer_slots_than_groups_keeps_the_largest_groups():
    groups = {"a": list(range(10)), "b": list(range(30)), "c": list(range(20))}
    picked = report.stratified_sample(groups, 2, 0)
    assert sorted(g for g, _ in picked) == ["b", "c"]
