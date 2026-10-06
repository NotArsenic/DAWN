"""dawn/data/align.py, the per-pair I/O layer: align_pair, the log, re-evaluation (fake tree)."""

import dataclasses
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("PIL")
pytest.importorskip("omegaconf")

from dawn.data import align  # noqa: E402
from dawn.data import dronevehicle as dv  # noqa: E402
from tests.fake_dronevehicle import EXPECTED_FINAL, fake_config  # noqa: E402


@pytest.fixture(scope="module")
def setup(fake_dronevehicle, tmp_path_factory):
    cfg = fake_config(tmp_path_factory.mktemp("cfg"), fake_dronevehicle.root)
    layout = dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])
    return layout, align.AlignConfig.from_config(cfg["align"]), fake_dronevehicle


@pytest.fixture(scope="module")
def rows(setup):
    layout, cfg, tree = setup
    out = {}
    for spec in tree.specs:
        if spec.has_thermal and spec.image_id not in ("train_00009", "val_00001"):
            out[spec.image_id] = align.align_pair(layout, cfg, spec.split, spec.stem)
    return out


def _same_map(fit, truth, tol=0.7):
    xs, ys = np.meshgrid(np.linspace(0, 640, 9), np.linspace(0, 512, 7))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    err = np.abs(align.apply_transform(fit, pts) - align.apply_transform(truth, pts)).max()
    assert err < tol, f"fitted transform is {err:.2f} px from the truth"


# --- align_pair ----------------------------------------------------------------------------------


@pytest.mark.parametrize("image_id", ["train_00001", "train_00010", "val_00002", "test_00001"])
def test_good_pairs_are_accepted_with_the_true_transform_recovered(setup, rows, image_id):
    _, _, tree = setup
    row = rows[image_id]
    assert row.reason == "ok" and row.accepted
    assert row.model == "similarity"
    _same_map(align.transform_of(row), tree.truth[image_id])
    assert row.edge_gain > 0.05  # the warp visibly improves the edges
    assert (row.edge_dx, row.edge_dy) == (0, 0)


def test_the_first_pair_has_a_moving_vehicle_a_copied_box_and_skipped_objects(rows):
    row = rows["train_00001"]
    assert row.n_moving == 1
    assert row.n_copied == 1
    assert row.n_matches == 9  # 10 vehicles, one copied (excluded)
    assert row.n_inliers == 8  # the mover is not an inlier
    assert row.n_rgb == 10 and row.n_thermal == 10


@pytest.mark.parametrize(
    "image_id, reason",
    [
        ("train_00002", "labels_copied"),
        ("train_00003", "few_matches"),
        ("train_00006", "scale"),
        ("train_00007", "residual"),
        ("train_00008", "edge_check"),
    ],
)
def test_each_failing_pair_gets_its_own_reason_code(rows, image_id, reason):
    assert rows[image_id].reason == reason
    assert EXPECTED_FINAL[image_id] == reason
    assert not rows[image_id].accepted


def test_the_edge_check_decides_only_after_the_geometry_passes(rows):
    row = rows["train_00008"]
    assert row.n_matches == 10 and row.median_residual < 1.0  # the boxes were fine
    # the content is 6 px from what the boxes say: the best shift saturates at the search window
    assert max(abs(row.edge_dx), abs(row.edge_dy)) == 4
    # geometry failures still log the edge numbers (for the audit), but the reason stays geometric
    assert not math.isnan(rows["train_00006"].edge_gain)
    assert rows["train_00006"].reason == "scale"


def test_copied_labels_are_accepted_only_in_a_dry_run(setup):
    layout, cfg, _ = setup
    dry = align.AlignConfig(**{**cfg.__dict__, "exclude_copied_boxes": False})
    row = align.align_pair(layout, dry, "train", "00002")
    assert row.n_copied == 10 and row.n_matches == 10
    assert row.median_residual == pytest.approx(0.0, abs=1e-6)
    assert row.reason == "ok"


def test_alignment_is_deterministic(setup):
    layout, cfg, _ = setup
    a = align.align_pair(layout, cfg, "train", "00001")
    b = align.align_pair(layout, cfg, "train", "00001")
    np.testing.assert_equal(dataclasses.asdict(a), dataclasses.asdict(b))  # NaN-safe equality


def test_missing_or_unreadable_files_are_a_read_error_not_a_crash(setup):
    layout, cfg, _ = setup
    row = align.align_pair(layout, cfg, "train", "99999")
    assert row.reason == "read_error" and not row.accepted
    assert math.isnan(row.median_residual)


def test_disabling_the_edge_check_skips_it(setup):
    layout, cfg, _ = setup
    off = align.AlignConfig(**{**cfg.__dict__, "grad_enabled": False})
    row = align.align_pair(layout, off, "train", "00008")
    assert row.reason == "ok" and math.isnan(row.edge_gain)


# --- the log -------------------------------------------------------------------------------------


def test_log_round_trips_the_transform_exactly_and_metrics_to_three_decimals(setup, rows, tmp_path):
    ordered = sorted(rows.values(), key=lambda r: r.image_id)
    path = tmp_path / "logs" / "align.csv"
    align.write_align_log(path, ordered)
    back = align.read_align_log(path)
    assert list(back) == [r.image_id for r in ordered]
    for row in ordered:
        got = back[row.image_id]
        assert got.reason == row.reason and got.accepted == row.accepted
        assert got.median_residual == pytest.approx(row.median_residual, abs=5e-4, nan_ok=True)
        if row.model != "none":
            np.testing.assert_allclose(align.transform_of(got), align.transform_of(row), rtol=1e-9)
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text


def test_transform_text_has_six_numbers_for_a_similarity_and_nine_for_a_homography():
    sim = np.array([[1.0, -0.01, 3.0], [0.01, 1.0, -2.0], [0.0, 0.0, 1.0]])
    hom = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [1e-5, 2e-5, 1.0]])
    assert len(align.format_transform("similarity", sim).split()) == 6
    assert len(align.format_transform("homography", hom).split()) == 9
    np.testing.assert_allclose(
        align.parse_transform("homography", align.format_transform("homography", hom)), hom
    )
    assert align.format_transform("none", None) == ""
    assert align.parse_transform("none", "") is None


# --- re-evaluation under other thresholds ------------------------------------------------------


def test_reevaluate_matches_the_run_under_the_same_thresholds(setup, rows):
    _, cfg, _ = setup
    for row in rows.values():
        assert align.reevaluate(row, cfg) == row.reason


def test_reevaluate_loosens_and_tightens_from_the_logged_numbers(setup, rows):
    _, cfg, _ = setup
    row = rows["train_00007"]  # median residual a few px
    assert row.median_residual > 2.0
    loose = align.AlignConfig(
        **{**cfg.__dict__, "max_median_residual_px": row.median_residual + 0.1}
    )
    assert align.reevaluate(row, loose) in ("ok", "edge_check")  # no longer the residual
    tight = align.AlignConfig(**{**cfg.__dict__, "max_median_residual_px": 0.0001})
    assert align.reevaluate(rows["train_00001"], tight) == "residual"
    few = align.AlignConfig(**{**cfg.__dict__, "min_matches": 3})
    assert align.reevaluate(rows["train_00003"], few) != "few_matches"


def test_config_fingerprint_changes_with_any_threshold_and_is_stable(setup):
    _, cfg, _ = setup
    assert align.config_fingerprint(cfg) == align.config_fingerprint(
        align.AlignConfig(**cfg.__dict__)
    )
    for change in (
        {"max_median_residual_px": 2.5},
        {"exclude_copied_boxes": False},
        {"grad_search_px": 5},
    ):
        other = align.AlignConfig(**{**cfg.__dict__, **change})
        assert align.config_fingerprint(other) != align.config_fingerprint(cfg)
