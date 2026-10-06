"""End to end: prepare_dronevehicle.py stats -> day -> align -> dedup -> write on the fake tree.

The pipeline is driven through the script's own `main()`, as a user would, with `--workers 1`.
Nothing here reads data/DroneVehicle.
"""

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")
omegaconf = pytest.importorskip("omegaconf")

from PIL import Image  # noqa: E402

from dawn.data import align, dedup, pairs, report  # noqa: E402
from dawn.data import day_filter as df  # noqa: E402
from dawn.data import dronevehicle as dv  # noqa: E402
from tests.fake_dronevehicle import (  # noqa: E402
    EXPECTED_FALLBACK,
    EXPECTED_FINAL,
    EXPECTED_MANIFESTS,
    FAKE_PAIRS,
    fake_config,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_dronevehicle.py"


def _module():
    spec = importlib.util.spec_from_file_location("script_prepare_dronevehicle", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(config_path, *argv):
    """Run `prepare_dronevehicle.py <argv> --data-config <config> --workers 1` in-process."""
    full = [str(SCRIPT), *argv, "--data-config", str(config_path), "--workers", "1"]
    with mock.patch.object(sys, "argv", full):
        _module().main()


class Pipeline:
    def __init__(self, base, tree):
        self.base = base
        self.cfg = fake_config(base, tree.root)
        self.config_path = base / "data.yaml"
        self.save()
        self.layout = dv.DroneVehicleLayout.from_config(self.cfg["dronevehicle"])
        self.tree = tree

    def save(self):
        omegaconf.OmegaConf.save(omegaconf.OmegaConf.create(self.cfg), self.config_path)

    def run(self, *argv):
        run_script(self.config_path, *argv)


@pytest.fixture(scope="module")
def pipe(fake_dronevehicle, tmp_path_factory):
    p = Pipeline(tmp_path_factory.mktemp("e2e"), fake_dronevehicle)
    for stage in ("stats", "day", "align", "dedup", "write"):
        p.run(stage)
    return p


# --- stats and day --------------------------------------------------------------------------------


def test_stats_has_a_row_for_every_stem_including_the_unpaired_one(pipe):
    rows = {r.image_id: r for r in df.read_stats_csv(pipe.layout.stats_csv)}
    assert set(rows) == {spec.image_id for spec in FAKE_PAIRS}
    assert rows["train_00005"].status == "missing_thermal"
    assert rows["train_00009"].border_rgb == "dark"
    assert rows["train_00001"].folder == "flight_00001"
    assert all(r.status == "ok" for i, r in rows.items() if i != "train_00005")


def test_day_log_gives_each_reason_and_the_outputs_exist(pipe):
    log = df.read_day_log(pipe.layout.log("day"))
    assert log["train_00004"] == (False, "not_day")
    assert log["train_00005"] == (False, "missing_thermal")
    assert log["train_00009"] == (False, "bad_border")
    assert all(log[i][0] for i in ("train_00001", "train_00002", "val_00001", "test_00001"))
    out = pipe.layout.outputs_dir
    assert (out / "day_histogram.png").stat().st_size > 1000
    assert (out / "day_sheets" / "day_random.jpg").exists()
    assert (out / "day_folders.csv").exists()


# --- align and dedup ------------------------------------------------------------------------------


def test_align_log_has_the_expected_reason_for_every_day_pair(pipe):
    log = align.read_align_log(pipe.layout.log("align"))
    day = {i for i, (is_day, _) in df.read_day_log(pipe.layout.log("day")).items() if is_day}
    assert set(log) == day
    for image_id, row in log.items():
        expected = EXPECTED_FINAL[image_id]
        # the duplicate passes alignment and is only dropped by dedup
        assert row.reason == ("ok" if expected == "near_duplicate" else expected), image_id


def test_accepted_transforms_are_close_to_the_ground_truth(pipe):
    log = align.read_align_log(pipe.layout.log("align"))
    xs, ys = np.meshgrid(np.linspace(0, 640, 9), np.linspace(0, 512, 7))
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)
    centre = np.array([[320.0, 256.0]])
    for image_id, row in log.items():
        if not row.accepted:
            continue
        T, truth = align.transform_of(row), pipe.tree.truth[image_id]
        if row.model == "edge_translation":  # a translation only matches the truth at the centre
            moved = align.apply_transform(T, centre) - align.apply_transform(truth, centre)
            assert np.abs(moved).max() < 1.0, image_id
            continue
        fit = align.apply_transform(T, pts)
        assert np.abs(fit - align.apply_transform(truth, pts)).max() < 0.7, image_id


def test_the_fallback_aligned_exactly_the_planted_pairs(pipe):
    log = align.read_align_log(pipe.layout.log("align"))
    by_fallback = sorted(i for i, r in log.items() if r.model == "edge_translation")
    assert by_fallback == EXPECTED_FALLBACK
    tried = sorted(i for i, r in log.items() if r.fb_tried)
    assert tried == ["train_00002", "train_00003", "train_00011", "train_00012"]


def test_the_fallback_manifest_dataset_flag_and_empty_moving_masks(pipe):
    path = pipe.layout.manifest("fallback_pairs")
    assert pairs.read_manifest(path) == EXPECTED_FALLBACK
    ds = pairs.AlignedPairs(pipe.layout, "train_pairs")
    flags = {ds[i]["image_id"]: ds[i]["fallback"] for i in range(len(ds))}
    assert {k for k, v in flags.items() if v} == set(EXPECTED_FALLBACK)
    for image_id in EXPECTED_FALLBACK:
        moving = np.asarray(Image.open(pipe.layout.aligned_path(image_id, "moving")))
        assert not moving.any()  # nothing is known about which vehicles moved
    assert np.asarray(Image.open(pipe.layout.aligned_path("train_00001", "moving"))).any()


def test_dedup_drops_the_planted_val_duplicate_and_never_touches_train(pipe):
    log = dedup.read_dedup_log(pipe.layout.log("dedup"))
    assert set(log) == {"val_00001", "val_00002", "test_00001"}  # no train pair is ever a candidate
    dup = log["val_00001"]
    assert (dup.matched_id, dup.distance, dup.near_duplicate) == ("train_00001", 0, True)
    assert not log["val_00002"].near_duplicate and not log["test_00001"].near_duplicate
    assert log["val_00002"].distance > 6


# --- write ----------------------------------------------------------------------------------------


def test_manifests_are_exactly_the_expected_sorted_ids(pipe):
    for split, expected in EXPECTED_MANIFESTS.items():
        path = pipe.layout.manifest(f"{split}_pairs")
        assert pairs.read_manifest(path) == expected
        assert path.read_bytes() == ("\n".join(expected) + "\n").encode()


def test_pngs_exist_for_manifest_pairs_only(pipe):
    wanted = {i for ids in EXPECTED_MANIFESTS.values() for i in ids}
    for spec in FAKE_PAIRS:
        for kind in dv.ASSET_KINDS:
            exists = pipe.layout.aligned_path(spec.image_id, kind).exists()
            assert exists == (spec.image_id in wanted), (spec.image_id, kind)


def test_the_warped_thermal_matches_the_ground_truth_aligned_thermal(pipe):
    from dawn.data.align import warp_thermal

    for image_id in ("train_00001", "val_00002", "test_00001"):
        split, stem = dv.parse_image_id(image_id)
        thermal = (
            dv.crop_border(dv.read_thermal8(pipe.layout.thermal_image(split, stem))).float() / 255
        )
        truth, truth_valid = warp_thermal(thermal, pipe.tree.truth[image_id], (512, 640))
        got = np.asarray(Image.open(pipe.layout.aligned_path(image_id, "thermal"))) / 255.0
        valid = np.asarray(Image.open(pipe.layout.aligned_path(image_id, "valid"))) > 127
        both = valid & truth_valid[0].numpy()
        err = np.abs(got - truth[0].numpy())[both]
        assert err.mean() < 0.01, (image_id, err.mean())  # sub-pixel fit error at edges only
        raw_err = np.abs(thermal[0].numpy() - truth[0].numpy())[both].mean()
        assert err.mean() < raw_err / 3


def test_the_dataset_serves_what_write_stored(pipe):
    ds = pairs.AlignedPairs(pipe.layout, "train_pairs")
    assert [ds[i]["image_id"] for i in range(len(ds))] == EXPECTED_MANIFESTS["train"]
    item = ds[0]
    assert item["thermal"].shape == (1, 512, 640) and item["moving"].any()  # train_00001's mover


def test_summary_counts_every_stage_and_lists_the_thresholds(pipe):
    text = (pipe.layout.manifest_dir / "logs" / "summary.md").read_text(encoding="utf-8")
    assert "| final pairs (manifests) | 4 | 1 | 1 | 6 |" in text
    assert "aligned by the edge-correlation fallback: 2 (tried: 4)" in text
    for reason in (
        "labels_copied",
        "few_matches",
        "not_day",
        "scale",
        "residual",
        "edge_check",
        "near_duplicate",
        "bad_border",
        "missing_thermal",
    ):
        assert f"| {reason} | " in text, reason
    assert "max_match_dist_px: 40" in text and "min_lum_median" in text


def test_final_reasons_match_the_documented_expectations(pipe):
    stats = df.read_stats_csv(pipe.layout.stats_csv)
    final = report.final_reasons(
        stats,
        df.read_day_log(pipe.layout.log("day")),
        align.read_align_log(pipe.layout.log("align")),
        dedup.read_dedup_log(pipe.layout.log("dedup")),
    )
    assert final == EXPECTED_FINAL


# --- re-running, thresholds, dry run, ordering ----------------------------------------------------


def test_rerunning_is_idempotent(pipe):
    before = {name: pipe.layout.log(name).read_bytes() for name in ("day", "align", "dedup")} | {
        "train": pipe.layout.manifest("train_pairs").read_bytes(),
        "fallback": pipe.layout.manifest("fallback_pairs").read_bytes(),
    }
    png = pipe.layout.aligned_path("train_00001", "thermal")
    mtime = png.stat().st_mtime_ns
    pipe.run("stats")
    pipe.run("day", "--no-sheets")
    pipe.run("align")
    pipe.run("dedup")
    pipe.run("write")
    after = {name: pipe.layout.log(name).read_bytes() for name in ("day", "align", "dedup")} | {
        "train": pipe.layout.manifest("train_pairs").read_bytes(),
        "fallback": pipe.layout.manifest("fallback_pairs").read_bytes(),
    }
    assert after == before
    assert png.stat().st_mtime_ns == mtime  # write skipped the finished pair


def test_the_dry_run_with_copied_boxes_writes_only_to_outputs(pipe):
    committed = pipe.layout.log("align").read_bytes()
    pipe.run("align", "--include-copied")
    assert pipe.layout.log("align").read_bytes() == committed
    dry = align.read_align_log(pipe.layout.outputs_dir / "align_include_copied.csv")
    assert dry["train_00002"].reason == "ok"  # zero residual by construction: why copies are barred
    assert dry["train_00002"].n_copied == 10


def test_changed_thresholds_recompute_the_align_log(fake_dronevehicle, tmp_path):
    p = Pipeline(tmp_path, fake_dronevehicle)
    for stage in ("stats", "day", "align"):
        p.run(stage)
    assert align.read_align_log(p.layout.log("align"))["train_00007"].reason == "residual"
    p.cfg["align"]["max_median_residual_px"] = 3.0
    p.save()
    p.run("align")  # no --force: the fingerprint notices
    assert align.read_align_log(p.layout.log("align"))["train_00007"].reason != "residual"


def test_write_refuses_to_run_before_dedup(fake_dronevehicle, tmp_path):
    p = Pipeline(tmp_path, fake_dronevehicle)
    for stage in ("stats", "day", "align"):
        p.run(stage)
    with pytest.raises(SystemExit, match="dedup"):
        p.run("write")


def test_the_stats_stage_skips_when_the_csv_exists(pipe, caplog):
    with caplog.at_level("INFO"):
        pipe.run("stats")
    assert "skipping stats" in caplog.text


@pytest.mark.parametrize("stage", ["stats", "day", "align", "dedup", "write", "all"])
def test_every_stage_has_help(stage):
    with mock.patch.object(sys, "argv", [str(SCRIPT), stage, "--help"]):
        with pytest.raises(SystemExit) as exit_info:
            _module().main()
    assert exit_info.value.code == 0


AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_alignment.py"


def test_the_audit_script_writes_sheets_histograms_and_the_sample_list(pipe):
    spec = importlib.util.spec_from_file_location("script_audit_alignment", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        str(AUDIT_SCRIPT),
        "--data-config",
        str(pipe.config_path),
        "--workers",
        "1",
        "--accepted",
        "3",
        "--rejected",
        "4",
        "--fallback",
        "3",
        "--per-sheet",
        "2",
        "--seed",
        "1",
    ]
    with mock.patch.object(sys, "argv", argv):
        module.main()
    out = pipe.layout.outputs_dir / "audit"
    assert (out / "fit_histograms.png").stat().st_size > 1000
    # the fallback pairs get sheets of their own: 2 aligned by edge correlation, 2 it rejected
    assert (out / "fallback_accepted_01.jpg").stat().st_size > 1000
    assert (out / "fallback_rejected_01.jpg").stat().st_size > 1000
    assert not list(out.glob("fallback_accepted_02.jpg"))
    for name in ("usable_luminance.png", "usable_darkest.jpg", "usable_random.jpg"):
        assert (out / name).stat().st_size > 1000, name
    assert sorted(p.name for p in out.glob("accepted_*.jpg")) == [
        "accepted_01.jpg",
        "accepted_02.jpg",
    ]
    assert sorted(p.name for p in out.glob("rejected_*.jpg")) == [
        "rejected_01.jpg",
        "rejected_02.jpg",
    ]
    lines = (out / "sampled.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "sheet,position,image_id,reason"
    assert len(lines) == 1 + 3 + 4 + 2 + 2  # + the fallback sheets
    rejected_reasons = {line.split(",")[3] for line in lines[1:] if line.startswith("rejected")}
    assert len(rejected_reasons) >= 3  # stratified: more than one reason on the sheets
