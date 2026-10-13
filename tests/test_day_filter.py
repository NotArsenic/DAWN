"""Tests for daytime filtering criteria, luminance statistics, and logging."""

import dataclasses
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("omegaconf")
pytest.importorskip("matplotlib")

from dawn.data import day_filter as df  # noqa: E402
from dawn.data import dedup  # noqa: E402
from dawn.data import dronevehicle as dv  # noqa: E402
from dawn.data.color import luminance, srgb8_to_linear  # noqa: E402
from tests.fake_dronevehicle import DAY_CUTOFF, fake_config  # noqa: E402


@pytest.fixture(scope="module")
def layout(fake_dronevehicle, tmp_path_factory):
    cfg = fake_config(tmp_path_factory.mktemp("cfg"), fake_dronevehicle.root)
    return dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])


def _row(**kw):
    base = dict(
        image_id="train_00001",
        split="train",
        stem="00001",
        status="ok",
        folder="f",
        rgb_filename="a",
        thermal_filename="b",
        border_rgb="",
        border_thermal="",
        thermal_channels_ok=True,
        lum_median=0.2,
        lum_p05=0.05,
        lum_p95=0.5,
        lum_mean=0.22,
        rgb_hash="0" * 16,
        thermal_hash="f" * 16,
    )
    base.update(kw)
    return df.PairStats(**base)


# --- the decision ---------------------------------------------------------------------------------


def test_is_day_is_at_or_above_the_cutoff():
    assert df.is_day(0.10, 0.10)
    assert df.is_day(0.1000001, 0.10)
    assert not df.is_day(0.0999999, 0.10)


def test_is_day_is_false_for_nan():
    assert not df.is_day(float("nan"), 0.10)


@pytest.mark.parametrize(
    "changes, reason",
    [
        ({}, "ok"),
        ({"lum_median": 0.01}, "not_day"),
        ({"border_rgb": "dark"}, "bad_border"),
        ({"border_thermal": "wide"}, "bad_border"),
        ({"thermal_channels_ok": False}, "thermal_channels"),
        ({"status": "missing_thermal", "lum_median": float("nan")}, "missing_thermal"),
        ({"status": "missing_thermal", "lum_median": 0.01}, "missing_thermal"),  # status first
        ({"border_rgb": "dark", "lum_median": 0.01}, "bad_border"),  # a defect beats "not day"
    ],
)
def test_day_reason_orders_the_failures(changes, reason):
    assert df.day_reason(_row(**changes), 0.10) == reason


# --- stats ----------------------------------------------------------------------------------------


def test_stats_of_a_good_pair_match_an_independent_measurement(layout):
    row = df.compute_pair_stats(layout, "train", "00001")
    assert row.status == "ok"
    assert row.image_id == "train_00001"
    assert row.folder == "flight_00001"
    assert (row.rgb_filename, row.thermal_filename) == ("DJI_00001", "DJI_00001_R")
    crop = dv.crop_border(dv.read_rgb8(layout.rgb_image("train", "00001")))
    lum = luminance(srgb8_to_linear(crop)).flatten()
    p05, med, p95 = torch.quantile(lum, torch.tensor([0.05, 0.5, 0.95])).tolist()
    assert row.lum_median == pytest.approx(med, rel=1e-5)
    assert row.lum_p05 == pytest.approx(p05, rel=1e-5)
    assert row.lum_p95 == pytest.approx(p95, rel=1e-5)
    assert row.border_rgb == row.border_thermal == ""
    assert row.thermal_channels_ok
    assert len(row.rgb_hash) == len(row.thermal_hash) == 16


def test_the_night_frame_is_below_the_fake_cutoff_and_the_day_frames_above(layout):
    night = df.compute_pair_stats(layout, "train", "00004")
    day = df.compute_pair_stats(layout, "train", "00001")
    assert night.lum_median < DAY_CUTOFF <= day.lum_median


def test_stats_flag_a_stem_missing_its_thermal_image(layout):
    row = df.compute_pair_stats(layout, "train", "00005")
    assert row.status == "missing_thermal"
    assert math.isnan(row.lum_median)
    assert row.rgb_hash == row.thermal_hash == ""


def test_stats_flag_a_bad_border(layout):
    row = df.compute_pair_stats(layout, "train", "00009")
    assert row.status == "ok"  # readable; the defect is its own field
    assert row.border_rgb == "dark"
    assert df.day_reason(row, DAY_CUTOFF) == "bad_border"


def test_duplicate_pairs_hash_identically_and_other_pairs_do_not(layout):
    a = df.compute_pair_stats(layout, "train", "00001")
    dup = df.compute_pair_stats(layout, "val", "00001")
    other = df.compute_pair_stats(layout, "val", "00002")
    assert (a.rgb_hash, a.thermal_hash) == (dup.rgb_hash, dup.thermal_hash)
    assert dedup.hamming(dedup.from_hex(a.rgb_hash), dedup.from_hex(other.rgb_hash)) > 10


def test_stats_flag_a_missing_label_file(layout, tmp_path):
    import shutil

    cfg_layout = dv.DroneVehicleLayout(
        root=tmp_path,
        aligned_dir=tmp_path,
        stats_csv=tmp_path / "s.csv",
        manifest_dir=tmp_path,
        outputs_dir=tmp_path,
    )
    for kind in ("img", "imgr"):
        (tmp_path / "train" / f"train{kind}").mkdir(parents=True)
    shutil.copy(layout.rgb_image("train", "00001"), cfg_layout.rgb_image("train", "00001"))
    shutil.copy(layout.thermal_image("train", "00001"), cfg_layout.thermal_image("train", "00001"))
    assert df.compute_pair_stats(cfg_layout, "train", "00001").status == "missing_label"


# --- CSV ------------------------------------------------------------------------------------------


def test_stats_csv_round_trips_including_nan_and_empty_fields(tmp_path):
    rows = [
        _row(image_id="val_00002", split="val", stem="00002"),
        _row(),
        _row(
            image_id="train_00005",
            split="train",
            stem="00005",
            status="missing_thermal",
            lum_median=float("nan"),
            lum_p05=float("nan"),
            lum_p95=float("nan"),
            lum_mean=float("nan"),
            rgb_hash="",
            thermal_hash="",
            border_rgb="dark",
        ),
    ]
    path = tmp_path / "s" / "stats.csv"
    df.write_stats_csv(path, rows)
    back = df.read_stats_csv(path)
    assert [r.image_id for r in back] == ["train_00001", "train_00005", "val_00002"]  # sorted
    assert back[1].status == "missing_thermal" and math.isnan(back[1].lum_median)
    assert back[1].rgb_hash == "" and back[1].border_rgb == "dark"
    assert back[0] == rows[1]
    assert path.read_text(encoding="utf-8").count("\r") == 0  # LF line endings


def test_day_log_has_three_decimals_and_reads_back(tmp_path):
    rows = [_row(lum_median=0.123456), _row(image_id="train_00002", stem="00002", lum_median=0.01)]
    path = tmp_path / "day.csv"
    df.write_day_log(path, rows, 0.10)
    text = path.read_text(encoding="utf-8")
    assert (
        text.splitlines()[0]
        == "image_id,folder,log10_lum_median,log10_lum_p05,log10_lum_p95,day,reason"
    )
    assert "-0.908" in text  # log10(0.123456), 3 decimals
    log = df.read_day_log(path)
    assert log["train_00001"] == (True, "ok")
    assert log["train_00002"] == (False, "not_day")


# --- tables for checkpoint 1 ----------------------------------------------------------------------


def test_counts_at_cutoffs_per_split():
    rows = [
        _row(image_id="train_1", stem="1", lum_median=0.30),
        _row(image_id="train_2", stem="2", lum_median=0.12),
        _row(image_id="train_3", stem="3", lum_median=0.05),
        _row(image_id="val_1", split="val", stem="1", lum_median=0.20),
        _row(image_id="val_2", split="val", stem="2", status="missing_rgb", lum_median=0.9),
    ]
    counts = df.counts_at_cutoffs(rows, [0.04, 0.10, 0.25])
    assert counts[0.04] == {"train": 3, "val": 1}
    assert counts[0.10] == {"train": 2, "val": 1}
    assert counts[0.25] == {"train": 1, "val": 0}


def test_folder_table_reports_each_folders_day_share():
    rows = [
        _row(image_id="train_1", stem="1", folder="a", lum_median=0.3),
        _row(image_id="train_2", stem="2", folder="a", lum_median=0.2),
        _row(image_id="train_3", stem="3", folder="b", lum_median=0.3),
        _row(image_id="train_4", stem="4", folder="b", lum_median=0.01),
    ]
    table = {t["folder"]: t for t in df.folder_table(rows, 0.1)}
    assert table["a"]["frames"] == 2 and table["a"]["day_share"] == 1.0
    assert table["b"]["day_share"] == 0.5
    assert table["b"]["median_log10_lum"] == pytest.approx((math.log10(0.01) + math.log10(0.3)) / 2)
    assert [t["folder"] for t in df.folder_table(rows, 0.1, mixed_only=True)] == ["b"]


# --- plots and sheets -----------------------------------------------------------------------------


def test_histogram_is_written_as_a_png(tmp_path):
    rows = [
        _row(image_id=f"train_{i}", stem=str(i), lum_median=10 ** (-3 + i / 10)) for i in range(30)
    ]
    rows += [_row(image_id=f"val_{i}", split="val", stem=str(i), lum_median=0.3) for i in range(5)]
    out = tmp_path / "plots" / "hist.png"
    df.plot_log_luminance_histogram(rows, 0.1, out, bin_log10=0.1)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_contact_sheets_cover_random_bands_and_night_folders(layout, tmp_path):
    ids = [
        ("train", "00001"),
        ("train", "00004"),
        ("train", "00010"),
        ("val", "00002"),
        ("test", "00001"),
    ]
    rows = [df.compute_pair_stats(layout, s, t) for s, t in ids]
    rows[0] = dataclasses.replace(rows[0], folder="flight_night_1")
    written = df.save_day_sheets(
        layout, rows, DAY_CUTOFF, tmp_path, count=4, band_log10=2.0, seed=0, workers=1
    )
    names = {p.name for p in written}
    assert {"day_random.jpg", "band_above.jpg", "band_below.jpg", "night_folders.jpg"} <= names
    assert all(p.exists() and p.stat().st_size > 0 for p in written)


def test_the_night_folder_sheet_is_still_written_when_there_are_none(layout, tmp_path):
    rows = [df.compute_pair_stats(layout, "train", "00001")]
    written = df.save_day_sheets(
        layout, rows, DAY_CUTOFF, tmp_path, count=4, band_log10=0.1, workers=1
    )
    sheet = next(p for p in written if p.name == "night_folders.jpg")
    assert sheet.stat().st_size > 0  # an empty sheet replaces any stale one from an earlier cutoff
