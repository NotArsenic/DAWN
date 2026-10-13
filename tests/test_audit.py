"""Tests for alignment audit visualizations, edge overlays, and fit metric plots."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")
pytest.importorskip("omegaconf")

from dawn.data import align, audit  # noqa: E402
from dawn.data import dronevehicle as dv  # noqa: E402
from tests.fake_dronevehicle import fake_config  # noqa: E402


def test_edge_overlay_paints_thermal_edges_and_nothing_else():
    rgb = torch.full((3, 64, 80), 100, dtype=torch.uint8)
    thermal = torch.zeros(1, 64, 80)
    thermal[:, 20:44, 30:50] = 1.0  # a hot rectangle: strong edges around its border
    out = audit.edge_overlay(rgb, thermal, keep_fraction=0.06)
    changed = (out != rgb).any(dim=0)
    assert changed.any()
    assert (out[:, changed] == torch.tensor(audit.EDGE_COLOR, dtype=torch.uint8)[:, None]).all()
    ys, xs = torch.nonzero(changed, as_tuple=True)
    assert ys.min() >= 17 and ys.max() <= 47 and xs.min() >= 27 and xs.max() <= 53  # near the box
    assert not changed[0:10].any() and not changed[:, 0:10].any()  # flat areas are untouched
    assert torch.equal(rgb, torch.full((3, 64, 80), 100, dtype=torch.uint8))  # input unchanged


def test_edge_overlay_respects_the_valid_mask():
    rgb = torch.full((3, 64, 80), 100, dtype=torch.uint8)
    thermal = torch.zeros(1, 64, 80)
    thermal[:, 20:44, 30:50] = 1.0
    valid = torch.ones(1, 64, 80, dtype=torch.bool)
    valid[:, :, 40:] = False
    out = audit.edge_overlay(rgb, thermal, valid)
    assert not (out != rgb).any(dim=0)[:, 40:].any()


@pytest.fixture(scope="module")
def rows(fake_dronevehicle, tmp_path_factory):
    base = tmp_path_factory.mktemp("audit")
    cfg = fake_config(base, fake_dronevehicle.root)
    layout = dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])
    acfg = align.AlignConfig.from_config(cfg["align"])
    ids = ["train_00001", "train_00010", "val_00002", "train_00011", "train_00006", "train_00008"]
    out = [align.align_pair(layout, acfg, *dv.parse_image_id(i)) for i in ids]
    return layout, acfg, out


def test_audit_tile_is_a_uint8_thumbnail_within_the_box(rows):
    layout, _, aligned = rows
    for row in aligned:
        tile = audit.audit_tile(layout, row)
        assert tile.dtype == torch.uint8 and tile.shape[0] == 3
        assert tile.shape[1] <= 256 and tile.shape[2] <= 320


def test_a_pair_without_a_fit_still_gets_a_tile(rows):
    layout, _, aligned = rows
    row = dataclasses.replace(aligned[3], model="none", transform="")  # as if only one match
    assert audit.audit_tile(layout, row).shape[0] == 3


def test_sheets_are_written_in_pages(rows, tmp_path):
    layout, _, aligned = rows
    accepted = [r for r in aligned if r.accepted]
    rejected = [r for r in aligned if not r.accepted]
    written = audit.save_audit_sheets(layout, accepted, rejected, tmp_path, per_sheet=2, workers=1)
    names = [p.name for p in written]
    assert len(accepted) == 3 and len(rejected) == 3
    assert names == ["accepted_01.jpg", "accepted_02.jpg", "rejected_01.jpg", "rejected_02.jpg"]
    assert all(p.stat().st_size > 1000 for p in written)


def test_histograms_are_written_as_a_png(rows, tmp_path):
    _, acfg, aligned = rows
    out = tmp_path / "audit" / "hist.png"
    audit.plot_fit_histograms(aligned, acfg, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_dedup_examples_are_split_between_just_under_and_just_over():
    from dawn.data.dedup import DedupRow

    rows = [
        DedupRow(f"val_{d}_{i}", "train_1", d, d, d, d <= 6) for d in range(0, 14) for i in range(3)
    ]
    picked = audit.pick_dedup_examples(rows, max_hamming=6, count=8)
    distances = sorted(r.distance for r in picked)
    assert len(picked) == 8
    # four from each side, the closest to the threshold first on each
    assert distances == [5, 6, 6, 6, 7, 7, 7, 8]
    assert any(r.distance <= 6 for r in picked) and any(r.distance > 6 for r in picked)


def test_dedup_examples_are_topped_up_with_the_closest_matches_when_none_are_near():
    from dawn.data.dedup import DedupRow

    rows = [DedupRow(f"val_{d}", "train_1", d, d, d, False) for d in (12, 14, 16, 20, 22, 24)]
    picked = audit.pick_dedup_examples(rows, max_hamming=6, count=4)
    assert [r.distance for r in picked] == [12, 14, 16, 20]  # nothing near 6: the closest four


def test_dedup_examples_fall_back_to_whatever_exists():
    from dawn.data.dedup import DedupRow

    rows = [DedupRow("val_1", "train_1", 5, 5, 5, True), DedupRow("val_2", "", 65, 64, 64, False)]
    assert [r.image_id for r in audit.pick_dedup_examples(rows, 6, 8)] == ["val_1"]


def test_dedup_sheet_is_written_for_real_pairs(rows, tmp_path):
    from dawn.data.dedup import DedupRow

    layout, _, _ = rows
    near = [
        DedupRow("val_00001", "train_00001", 6, 5, 6, True),
        DedupRow("val_00002", "train_00001", 7, 7, 6, False),
    ]
    path = audit.save_dedup_sheet(layout, near, 6, tmp_path / "d.jpg", count=4, workers=1)
    assert path is not None and path.stat().st_size > 1000
    assert audit.save_dedup_sheet(layout, [], 6, tmp_path / "none.jpg") is None


# --- usable pairs: luminance plot and RGB | thermal sheets -------------------------------------


def _stats_row(image_id, lum):
    from dawn.data.day_filter import PairStats

    split, stem = image_id.split("_")
    return PairStats(
        image_id=image_id,
        split=split,
        stem=stem,
        status="ok",
        folder="f",
        rgb_filename="a",
        thermal_filename="b",
        border_rgb="",
        border_thermal="",
        thermal_channels_ok=True,
        lum_median=lum,
        lum_p05=0.1,
        lum_p95=0.5,
        lum_mean=0.2,
        rgb_hash="0" * 16,
        thermal_hash="0" * 16,
    )


def test_usable_luminance_plot_is_written_and_counts_usable_against_rejected_day_pairs(tmp_path):
    stats = [_stats_row(f"train_{i}", 0.18 + 0.01 * i) for i in range(40)]
    stats += [_stats_row(f"val_{i}", 0.2 + 0.02 * i) for i in range(10)]
    stats += [_stats_row("test_0", 0.3), _stats_row("test_1", 0.01)]  # test_1 is not day
    day_log = {
        r.image_id: (r.lum_median >= 0.18, "ok" if r.lum_median >= 0.18 else "not_day")
        for r in stats
    }
    usable = {r.image_id for r in stats if r.image_id.endswith(("0", "2", "4", "6"))}
    out = tmp_path / "plots" / "usable.png"
    counts = audit.plot_usable_luminance(stats, day_log, usable, 0.18, out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert counts["train"] == (sum(i in usable for i in {f"train_{k}" for k in range(40)}), 40)
    assert counts["val"][1] == 10 and counts["test"][1] == 1  # day pairs per split
    assert counts["test"][0] == 1  # test_0 is usable, test_1 was never a day pair


@pytest.fixture()
def aligned_layout(tmp_path):
    import numpy as np
    from PIL import Image

    cfg = {
        "root": str(tmp_path / "raw"),
        "aligned_dir": str(tmp_path / "aligned"),
        "stats_csv": str(tmp_path / "s.csv"),
        "manifest_dir": str(tmp_path / "m"),
        "outputs_dir": str(tmp_path / "o"),
    }
    layout = dv.DroneVehicleLayout.from_config(cfg)
    rng = np.random.default_rng(0)
    for i in range(5):
        iid = f"train_{i:05d}"
        for kind, array in (
            ("rgb", rng.integers(0, 255, (512, 640, 3), dtype=np.uint8)),
            ("thermal", rng.integers(0, 255, (512, 640), dtype=np.uint8)),
        ):
            path = layout.aligned_path(iid, kind)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)
    return layout


def test_usable_tiles_are_rgb_then_thermal_thumbnails_of_the_written_pngs(aligned_layout):
    rgb, thermal = audit.usable_tiles(aligned_layout, "train_00001")
    assert rgb.dtype == thermal.dtype == torch.uint8
    assert rgb.shape[0] == thermal.shape[0] == 3
    assert rgb.shape == thermal.shape


def test_usable_sheet_shows_the_requested_pairs_with_their_luminance(aligned_layout, tmp_path):
    ids = [f"train_{i:05d}" for i in range(5)]
    lum = {i: 0.2 + 0.01 * n for n, i in enumerate(ids)}
    out = audit.save_usable_sheet(
        aligned_layout, ids[:4], lum, tmp_path / "sheet.jpg", title="darkest usable", workers=1
    )
    assert out is not None and out.stat().st_size > 1000
    assert (
        audit.save_usable_sheet(aligned_layout, [], lum, tmp_path / "none.jpg", title="t") is None
    )


def test_pick_darkest_and_random_usable_ids():
    lum = {f"train_{i}": 0.18 + i / 100 for i in range(50)}
    darkest = audit.pick_usable(lum, "darkest", 5, seed=0)
    assert darkest == [f"train_{i}" for i in range(5)]
    sample = audit.pick_usable(lum, "random", 5, seed=0)
    assert len(sample) == 5 and sample == audit.pick_usable(lum, "random", 5, seed=0)
    assert sample != audit.pick_usable(lum, "random", 5, seed=1)
    with pytest.raises(ValueError, match="how"):
        audit.pick_usable(lum, "brightest", 5, seed=0)


def _fb_row(**kw):
    from dawn.data.align import _failed_row

    base = dataclasses.replace(
        _failed_row("train_00002", "ok"),
        accepted=True,
        model="edge_translation",
        fb_tried=True,
        fb_dx=3.0,
        fb_dy=-1.0,
        fb_corr=0.7,
        fb_prominence=0.05,
        fb_quad_dev=1.0,
        transform="1 0 3 0 1 -1",
    )
    return dataclasses.replace(base, **kw)


def test_a_fallback_row_caption_shows_the_estimate_not_a_box_residual():
    caption = audit._caption(_fb_row())
    assert "fb" in caption and "(+3,-1)" in caption and "p0.05" in caption and "q1" in caption


def test_fit_histograms_leave_fallback_rows_out_of_the_box_fit_panels(rows, tmp_path):
    _, acfg, aligned = rows
    out = tmp_path / "h.png"
    audit.plot_fit_histograms([*aligned, _fb_row()], acfg, out)  # must not trip on NaN residuals
    assert out.stat().st_size > 1000
    assert audit.box_fitted([*aligned, _fb_row()]) == [r for r in aligned if r.model != "none"]


def test_sheet_names_can_carry_a_prefix(rows, tmp_path):
    layout, _, aligned = rows
    written = audit.save_audit_sheets(
        layout, aligned[:1], aligned[3:4], tmp_path, per_sheet=2, workers=1, prefix="fallback_"
    )
    assert [p.name for p in written] == ["fallback_accepted_01.jpg", "fallback_rejected_01.jpg"]


def test_vehicle_tile_tints_the_masked_pixels(aligned_layout):
    import numpy as np
    from PIL import Image

    iid = "train_00001"
    mask = np.zeros((512, 640), np.uint8)
    mask[100:300, 100:300] = 255
    Image.fromarray(mask).save(aligned_layout.aligned_path(iid, "vehicles"))
    tile = audit.vehicles_tile(aligned_layout, iid, box=(256, 320))
    assert tile.dtype == torch.uint8 and tile.shape == (3, 256, 320)
    # inside the mask the red channel dominates the (random) RGB; outside it is untouched noise
    inside = tile[:, 70:130, 60:100].float().mean(dim=(1, 2))
    assert inside[0] > inside[1] and inside[0] > inside[2]


def test_vehicle_sheet_is_written_with_the_fractions_in_the_captions(aligned_layout, tmp_path):
    import numpy as np
    from PIL import Image

    ids = [f"train_{i:05d}" for i in range(3)]
    for i in ids:
        Image.fromarray(np.zeros((512, 640), np.uint8)).save(
            aligned_layout.aligned_path(i, "vehicles")
        )
    out = audit.save_vehicle_sheet(
        aligned_layout,
        [(ids[0], 0.41), (ids[1], 0.33)],
        tmp_path / "dense.jpg",
        title="dense frames",
    )
    assert out is not None and out.stat().st_size > 1000
    assert audit.save_vehicle_sheet(aligned_layout, [], tmp_path / "none.jpg", title="t") is None
