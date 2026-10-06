"""dawn/data/dronevehicle.py: layout, pair listing, raw readers, border check, XML parsing."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("omegaconf")

from dawn.data import dronevehicle as dv  # noqa: E402
from tests.fake_dronevehicle import (  # noqa: E402
    BORDER,
    FRAME_H,
    FRAME_W,
    H,
    W,
    _box,
    _vehicles,
    fake_config,
    write_framed_jpeg,
)


@pytest.fixture(scope="module")
def layout(fake_dronevehicle, tmp_path_factory):
    cfg = fake_config(tmp_path_factory.mktemp("cfg"), fake_dronevehicle.root)
    return dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])


def _spec(tree, image_id):
    return next(s for s in tree.specs if s.image_id == image_id)


# --- ids and layout ------------------------------------------------------------------------------


def test_image_id_round_trips_and_keeps_splits_apart():
    assert dv.image_id("train", "00001") == "train_00001"
    assert dv.parse_image_id("val_00001") == ("val", "00001")
    assert dv.image_id("train", "00001") != dv.image_id(
        "val", "00001"
    )  # stems repeat across splits


@pytest.mark.parametrize("bad", ["00001", "dev_00001", "train", ""])
def test_parse_image_id_rejects_ids_without_a_known_split(bad):
    with pytest.raises(ValueError, match="image ID"):
        dv.parse_image_id(bad)


def test_layout_paths_follow_the_raw_tree(layout):
    assert layout.rgb_image("train", "00001").parts[-2:] == ("trainimg", "00001.jpg")
    assert layout.thermal_image("val", "00002").parts[-2:] == ("valimgr", "00002.jpg")
    assert layout.rgb_label("test", "00001").parts[-2:] == ("testlabel", "00001.xml")
    assert layout.thermal_label("train", "00001").parts[-2:] == ("trainlabelr", "00001.xml")
    assert layout.crop_size == (512, 640)


def test_layout_aligned_paths_use_the_global_image_id(layout):
    path = layout.aligned_path("train_00001", "thermal")
    assert path.parts[-2:] == ("train", "train_00001_thermal.png")
    assert layout.aligned_path("val_00002", "valid").name == "val_00002_valid.png"
    with pytest.raises(ValueError, match="kind"):
        layout.aligned_path("train_00001", "depth")


# --- pair listing --------------------------------------------------------------------------------


def test_list_pairs_reports_stems_missing_a_modality(tmp_path):
    root = tmp_path / "raw" / "train"
    for kind in ("trainimg", "trainimgr"):
        (root / kind).mkdir(parents=True)
    for stem in ("00001", "00002", "00004"):
        (root / "trainimg" / f"{stem}.jpg").touch()
    for stem in ("00001", "00003", "00002"):
        (root / "trainimgr" / f"{stem}.jpg").touch()
    layout = dv.DroneVehicleLayout.from_config({**_minimal_cfg(), "root": str(tmp_path / "raw")})
    listing = dv.list_pairs(layout, "train")
    assert listing.paired == ["00001", "00002"]
    assert listing.rgb_only == ["00004"]
    assert listing.thermal_only == ["00003"]


def _minimal_cfg():
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "data.yaml")
    )
    return cfg["dronevehicle"]


def test_list_pairs_on_the_fake_tree(layout):
    listing = dv.list_pairs(layout, "train")
    assert "00005" in listing.rgb_only  # the stem with no thermal image
    assert "00001" in listing.paired and "00005" not in listing.paired
    assert listing.thermal_only == []
    assert dv.list_pairs(layout, "val").paired == ["00001", "00002"]


# --- raw readers ---------------------------------------------------------------------------------


def test_read_rgb8_is_uint8_channels_first(layout):
    img = dv.read_rgb8(layout.rgb_image("train", "00001"))
    assert img.dtype == torch.uint8
    assert img.shape == (3, FRAME_H, FRAME_W)


@pytest.mark.parametrize("stem, mode", [("00001", "L"), ("00010", "RGB")])
def test_read_thermal8_gives_one_channel_for_either_storage_mode(
    layout, fake_dronevehicle, stem, mode
):
    spec = _spec(fake_dronevehicle, f"train_{stem}")
    assert spec.thermal_mode == mode
    img = dv.read_thermal8(layout.thermal_image("train", stem))
    assert img.dtype == torch.uint8
    assert img.shape == (1, FRAME_H, FRAME_W)


def test_read_thermal8_takes_the_channel_and_not_a_luma_mix(tmp_path):
    content = np.random.default_rng(0).integers(0, 255, (H, W), dtype=np.uint8)
    write_framed_jpeg(tmp_path / "t.jpg", content, mode="RGB")
    one = dv.read_thermal8(tmp_path / "t.jpg")
    write_framed_jpeg(tmp_path / "l.jpg", content, mode="L")
    assert torch.equal(one, dv.read_thermal8(tmp_path / "l.jpg"))


def test_read_thermal8_rejects_channels_that_differ(tmp_path):
    from PIL import Image

    frame = np.full((FRAME_H, FRAME_W, 3), 255, np.uint8)
    frame[BORDER:-BORDER, BORDER:-BORDER] = (200, 100, 50)  # a coloured "thermal" image
    Image.fromarray(frame).save(tmp_path / "color.jpg", quality=95)
    with pytest.raises(dv.ThermalChannelError, match="color.jpg"):
        dv.read_thermal8(tmp_path / "color.jpg")


# --- crop and border -----------------------------------------------------------------------------


def test_crop_border_keeps_the_content_window():
    frame = torch.arange(FRAME_H * FRAME_W, dtype=torch.float32).view(1, FRAME_H, FRAME_W)
    crop = dv.crop_border(frame)
    assert crop.shape == (1, 512, 640)
    assert crop[0, 0, 0] == frame[0, 100, 100]
    assert crop[0, -1, -1] == frame[0, 611, 739]


def test_crop_border_refuses_frames_of_another_size():
    with pytest.raises(dv.FrameError, match="840"):
        dv.crop_border(torch.zeros(3, 700, 840))


def test_good_frames_pass_the_border_check(layout):
    for split, stem in (("train", "00001"), ("val", "00002")):
        assert dv.check_border(dv.read_rgb8(layout.rgb_image(split, stem))) is None
        assert dv.check_border(dv.read_thermal8(layout.thermal_image(split, stem))) is None


def test_a_dark_stripe_in_the_border_fails(layout):
    assert dv.check_border(dv.read_rgb8(layout.rgb_image("train", "00009"))) == "dark"


def test_a_border_wider_than_100_px_fails_through_the_ring(tmp_path):
    content = np.full((H - 8, W - 8, 3), 255, np.uint8)  # 4 more white px on every side
    content[10:-10, 10:-10] = 90
    write_framed_jpeg(
        tmp_path / "wide.jpg", np.pad(content, ((4, 4), (4, 4), (0, 0)), constant_values=255)
    )
    assert dv.check_border(dv.read_rgb8(tmp_path / "wide.jpg")) == "wide"


def test_border_check_bounds_come_from_the_config(layout):
    strict = dv.BorderCheck(min_value=256, max_ring_white=0.5)
    img = dv.read_rgb8(layout.rgb_image("train", "00001"))
    assert dv.check_border(img, strict) == "dark"


# --- XML -----------------------------------------------------------------------------------------


def test_parse_annotation_crops_coordinates_and_maps_classes(layout, fake_dronevehicle):
    ann = dv.parse_annotation(layout.rgb_label("train", "00001"), layout.class_map)
    spec = _spec(fake_dronevehicle, "train_00001")
    vehicles = _vehicles(spec)
    assert len(ann.boxes) == len(vehicles)
    first = ann.boxes[0]
    assert first.kind == "polygon"
    np.testing.assert_allclose(first.vertices, np.array(_box(vehicles[0]), float))
    assert ann.boxes[1].kind == "bndbox"  # the RGB bndbox form becomes a 4-vertex rectangle
    np.testing.assert_allclose(ann.boxes[1].vertices, np.array(_box(vehicles[1]), float))
    assert {b.category for b in ann.boxes} <= {"car", "truck", "bus", "van", "freight_car"}
    assert "freight_car" in {b.category for b in ann.boxes}  # the "feright car" typo is mapped


def test_parse_annotation_counts_what_it_skips(layout):
    ann = dv.parse_annotation(layout.rgb_label("train", "00001"), layout.class_map)
    assert ann.num_point_only == 1  # the <point> object
    assert ann.num_dropped == 1  # the "*" class
    assert ann.unknown_classes == ()


def test_thermal_label_typo_class_is_mapped(layout):
    ann = dv.parse_annotation(layout.thermal_label("train", "00001"), layout.class_map)
    assert "truck" in {b.category for b in ann.boxes}  # "truvk" is a truck


def test_unknown_class_names_are_dropped_and_listed(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text(
        "<annotation><folder>f</folder><filename>x</filename>"
        "<object><name>tractor</name><polygon><x1>110</x1><y1>110</y1><x2>120</x2><y2>110</y2>"
        "<x3>120</x3><y3>120</y3><x4>110</x4><y4>120</y4></polygon></object></annotation>"
    )
    ann = dv.parse_annotation(xml, {"car": "car"})
    assert ann.boxes == ()
    assert ann.unknown_classes == ("tractor",)


def test_object_without_any_geometry_is_malformed(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text(
        "<annotation><folder>f</folder><filename>x</filename>"
        "<object><name>car</name></object></annotation>"
    )
    with pytest.raises(ValueError, match="geometry"):
        dv.parse_annotation(xml, {"car": "car"})


def test_annotation_carries_folder_and_filename(layout):
    rgb = dv.parse_annotation(layout.rgb_label("train", "00002"), layout.class_map)
    thermal = dv.parse_annotation(layout.thermal_label("train", "00002"), layout.class_map)
    assert rgb.folder == thermal.folder == "flight_00002"
    assert rgb.filename == "DJI_00002"
    assert thermal.filename == "DJI_00002_R"


def test_box_centre_is_the_mean_of_the_four_vertices():
    box = dv.Box("car", ((0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)), "polygon")
    np.testing.assert_allclose(box.center, [2.0, 1.0])
    assert box.vertices.shape == (4, 2)
