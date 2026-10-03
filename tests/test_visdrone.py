"""dawn/data/visdrone.py: layout, file formats, box filtering and the dataset's three modes."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dawn.data.visdrone import (  # noqa: E402
    BoxFilter,
    ImageStats,
    VisDroneDataset,
    VisDroneLayout,
    is_still_collection,
    read_annotations,
    read_labels,
    read_manifest,
    read_srgb8,
    read_stats_csv,
    sequence_id,
    set_labels,
    write_manifest,
    write_stats_csv,
)
from tests.fake_visdrone import FAKE_IMAGES, NUM_EVAL_BOXES  # noqa: E402

TRAIN_IDS = [image_id for split, image_id, *_ in FAKE_IMAGES if split == "train"]
NATIVE_SIZE = {image_id: size for _, image_id, size, _ in FAKE_IMAGES}


def _all_train_manifest(layout, order=None):
    """A manifest listing every fake train image, in the given (possibly unsorted) order."""
    path = layout.manifest_dir / "train" / "all_unsorted.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image_id}\n" for image_id in (order or TRAIN_IDS[::-1])))
    return path


def _ann(*rows):
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 8)


# --- layout and file formats -------------------------------------------------------------------


def test_layout_paths_follow_the_documented_tree(tmp_path):
    layout = VisDroneLayout(tmp_path / "raw", tmp_path / "s.csv", tmp_path / "m", tmp_path / "p")
    assert layout.raw_image("val", "x") == tmp_path / "raw/VisDrone2019-DET-val/images/x.jpg"
    assert layout.raw_annotation("val", "x") == (
        tmp_path / "raw/VisDrone2019-DET-val/annotations/x.txt"
    )
    assert layout.lowres_image("test-dev", "x") == tmp_path / "p/test-dev/x.png"
    assert layout.manifest("train", "day") == tmp_path / "m/train/day.txt"


def test_layout_from_config():
    cfg = {
        "root": "raw",
        "stats_csv": "data/stats/s.csv",
        "manifest_dir": "data/manifests",
        "prepared_dir": "data/p",
        "image_size": 512,  # other keys of the section are ignored
    }
    layout = VisDroneLayout.from_config(cfg)
    assert layout.root.as_posix() == "raw"
    assert layout.stats_csv.as_posix() == "data/stats/s.csv"
    assert layout.manifest_dir.as_posix() == "data/manifests"
    assert layout.prepared_dir.as_posix() == "data/p"


def test_raw_image_ids_are_sorted_stems(fake_visdrone):
    assert fake_visdrone.raw_image_ids("train") == sorted(TRAIN_IDS)


def test_read_annotations_parses_rows_and_tolerates_trailing_commas(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("1,2,3,4,1,4,0,0\n5,6,7,8,0,11,1,2,\n\n")
    ann = read_annotations(path)
    assert ann.dtype == torch.int64
    assert ann.tolist() == [[1, 2, 3, 4, 1, 4, 0, 0], [5, 6, 7, 8, 0, 11, 1, 2]]


def test_read_annotations_of_an_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    assert read_annotations(path).shape == (0, 8)


def test_read_srgb8_returns_chw_uint8_as_stored(tmp_path):
    pixels = np.random.default_rng(0).integers(0, 256, (5, 7, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(tmp_path / "x.png")
    img = read_srgb8(tmp_path / "x.png")
    assert img.dtype == torch.uint8 and img.shape == (3, 5, 7)
    assert torch.equal(img, torch.from_numpy(pixels).permute(2, 0, 1))


def _stats(image_id, split, median=0.1):
    return ImageStats(image_id, split, 640, 480, median, 0.12, 0.01, 0.5, 3, 0.02, 7.5)


def test_stats_csv_is_sorted_by_split_then_id_and_round_trips(tmp_path):
    rows = [_stats("b", "val"), _stats("z", "train"), _stats("c", "test-dev"), _stats("a", "train")]
    write_stats_csv(tmp_path / "s.csv", rows)
    back = read_stats_csv(tmp_path / "s.csv")
    assert list(back) == ["a", "z", "b", "c"]  # train, val, test-dev; IDs sorted within a split
    assert back["b"] == rows[0]


def test_stats_csv_does_not_depend_on_row_order(tmp_path):
    rows = [_stats(f"id{i}", split) for i in range(5) for split in ("val", "train")]
    write_stats_csv(tmp_path / "a.csv", rows)
    write_stats_csv(tmp_path / "b.csv", rows[::-1])
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_write_manifest_sorts_and_read_manifest_keeps_file_order(tmp_path):
    write_manifest(tmp_path / "m.txt", ["b", "c", "a"])
    assert read_manifest(tmp_path / "m.txt") == ["a", "b", "c"]
    (tmp_path / "n.txt").write_text("c\n\na\nb\n")
    assert read_manifest(tmp_path / "n.txt") == ["c", "a", "b"]


def test_sequence_id_is_the_first_filename_field():
    assert sequence_id("0000287_00759_d_0000759") == "0000287"


def test_99999_groups_are_still_collections_not_videos():
    assert is_still_collection("9999981")
    assert not is_still_collection("0000287")


def test_read_labels_keeps_key_and_label_and_ignores_helper_columns(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label,note,frames\n0000001, Day ,sunny,4\n0000002,,,7\n")
    assert read_labels(path) == {"0000001": "day", "0000002": ""}


def test_read_labels_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label\n0000001,day\n0000001,night\n")
    with pytest.raises(ValueError, match="0000001"):
        read_labels(path)


def test_read_labels_needs_key_and_label_columns(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("sequence,label\n0000001,day\n")
    with pytest.raises(ValueError, match="key"):
        read_labels(path)


def test_set_labels_updates_in_place_and_keeps_every_other_column_and_row(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label,note,frames\n0000001,,sunny,4\n0000002,night,,7\n")
    set_labels(path, {"0000001": "day"})
    assert path.read_text() == "key,label,note,frames\n0000001,day,sunny,4\n0000002,night,,7\n"


def test_set_labels_appends_keys_it_does_not_find(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label,note\n0000001,day,\n")
    set_labels(path, {"9999901_00000_d_0000001": "night"})
    assert read_labels(path) == {"0000001": "day", "9999901_00000_d_0000001": "night"}


def test_set_labels_can_clear_a_label(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label\n0000001,day\n")
    set_labels(path, {"0000001": ""})
    assert read_labels(path) == {"0000001": ""}


def test_set_labels_rejects_unknown_labels_and_leaves_the_file_alone(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label\n0000001,day\n")
    with pytest.raises(ValueError, match="sunset"):
        set_labels(path, {"0000001": "sunset"})
    assert path.read_text() == "key,label\n0000001,day\n"
    assert list(tmp_path.iterdir()) == [path]  # no temp file left behind


def test_set_labels_starts_a_missing_file(tmp_path):
    path = tmp_path / "labels" / "labels.csv"
    set_labels(path, {"0000001": "day"})
    assert read_labels(path) == {"0000001": "day"}


# --- box filter --------------------------------------------------------------------------------


def test_label_rules_drop_ignored_regions_others_and_score_zero():
    ann = _ann(
        [0, 0, 20, 20, 1, 0, 0, 0],  # category 0
        [0, 0, 20, 20, 1, 11, 0, 0],  # category 11
        [0, 0, 20, 20, 0, 4, 0, 0],  # score 0
        [0, 0, 20, 20, 1, 4, 0, 0],  # kept
    )
    assert BoxFilter().keep_labels(ann).tolist() == [False, False, False, True]


def test_min_size_is_measured_at_the_returned_resolution():
    ann = _ann([10, 10, 20, 20, 1, 4, 0, 0])  # 20 x 20 native
    box_filter = BoxFilter()
    assert len(box_filter(ann, (1.0, 1.0), (100, 100))[0]) == 1
    assert len(box_filter(ann, (0.5, 0.5), (50, 50))[0]) == 0  # 10 x 10 once returned


def test_boxes_are_rescaled_per_axis_and_clipped_to_the_image():
    ann = _ann(
        [10, 20, 40, 80, 1, 4, 0, 0],  # xyxy 10,20,50,100 -> 5,5,25,25
        [160, 120, 100, 200, 1, 5, 0, 0],  # xyxy 160,120,260,320 -> 80,30,130,80 -> clipped
        [-20, -8, 60, 100, 1, 6, 0, 0],  # xyxy -20,-8,40,92 -> -10,-2,20,23 -> clipped at 0
    )
    boxes, categories = BoxFilter()(ann, (0.5, 0.25), (100, 50))
    assert boxes.dtype == torch.float32 and categories.dtype == torch.int64
    assert boxes.tolist() == [
        [5.0, 5.0, 25.0, 25.0],
        [80.0, 30.0, 100.0, 50.0],
        [0.0, 0.0, 20.0, 23.0],
    ]
    assert categories.tolist() == [4, 5, 6]


def test_small_box_after_clipping_is_dropped():
    ann = _ann([95, 10, 50, 50, 1, 4, 0, 0])  # only 5 px of it is inside a 100-px-wide image
    assert len(BoxFilter()(ann, (1.0, 1.0), (100, 100))[0]) == 0


def test_box_filter_from_config():
    box_filter = BoxFilter.from_config({"drop_categories": [0], "min_size": 8})
    assert box_filter == BoxFilter(drop_categories=(0,), min_size=8.0)
    assert BoxFilter.from_config(None) == BoxFilter()


# --- dataset -----------------------------------------------------------------------------------


def test_lowres_items_have_the_documented_shapes_and_ranges(prepared_visdrone):
    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "lowres")
    assert len(ds) == len(TRAIN_IDS)
    for item in ds:
        day = item["day"]
        native_w, native_h = NATIVE_SIZE[item["image_id"]]
        assert day.dtype == torch.float32 and day.shape[0] == 3
        assert 0.0 <= day.min() and day.max() <= 1.0
        assert max(day.shape[1:]) == 512
        assert day.shape[2] / day.shape[1] == pytest.approx(native_w / native_h, rel=0.01)
        assert item["scale"] == pytest.approx(512 / max(native_w, native_h))
        assert "boxes" not in item


def test_lowres_day_is_the_linearized_png_copy(prepared_visdrone):
    from dawn.data.color import srgb8_to_linear

    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "lowres")
    item = ds[0]
    png = read_srgb8(prepared_visdrone.lowres_image("train", item["image_id"]))
    assert torch.equal(item["day"], srgb8_to_linear(png))


@pytest.mark.parametrize("mode", ["lowres", "native_crops", "eval"])
def test_native_size_matches_the_csv(prepared_visdrone, mode):
    csv = read_stats_csv(prepared_visdrone.stats_csv)
    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, mode)
    for item in ds:
        row = csv[item["image_id"]]
        assert item["native_size"].tolist() == [row.native_width, row.native_height]
        assert item["native_size"].dtype == torch.int64


def test_native_crops_are_crop_size_squares_at_native_scale(prepared_visdrone):
    ds = VisDroneDataset(
        _all_train_manifest(prepared_visdrone), prepared_visdrone, "native_crops", crop_size=256
    )
    for item in ds:
        assert item["day"].shape == (3, 256, 256)
        assert item["day"].dtype == torch.float32
        assert 0.0 <= item["day"].min() and item["day"].max() <= 1.0
        assert item["scale"] == 1.0


def test_native_crops_come_from_different_places(prepared_visdrone):
    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "native_crops")
    crops = [ds[0]["day"] for _ in range(4)]
    assert any(not torch.equal(crops[0], other) for other in crops[1:])


def test_eval_returns_the_whole_raw_image_with_filtered_boxes(prepared_visdrone):
    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "eval")
    for item in ds:
        native_w, native_h = NATIVE_SIZE[item["image_id"]]
        assert item["day"].shape == (3, native_h, native_w)
        assert item["scale"] == 1.0
        assert item["boxes"].shape == (NUM_EVAL_BOXES, 4)
        assert item["categories"].tolist() == [4, 2]  # the car and the clipped corner box


def test_eval_boxes_stay_inside_the_image(prepared_visdrone):
    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "eval")
    for item in ds:
        _, height, width = item["day"].shape
        boxes = item["boxes"]
        assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
        assert (boxes[:, 2] <= width).all() and (boxes[:, 3] <= height).all()
        assert (boxes[:, 2] - boxes[:, 0] >= 16).all() and (boxes[:, 3] - boxes[:, 1] >= 16).all()


def test_eval_mode_loads_with_batch_size_1(prepared_visdrone):
    ds = VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "eval")
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=1)))
    assert batch["day"].shape[0] == 1 and batch["boxes"].shape == (1, NUM_EVAL_BOXES, 4)


def test_ram_cache_items_are_identical_to_disk_items(prepared_visdrone):
    manifest = _all_train_manifest(prepared_visdrone)
    disk = VisDroneDataset(manifest, prepared_visdrone, "lowres", ram_cache=False)
    cached = VisDroneDataset(manifest, prepared_visdrone, "lowres", ram_cache=True)
    assert len(disk) == len(cached)
    for a, b in zip(disk, cached):
        assert a["image_id"] == b["image_id"]
        assert torch.equal(a["day"], b["day"])
        assert torch.equal(a["native_size"], b["native_size"]) and a["scale"] == b["scale"]


def test_item_order_is_the_manifest_order_every_time(prepared_visdrone):
    order = [TRAIN_IDS[1], TRAIN_IDS[2], TRAIN_IDS[0]]  # deliberately unsorted
    manifest = _all_train_manifest(prepared_visdrone, order)
    for _ in range(2):
        ds = VisDroneDataset(manifest, prepared_visdrone, "lowres")
        assert [ds[i]["image_id"] for i in range(len(ds))] == order


def test_phase_2_manifests_are_readable_by_the_dataset(prepared_visdrone):
    ds = VisDroneDataset(prepared_visdrone.manifest("train", "day"), prepared_visdrone)
    assert [item["image_id"] for item in ds] == [FAKE_IMAGES[0][1], FAKE_IMAGES[1][1]]


def test_max_images_keeps_the_first_n(prepared_visdrone):
    manifest = _all_train_manifest(prepared_visdrone)
    ds = VisDroneDataset(manifest, prepared_visdrone, "lowres", max_images=2)
    assert [ds[i]["image_id"] for i in range(len(ds))] == read_manifest(manifest)[:2]


def test_ram_cache_is_lowres_only(prepared_visdrone):
    with pytest.raises(ValueError, match="ram_cache"):
        VisDroneDataset(
            _all_train_manifest(prepared_visdrone), prepared_visdrone, "eval", ram_cache=True
        )


def test_unknown_mode_is_rejected(prepared_visdrone):
    with pytest.raises(ValueError, match="mode"):
        VisDroneDataset(_all_train_manifest(prepared_visdrone), prepared_visdrone, "fullres")


def test_manifest_id_missing_from_the_stats_csv_is_rejected(prepared_visdrone, tmp_path):
    manifest = tmp_path / "bad.txt"
    manifest.write_text(f"{TRAIN_IDS[0]}\nnot_an_image\n")
    with pytest.raises(ValueError, match="not_an_image"):
        VisDroneDataset(manifest, prepared_visdrone)
