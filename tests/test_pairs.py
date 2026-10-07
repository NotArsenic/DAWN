"""dawn/data/pairs.py: the aligned-pair PNGs (writer) and AlignedPairs (dataset, augmentation)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("scipy")
omegaconf = pytest.importorskip("omegaconf")

from PIL import Image  # noqa: E402

from dawn.data import align, pairs  # noqa: E402
from dawn.data import dronevehicle as dv  # noqa: E402
from dawn.data.color import srgb8_to_linear  # noqa: E402
from tests.fake_dronevehicle import fake_config  # noqa: E402

H, W = 512, 640


# --- a hand-made aligned directory ---------------------------------------------------------------


def _write_pair(layout, image_id, seed=0, thermal=None):
    """The four PNGs of one pair with asymmetric content, so any flip is visible."""
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    thermal = (
        thermal
        if thermal is not None
        else np.tile(
            np.linspace(0, 255, W).astype(np.uint8), (H, 1)
        )  # a ramp: monotone, asymmetric
    )
    valid = np.full((H, W), 255, np.uint8)
    valid[:10, :40] = 0  # an invalid corner (top-left)
    moving = np.zeros((H, W), np.uint8)
    moving[100:130, 200:260] = 255  # a block, off-centre
    vehicles = np.zeros((H, W), np.uint8)
    vehicles[300:340, 50:130] = 255  # another block, so a flip of it is visible too
    for kind, array in (
        ("rgb", rgb),
        ("thermal", thermal),
        ("valid", valid),
        ("moving", moving),
        ("vehicles", vehicles),
    ):
        path = layout.aligned_path(image_id, kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(array).save(path)
    return rgb, thermal, valid, moving


@pytest.fixture()
def layout(tmp_path):
    cfg = {
        "root": str(tmp_path / "raw"),
        "aligned_dir": str(tmp_path / "aligned"),
        "stats_csv": str(tmp_path / "s.csv"),
        "manifest_dir": str(tmp_path / "manifests"),
        "outputs_dir": str(tmp_path / "out"),
    }
    return dv.DroneVehicleLayout.from_config(cfg)


@pytest.fixture()
def made(layout):
    ids = ["train_00003", "train_00001", "train_00002"]
    data = {i: _write_pair(layout, i, seed=n) for n, i in enumerate(sorted(ids))}
    pairs.write_manifest(layout.manifest("train_pairs"), ids)
    return layout, data


NO_AUG = pairs.AugmentConfig(hflip=0.0, vflip=0.0, contrast_prob=0.0)


# --- manifests -----------------------------------------------------------------------------------


def test_manifest_is_sorted_unique_one_id_per_line_with_lf(tmp_path):
    path = tmp_path / "m" / "x.txt"
    pairs.write_manifest(path, ["val_2", "val_1", "val_2"])
    assert path.read_bytes() == b"val_1\nval_2\n"
    assert pairs.read_manifest(path) == ["val_1", "val_2"]


def test_empty_manifest_round_trips(tmp_path):
    path = tmp_path / "x.txt"
    pairs.write_manifest(path, [])
    assert pairs.read_manifest(path) == []


# --- the dataset ---------------------------------------------------------------------------------


def test_items_have_the_documented_shapes_dtypes_and_ranges(made):
    layout, _ = made
    ds = pairs.AlignedPairs(layout, "train_pairs")
    assert len(ds) == 3
    item = ds[0]
    assert set(item) == {"thermal", "rgb", "valid", "moving", "vehicles", "image_id", "fallback"}
    assert item["vehicles"].shape == (1, H, W) and item["vehicles"].dtype == torch.bool
    assert item["fallback"] is False
    assert item["thermal"].shape == (1, H, W) and item["thermal"].dtype == torch.float32
    assert item["rgb"].shape == (3, H, W) and item["rgb"].dtype == torch.float32
    assert item["valid"].shape == (1, H, W) and item["valid"].dtype == torch.bool
    assert item["moving"].shape == (1, H, W) and item["moving"].dtype == torch.bool
    assert isinstance(item["image_id"], str)
    for key in ("thermal", "rgb"):
        assert 0.0 <= item[key].min() and item[key].max() <= 1.0


def test_ids_come_from_the_manifest_in_sorted_order(made):
    layout, _ = made
    ds = pairs.AlignedPairs(layout, "train_pairs")
    assert [ds[i]["image_id"] for i in range(3)] == ["train_00001", "train_00002", "train_00003"]


def test_thermal_is_code_over_255_with_no_gamma_and_rgb_is_linearized(made):
    layout, data = made
    ds = pairs.AlignedPairs(layout, "train_pairs")
    rgb, thermal, valid, moving = data["train_00001"]
    item = ds[0]
    assert torch.equal(item["thermal"][0], torch.from_numpy(thermal).float() / 255)
    expected = srgb8_to_linear(torch.from_numpy(rgb).permute(2, 0, 1))
    assert torch.equal(item["rgb"], expected)
    assert torch.equal(item["valid"][0], torch.from_numpy(valid > 127))
    assert torch.equal(item["moving"][0], torch.from_numpy(moving > 127))
    assert not item["valid"][0, 0, 0] and item["valid"][0, 20, 100]


def test_a_thermal_code_of_128_is_exactly_128_over_255(layout):
    _write_pair(layout, "train_00001", thermal=np.full((H, W), 128, np.uint8))
    pairs.write_manifest(layout.manifest("train_pairs"), ["train_00001"])
    item = pairs.AlignedPairs(layout, "train_pairs")[0]
    assert (item["thermal"] == 128 / 255).all()


def test_validation_is_deterministic_and_never_augments(made):
    layout, data = made
    ds = pairs.AlignedPairs(
        layout,
        "train_pairs",
        train=False,
        augment=pairs.AugmentConfig(hflip=1.0, vflip=1.0, contrast_prob=1.0),
    )
    a, b = ds[1], ds[1]
    for key in ("thermal", "rgb", "valid", "moving"):
        assert torch.equal(a[key], b[key])
    assert torch.equal(a["thermal"][0], torch.from_numpy(data["train_00002"][1]).float() / 255)


def test_max_pairs_keeps_the_first_n_of_the_manifest(made):
    layout, _ = made
    capped = dv.DroneVehicleLayout(**{**layout.__dict__, "max_pairs": 2})
    ds = pairs.AlignedPairs(capped, "train_pairs")
    assert len(ds) == 2 and ds[1]["image_id"] == "train_00002"
    assert len(pairs.AlignedPairs(capped, "train_pairs", max_pairs=3)) == 3  # explicit wins


def test_a_missing_manifest_or_png_names_what_is_missing(made):
    layout, _ = made
    with pytest.raises(FileNotFoundError, match="nope_pairs"):
        pairs.AlignedPairs(layout, "nope_pairs")
    os.remove(layout.aligned_path("train_00002", "moving"))
    ds = pairs.AlignedPairs(layout, "train_pairs")
    with pytest.raises(FileNotFoundError, match="train_00002"):
        ds[1]


def test_dataloader_batches_the_items(made):
    from torch.utils.data import DataLoader

    layout, _ = made
    batch = next(iter(DataLoader(pairs.AlignedPairs(layout, "train_pairs"), batch_size=2)))
    assert batch["thermal"].shape == (2, 1, H, W)
    assert batch["valid"].dtype == torch.bool
    assert batch["image_id"] == ["train_00001", "train_00002"]


def test_rot90_is_refused_because_it_would_change_the_frame_shape():
    with pytest.raises(ValueError, match="rot90"):
        pairs.AugmentConfig(rot90=True)


def test_augment_config_reads_the_augment_section_of_data_yaml():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path))["augment"]
    aug = pairs.AugmentConfig.from_config(cfg)
    assert (aug.hflip, aug.vflip, aug.rot90) == (0.5, 0.5, False)
    assert 0 < aug.contrast_prob <= 1 and aug.gamma[0] < 1 < aug.gamma[1]


# --- augmentation --------------------------------------------------------------------------------


def _items(layout, aug, generator=None, train=True):
    ds = pairs.AlignedPairs(layout, "train_pairs", train=train, augment=aug, generator=generator)
    return [ds[i] for i in range(len(ds))]


@pytest.mark.parametrize(
    "h, v, dims",
    [(1.0, 0.0, (-1,)), (0.0, 1.0, (-2,)), (1.0, 1.0, (-2, -1)), (0.0, 0.0, ())],
)
def test_flips_apply_to_all_four_tensors_alike(made, h, v, dims):
    layout, data = made
    aug = pairs.AugmentConfig(hflip=h, vflip=v, contrast_prob=0.0)
    for item in _items(layout, aug):
        rgb, thermal, valid, moving = data[item["image_id"]]
        flip = lambda x: torch.flip(x, dims) if dims else x  # noqa: E731
        assert torch.equal(item["thermal"], flip(torch.from_numpy(thermal).float()[None] / 255))
        assert torch.equal(item["valid"], flip(torch.from_numpy(valid > 127)[None]))
        assert torch.equal(item["moving"], flip(torch.from_numpy(moving > 127)[None]))
        expected_rgb = srgb8_to_linear(torch.from_numpy(rgb).permute(2, 0, 1))
        assert torch.equal(item["rgb"], flip(expected_rgb))


def _flip_states(item, rgb_original):
    """`(vflip, hflip)` as read from each of the four tensors of an augmented item."""
    moving, valid = item["moving"][0], item["valid"][0]
    # unflipped: moving block at rows 100..129 / columns 200..259, invalid corner at rows 0..9 /
    # columns 0..39, thermal ramp increasing along x
    from_moving = (not moving[100:130].any().item(), not moving[:, 200:260].any().item())
    from_valid = (valid[:10].all().item(), valid[:, :40].all().item())
    ramp_h = (item["thermal"][0, 0, 0] > item["thermal"][0, 0, -1]).item()
    variants = {
        (v, h): torch.flip(rgb_original, [d for d, on in ((-2, v), (-1, h)) if on])
        for v in (False, True)
        for h in (False, True)
    }
    from_rgb = next(k for k, img in variants.items() if torch.equal(item["rgb"], img))
    return from_moving, from_valid, ramp_h, from_rgb


def test_random_flips_never_split_the_modalities(made):
    layout, data = made
    aug = pairs.AugmentConfig(hflip=0.5, vflip=0.5, contrast_prob=0.0)
    ds = pairs.AlignedPairs(
        layout, "train_pairs", train=True, augment=aug, generator=torch.Generator().manual_seed(7)
    )
    rgb_original = srgb8_to_linear(torch.from_numpy(data["train_00001"][0]).permute(2, 0, 1))
    seen = set()
    for _ in range(40):
        from_moving, from_valid, ramp_h, from_rgb = _flip_states(ds[0], rgb_original)
        assert from_moving == from_valid == from_rgb  # all four tensors flipped alike
        assert ramp_h == from_rgb[1]
        seen.add(from_rgb)
    assert len(seen) == 4  # all four of {none, h, v, h+v} occur


def test_same_generator_seed_gives_the_same_augmentation_and_global_rng_is_untouched(made):
    layout, _ = made
    aug = pairs.AugmentConfig(hflip=0.5, vflip=0.5, contrast_prob=1.0)
    state = torch.get_rng_state()
    a = _items(layout, aug, torch.Generator().manual_seed(3))
    b = _items(layout, aug, torch.Generator().manual_seed(3))
    assert torch.equal(torch.get_rng_state(), state)  # the global RNG was not touched
    for x, y in zip(a, b):
        assert torch.equal(x["thermal"], y["thermal"]) and torch.equal(x["moving"], y["moving"])
    c = _items(layout, aug, torch.Generator().manual_seed(4))
    assert any(not torch.equal(x["thermal"], z["thermal"]) for x, z in zip(a, c))


def test_without_a_generator_the_default_rng_is_used(made):
    layout, _ = made
    aug = pairs.AugmentConfig(hflip=0.5, vflip=0.5, contrast_prob=1.0)
    torch.manual_seed(11)
    a = _items(layout, aug)
    torch.manual_seed(11)
    b = _items(layout, aug)
    for x, y in zip(a, b):
        assert torch.equal(x["thermal"], y["thermal"])


def test_the_contrast_curve_touches_thermal_only_and_is_monotone(made):
    layout, data = made
    aug = pairs.AugmentConfig(hflip=0.0, vflip=0.0, contrast_prob=1.0)
    plain = _items(layout, NO_AUG)
    curved = _items(layout, aug, torch.Generator().manual_seed(5))
    for p, c in zip(plain, curved):
        for key in ("rgb", "valid", "moving"):
            assert torch.equal(p[key], c[key])
        assert not torch.equal(p["thermal"], c["thermal"])
        assert 0.0 <= c["thermal"].min() and c["thermal"].max() <= 1.0
        # monotone non-decreasing in the original thermal value: sort by original, look at output
        order = torch.argsort(p["thermal"].flatten(), stable=True)
        out = c["thermal"].flatten()[order]
        assert (out[1:] >= out[:-1]).all()


def test_contrast_probability_zero_leaves_thermal_alone(made):
    layout, _ = made
    for p, c in zip(
        _items(layout, NO_AUG), _items(layout, NO_AUG, torch.Generator().manual_seed(1))
    ):
        assert torch.equal(p["thermal"], c["thermal"])


def test_contrast_curve_values():
    x = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    out = pairs.apply_contrast_curve(x, black=0.1, white=0.9, gamma=1.0)
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    sq = pairs.apply_contrast_curve(x, black=0.0, white=1.0, gamma=2.0)
    torch.testing.assert_close(sq, x**2)


def test_sampled_curve_parameters_stay_in_the_configured_ranges():
    aug = pairs.AugmentConfig(
        contrast_prob=1.0, black_point=(0.0, 0.1), white_point=(0.9, 1.0), gamma=(0.7, 1.4)
    )
    g = torch.Generator().manual_seed(0)
    draws = [pairs.sample_contrast_curve(aug, g) for _ in range(200)]
    assert all(d is not None for d in draws)
    for black, white, gamma in draws:
        assert 0.0 <= black <= 0.1 and 0.9 <= white <= 1.0 and 0.7 <= gamma <= 1.4
    assert pairs.sample_contrast_curve(pairs.AugmentConfig(contrast_prob=0.0), g) is None


# --- the writer ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw(fake_dronevehicle, tmp_path_factory):
    base = tmp_path_factory.mktemp("written")
    cfg = fake_config(base, fake_dronevehicle.root)
    layout = dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])
    acfg = align.AlignConfig.from_config(cfg["align"])
    return layout, acfg, fake_dronevehicle


@pytest.fixture(scope="module")
def written(raw):
    layout, acfg, tree = raw
    out = {}
    for iid in ("train_00001", "train_00010"):
        split, stem = dv.parse_image_id(iid)
        row = align.align_pair(layout, acfg, split, stem)
        assert row.accepted
        assert pairs.write_aligned_pair(layout, acfg, row) is True
        out[iid] = row
    return out


def test_four_pngs_are_written_at_the_crop_size(raw, written):
    layout, _, _ = raw
    for iid in written:
        for kind, mode in (("rgb", "RGB"), ("thermal", "L"), ("valid", "L"), ("moving", "L")):
            with Image.open(layout.aligned_path(iid, kind)) as img:
                assert (img.size, img.mode) == ((640, 512), mode), (iid, kind)


def test_the_rgb_png_is_the_lossless_crop_of_the_decoded_jpeg(raw, written):
    layout, _, _ = raw
    expected = dv.crop_border(dv.read_rgb8(layout.rgb_image("train", "00001")))
    got = np.asarray(Image.open(layout.aligned_path("train_00001", "rgb")))
    assert np.array_equal(got, expected.permute(1, 2, 0).numpy())


def test_masks_are_zero_or_255(raw, written):
    layout, _, _ = raw
    for kind in ("valid", "moving"):
        values = np.unique(np.asarray(Image.open(layout.aligned_path("train_00001", kind))))
        assert set(values.tolist()) <= {0, 255}


def test_the_warped_thermal_matches_the_ground_truth_aligned_thermal_on_valid_pixels(raw, written):
    layout, _, tree = raw
    for iid in written:
        split, stem = dv.parse_image_id(iid)
        # Ground truth: the thermal's gain curve applied to the RGB luma (what the fake thermal
        # was made from), which is exactly what a perfectly aligned thermal would show.
        rgb = np.asarray(Image.open(layout.aligned_path(iid, "rgb"))).astype(np.float64) / 255
        luma = rgb @ np.array([0.299, 0.587, 0.114])
        ideal = np.clip(luma, 0, 1) ** 0.8 * 0.9 + 0.05
        got = np.asarray(Image.open(layout.aligned_path(iid, "thermal"))).astype(np.float64) / 255
        valid = np.asarray(Image.open(layout.aligned_path(iid, "valid"))) > 127
        raw_thermal = dv.crop_border(dv.read_thermal8(layout.thermal_image(split, stem)))[0]
        raw_err = np.abs(raw_thermal.numpy() / 255 - ideal)[valid].mean()
        warp_err = np.abs(got - ideal)[valid].mean()
        assert warp_err < 0.02, (iid, warp_err)
        assert warp_err < raw_err / 2, "the warp must visibly beat the unwarped thermal"


def test_the_valid_mask_marks_what_the_fit_pulled_from_outside(raw, written):
    layout, _, tree = raw
    valid = np.asarray(Image.open(layout.aligned_path("train_00010", "valid"))) > 127
    assert valid.mean() > 0.95  # a few px shift: almost the whole frame is valid
    assert not valid.all()  # but the edge strip that came from outside the thermal frame is not


def test_the_moving_mask_covers_the_moving_vehicle_and_nothing_else(raw, written):
    from tests.fake_dronevehicle import _box, _vehicles

    layout, _, tree = raw
    moving = np.asarray(Image.open(layout.aligned_path("train_00001", "moving"))) > 127
    spec = next(s for s in tree.specs if s.image_id == "train_00001")
    mover = _vehicles(spec)[9]
    (x0, y0), _, (x1, y1), _ = _box(mover)
    assert moving[(y0 + y1) // 2, (x0 + x1) // 2]  # the RGB position of the vehicle is masked
    assert moving.sum() < 4000  # one vehicle's two polygons, dilated: nowhere near a full frame
    none = np.asarray(Image.open(layout.aligned_path("train_00010", "moving"))) > 127
    assert not none.any()


def test_writing_is_idempotent_and_force_rewrites(raw, written):
    layout, acfg, _ = raw
    path = layout.aligned_path("train_00001", "thermal")
    before = path.stat().st_mtime_ns
    assert pairs.write_aligned_pair(layout, acfg, written["train_00001"]) is False
    assert path.stat().st_mtime_ns == before
    assert pairs.write_aligned_pair(layout, acfg, written["train_00001"], force=True) is True
    assert not list(path.parent.glob("*.partial.png"))  # no half-written files left behind


def test_only_accepted_pairs_can_be_written(raw):
    layout, acfg, _ = raw
    row = align.align_pair(layout, acfg, "train", "00006")  # scale failure
    assert not row.accepted
    with pytest.raises(ValueError, match="accepted"):
        pairs.write_aligned_pair(layout, acfg, row)


def test_a_written_pair_loads_through_the_dataset(raw, written, vehicles_written):
    layout, _, _ = raw
    pairs.write_manifest(layout.manifest("train_pairs"), list(written))
    item = pairs.AlignedPairs(layout, "train_pairs")[0]
    assert item["thermal"].shape == (1, 512, 640) and item["image_id"] == "train_00001"


# --- fallback pairs ------------------------------------------------------------------------------


def test_the_dataset_flags_fallback_pairs_from_their_manifest(made):
    layout, _ = made
    pairs.write_manifest(layout.manifest("fallback_pairs"), ["train_00002"])
    ds = pairs.AlignedPairs(layout, "train_pairs")
    flags = {ds[i]["image_id"]: ds[i]["fallback"] for i in range(len(ds))}
    assert flags == {"train_00001": False, "train_00002": True, "train_00003": False}


def test_without_a_fallback_manifest_nothing_is_flagged(made):
    layout, _ = made
    assert not layout.manifest("fallback_pairs").exists()
    assert not any(pairs.AlignedPairs(layout, "train_pairs")[i]["fallback"] for i in range(3))


def test_the_flag_survives_augmentation_and_collation(made):
    from torch.utils.data import DataLoader

    layout, _ = made
    pairs.write_manifest(layout.manifest("fallback_pairs"), ["train_00002"])
    aug = pairs.AugmentConfig(hflip=1.0, vflip=1.0, contrast_prob=1.0)
    ds = pairs.AlignedPairs(layout, "train_pairs", train=True, augment=aug)
    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert batch["fallback"].dtype == torch.bool
    assert batch["fallback"].tolist() == [False, True, False]


@pytest.mark.parametrize("stem", ["00002", "00003"])  # 00003 has 3 matched boxes, 00002 none
def test_a_fallback_pair_is_written_as_an_exact_shift_with_an_empty_moving_mask(raw, stem):
    layout, acfg, _ = raw
    image_id = f"train_{stem}"
    row = align.align_pair(layout, acfg, "train", stem)
    assert row.model == "edge_translation" and row.accepted
    assert pairs.write_aligned_pair(layout, acfg, row) is True
    T = align.transform_of(row)
    dx, dy = int(T[0, 2]), int(T[1, 2])
    moving = np.asarray(Image.open(layout.aligned_path(image_id, "moving")))
    assert not moving.any()  # a translation cannot say which matched vehicles moved
    raw_thermal = dv.crop_border(dv.read_thermal8(layout.thermal_image("train", stem)))[0].numpy()
    got = np.asarray(Image.open(layout.aligned_path(image_id, "thermal")))
    h, w = raw_thermal.shape
    r0, r1 = max(dy, 0) + 1, h + min(dy, 0) - 1
    c0, c1 = max(dx, 0) + 1, w + min(dx, 0) - 1
    np.testing.assert_array_equal(
        got[r0:r1, c0:c1], raw_thermal[r0 - dy : r1 - dy, c0 - dx : c1 - dx]
    )


# --- the vehicles mask --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vmasks():
    return pairs.VehicleMaskConfig()  # the defaults: dilate 3 px, extend 17.4 px


@pytest.fixture(scope="module")
def vehicles_written(raw, written, vmasks):
    layout, acfg, _ = raw
    return {
        image_id: pairs.write_vehicles(layout, acfg, vmasks, row)
        for image_id, row in written.items()
    }


def test_the_dataset_serves_the_vehicles_png_and_flips_it_with_the_other_tensors(made):
    layout, _ = made
    plain = pairs.AlignedPairs(layout, "train_pairs")[0]["vehicles"][0]
    assert plain[300:340, 50:130].all() and int(plain.sum()) == 40 * 80
    aug = pairs.AugmentConfig(hflip=1.0, vflip=1.0, contrast_prob=0.0)
    flipped = _items(layout, aug)[0]["vehicles"][
        0
    ]  # 180 degrees: the block moves to the far corner
    assert flipped[H - 340 : H - 300, W - 130 : W - 50].all() and int(flipped.sum()) == 40 * 80


def test_vehicle_mask_config_reads_data_yaml():
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path))
    masks = pairs.VehicleMaskConfig.from_config(cfg["masks"]["vehicles"])
    assert masks.dilate_px == 3  # the fallback's alignment error, not a motion margin
    assert masks.extend_px == pytest.approx(17.4)  # the p95 of moving-vehicle displacement
    assert masks.max_frame_fraction == 0.30
    p = masks.params
    assert (p.window_margin_px, p.search_px, p.max_residual_px) == (12, 6, 2)
    assert p.min_prominence == 0.02 and p.min_window_px == 32


def _box_polygons(layout, split, stem):
    ann = dv.parse_annotation(layout.rgb_label(split, stem), layout.class_map)
    return [b.vertices for b in ann.boxes]


def test_every_rgb_polygon_is_covered_in_the_written_vehicles_png(raw, vehicles_written):
    layout, _, _ = raw
    for image_id in vehicles_written:
        split, stem = dv.parse_image_id(image_id)
        mask = np.asarray(Image.open(layout.aligned_path(image_id, "vehicles"))) > 127
        inside = align.rasterize(_box_polygons(layout, split, stem), (512, 640))
        assert mask[inside].all(), image_id  # every pixel of every box is masked
        assert mask.sum() > inside.sum()  # grown by the margin


def test_a_box_pair_without_movers_is_masked_tightly(raw, vehicles_written):
    from scipy import ndimage

    layout, _, _ = raw
    mask = np.asarray(Image.open(layout.aligned_path("train_00010", "vehicles"))) > 127
    inside = align.rasterize(_box_polygons(layout, "train", "00010"), (512, 640))
    distance = ndimage.distance_transform_edt(~inside)
    assert distance[mask].max() <= 3 + 2.0  # the 3 px margin plus the rounding of the labels
    branches = vehicles_written["train_00010"].branches
    assert branches.matched == 9 and branches.moved == branches.unsure == 0


def test_the_branches_are_counted_per_vehicle(raw, vehicles_written):
    first = vehicles_written["train_00001"].branches
    # 10 vehicles: 9 matched (one of them the 14 px mover), 1 whose thermal label is a copy
    assert first.matched == 9 and first.matched + first.static + first.moved + first.unsure == 10
    assert first.thermal_only == 0


def test_a_moving_vehicle_is_masked_where_rgb_and_thermal_see_it(raw, vehicles_written):
    layout, _, tree = raw
    from tests.fake_dronevehicle import _box, _vehicles

    spec = next(s for s in tree.specs if s.image_id == "train_00001")
    mover = _vehicles(spec)[9]  # displaced 14 px in the thermal labels
    (x0, y0), _, (x1, y1), _ = _box(mover)
    mask = np.asarray(Image.open(layout.aligned_path("train_00001", "vehicles"))) > 127
    assert mask[(y0 + y1) // 2, (x0 + x1) // 2]  # its RGB position
    assert mask[(y0 + y1) // 2, (x0 + x1) // 2 + 14]  # and the position its thermal pixels moved to


def test_a_fallback_pair_with_copied_labels_is_masked_per_vehicle_from_the_rgb_boxes(raw, vmasks):
    layout, acfg, _ = raw
    row = align.align_pair(layout, acfg, "train", "00002")  # labels_copied, aligned by the fallback
    assert row.model == "edge_translation" and row.labels_identical
    result = pairs.write_vehicles(layout, acfg, vmasks, row)
    b = result.branches
    assert b.matched == 0 and b.thermal_only == 0  # the thermal boxes are copies: nothing matches
    assert b.static + b.moved + b.unsure == 10  # every RGB vehicle took the local test
    mask = np.asarray(Image.open(layout.aligned_path("train_00002", "vehicles"))) > 127
    assert mask[align.rasterize(_box_polygons(layout, "train", "00002"), (512, 640))].all()


def test_the_result_carries_the_masked_fraction(raw, vehicles_written):
    layout, _, _ = raw
    result = vehicles_written["train_00001"]
    mask = np.asarray(Image.open(layout.aligned_path("train_00001", "vehicles"))) > 127
    assert result.fraction == pytest.approx(mask.mean())
    assert 0.0 < result.fraction < 0.3


def test_the_vehicles_mask_is_recomputed_on_every_call_so_its_branches_are_always_known(
    raw, vmasks
):
    layout, acfg, _ = raw
    row = align.align_pair(layout, acfg, "train", "00010")
    first = pairs.write_vehicles(layout, acfg, vmasks, row)
    assert pairs.write_vehicles(layout, acfg, vmasks, row) == first  # deterministic
