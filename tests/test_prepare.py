"""dawn/data/prepare.py: the three phases behind scripts/prepare_visdrone.py."""

import math
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dawn.data.color import srgb8_to_linear  # noqa: E402
from dawn.data.prepare import (  # noqa: E402
    compute_image_stats,
    frames_for_key,
    highlight_ratio,
    make_lowres_copy,
    noise_level,
    pick_three,
    plot_feature_scatter,
    plot_log_luminance_histogram,
    preview_image,
    resolve_labels,
    run_parallel,
    save_contact_sheets,
    save_sequence_sheets,
    write_labels_template,
    write_manifests,
)
from dawn.data.visdrone import (  # noqa: E402
    ImageStats,
    VisDroneLayout,
    read_labels,
    read_manifest,
    read_srgb8,
    read_stats_csv,
)
from tests.fake_visdrone import FAKE_IMAGES, FAKE_LABELS, NUM_LABEL_KEPT  # noqa: E402


def _code(c):
    return srgb8_to_linear(torch.tensor(c, dtype=torch.uint8)).item()


def _layout(tmp_path):
    return VisDroneLayout(tmp_path / "raw", tmp_path / "s.csv", tmp_path / "m", tmp_path / "p")


def _write_raw(layout, split, image_id, pixels, annotations=""):
    path = layout.raw_image(split, image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path, quality=100, subsampling=0)
    ann = layout.raw_annotation(split, image_id)
    ann.parent.mkdir(parents=True, exist_ok=True)
    ann.write_text(annotations)


# --- phase 1: stats ----------------------------------------------------------------------------


def test_image_stats_measure_linear_luminance_percentiles(tmp_path):
    # 512 x 256 gray bands on 16-px (JPEG block) boundaries: rows 0-63 code 20, 64-191 code 128,
    # 192-255 code 240. The 2x area downsample keeps the bands exact: 25% / 50% / 25%.
    pixels = np.empty((256, 512, 3), np.uint8)
    pixels[:64], pixels[64:192], pixels[192:] = 20, 128, 240
    layout = _layout(tmp_path)
    _write_raw(layout, "train", "bands", pixels)
    row = compute_image_stats(layout, "train", "bands")
    assert (row.native_width, row.native_height) == (512, 256)
    assert row.lum_median == pytest.approx(_code(128), rel=0.02)
    assert row.lum_p05 == pytest.approx(_code(20), rel=0.02)
    assert row.lum_p95 == pytest.approx(_code(240), rel=0.02)
    expected_mean = 0.25 * _code(20) + 0.5 * _code(128) + 0.25 * _code(240)
    assert row.lum_mean == pytest.approx(expected_mean, rel=0.02)


def test_image_stats_record_split_native_size_and_label_filtered_box_count(fake_visdrone):
    split, image_id, (width, height), _ = FAKE_IMAGES[3]
    row = compute_image_stats(fake_visdrone, split, image_id)
    assert (row.image_id, row.split) == (image_id, split)
    assert (row.native_width, row.native_height) == (width, height)
    assert row.num_boxes == NUM_LABEL_KEPT
    assert row.noise > 0 and row.highlight_ratio >= 1.0


# --- phase 1 features: noise level and highlight ratio -------------------------------------------


def _noisy(level, gain, size=(3, 200, 300), read_var=0.0):
    """Linear gray image with Poisson-Gaussian-like noise, Var = gain * level + read_var, the
    same draw on all three channels so luminance carries all of it."""
    std = (gain * level + read_var) ** 0.5
    noise = torch.randn(1, *size[1:], dtype=torch.float64) * std
    return (torch.full(size, level, dtype=torch.float64) + noise).clamp(0, 1).float()


@pytest.mark.parametrize("gain", [1e-4, 1e-3, 1e-2])
@pytest.mark.parametrize("level", [0.05, 0.2, 0.5])
def test_noise_level_estimates_the_square_root_of_the_gain_at_any_brightness(gain, level):
    # "Relative to brightness": the same sensor gain reads the same at any exposure.
    assert noise_level(_noisy(level, gain)) == pytest.approx(gain**0.5, rel=0.1)


def test_noise_level_ranks_high_gain_above_low_gain_even_when_brighter():
    bright_high_gain = noise_level(_noisy(0.4, 1e-2))  # auto-exposure made a night shot bright
    dim_low_gain = noise_level(_noisy(0.05, 1e-4))  # a dim day shot
    assert bright_high_gain > 5 * dim_low_gain


def test_noise_level_ignores_crushed_blacks_and_clipped_highlights():
    img = _noisy(0.2, 1e-3, size=(3, 200, 300))
    reference = noise_level(img)
    img[:, :, :100] = 0.0  # crushed black: no noise left to measure
    img[:, :, 200:] = 1.0  # clipped highlight: same
    assert noise_level(img) == pytest.approx(reference, rel=0.1)


def test_noise_level_of_a_frame_with_nothing_measurable_is_nan():
    assert math.isnan(noise_level(torch.zeros(3, 50, 50)))


def test_highlight_ratio_is_the_995th_percentile_over_the_median():
    img = torch.full((3, 100, 100), 0.01)
    img[:, :2, :] = 1.0  # 2% of the pixels are lamps
    assert highlight_ratio(img) == pytest.approx(100.0, rel=1e-4)
    assert highlight_ratio(torch.full((3, 10, 10), 0.3)) == pytest.approx(1.0)


def test_highlight_ratio_floors_the_median_at_1e_4():
    img = torch.zeros(3, 100, 100)
    img[:, :2, :] = 0.5
    assert highlight_ratio(img) == pytest.approx(0.5 / 1e-4, rel=1e-4)


def _stats_job(layout, split, image_id):
    return compute_image_stats(layout, split, image_id)


def test_run_parallel_on_processes_matches_in_process(fake_visdrone):
    # Same values in job order. Not bit-identical: torch sums in a different order with one
    # thread per worker than with the parent's thread pool.
    jobs = [(fake_visdrone, split, image_id) for split, image_id, *_ in FAKE_IMAGES]
    pooled = run_parallel(compute_image_stats, jobs, workers=2)
    inline = run_parallel(compute_image_stats, jobs, workers=1)
    assert [r.image_id for r in pooled] == [image_id for _, image_id, *_ in FAKE_IMAGES]
    for a, b in zip(pooled, inline):
        for field, value in vars(a).items():
            assert value == pytest.approx(getattr(b, field), rel=1e-5), field


# --- phase 2: labels -> manifests -------------------------------------------------------------


def _row(image_id, split, median=0.1):
    return ImageStats(
        image_id, split, 640, 480, median, median, median / 2, median * 2, 1, 0.01, 5.0
    )


ROWS = [
    _row("0000001_00001_d_0000001", "train", 0.5),
    _row("0000001_00002_d_0000002", "train", 0.4),
    _row("0000002_00001_d_0000003", "train", 0.01),
    _row("0000003_00001_d_0000004", "train", 0.05),
    _row("9999901_00000_d_0000001", "train", 0.3),
    _row("9999901_00000_d_0000002", "train", 0.02),
    _row("0000009_00001_d_0000009", "val", 0.6),
]
LABELS = {
    "0000001": "day",
    "0000002": "night",
    "0000003": "exclude",
    "9999901_00000_d_0000001": "day",
    "0000009": "dusk",
}


def test_sequence_labels_cover_every_frame_and_image_labels_take_precedence():
    labels = resolve_labels(ROWS, {**LABELS, "0000001_00002_d_0000002": "exclude"})
    assert labels["0000001_00001_d_0000001"] == "day"
    assert labels["0000001_00002_d_0000002"] == "exclude"  # image key beats its sequence
    assert labels["0000002_00001_d_0000003"] == "night"
    assert labels["9999901_00000_d_0000002"] is None  # nothing names it: unlabelled


def test_blank_labels_count_as_unlabelled():
    assert resolve_labels(ROWS, {"0000002": ""})["0000002_00001_d_0000003"] is None


@pytest.mark.parametrize("key", ["0000077", "0000001_00009_d_0000099"])
def test_a_label_key_that_matches_nothing_is_rejected(key):
    with pytest.raises(ValueError, match=key):
        resolve_labels(ROWS, {key: "day"})


def test_an_unknown_label_is_rejected():
    with pytest.raises(ValueError, match="twilight"):
        resolve_labels(ROWS, {"0000001": "twilight"})


def test_manifests_come_from_labels_and_leave_out_excluded_and_unlabelled(tmp_path):
    counts = write_manifests(ROWS, resolve_labels(ROWS, LABELS), tmp_path)
    assert counts == {
        "train": {"day": 3, "night": 1, "dusk": 0, "exclude": 1, "unlabelled": 1},
        "val": {"day": 0, "night": 0, "dusk": 1, "exclude": 0, "unlabelled": 0},
    }
    assert read_manifest(tmp_path / "train" / "day.txt") == [
        "0000001_00001_d_0000001",
        "0000001_00002_d_0000002",
        "9999901_00000_d_0000001",
    ]
    assert read_manifest(tmp_path / "train" / "dusk.txt") == []  # written even when empty
    assert sorted(p.name for p in (tmp_path / "train").iterdir()) == [
        "day.txt",
        "dusk.txt",
        "night.txt",
    ]


def test_manifests_do_not_depend_on_row_order(tmp_path):
    shuffled = ROWS[:]
    random.Random(0).shuffle(shuffled)
    write_manifests(ROWS, resolve_labels(ROWS, LABELS), tmp_path / "a")
    write_manifests(shuffled, resolve_labels(shuffled, LABELS), tmp_path / "b")
    for path in sorted((tmp_path / "a").rglob("*.txt")):
        twin = tmp_path / "b" / path.relative_to(tmp_path / "a")
        assert path.read_bytes() == twin.read_bytes()


def test_labels_template_lists_video_sequences_darkest_first_with_blank_labels(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels_template(ROWS, path)
    template = read_labels(path)
    assert list(template) == ["0000002", "0000003", "0000001", "0000009"]  # no still collection
    assert set(template.values()) == {""}


def test_labels_template_never_overwrites_existing_labels(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("key,label\n0000001,day\n")
    with pytest.raises(FileExistsError):
        write_labels_template(ROWS, path)
    assert path.read_text() == "key,label\n0000001,day\n"


def test_a_sequence_key_brings_all_its_frames_darkest_first():
    frames = frames_for_key(ROWS, "0000001")
    assert [r.image_id for r in frames] == ["0000001_00002_d_0000002", "0000001_00001_d_0000001"]


def test_an_image_key_brings_just_that_image():
    assert [r.image_id for r in frames_for_key(ROWS, "9999901_00000_d_0000002")] == [
        "9999901_00000_d_0000002"
    ]


def test_an_unknown_key_brings_an_error():
    with pytest.raises(ValueError, match="0000077"):
        frames_for_key(ROWS, "0000077")


@pytest.mark.parametrize("n, expected", [(5, [0, 2, 4]), (4, [0, 2, 3]), (2, [0, 1]), (1, [0])])
def test_pick_three_takes_the_darkest_median_and_brightest_frame_once_each(n, expected):
    rows = [_row(f"0000001_{i:05d}_d_0000001", "train", 0.1 * (i + 1)) for i in range(n)]
    assert pick_three(rows) == [rows[i] for i in expected]


def test_preview_image_puts_the_frames_side_by_side(prepared_visdrone):
    stats = list(read_stats_csv(prepared_visdrone.stats_csv).values())
    frames = frames_for_key(stats, "0000001")  # two 640x480 frames
    image = preview_image(prepared_visdrone, frames, box=(240, 320))
    assert 2 * 320 <= image.width < 3 * 320
    assert 240 <= image.height < 2 * 240


def test_plots_and_sheets_are_written(prepared_visdrone, tmp_path):
    pytest.importorskip("matplotlib")
    stats = list(read_stats_csv(prepared_visdrone.stats_csv).values())
    labels = resolve_labels(stats, FAKE_LABELS)
    for plot, name in [(plot_log_luminance_histogram, "hist.png"), (plot_feature_scatter, "s.png")]:
        plot(stats, labels, tmp_path / name)
        assert Image.open(tmp_path / name).size[0] > 0
    plot_feature_scatter(stats, resolve_labels(stats, {}), tmp_path / "unlabelled.png")

    sheets = save_contact_sheets(prepared_visdrone, stats, labels, tmp_path / "lists")
    assert sorted(p.name for p in sheets) == [
        "train_day.jpg",
        "train_dusk.jpg",
        "train_night.jpg",
        "val_day.jpg",
    ]
    again = save_contact_sheets(prepared_visdrone, stats, labels, tmp_path / "again")
    assert [p.read_bytes() for p in sheets] == [p.read_bytes() for p in again]  # seeded


def test_sequence_sheets_show_every_video_sequence(prepared_visdrone, tmp_path):
    stats = list(read_stats_csv(prepared_visdrone.stats_csv).values())
    pages = save_sequence_sheets(prepared_visdrone, stats, tmp_path, per_page=2)
    assert [p.name for p in pages] == ["sequences_01.jpg", "sequences_02.jpg"]  # 3 videos
    assert all(Image.open(p).size[1] > 0 for p in pages)


# --- phase 3: 512-px copies --------------------------------------------------------------------


@pytest.mark.parametrize("index", range(len(FAKE_IMAGES)))
def test_lowres_copy_is_512_on_the_long_side_with_the_aspect_kept(prepared_visdrone, index):
    split, image_id, (width, height), _ = FAKE_IMAGES[index]
    img = read_srgb8(prepared_visdrone.lowres_image(split, image_id))
    assert max(img.shape[1:]) == 512
    assert img.shape[2] / img.shape[1] == pytest.approx(width / height, rel=0.01)


def test_lowres_copy_averages_in_linear_space(tmp_path):
    # A 1-px black/white checkerboard, downsampled 2x. Linear averaging gives linear 0.5, which
    # is sRGB code ~188. Averaging the sRGB codes would give 128: bright lights too dim.
    checker = (np.indices((512, 1024)).sum(axis=0) % 2 * 255).astype(np.uint8)
    layout = _layout(tmp_path)
    _write_raw(layout, "train", "checker", np.repeat(checker[..., None], 3, axis=2))
    out = read_srgb8(make_lowres_copy(layout, "train", "checker"))
    assert out.shape == (3, 256, 512)
    interior = out[:, 8:-8, 8:-8].float()
    assert 184 <= interior.min() and interior.max() <= 192


def test_lowres_copy_is_a_png_and_is_kept_unless_overwrite(tmp_path):
    layout = _layout(tmp_path)
    _write_raw(layout, "val", "img", np.full((600, 800, 3), 100, np.uint8))
    path = make_lowres_copy(layout, "val", "img")
    assert path == layout.lowres_image("val", "img") and Image.open(path).format == "PNG"
    path.write_bytes(b"sentinel")
    make_lowres_copy(layout, "val", "img")
    assert path.read_bytes() == b"sentinel"
    make_lowres_copy(layout, "val", "img", overwrite=True)
    assert Image.open(path).format == "PNG"
