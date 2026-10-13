"""Tests for contact sheet generation and parallel execution helpers."""

import operator

import pytest

pytest.importorskip("torch")
pytest.importorskip("PIL")

import torch  # noqa: E402

from dawn.data.sheets import (  # noqa: E402
    run_parallel,
    thumbnail,
    thumbnail_thermal,
    tile_sheet,
)


def test_thumbnail_fits_the_box_keeps_the_aspect_and_never_upsamples():
    img = torch.rand(3, 512, 640)
    thumb = thumbnail(img, box=(128, 128))
    assert thumb.dtype == torch.uint8
    assert thumb.shape == (3, 102, 128)  # 640 wide is the binding side: 512 * 128 / 640 = 102.4

    small = torch.rand(3, 20, 30)
    assert thumbnail(small, box=(128, 128)).shape == (3, 20, 30)


def test_thumbnail_averages_in_linear_light():
    # Alternating linear 0 / 1 columns average to linear 0.5, which is sRGB code 188, not 128.
    img = torch.zeros(3, 8, 8)
    img[:, :, ::2] = 1.0
    thumb = thumbnail(img, box=(4, 4))
    assert thumb.shape == (3, 4, 4)
    assert (thumb == 188).all()


def test_thumbnail_thermal_is_a_plain_code_average_in_three_channels():
    # Thermal values are display codes, not sRGB: 0 and 1 average to 0.5, which is code 128.
    thermal = torch.zeros(1, 8, 8)
    thermal[:, :, ::2] = 1.0
    thumb = thumbnail_thermal(thermal, box=(4, 4))
    assert thumb.dtype == torch.uint8
    assert thumb.shape == (3, 4, 4)
    assert (thumb == 128).all()


def test_tile_sheet_has_the_expected_size():
    thumbs = [torch.full((3, 10, 12), 100, dtype=torch.uint8) for _ in range(5)]
    sheet = tile_sheet(thumbs, [f"{i}" for i in range(5)], "title", columns=3, box=(10, 12))
    width = 3 * (12 + 6) + 6
    assert sheet.size[0] == width
    assert sheet.size[1] > 2 * 10  # two rows plus the title and label strips


def test_tile_sheet_rejects_mismatched_labels():
    with pytest.raises(ValueError, match="labels"):
        tile_sheet([torch.zeros(3, 4, 4, dtype=torch.uint8)], [], "t", box=(4, 4))


@pytest.mark.parametrize("workers", [1, 2])
def test_run_parallel_keeps_job_order(workers):
    jobs = [(i, 10) for i in range(7)]
    assert run_parallel(operator.mul, jobs, workers) == [i * 10 for i in range(7)]
