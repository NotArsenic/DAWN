"""Tests for evaluation comparison grids, color mapping, and visualization sheet export."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("PIL")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dawn.data.color import linear_to_srgb  # noqa: E402
from dawn.eval.visualize import (  # noqa: E402
    colorize,
    grey_to_rgb,
    make_comparison_grid,
    save_comparison_sheets,
    save_grid,
)

H, W = 12, 16


def _batch(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        thermal=torch.rand(n, 1, H, W, generator=g),
        generated=torch.rand(n, 3, H, W, generator=g),
        real=torch.rand(n, 3, H, W, generator=g),
        l_std=torch.rand(n, 1, H, W, generator=g) * 12,
        ab_std=torch.rand(n, 1, H, W, generator=g) * 12,
    )


def test_the_ramp_is_fixed_so_two_frames_are_comparable():
    """Verify color mapping uses fixed absolute scale independent of other pixel values."""
    quiet = torch.full((1, 1, 4, 4), 2.0)
    loud = torch.full((1, 1, 4, 4), 2.0)
    loud[0, 0, 0, 0] = 9.0
    assert torch.allclose(colorize(quiet, 10.0)[:, :, 1, 1], colorize(loud, 10.0)[:, :, 1, 1])


def test_values_above_the_ramp_saturate_rather_than_rescale():
    a = colorize(torch.full((1, 1, 2, 2), 10.0), 10.0)
    b = colorize(torch.full((1, 1, 2, 2), 1e6), 10.0)
    assert torch.allclose(a, b)


def test_colorize_output_shape_range_and_monotonicity():
    out = colorize(torch.linspace(0, 10, 16).reshape(1, 1, 4, 4), 10.0)
    assert out.shape == (1, 3, 4, 4) and out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    flat = out[0].reshape(3, -1)
    assert flat[:, 0].sum() < flat[:, -1].sum(), "magma gets brighter with value"


def test_negatives_clamp_to_the_bottom_of_the_ramp():
    assert torch.allclose(colorize(torch.full((1, 1, 2, 2), -5.0), 10.0),
                          colorize(torch.zeros(1, 1, 2, 2), 10.0))


@pytest.mark.parametrize(
    "value, vmax, message",
    [
        (torch.rand(1, 3, 4, 4), 10.0, "B, 1, H, W"),
        (torch.rand(1, 1, 4, 4), 0.0, "vmax must be positive"),
        (torch.rand(1, 1, 4, 4), -1.0, "vmax must be positive"),
    ],
)
def test_colorize_rejects_bad_input(value, vmax, message):
    with pytest.raises(ValueError, match=message):
        colorize(value, vmax)


def test_thermal_is_repeated_not_false_coloured():
    """Verify single-channel thermal image is converted to grayscale RGB representation."""
    t = torch.rand(2, 1, H, W)
    out = grey_to_rgb(t)
    assert out.shape == (2, 3, H, W)
    assert torch.allclose(out[:, 0], out[:, 1]) and torch.allclose(out[:, 1], out[:, 2])
    assert torch.allclose(out[:, 0], t[:, 0])


def test_grid_has_one_row_per_pair_and_one_column_per_panel():
    b = _batch(n=3)
    grid = make_comparison_grid(**b, pad=2)
    assert grid.shape == (3, 3 * (H + 2) - 2, 5 * (W + 2) - 2)
    grid3 = make_comparison_grid(b["thermal"], b["generated"], b["real"], pad=2)
    assert grid3.shape == (3, 3 * (H + 2) - 2, 3 * (W + 2) - 2), "spread columns are optional"


def test_grid_leaves_as_srgb_not_linear():
    b = _batch(n=1)
    grid = make_comparison_grid(b["thermal"], b["generated"], b["real"], pad=0)
    panel = grid[:, :H, W : 2 * W]
    assert torch.allclose(panel, linear_to_srgb(b["generated"][0].clamp(0, 1)), atol=1e-6)
    assert not torch.allclose(panel, b["generated"][0], atol=1e-3), "must not be raw linear"


def test_grid_values_stay_in_range_and_padding_is_white():
    grid = make_comparison_grid(**_batch(n=2), pad=3)
    assert grid.min() >= 0.0 and grid.max() <= 1.0
    assert torch.allclose(grid[:, H : H + 3, :], torch.ones(3, 3, grid.shape[-1]))


def test_mismatched_panels_are_a_clear_error():
    b = _batch(n=2)
    with pytest.raises(ValueError, match="disagree on"):
        make_comparison_grid(b["thermal"], b["generated"], torch.rand(2, 3, H + 4, W))
    with pytest.raises(ValueError, match="linear RGB"):
        make_comparison_grid(b["thermal"], torch.rand(2, 1, H, W), b["real"])


def test_save_grid_writes_a_readable_png(tmp_path):
    grid = make_comparison_grid(**_batch(n=2))
    p = save_grid(grid, tmp_path / "g.png")
    assert p.exists() and not list(tmp_path.glob("*.partial.png"))
    img = np.asarray(Image.open(p))
    assert img.shape == (grid.shape[1], grid.shape[2], 3) and img.dtype == np.uint8


def test_sheets_split_the_batch_and_label_each_row(tmp_path):
    b = _batch(n=5)
    ids = [f"val_{i:05d}" for i in range(5)]
    paths = save_comparison_sheets(tmp_path, **b, image_ids=ids, rows_per_sheet=2)
    assert [p.name for p in paths] == ["examples_000.png", "examples_001.png", "examples_002.png"]
    assert all(p.exists() for p in paths)
    assert (tmp_path / "examples_000.txt").read_text().split() == ids[:2]
    assert (tmp_path / "examples_002.txt").read_text().split() == ids[4:], "the short last sheet"
    last = np.asarray(Image.open(paths[-1]))
    assert last.shape[0] == H, "one row, not padded out to a full sheet"


def test_sheets_work_without_ids_or_spread(tmp_path):
    b = _batch(n=2)
    paths = save_comparison_sheets(
        tmp_path, b["thermal"], b["generated"], b["real"], prefix="plain", rows_per_sheet=4
    )
    assert len(paths) == 1 and Path(paths[0]).name == "plain_000.png"
    assert not list(tmp_path.glob("*.txt"))
