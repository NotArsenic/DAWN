"""docs/conventions.md sections 3 and 8: one trilinear implementation, cell-center coordinates.

`fit.py` and `upsample.py` take their weights from `slice.trilinear_weights` only, and
`slice_grid` (via `F.grid_sample`) must agree with it.
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from dawn.grid import fit as fit_mod  # noqa: E402
from dawn.grid import slice as slice_mod  # noqa: E402
from dawn.grid import upsample as upsample_mod  # noqa: E402
from dawn.grid.guidance import compute_guidance  # noqa: E402

K, GY, GX = 8, 4, 5  # deliberately not square, and not the project's grid


def test_trilinear_weights_are_a_partition_of_unity():
    guidance = torch.rand(2, 1, 40, 56)
    node_index, weight = slice_mod.trilinear_weights(guidance, (K, GY, GX))
    assert node_index.shape == weight.shape == (2, 8, 40, 56)
    assert node_index.dtype == torch.int64
    assert node_index.min() >= 0 and node_index.max() < K * GY * GX
    assert (weight >= 0).all()
    assert torch.allclose(weight.sum(dim=1), torch.ones(2, 40, 56), atol=1e-5)


def test_pixel_on_a_node_center_gets_exactly_that_node():
    # 3 px per cell, so pixel (3y + 1, 3x + 1) is the center of cell (y, x).
    k = 3
    guidance = torch.full((1, 1, 3 * GY, 3 * GX), (k + 0.5) / K)  # bin k's center
    grid = torch.randn(1, 2, K, GY, GX)
    coeffs = slice_mod.slice_grid(grid, guidance)
    rows, cols = torch.arange(GY) * 3 + 1, torch.arange(GX) * 3 + 1
    assert torch.allclose(coeffs[0][:, rows][:, :, cols], grid[0, :, k], atol=1e-5)


def test_outside_the_outer_centers_the_edge_node_wins():
    guidance = torch.zeros(1, 1, 3 * GY, 3 * GX)  # below bin 0's center
    grid = torch.randn(1, 2, K, GY, GX)
    coeffs = slice_mod.slice_grid(grid, guidance)
    assert torch.allclose(coeffs[0, :, 0, 0], grid[0, :, 0, 0, 0], atol=1e-6)  # the corner pixel
    assert torch.allclose(coeffs[0, :, -1, -1], grid[0, :, 0, -1, -1], atol=1e-6)


@pytest.mark.parametrize("channels", [2, 4])
def test_slice_grid_matches_trilinear_weights_and_grid_sample(channels):
    height, width = 40, 56
    grid = torch.randn(2, channels, K, GY, GX)
    guidance = torch.rand(2, 1, height, width)
    sliced = slice_mod.slice_grid(grid, guidance)
    assert sliced.shape == (2, channels, height, width)

    # Reference 1: the single source of corner nodes and weights.
    node_index, weight = slice_mod.trilinear_weights(guidance, (K, GY, GX))
    flat = grid.reshape(2, channels, K * GY * GX)
    via_weights = torch.stack(
        [(flat[b][:, node_index[b]] * weight[b]).sum(dim=1) for b in range(2)]
    )
    assert torch.allclose(sliced, via_weights, atol=1e-5)

    # Reference 2: F.grid_sample with the image's own normalized pixel centers and 2g - 1.
    ys = (2 * torch.arange(height) + 1) / height - 1
    xs = (2 * torch.arange(width) + 1) / width - 1
    coords = torch.stack(
        [
            xs.view(1, 1, width).expand(2, height, width),
            ys.view(1, height, 1).expand(2, height, width),
            2 * guidance[:, 0] - 1,
        ],
        dim=-1,
    ).unsqueeze(1)
    via_grid_sample = F.grid_sample(
        grid, coords, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(2)
    assert torch.allclose(sliced, via_grid_sample, atol=1e-5)


def test_slice_grid_rejects_bad_shapes():
    guidance = torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="grid"):
        slice_mod.slice_grid(torch.randn(1, 2, K, GY), guidance)
    with pytest.raises(ValueError, match="guidance"):
        slice_mod.slice_grid(torch.randn(1, 2, K, GY, GX), torch.rand(1, 3, 8, 8))
    with pytest.raises(ValueError, match="batch"):
        slice_mod.slice_grid(torch.randn(2, 2, K, GY, GX), guidance)


def _spy_on_trilinear_weights(monkeypatch, *modules):
    calls = []
    real = slice_mod.trilinear_weights

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(slice_mod, "trilinear_weights", spy)
    for module in modules:
        if hasattr(module, "trilinear_weights"):  # `from dawn.grid.slice import trilinear_weights`
            monkeypatch.setattr(module, "trilinear_weights", spy)
    return calls


def test_fit_gets_its_weights_from_slice(monkeypatch):
    calls = _spy_on_trilinear_weights(monkeypatch, fit_mod)
    thermal = torch.rand(1, 1, 32, 32)
    fit_mod.fit_affine_grid(
        thermal, torch.rand(1, 1, 32, 32), compute_guidance(thermal), grid_size=(K, GY, GX)
    )
    assert calls, "fit_affine_grid must call slice.trilinear_weights (docs/conventions.md rule 1)"


def test_upsample_gets_its_weights_from_slice(monkeypatch):
    calls = _spy_on_trilinear_weights(monkeypatch, fit_mod, upsample_mod)
    guide_lo, guide_hi = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 16, 16)
    upsample_mod.bilateral_guided_upsample(
        torch.rand(1, 2, 8, 8), guide_lo, guide_hi, grid_size=(K, 2, 2)
    )
    # Slicing goes through grid_sample, which test_slice_grid_matches_trilinear_weights_and_
    # grid_sample pins to trilinear_weights; the fit is the part that calls it.
    assert calls, "the BGU fit must use slice.trilinear_weights"


def test_fit_and_upsample_never_compute_corner_weights_themselves():
    for module in (fit_mod, upsample_mod):
        source = open(module.__file__).read()
        assert "floor" not in source, f"{module.__name__} computes grid coordinates itself"
