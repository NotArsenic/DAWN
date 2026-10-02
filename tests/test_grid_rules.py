"""docs/conventions.md sections 3 and 8: one trilinear implementation, cell-center coordinates.

These tests skip while fit.py / slice.py are still stubs and run as soon as they are implemented.
"""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from dawn.grid import fit as fit_mod  # noqa: E402
from dawn.grid import slice as slice_mod  # noqa: E402
from dawn.grid.guidance import compute_guidance  # noqa: E402
from tests.helpers import skip_until_implemented  # noqa: E402

K, GY, GX = 8, 16, 16


@skip_until_implemented
def test_trilinear_weights_are_a_partition_of_unity():
    guidance = torch.rand(2, 1, 40, 56)
    node_index, weight = slice_mod.trilinear_weights(guidance, (K, GY, GX))
    assert node_index.shape == weight.shape == (2, 8, 40, 56)
    assert node_index.dtype == torch.int64
    assert node_index.min() >= 0 and node_index.max() < K * GY * GX
    assert (weight >= 0).all()
    assert torch.allclose(weight.sum(dim=1), torch.ones(2, 40, 56), atol=1e-5)


@skip_until_implemented
def test_pixel_on_a_node_center_gets_exactly_that_node():
    # 48 px / 16 cells = 3 px per cell, so pixel (3y + 1, 3x + 1) is the center of cell (y, x).
    k = 3
    guidance = torch.full((1, 1, 48, 48), (k + 0.5) / K)  # bin k's center
    grid = torch.randn(1, 12, K, GY, GX)
    coeffs = slice_mod.slice_grid(grid, guidance)
    rows = torch.arange(GY) * 3 + 1
    cols = torch.arange(GX) * 3 + 1
    assert torch.allclose(coeffs[0][:, rows][:, :, cols], grid[0, :, k], atol=1e-5)


@skip_until_implemented
def test_slice_grid_matches_trilinear_weights_and_grid_sample():
    height, width = 40, 56
    grid = torch.randn(2, 12, K, GY, GX)
    guidance = torch.rand(2, 1, height, width)
    sliced = slice_mod.slice_grid(grid, guidance)
    assert sliced.shape == (2, 12, height, width)

    # Reference 1: the single source of corner nodes and weights.
    node_index, weight = slice_mod.trilinear_weights(guidance, (K, GY, GX))
    flat = grid.reshape(2, 12, K * GY * GX)
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


@skip_until_implemented
def test_fit_gets_its_weights_from_slice(monkeypatch):
    calls = []
    real = slice_mod.trilinear_weights

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(slice_mod, "trilinear_weights", spy)
    if hasattr(fit_mod, "trilinear_weights"):  # `from dawn.grid.slice import trilinear_weights`
        monkeypatch.setattr(fit_mod, "trilinear_weights", spy)

    night = torch.rand(1, 3, 32, 32)
    day = torch.rand(1, 3, 32, 32)
    fit_mod.fit_grid(night, day, compute_guidance(night))
    assert calls, "fit_grid must call slice.trilinear_weights (docs/conventions.md section 8)"


@skip_until_implemented
def test_oracle_recovers_a_global_affine_transform():
    night = torch.rand(1, 3, 64, 64) * 0.3 + 0.01
    # Strong off-diagonal terms: a transposed M (the usual fit bug) would be far off.
    matrix = torch.tensor([[1.5, 0.8, 0.1], [0.2, 1.2, 0.9], [0.7, 0.1, 1.4]])
    offset = torch.tensor([0.02, 0.01, 0.03]).view(1, 3, 1, 1)
    day = torch.einsum("ij,bjhw->bihw", matrix, night) + offset

    guidance = compute_guidance(night)
    grid, node_weights = fit_mod.fit_grid(night, day, guidance)
    assert grid.shape == (1, 12, K, GY, GX)
    assert node_weights.shape == (1, K, GY, GX)
    # Every pixel spreads a total weight of 1 over its 8 corner nodes.
    assert torch.allclose(node_weights.sum(), torch.tensor(64.0 * 64.0), rtol=1e-4)

    rendered = slice_mod.apply_affine(slice_mod.slice_grid(grid, guidance), night)
    psnr = -10 * torch.log10((rendered - day).pow(2).mean())
    assert psnr > 35, f"oracle PSNR {psnr:.1f} dB; a global affine map should be recovered"
