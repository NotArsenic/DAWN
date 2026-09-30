"""docs/conventions.md section 5: raw vs standardized grids."""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.normalize import standardize, unstandardize  # noqa: E402

K = 8


def test_standardize_uses_per_coefficient_and_bin_stats():
    grid = torch.randn(4, 12, K, 16, 16) * 3 + 1
    mean = grid.mean(dim=(0, 3, 4))  # (12, K)
    std = grid.std(dim=(0, 3, 4))
    z = standardize(grid, mean, std)
    assert z.shape == grid.shape
    assert torch.allclose(z.mean(dim=(0, 3, 4)), torch.zeros(12, K), atol=1e-5)
    assert torch.allclose(z.std(dim=(0, 3, 4)), torch.ones(12, K), atol=1e-4)
    assert torch.allclose(unstandardize(z, mean, std), grid, atol=1e-4)


def test_stats_index_coefficient_then_bin():
    grid = torch.zeros(1, 12, K, 2, 2)
    mean = torch.arange(12 * K, dtype=torch.float32).reshape(12, K)
    z = standardize(grid, mean, torch.ones(12, K))
    assert z[0, 3, 5, 1, 1] == -mean[3, 5]


def test_roundtrip_is_exact_below_the_std_floor():
    grid = torch.randn(1, 12, K, 2, 2) * 1e-7
    mean, std = torch.zeros(12, K), torch.full((12, K), 1e-9)
    assert torch.allclose(unstandardize(standardize(grid, mean, std), mean, std), grid, atol=1e-12)
