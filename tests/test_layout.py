"""docs/conventions.md section 2: grid layout, coefficient order and tokens."""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid import layout  # noqa: E402

K, GY, GX = 8, 16, 16


def test_token_shapes_and_roundtrip():
    grid = torch.randn(2, 12, K, GY, GX)
    tokens = layout.grid_to_tokens(grid)
    assert tokens.shape == (2, 256, 96)
    assert torch.equal(layout.tokens_to_grid(tokens, GY, GX, K), grid)


def test_token_order_is_row_major_cells_with_bin_major_values():
    grid = torch.randn(1, 12, K, GY, GX)
    tokens = layout.grid_to_tokens(grid)
    for y, x, k, c in [(0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (3, 5, 2, 7), (15, 15, 7, 11)]:
        assert tokens[0, y * GX + x, k * 12 + c] == grid[0, c, k, y, x]


def test_coefficient_order():
    assert layout.NUM_COEFFS == len(layout.IDENTITY_COEFFS) == 12
    coeffs = torch.tensor(layout.IDENTITY_COEFFS)
    assert torch.equal(coeffs[layout.MATRIX].reshape(3, 3), torch.eye(3))
    assert torch.equal(coeffs[layout.OFFSET], torch.zeros(3))
