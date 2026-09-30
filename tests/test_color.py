"""docs/conventions.md section 1: sRGB <-> linear."""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import linear_to_srgb, srgb_gamma_slope, srgb_to_linear  # noqa: E402


def test_srgb_roundtrip():
    x = torch.rand(2, 3, 16, 16)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_gamma_slope_matches_finite_difference():
    x = torch.tensor([0.01, 0.05, 0.2, 0.5, 0.9], dtype=torch.float64)
    eps = 1e-6
    numeric = (linear_to_srgb(x + eps) - linear_to_srgb(x - eps)) / (2 * eps)
    assert torch.allclose(srgb_gamma_slope(x), numeric, rtol=1e-4)
