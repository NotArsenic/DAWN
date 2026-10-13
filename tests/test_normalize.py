"""Tests for token standardization, unstandardization, and weighted moments estimation."""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.normalize import (  # noqa: E402
    STD_FLOOR,
    WeightedMoments,
    standardize,
    unstandardize,
)

N, D = 6, 5


def test_standardize_uses_per_value_stats_and_round_trips():
    tokens = torch.randn(4, N, D) * torch.arange(1, D + 1) + torch.arange(D)
    mean, std = tokens.mean(dim=(0, 1)), tokens.std(dim=(0, 1))
    z = standardize(tokens, mean, std)
    assert z.shape == tokens.shape
    assert torch.allclose(z.mean(dim=(0, 1)), torch.zeros(D), atol=1e-5)
    assert torch.allclose(z.std(dim=(0, 1)), torch.ones(D), atol=1e-4)
    assert torch.allclose(unstandardize(z, mean, std), tokens, atol=1e-4)


def test_stats_index_the_last_dimension():
    mean = torch.arange(D, dtype=torch.float32)
    z = standardize(torch.zeros(1, N, D), mean, torch.ones(D))
    assert z[0, 3, 4] == -4.0


def test_roundtrip_is_exact_below_the_std_floor():
    tokens = torch.randn(1, N, D) * 1e-7
    mean, std = torch.zeros(D), torch.full((D,), 1e-9)
    assert STD_FLOOR > 1e-9
    assert torch.allclose(
        unstandardize(standardize(tokens, mean, std), mean, std), tokens, atol=1e-12
    )


def test_stats_of_the_wrong_width_are_rejected():
    with pytest.raises(ValueError, match="stats"):
        standardize(torch.zeros(1, N, D), torch.zeros(D + 1), torch.ones(D + 1))


def test_weighted_moments_match_torch_and_ignore_zero_weight_values():
    values = torch.randn(200, 3)
    moments = WeightedMoments(3)
    moments.add(values[:100], None)
    moments.add(values[100:], torch.ones(100))
    mean, std = moments.finalize()
    assert torch.allclose(mean, values.mean(dim=0), atol=1e-5)
    assert torch.allclose(std, values.std(dim=0), atol=1e-4)

    junk = torch.cat([values, torch.full((50, 3), 1e3)])
    weights = torch.cat([torch.ones(200), torch.zeros(50)])
    moments = WeightedMoments(3)
    moments.add(junk, weights)
    mean2, std2 = moments.finalize()
    assert torch.allclose(mean2, mean, atol=1e-5) and torch.allclose(std2, std, atol=1e-4)


def test_weighted_moments_without_data_is_an_error():
    with pytest.raises(ValueError, match="no weight"):
        WeightedMoments(2).finalize()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU (MPS)")
def test_weighted_moments_of_mps_tensors_match_the_cpu():
    """Verify MPS weighted moments agree with CPU calculations across device transfers."""
    g = torch.Generator().manual_seed(0)
    values = torch.randn(300, 16, generator=g).reshape(10, 30, 16).permute(0, 2, 1).reshape(-1, 30)
    weights = torch.rand(values.shape[0], generator=g)
    cpu = WeightedMoments(30)
    cpu.add(values, weights)
    mps = WeightedMoments(30)
    mps.add(values.to("mps"), weights.to("mps"))
    for want, got in zip(cpu.finalize(), mps.finalize()):
        assert torch.allclose(want, got, atol=1e-5)
