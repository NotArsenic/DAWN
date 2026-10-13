"""Tests for uncertainty calibration and rank correlation between spread and error."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")
import numpy as np  # noqa: E402

from dawn.eval.calibration import (  # noqa: E402
    calibration_curve,
    expected_calibration_error,
    fit_scale,
    rank_correlation,
)


def _pair(n=4000, noise=0.0, seed=0):
    """Generate a synthetic spread map and correlated absolute error map."""
    g = torch.Generator().manual_seed(seed)
    spread = torch.rand(1, 1, n, generator=g) * 10
    error = spread * 2.0 + torch.randn(1, 1, n, generator=g) * noise
    return spread, error.abs()


def test_perfectly_ordered_maps_score_rho_one():
    spread, error = _pair(noise=0.0)
    assert rank_correlation(spread, error) == pytest.approx(1.0, abs=1e-6)


def test_rho_is_invariant_to_rescaling_the_spread():
    """Verify rank correlation is invariant to monotone rescaling of spread."""
    spread, error = _pair(noise=3.0)
    base = rank_correlation(spread, error)
    for factor in (0.01, 7.0, 1000.0):
        assert rank_correlation(spread * factor, error) == pytest.approx(base, abs=1e-9)
    # any monotone map, not just a scale
    assert rank_correlation(spread.exp(), error) == pytest.approx(base, abs=1e-9)


def test_rho_falls_as_the_maps_decouple():
    scores = [rank_correlation(*_pair(noise=n, seed=1)) for n in (0.5, 5.0, 50.0)]
    assert scores == sorted(scores, reverse=True), scores


def test_an_uninformative_spread_scores_about_zero():
    g = torch.Generator().manual_seed(0)
    spread = torch.rand(1, 1, 5000, generator=g)
    error = torch.rand(1, 1, 5000, generator=g)
    assert abs(rank_correlation(spread, error)) < 0.08


def test_anticorrelated_maps_score_negative():
    spread, error = _pair(noise=0.0)
    assert rank_correlation(spread, -error) == pytest.approx(-1.0, abs=1e-6)


@pytest.mark.parametrize(
    "spread, error",
    [
        (torch.zeros(1, 1, 100), torch.rand(1, 1, 100)),   # deterministic sampler
        (torch.rand(1, 1, 100), torch.zeros(1, 1, 100)),   # no error at all
        (torch.rand(1, 1, 1), torch.rand(1, 1, 1)),        # one pixel
    ],
)
def test_undefined_cases_return_zero_not_nan(spread, error):
    out = rank_correlation(spread, error)
    assert out == 0.0 and out == out


def test_the_mask_selects_which_pixels_count():
    """Verify masked-out pixels do not influence rank correlation."""
    spread, error = _pair(n=2000, noise=0.0)
    mask = torch.ones_like(spread)
    mask[..., 1000:] = 0.0
    clean = rank_correlation(spread, error, mask)
    wrecked = error.clone()
    wrecked[..., 1000:] = torch.rand(1000) * 1e6  # garbage, all masked out
    assert rank_correlation(spread, wrecked, mask) == pytest.approx(clean, abs=1e-9)


def test_mismatched_shapes_are_clear_errors():
    with pytest.raises(ValueError, match="same map"):
        rank_correlation(torch.rand(1, 1, 10), torch.rand(1, 1, 11))
    with pytest.raises(ValueError, match="mask .* must match"):
        rank_correlation(torch.rand(1, 1, 10), torch.rand(1, 1, 10), torch.ones(1, 1, 11))


def test_fit_scale_recovers_a_known_factor():
    g = torch.Generator().manual_seed(0)
    spread = torch.rand(1, 1, 5000, generator=g) * 10
    assert fit_scale(spread, spread * 3.0) == pytest.approx(3.0, rel=1e-5)
    assert fit_scale(torch.zeros(1, 1, 10), torch.rand(1, 1, 10)) == 0.0


def test_calibration_curve_bins_by_quantile_and_rises():
    spread, error = _pair(n=8000, noise=1.0)
    c = calibration_curve(spread, error, num_bins=10)
    assert c["bin_spread"].size == 10 and c["bin_errors"].size == 10
    assert c["bin_counts"].sum() == 8000
    assert c["bin_counts"].std() < 1.0, "equal-FREQUENCY bins: counts are near-identical"
    assert list(c["bin_spread"]) == sorted(c["bin_spread"])
    assert c["monotonicity"] > 0.9 and c["spearman"] > 0.9


def test_a_useless_spread_has_a_flat_curve():
    g = torch.Generator().manual_seed(0)
    spread, error = torch.rand(1, 1, 8000, generator=g), torch.rand(1, 1, 8000, generator=g)
    c = calibration_curve(spread, error, num_bins=10)
    assert c["monotonicity"] < 0.8
    assert abs(c["spearman"]) < 0.1


def test_calibration_curve_survives_too_few_pixels():
    c = calibration_curve(torch.rand(1, 1, 3), torch.rand(1, 1, 3), num_bins=20)
    assert c["bin_spread"].size == 0 and c["spearman"] == 0.0


def test_ece_reports_the_scale_it_used():
    spread, error = _pair(n=5000, noise=0.5)
    out = expected_calibration_error(spread, error, num_bins=10)
    assert set(out) == {"ece", "scale"}
    assert out["scale"] == pytest.approx(2.0, rel=0.05), "|error| ~ 2 * spread by construction"
    assert out["ece"] < 1.0, "with the right scale the gap is small"


def test_an_unscaled_ece_is_large_which_is_why_the_scale_exists():
    """Verify unscaled ECE reflects scale mismatch when spread magnitude differs from error."""
    spread, error = _pair(n=5000, noise=0.0)
    fitted = expected_calibration_error(spread, error, num_bins=10)
    unscaled = expected_calibration_error(spread, error, num_bins=10, scale=1.0)
    assert unscaled["ece"] > 10 * fitted["ece"]
    assert unscaled["scale"] == 1.0


def test_ece_accepts_a_scale_fitted_elsewhere():
    spread, error = _pair(n=4000, noise=0.5, seed=2)
    held_out = fit_scale(*_pair(n=4000, noise=0.5, seed=3))
    out = expected_calibration_error(spread, error, num_bins=10, scale=held_out)
    assert out["scale"] == pytest.approx(held_out)


def test_subsampling_is_reproducible_and_bounded():
    spread, error = _pair(n=20000, noise=2.0)
    a = rank_correlation(spread, error, max_pixels=1000, seed=7)
    b = rank_correlation(spread, error, max_pixels=1000, seed=7)
    assert a == b
    assert a != rank_correlation(spread, error, max_pixels=1000, seed=8)
    full = rank_correlation(spread, error, max_pixels=0)
    assert abs(a - full) < 0.1, "a 1k subsample should track the full map"


def test_non_finite_pixels_are_dropped():
    spread, error = _pair(n=2000, noise=0.0)
    error[..., :50] = float("nan")
    spread[..., 50:100] = float("inf")
    out = rank_correlation(spread, error)
    assert np.isfinite(out) and out > 0.99
