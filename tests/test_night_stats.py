"""Tests for dawn/eval/night_stats.py.

Fast CPU tests with synthetic arrays to verify metric calculations,
boundary cases (black/white/saturated/clipped), and noise recovery.
"""

from __future__ import annotations

import math

import pytest
import torch

from dawn.data.color import srgb_to_linear
from dawn.eval.night_stats import (
    NightStats,
    compute_code0_fraction,
    compute_flat_patch_noise,
    compute_luminance_stats,
    compute_night_stats,
    compute_saturation_stats,
)


def test_invalid_shape_raises_error():
    with pytest.raises(ValueError, match="expected"):
        compute_night_stats(torch.zeros(1, 3, 32, 32))
    with pytest.raises(ValueError, match="expected"):
        compute_night_stats(torch.zeros(1, 32, 32))


def test_black_image_statistics():
    black = torch.zeros(3, 64, 64)
    stats = compute_night_stats(black)

    assert stats.lum_median == 0.0
    assert stats.lum_p90 == 0.0
    assert stats.code0_fraction == 1.0
    assert stats.sat_pixel_fraction == 0.0
    assert math.isnan(stats.sat_mean)
    assert math.isnan(stats.flat_noise_srgb)
    assert math.isnan(stats.flat_noise_linear)
    assert math.isnan(stats.flat_patch_mean)
    assert math.isnan(stats.noise_level)
    assert stats.highlight_ratio == pytest.approx(0.0)


def test_white_image_statistics():
    white = torch.ones(3, 64, 64)
    stats = compute_night_stats(white)

    assert stats.lum_median == pytest.approx(1.0, abs=1e-5)
    assert stats.lum_p90 == pytest.approx(1.0, abs=1e-5)
    assert stats.code0_fraction == 0.0
    assert stats.sat_pixel_fraction == pytest.approx(1.0)
    assert stats.sat_mean == pytest.approx(0.0, abs=1e-5)
    # White image is clipped at 255, so all patches are dropped
    assert math.isnan(stats.flat_noise_srgb)


def test_saturation_pure_colors():
    # Pure red in linear space (corresponds to [1, 0, 0] in sRGB)
    red = torch.zeros(3, 32, 32)
    red[0] = 1.0

    sat_mean, frac = compute_saturation_stats(red)
    assert frac == pytest.approx(1.0)
    assert sat_mean == pytest.approx(1.0, abs=1e-4)

    # Pure neutral gray in linear space
    gray_linear = srgb_to_linear(torch.full((3, 32, 32), 0.5))
    sat_mean, frac = compute_saturation_stats(gray_linear)
    assert frac == pytest.approx(1.0)
    assert sat_mean == pytest.approx(0.0, abs=1e-4)


def test_code0_fraction_partial():
    img = torch.zeros(3, 20, 20)
    # Set top half to non-zero (e.g. sRGB code 100)
    img[:, :10, :] = 0.2
    frac = compute_code0_fraction(img)
    assert frac == pytest.approx(0.5)


def test_luminance_percentiles():
    # Two halves: luminance 0.1 and 0.9
    img = torch.zeros(3, 100, 100)
    img[:, :50, :] = 0.1
    img[:, 50:, :] = 0.9

    median, p90 = compute_luminance_stats(img)
    assert median == pytest.approx(0.1, abs=0.05)
    assert p90 == pytest.approx(0.9, abs=0.05)


def test_flat_patch_noise_recovers_synthetic_noise():
    # Create an image with a planar ramp across the entire image
    # and add known Gaussian noise in sRGB space
    h, w = 64, 64
    y = torch.linspace(40, 120, h).view(h, 1)
    x = torch.linspace(40, 120, w).view(1, w)
    ramp_srgb = (y + x) / 2.0  # in [40, 120]
    ramp_srgb = ramp_srgb.expand(3, h, w).clone()

    torch.manual_seed(42)
    true_sigma_srgb = 4.0
    noise = torch.randn_like(ramp_srgb) * true_sigma_srgb
    noisy_srgb = (ramp_srgb + noise).clamp(10, 240)

    # Convert to linear
    noisy_linear = srgb_to_linear(noisy_srgb / 255.0)

    flat_noise_srgb, flat_noise_lin, patch_mean = compute_flat_patch_noise(
        noisy_linear, patch_size=8, flat_fraction=0.20
    )

    # Should recover noise close to true_sigma_srgb (~4.0 codes)
    assert not math.isnan(flat_noise_srgb)
    assert flat_noise_srgb == pytest.approx(true_sigma_srgb, rel=0.15)
    assert not math.isnan(flat_noise_lin)
    assert not math.isnan(patch_mean)
    assert 0.01 < patch_mean < 0.5


def test_flat_patch_noise_texture_rejection():
    # Construct an image with half textured (checkerboard) and half smooth with noise
    h, w = 64, 64
    img_srgb = torch.full((3, h, w), 100.0)

    # Left half: smooth + small noise (sigma = 3.0)
    torch.manual_seed(123)
    img_srgb[:, :, :32] += torch.randn(3, h, 32) * 3.0

    # Right half: large texture (amplitude = 50.0)
    checker = torch.zeros(h, 32)
    checker[::2, ::2] = 50.0
    checker[1::2, 1::2] = 50.0
    img_srgb[:, :, 32:] += checker.unsqueeze(0)

    img_linear = srgb_to_linear(img_srgb.clamp(10, 240) / 255.0)
    flat_noise_srgb, _, _ = compute_flat_patch_noise(
        img_linear, patch_size=8, flat_fraction=0.15
    )

    # The flattest 15% patches must come from the smooth half
    assert flat_noise_srgb == pytest.approx(3.0, rel=0.25)


def test_night_stats_as_dict():
    img = srgb_to_linear(torch.full((3, 32, 32), 0.2))
    stats = compute_night_stats(img)
    assert isinstance(stats, NightStats)
    d = stats.as_dict()

    expected_keys = {
        "lum_median",
        "lum_p90",
        "sat_mean",
        "sat_pixel_fraction",
        "highlight_ratio",
        "flat_noise_srgb",
        "flat_noise_linear",
        "flat_patch_mean",
        "noise_level",
        "code0_fraction",
    }
    assert set(d.keys()) == expected_keys
    for v in d.values():
        assert isinstance(v, float)
