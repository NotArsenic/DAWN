"""Tests for bilateral guided upsampling and bilinear spatial interpolation."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from dawn.data.resample import area_downsample  # noqa: E402
from dawn.grid.upsample import UPSAMPLE_MODES, bilateral_guided_upsample, upsample_map  # noqa: E402

GRID = (8, 4, 4)


def _smooth_guide(height, width, seed=0):
    generator = torch.Generator().manual_seed(seed)
    noise = torch.rand(1, 1, 8, 8, generator=generator)
    return (
        F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False) * 0.8 + 0.1
    )


def test_exact_when_the_low_res_output_is_an_affine_function_of_the_guide():
    guide_hi = _smooth_guide(64, 80)
    guide_lo = area_downsample(guide_hi, (16, 20))
    low = torch.cat([0.7 * guide_lo + 0.1, -0.4 * guide_lo + 0.5], dim=1)
    out = bilateral_guided_upsample(low, guide_lo, guide_hi, grid_size=GRID)
    expected = torch.cat([0.7 * guide_hi + 0.1, -0.4 * guide_hi + 0.5], dim=1)
    assert out.shape == (1, 2, 64, 80)
    assert torch.allclose(out, expected, atol=2e-3)


def test_better_than_bilinear_on_a_step_edge_in_the_guide():
    height, width = 32, 64
    guide_hi = torch.full((1, 1, height, width), 0.2)
    guide_hi[..., 27:] = 0.8  # the edge falls inside a low-res pixel (27 / 4 = 6.75)
    truth = torch.where(guide_hi > 0.5, torch.tensor(0.9), torch.tensor(0.1))
    guide_lo, low = area_downsample(guide_hi, (8, 16)), area_downsample(truth, (8, 16))
    bilinear = F.interpolate(low, size=(height, width), mode="bilinear", align_corners=False)
    guided = bilateral_guided_upsample(low, guide_lo, guide_hi, grid_size=(8, 2, 4))
    err_guided = float((guided - truth).abs().mean())
    err_bilinear = float((bilinear - truth).abs().mean())
    assert err_guided < 0.5 * err_bilinear, (err_guided, err_bilinear)


def test_gradient_flows_to_low_and_matches_a_numerical_gradient():
    guide_hi = _smooth_guide(16, 20).double()
    guide_lo = area_downsample(guide_hi, (8, 10))
    low = torch.rand(1, 2, 8, 10, dtype=torch.float64, requires_grad=True)

    def fn(low):
        return bilateral_guided_upsample(low, guide_lo, guide_hi, grid_size=(4, 2, 2))

    assert torch.autograd.gradcheck(fn, (low,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_the_guides_are_constants():
    guide_lo = torch.rand(1, 1, 8, 8, requires_grad=True)
    guide_hi = torch.rand(1, 1, 16, 16, requires_grad=True)
    low = torch.rand(1, 1, 8, 8, requires_grad=True)
    bilateral_guided_upsample(low, guide_lo, guide_hi, grid_size=GRID).sum().backward()
    assert low.grad is not None and guide_lo.grad is None and guide_hi.grad is None


def test_weights_remove_pixels_from_the_fit():
    guide_hi = _smooth_guide(32, 32)
    guide_lo = area_downsample(guide_hi, (8, 8))
    low = 0.5 * guide_lo + 0.2
    weights = torch.ones_like(low)
    weights[..., :4, :4] = 0.0
    corrupted = low.clone()
    corrupted[..., :4, :4] = 50.0
    clean = bilateral_guided_upsample(low, guide_lo, guide_hi, weights, grid_size=GRID)
    dirty = bilateral_guided_upsample(corrupted, guide_lo, guide_hi, weights, grid_size=GRID)
    assert torch.allclose(clean, dirty, atol=1e-5)


def test_rejects_bad_shapes():
    low, guide_lo, guide_hi = (
        torch.rand(1, 2, 8, 8),
        torch.rand(1, 1, 8, 8),
        torch.rand(1, 1, 16, 16),
    )
    with pytest.raises(ValueError, match="guide_lo"):
        bilateral_guided_upsample(low, torch.rand(1, 1, 4, 4), guide_hi, grid_size=GRID)
    with pytest.raises(ValueError, match="guide_hi"):
        bilateral_guided_upsample(low, guide_lo, torch.rand(2, 1, 16, 16), grid_size=GRID)


# --- the switch: bilinear (the default) or bilateral guided ----------------------------------


def test_the_modes_are_bilinear_and_bgu():
    assert UPSAMPLE_MODES == ("bilinear", "bgu")


def test_bilinear_mode_is_plain_align_corners_false_bilinear_and_ignores_the_guides():
    low = torch.rand(2, 2, 8, 10)
    hi = torch.rand(2, 1, 32, 40)
    out = upsample_map(low, torch.rand(2, 1, 8, 10), hi, "bilinear")
    assert out.shape == (2, 2, 32, 40)
    assert torch.equal(out, F.interpolate(low, size=(32, 40), mode="bilinear", align_corners=False))
    other = upsample_map(low, torch.zeros(2, 1, 8, 10), torch.zeros_like(hi), "bilinear")
    assert torch.equal(out, other)  # no guidance in this mode


def test_bilinear_mode_is_differentiable_in_low():
    low = torch.rand(1, 2, 4, 5, requires_grad=True)
    upsample_map(low, torch.rand(1, 1, 4, 5), torch.rand(1, 1, 16, 20), "bilinear").sum().backward()
    assert low.grad is not None and low.grad.abs().sum() > 0


def test_bgu_mode_is_the_bilateral_guided_upsample_with_the_given_grid():
    guide_hi = _smooth_guide(32, 40)
    guide_lo = area_downsample(guide_hi, (8, 10))
    low = torch.cat([0.7 * guide_lo + 0.1, -0.4 * guide_lo + 0.5], dim=1)
    kwargs = {"grid_size": (4, 2, 2), "ridge_alpha": 1e-3, "refine_steps": 1}
    via_switch = upsample_map(low, guide_lo, guide_hi, "bgu", **kwargs)
    direct = bilateral_guided_upsample(low, guide_lo, guide_hi, **kwargs)
    assert torch.equal(via_switch, direct)


def test_an_unknown_mode_is_an_error():
    low, g = torch.rand(1, 2, 4, 4), torch.rand(1, 1, 8, 8)
    with pytest.raises(ValueError, match="upsample"):
        upsample_map(low, torch.rand(1, 1, 4, 4), g, "bicubic")
