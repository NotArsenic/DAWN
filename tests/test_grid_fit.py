"""Tests for bilateral affine grid least-squares fitting and iterative refinement."""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.fit import fit_affine_grid  # noqa: E402
from dawn.grid.guidance import compute_guidance  # noqa: E402
from dawn.grid.slice import apply_gain_offset, slice_grid  # noqa: E402

K, GY, GX = 8, 6, 8
SIZE = (K, GY, GX)


def _render(grid, x, guidance, clamp=True):
    return apply_gain_offset(slice_grid(grid, guidance), x, clamp=clamp)


def _psnr(a, b, peak=1.0):
    return float(10 * torch.log10(peak**2 / (a - b).pow(2).mean().clamp(min=1e-20)))


def _smooth_grid(seed=0, cycles=1.0, gain=0.6, offset=0.2, channels=1):
    """Generate synthetic smoothly-varying gain and offset grid coefficients."""
    generator = torch.Generator().manual_seed(seed)
    k = torch.linspace(0, 1, K).view(1, K, 1, 1)
    y = torch.linspace(0, 1, GY).view(1, 1, GY, 1)
    x = torch.linspace(0, 1, GX).view(1, 1, 1, GX)
    phase = torch.rand(2 * channels, 3, generator=generator) * 6.283
    ripple = (
        torch.sin(cycles * 6.283 * k + phase[:, 0].view(-1, 1, 1, 1))
        + torch.sin(cycles * 6.283 * y + phase[:, 1].view(-1, 1, 1, 1))
        + torch.sin(cycles * 6.283 * x + phase[:, 2].view(-1, 1, 1, 1))
    ) / 3.0
    base = torch.tensor([gain, offset] * channels).view(-1, 1, 1, 1)
    scale = torch.tensor([0.15, 0.04] * channels).view(-1, 1, 1, 1)
    return (base + scale * ripple).unsqueeze(0)


def _unclipped(grid, x, guidance):
    raw = _render(grid, x, guidance, clamp=False)
    assert raw.min() > 0.0 and raw.max() < 1.0, f"target clips: [{raw.min():.3f}, {raw.max():.3f}]"


def _thermal(seed, size=(1, 1, 96, 128)):
    return torch.rand(size, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1


def test_a_constant_gain_and_offset_is_recovered_exactly_at_every_node():
    thermal = _thermal(1)
    y = 0.7 * thermal + 0.1
    grid, node_weights = fit_affine_grid(thermal, y, compute_guidance(thermal), grid_size=SIZE)
    assert grid.shape == (1, 2, K, GY, GX) and node_weights.shape == (1, K, GY, GX)
    # Starved nodes inherit their cell, and the cell the image: the same affine everywhere.
    assert torch.allclose(grid[0, 0], torch.full((K, GY, GX), 0.7), atol=2e-3)
    assert torch.allclose(grid[0, 1], torch.full((K, GY, GX), 0.1), atol=2e-3)


def test_fitting_a_known_smooth_grid_and_re_rendering_beats_45_db_with_refine():
    thermal = _thermal(4, (1, 1, 192, 256))
    guidance = compute_guidance(thermal)
    truth = _smooth_grid(seed=1)
    _unclipped(truth, thermal, guidance)
    target = _render(truth, thermal, guidance)
    fitted, _ = fit_affine_grid(
        thermal, target, guidance, grid_size=SIZE, ridge_alpha=1e-3, refine_steps=2
    )
    assert _psnr(_render(fitted, thermal, guidance), target) > 45


def test_masked_pixels_have_no_influence():
    thermal = _thermal(2)
    guidance = compute_guidance(thermal)
    target = _render(_smooth_grid(seed=3), thermal, guidance)
    mask = torch.ones_like(thermal)
    mask[..., :40, 20:90] = 0.0
    corrupted = target.clone()
    corrupted[..., :40, 20:90] = 1e3  # garbage under the mask
    clean_fit, w_clean = fit_affine_grid(thermal, target, guidance, mask, SIZE, refine_steps=1)
    dirty_fit, w_dirty = fit_affine_grid(thermal, corrupted, guidance, mask, SIZE, refine_steps=1)
    assert torch.equal(clean_fit, dirty_fit) or torch.allclose(clean_fit, dirty_fit, atol=1e-5)
    assert torch.allclose(w_clean, w_dirty)
    assert torch.isclose(w_clean.sum(), mask.sum(), rtol=1e-4)  # each pixel hands out its weight


def test_no_weights_means_every_pixel_counts_once():
    thermal = _thermal(5)
    target = 0.5 * thermal + 0.2
    guidance = compute_guidance(thermal)
    plain, w_plain = fit_affine_grid(thermal, target, guidance, None, SIZE)
    ones, w_ones = fit_affine_grid(thermal, target, guidance, torch.ones_like(thermal), SIZE)
    assert torch.allclose(plain, ones, atol=1e-6) and torch.allclose(w_plain, w_ones)
    assert torch.isclose(w_plain.sum(), torch.tensor(float(thermal.numel())), rtol=1e-4)


def test_refine_reduces_the_render_residual():
    # A grid that varies quickly from node to node is where splat-and-solve is biased.
    thermal = _thermal(6, (1, 1, 192, 256))
    guidance = compute_guidance(thermal)
    truth = _smooth_grid(seed=2, cycles=2.0)
    target = _render(truth, thermal, guidance)
    errors = []
    for refine in (0, 1, 2):
        grid, _ = fit_affine_grid(thermal, target, guidance, None, SIZE, refine_steps=refine)
        errors.append(float((_render(grid, thermal, guidance) - target).pow(2).mean()))
    assert errors[1] < errors[0] and errors[2] < errors[1], errors


def test_output_channels_share_the_normal_equations_but_not_the_answer():
    thermal = _thermal(7)
    guidance = compute_guidance(thermal)
    y = torch.cat([0.4 * thermal + 0.1, -0.3 * thermal + 0.6], dim=1)
    both, w_both = fit_affine_grid(thermal, y, guidance, None, SIZE, refine_steps=1)
    assert both.shape == (1, 4, K, GY, GX)  # channels 2c = gain, 2c + 1 = offset
    for channel in range(2):
        single, w_single = fit_affine_grid(
            thermal, y[:, channel : channel + 1], guidance, None, SIZE, refine_steps=1
        )
        assert torch.allclose(both[:, 2 * channel : 2 * channel + 2], single, atol=1e-5)
        assert torch.allclose(w_both, w_single)
    assert both[0, 0].mean() > 0 > both[0, 2].mean()


def test_an_empty_bin_inherits_its_cell_rather_than_zero():
    """Verify starved nodes inherit spatial cell parameters via ridge regularization."""
    thermal = torch.rand(1, 1, 96, 128, generator=torch.Generator().manual_seed(12)) * 0.05 + 0.01
    target = 3.0 * thermal + 0.05  # all of it lands in the darkest bin
    grid, node_weights = fit_affine_grid(thermal, target, compute_guidance(thermal), None, SIZE)
    starved = node_weights[0] < 1e-6
    assert starved.any()
    assert grid[0, 0][starved].mean() > 2.0, "starved nodes fell back toward gain 0"


def test_a_fully_masked_image_gives_a_finite_grid_and_zero_node_weights():
    thermal = _thermal(8)
    grid, node_weights = fit_affine_grid(
        thermal, thermal, compute_guidance(thermal), torch.zeros_like(thermal), SIZE, refine_steps=1
    )
    assert torch.isfinite(grid).all()
    assert node_weights.abs().max() == 0


def test_fit_rejects_mismatched_shapes():
    x = torch.rand(1, 1, 32, 32)
    g = compute_guidance(x)
    with pytest.raises(ValueError, match="y"):
        fit_affine_grid(x, torch.rand(1, 1, 16, 16), g, None, SIZE)
    with pytest.raises(ValueError, match="x"):
        fit_affine_grid(torch.rand(1, 3, 32, 32), x, g, None, SIZE)
    with pytest.raises(ValueError, match="guidance"):
        fit_affine_grid(x, x, torch.rand(1, 3, 32, 32), None, SIZE)
    with pytest.raises(ValueError, match="weights"):
        fit_affine_grid(x, x, g, torch.ones(1, 1, 8, 8), SIZE)


def test_apply_gain_offset_clamps_only_when_asked():
    x = torch.full((1, 1, 4, 4), 0.5)
    coeffs = torch.cat([torch.full((1, 1, 4, 4), 5.0), torch.full((1, 1, 4, 4), 0.5)], dim=1)
    assert apply_gain_offset(coeffs, x).max() == 1.0
    assert torch.allclose(apply_gain_offset(coeffs, x, clamp=False), torch.full_like(x, 3.0))
    with pytest.raises(ValueError, match="coeffs"):
        apply_gain_offset(torch.zeros(1, 3, 4, 4), x)
