"""Grid slicing and fitting against cases with a known answer.

`test_grid_rules.py` pins the conventions (coordinates, one trilinear implementation). This
file asks the harder question: does the fit actually recover a transform, measured where it
matters -- on the rendered image, not on the coefficients. Two different coefficient fields can
render identically wherever the data does not constrain them, so comparing renders is the only
test that fails for the right reason.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.fit import fit_grid  # noqa: E402
from dawn.grid.guidance import compute_guidance  # noqa: E402
from dawn.grid.layout import IDENTITY_COEFFS  # noqa: E402
from dawn.grid.slice import apply_affine, slice_grid  # noqa: E402

K, GY, GX = 8, 16, 16


def _constant_grid(coeffs, batch=1):
    """A (B, 12, K, gy, gx) grid holding the same 12 coefficients at every node."""
    return torch.tensor(coeffs, dtype=torch.float32).view(1, 12, 1, 1, 1).expand(
        batch, 12, K, GY, GX
    ).contiguous()


def _render(grid, image):
    return apply_affine(slice_grid(grid, compute_guidance(image)), image)


def _psnr(a, b):
    return float(-10 * torch.log10((a - b).pow(2).mean().clamp(min=1e-20)))


def test_identity_grid_returns_the_input():
    image = torch.rand(2, 3, 64, 96)
    out = _render(_constant_grid(IDENTITY_COEFFS, batch=2), image)
    assert torch.allclose(out, image, atol=1e-6)


def test_constant_five_times_grid_returns_five_times_the_input():
    image = torch.rand(2, 3, 64, 96) * 0.15  # stays under 1/5 so nothing clips
    five = [5.0, 0, 0, 0, 5.0, 0, 0, 0, 5.0, 0, 0, 0]
    out = _render(_constant_grid(five, batch=2), image)
    assert torch.allclose(out, image * 5.0, atol=1e-6)


def test_apply_affine_clamps_to_the_unit_interval():
    image = torch.full((1, 3, 8, 8), 0.5)
    five = [5.0, 0, 0, 0, 5.0, 0, 0, 0, 5.0, 0, 0, 0]
    coeffs = slice_grid(_constant_grid(five), compute_guidance(image))
    assert apply_affine(coeffs, image).max() == 1.0


def _smooth_grid(seed=0, cycles=1.0, offset_base=0.12):
    """Identity plus a low-frequency ripple on every coefficient.

    Two things make this a fair test rather than an easy or an impossible one.

    Smooth: a grid that jumps between neighbouring nodes is not something a trilinear fit can
    represent from pixel data at all, so re-rendering it would fail for a reason that has
    nothing to do with the fit being wrong.

    In range: the twelve coefficients live on three different scales, and a ripple big enough
    to push the rendered target outside [0, 1] would make the test unwinnable -- `apply_affine`
    clamps, and no affine map reproduces a clamped target. So the diagonal gains swing widely
    (+-0.25), the cross-channel terms gently (+-0.08), and the offsets sit on a positive lift
    (0.12 +- 0.03) so that dark pixels cannot go negative. `_assert_unclipped` checks it.
    """
    generator = torch.Generator().manual_seed(seed)
    k = torch.linspace(0, 1, K).view(1, K, 1, 1)
    y = torch.linspace(0, 1, GY).view(1, 1, GY, 1)
    x = torch.linspace(0, 1, GX).view(1, 1, 1, GX)
    phase = torch.rand(12, 3, generator=generator) * 6.283
    ripple = (
        torch.sin(cycles * 6.283 * k + phase[:, 0].view(12, 1, 1, 1))
        + torch.sin(cycles * 6.283 * y + phase[:, 1].view(12, 1, 1, 1))
        + torch.sin(cycles * 6.283 * x + phase[:, 2].view(12, 1, 1, 1))
    ) / 3.0
    diagonal = [0, 4, 8]
    scale = torch.full((12,), 0.08)
    scale[diagonal] = 0.25
    scale[9:] = 0.03
    base = torch.tensor(IDENTITY_COEFFS).clone()
    base[9:] = offset_base
    return (base.view(12, 1, 1, 1) + scale.view(12, 1, 1, 1) * ripple).unsqueeze(0)


def _assert_unclipped(grid, image, guidance):
    """The rendered target must sit strictly inside [0, 1], or the fit is chasing a clamp."""
    coeffs = slice_grid(grid, guidance)
    batch, _, height, width = image.shape
    raw = (coeffs[:, :9].reshape(batch, 3, 3, height, width) * image.unsqueeze(1)).sum(
        dim=2
    ) + coeffs[:, 9:12]
    assert raw.min() > 0.0 and raw.max() < 1.0, (
        f"target clips: range [{raw.min():.3f}, {raw.max():.3f}]"
    )


def test_fitting_a_known_smooth_grid_and_re_rendering_beats_45_db():
    generator = torch.Generator().manual_seed(4)
    night_clean = torch.rand(1, 3, 256, 256, generator=generator) * 0.28 + 0.02
    truth = _smooth_grid(seed=1)

    guidance = compute_guidance(night_clean)
    _assert_unclipped(truth, night_clean, guidance)
    day = apply_affine(slice_grid(truth, guidance), night_clean)

    fitted, node_weights = fit_grid(night_clean, day, guidance)
    assert fitted.shape == (1, 12, K, GY, GX)
    assert node_weights.shape == (1, K, GY, GX)

    rendered = apply_affine(slice_grid(fitted, guidance), night_clean)
    psnr = _psnr(rendered, day)
    assert psnr > 45, f"re-render PSNR {psnr:.1f} dB, expected > 45"


def test_the_fit_is_judged_on_renders_because_coefficients_are_underdetermined():
    """A node no pixel lands on is free; only the render is pinned down."""
    generator = torch.Generator().manual_seed(9)
    night_clean = torch.rand(1, 3, 128, 128, generator=generator) * 0.25 + 0.03
    truth = _smooth_grid(seed=2)
    guidance = compute_guidance(night_clean)
    _assert_unclipped(truth, night_clean, guidance)
    day = apply_affine(slice_grid(truth, guidance), night_clean)

    fitted, node_weights = fit_grid(night_clean, day, guidance)
    empty = node_weights[0] < 1e-6
    assert empty.any(), "this test needs at least one unvisited node to be meaningful"
    rendered = apply_affine(slice_grid(fitted, guidance), night_clean)
    assert _psnr(rendered, day) > 45


def test_an_empty_bin_inherits_its_cell_rather_than_the_identity():
    """Ridge pulls a starved node onto its cell's fit: "no data" is not "do not brighten"."""
    generator = torch.Generator().manual_seed(12)
    night_clean = torch.rand(1, 3, 128, 128, generator=generator) * 0.05 + 0.01  # one bin only
    day = (night_clean * 4.0).clamp(0, 1)
    guidance = compute_guidance(night_clean)
    grid, node_weights = fit_grid(night_clean, day, guidance)

    starved = node_weights[0] < 1e-6
    assert starved.any()
    gain = grid[0, 0][starved]  # M[0, 0] at the nodes no pixel reached
    assert gain.mean() > 2.0, (
        f"starved nodes fell back toward identity (mean M00 {gain.mean():.2f})"
    )


def test_fit_grid_rejects_mismatched_shapes():
    night = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError):
        fit_grid(night, torch.rand(1, 3, 16, 16), compute_guidance(night))
    with pytest.raises(ValueError):
        fit_grid(night, night, torch.rand(1, 3, 32, 32))
