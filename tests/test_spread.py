"""Sample-spread uncertainty (docs/conventions.md section 11)."""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow.render import RenderSettings  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.uncertainty.spread import spread_from_parts  # noqa: E402
from tests.flow_helpers import TINY_INPUT, TINY_RENDER, tiny_layout  # noqa: E402


def _settings(correction=False, upsample="bilinear"):
    return RenderSettings(layout=tiny_layout(correction), bgu_grid=(4, 2, 2), upsample=upsample)


def _parts(layout: TokenLayout, samples: int, batch: int = 2, seed: int = 0, spread: float = 0.0):
    """`samples` members that are identical when `spread` is 0 and jitter around that otherwise."""
    generator = torch.Generator().manual_seed(seed)
    gy, gx = layout.grid
    hc, wc = layout.color_size
    base = {
        "grid": torch.rand(1, batch, 2, layout.num_bins, gy, gx, generator=generator),
        "ab": torch.rand(1, batch, 2, hc, wc, generator=generator) * 40 - 20,
    }
    if layout.correction:
        base["corr"] = torch.zeros(1, batch, 1, hc, wc)
    parts = {}
    for name, value in base.items():
        member = value.repeat(samples, *([1] * (value.ndim - 1)))
        if spread:
            member = member + spread * torch.randn(member.shape, generator=generator)
        parts[name] = member
    return parts


def _thermal(batch=2, size=TINY_INPUT, seed=1):
    return torch.rand(batch, 1, *size, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1


def test_shapes_and_dtypes_are_the_documented_ones():
    settings = _settings()
    out = spread_from_parts(_parts(settings.layout, 4), _thermal(), TINY_RENDER, settings)
    assert out.num_samples == 4
    assert out.sample_rgb.shape == (2, 3, *TINY_RENDER)
    assert out.rgb.shape == (2, 3, *TINY_RENDER)
    assert out.lab.shape == (2, 3, *TINY_RENDER)
    assert out.l_std.shape == (2, 1, *TINY_RENDER)
    assert out.ab_std.shape == (2, 1, *TINY_RENDER)
    for tensor in (out.sample_rgb, out.rgb, out.lab, out.l_std, out.ab_std):
        assert tensor.dtype == torch.float32


def test_identical_members_have_zero_spread():
    """K copies of one draw disagree nowhere, so both maps are exactly 0."""
    settings = _settings()
    out = spread_from_parts(
        _parts(settings.layout, 5, spread=0.0), _thermal(), TINY_RENDER, settings
    )
    assert torch.allclose(out.l_std, torch.zeros_like(out.l_std), atol=1e-5)
    assert torch.allclose(out.ab_std, torch.zeros_like(out.ab_std), atol=1e-5)


def test_one_sample_gives_zeros_not_nan():
    """K=1 has nothing to disagree with: zeros, never NaN from an (n-1) divide."""
    settings = _settings()
    out = spread_from_parts(_parts(settings.layout, 1), _thermal(), TINY_RENDER, settings)
    assert out.num_samples == 1
    assert torch.all(torch.isfinite(out.l_std)) and torch.all(out.l_std == 0)
    assert torch.all(torch.isfinite(out.ab_std)) and torch.all(out.ab_std == 0)


def test_more_disagreement_means_more_spread():
    settings = _settings()
    thermal = _thermal()
    small = spread_from_parts(
        _parts(settings.layout, 6, spread=0.02), thermal, TINY_RENDER, settings
    )
    large = spread_from_parts(
        _parts(settings.layout, 6, spread=0.20), thermal, TINY_RENDER, settings
    )
    assert large.l_std.mean() > small.l_std.mean()
    assert large.ab_std.mean() > small.ab_std.mean()


def test_the_maps_match_an_explicit_std_over_rendered_members():
    """Welford must agree with stacking the renders and calling torch.std (unbiased, n-1)."""
    from dawn.flow.render import render

    settings = _settings()
    parts, thermal = _parts(settings.layout, 5, spread=0.1), _thermal()
    labs = torch.stack(
        [
            render({k: v[s] for k, v in parts.items()}, thermal, TINY_RENDER, settings)["lab"]
            for s in range(5)
        ]
    )
    out = spread_from_parts(parts, thermal, TINY_RENDER, settings)
    variance = labs.var(dim=0, unbiased=True)
    assert torch.allclose(out.lab, labs.mean(dim=0), atol=1e-5)
    assert torch.allclose(out.l_std, variance[:, 0:1].sqrt(), atol=1e-5)
    assert torch.allclose(out.ab_std, (variance[:, 1:2] + variance[:, 2:3]).sqrt(), atol=1e-5)


@pytest.mark.parametrize("delta_a, delta_b", [(10.0, 0.0), (6.0, 8.0), (0.0, -10.0)])
def test_ab_std_is_a_distance_in_the_ab_plane(delta_a, delta_b):
    """Two members a known a*b* offset apart: only the offset's LENGTH may show up.

    For K=2 the unbiased variance of two values d apart is d^2 / 2, so
    sqrt(var_a + var_b) = |offset| / sqrt(2). Three offsets of length 10 pointing different ways
    must all give the same number: that is what makes the map a chroma distance rather than a
    per-channel quantity, and it is what lets it be ranked against CIEDE2000.
    """
    settings = _settings()
    layout, thermal = settings.layout, _thermal(batch=1)
    gy, gx = layout.grid
    hc, wc = layout.color_size
    grid = torch.full((2, 1, 2, layout.num_bins, gy, gx), 0.5)
    ab = torch.zeros(2, 1, 2, hc, wc)
    ab[1, :, 0], ab[1, :, 1] = delta_a, delta_b
    out = spread_from_parts({"grid": grid, "ab": ab}, thermal, TINY_RENDER, settings)
    expected = 10.0 / (2**0.5)
    assert torch.allclose(out.ab_std, torch.full_like(out.ab_std, expected), atol=1e-3)
    assert torch.allclose(out.l_std, torch.zeros_like(out.l_std), atol=1e-5)


def test_sample_rgb_is_member_zero_and_differs_from_the_mean():
    from dawn.flow.render import render

    settings = _settings()
    parts, thermal = _parts(settings.layout, 4, spread=0.2), _thermal()
    out = spread_from_parts(parts, thermal, TINY_RENDER, settings)
    member_0 = render({k: v[0] for k, v in parts.items()}, thermal, TINY_RENDER, settings)
    assert torch.allclose(out.sample_rgb, member_0["rgb"])
    assert not torch.allclose(out.sample_rgb, out.rgb)


def test_rgb_is_the_mean_lab_brought_back_and_stays_in_gamut():
    from dawn.data.color import lab_to_linear

    settings = _settings()
    out = spread_from_parts(
        _parts(settings.layout, 4, spread=0.1), _thermal(), TINY_RENDER, settings
    )
    assert torch.allclose(out.rgb, lab_to_linear(out.lab).clamp(0.0, 1.0), atol=1e-6)
    assert out.rgb.min() >= 0.0 and out.rgb.max() <= 1.0


def test_the_correction_layout_is_carried_through():
    settings = _settings(correction=True)
    assert settings.layout.correction
    out = spread_from_parts(_parts(settings.layout, 3), _thermal(), TINY_RENDER, settings)
    assert out.l_std.shape == (2, 1, *TINY_RENDER)


def test_sample_with_spread_runs_the_whole_path_off_one_encoder_pass():
    """encode once -> K members -> maps, and the encoder is not touched again."""
    from dawn.models.velocity import VelocityTransformer
    from dawn.uncertainty.spread import sample_with_spread
    from tests.flow_helpers import wake_up

    settings = _settings()
    layout = settings.layout
    model = wake_up(VelocityTransformer(layout)).eval()
    thermal = _thermal(batch=2)
    encoded = {
        "patch_tokens": torch.randn(2, layout.num_tokens, 384),
        "thumb": torch.rand(2, 1, *layout.grid),
    }
    calls = []
    forward = model.forward
    model.forward = lambda *a, **k: (calls.append(1), forward(*a, **k))[1]

    out = sample_with_spread(
        model,
        encoded,
        layout,
        torch.zeros(layout.dim),
        torch.ones(layout.dim),
        thermal,
        TINY_RENDER,
        settings,
        num_samples=4,
        nfe=2,
        image_ids=["a", "b"],
    )
    assert out.num_samples == 4
    assert out.rgb.shape == (2, 3, *TINY_RENDER)
    assert len(calls) == 2, "2 NFE, with the K members batched into one velocity pass each"
    assert out.l_std.min() >= 0 and out.ab_std.min() >= 0


def test_sample_with_spread_is_reproducible_from_the_image_ids():
    from dawn.models.velocity import VelocityTransformer
    from dawn.uncertainty.spread import sample_with_spread
    from tests.flow_helpers import wake_up

    settings = _settings()
    layout = settings.layout
    model = wake_up(VelocityTransformer(layout)).eval()
    thermal = _thermal(batch=2)
    encoded = {
        "patch_tokens": torch.randn(2, layout.num_tokens, 384),
        "thumb": torch.rand(2, 1, *layout.grid),
    }
    kwargs = dict(num_samples=3, nfe=2, image_ids=["a", "b"])
    args = (
        model,
        encoded,
        layout,
        torch.zeros(layout.dim),
        torch.ones(layout.dim),
        thermal,
        TINY_RENDER,
        settings,
    )
    first = sample_with_spread(*args, **kwargs)
    second = sample_with_spread(*args, **kwargs)
    assert torch.allclose(first.l_std, second.l_std)
    assert torch.allclose(first.ab_std, second.ab_std)
    assert torch.allclose(first.sample_rgb, second.sample_rgb)


@pytest.mark.parametrize(
    "parts, message",
    [
        ({}, "empty"),
        ("mismatch", "disagree on the sample count"),
    ],
)
def test_malformed_parts_are_clear_errors(parts, message):
    settings = _settings()
    if parts == "mismatch":
        parts = _parts(settings.layout, 3)
        parts["ab"] = parts["ab"][:2]
    with pytest.raises(ValueError, match=message):
        spread_from_parts(parts, _thermal(), TINY_RENDER, settings)
