"""The one render path (docs/conventions.md section 6.4)."""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import lab_to_linear  # noqa: E402
from dawn.data.resample import area_downsample  # noqa: E402
from dawn.flow.render import RenderSettings, render  # noqa: E402
from tests.flow_helpers import TINY_INPUT, TINY_RENDER, load_configs, tiny_layout  # noqa: E402


def _settings(correction=False):
    return RenderSettings(layout=tiny_layout(correction), bgu_grid=(4, 2, 2), bgu_ridge=1e-3)


def _constant_parts(layout, gain, offset, a=0.0, b=0.0, corr=None, batch=1):
    gy, gx = layout.grid
    grid = torch.empty(batch, 2, layout.num_bins, gy, gx)
    grid[:, 0], grid[:, 1] = gain, offset
    h, w = layout.color_size
    parts = {
        "grid": grid,
        "ab": torch.cat([torch.full((batch, 1, h, w), a), torch.full((batch, 1, h, w), b)], dim=1),
    }
    if layout.correction:
        parts["corr"] = torch.full((batch, 1, h, w), 0.0 if corr is None else corr)
    return parts


def _thermal(batch=1, size=TINY_INPUT, seed=0):
    return torch.rand(batch, 1, *size, generator=torch.Generator().manual_seed(seed)) * 0.8 + 0.1


def test_outputs_have_the_documented_shapes_and_dtype():
    settings = _settings()
    out = render(_constant_parts(settings.layout, 0.5, 0.2), _thermal(), TINY_INPUT, settings)
    assert set(out) == {"lab", "rgb", "L_grid"}
    assert out["lab"].shape == out["rgb"].shape == (1, 3, *TINY_INPUT)
    assert out["L_grid"].shape == (1, 1, *TINY_INPUT)
    assert all(t.dtype == torch.float32 for t in out.values())


def test_brightness_is_gain_times_thermal_plus_offset_as_l_over_100():
    settings, thermal = _settings(), _thermal()
    out = render(_constant_parts(settings.layout, 0.4, 0.1), thermal, TINY_INPUT, settings)
    assert torch.allclose(out["L_grid"], 0.4 * thermal + 0.1, atol=1e-5)
    assert torch.allclose(out["lab"][:, :1], 100 * out["L_grid"], atol=1e-4)  # no correction


def test_lab_is_converted_to_linear_rgb_and_zero_chroma_is_grey():
    settings, thermal = _settings(), _thermal()
    out = render(_constant_parts(settings.layout, 0.4, 0.1), thermal, TINY_INPUT, settings)
    assert torch.allclose(out["rgb"], lab_to_linear(out["lab"]).clamp(0, 1), atol=1e-6)
    assert torch.allclose(out["rgb"][:, 0], out["rgb"][:, 1], atol=1e-5)
    assert torch.allclose(out["rgb"][:, 1], out["rgb"][:, 2], atol=1e-5)


def test_brightness_is_clamped_and_the_rgb_is_clipped_to_the_gamut():
    settings, thermal = _settings(), _thermal()
    parts = _constant_parts(settings.layout, 5.0, 0.5, a=120.0, b=-120.0)
    out = render(parts, thermal, TINY_INPUT, settings)
    assert out["L_grid"].min() >= 0 and out["L_grid"].max() == 1.0
    assert out["rgb"].min() >= 0 and out["rgb"].max() <= 1


def test_a_constant_colour_map_renders_that_colour_everywhere():
    settings = _settings()
    out = render(
        _constant_parts(settings.layout, 0.3, 0.2, a=15.0, b=-8.0), _thermal(), TINY_INPUT, settings
    )
    assert torch.allclose(out["lab"][:, 1], torch.full_like(out["lab"][:, 1], 15.0), atol=1e-3)
    assert torch.allclose(out["lab"][:, 2], torch.full_like(out["lab"][:, 2], -8.0), atol=1e-3)


def test_colour_edges_follow_the_output_lightness_edge():
    """The colour map is guided by the OUTPUT L: a sharp thermal edge gives a sharp colour edge."""
    settings = _settings()
    layout = settings.layout
    thermal = torch.full((1, 1, *TINY_INPUT), 0.2)
    thermal[..., 33:] = 0.8  # a vertical edge inside a colour pixel (33 / 4 = 8.25)
    h, w = layout.color_size
    parts = _constant_parts(layout, 1.0, 0.0)
    share = area_downsample((thermal > 0.5).float(), (h, w))  # how much of each pixel is bright
    parts["ab"] = torch.cat([share * 40 - 20, torch.zeros_like(share)], dim=1)  # a*: -20 | +20
    out = render(parts, thermal, TINY_INPUT, settings)
    a_star = out["lab"][0, 1, 10]  # one row
    assert float(a_star[20]) == pytest.approx(-20.0, abs=3) and float(a_star[60]) == pytest.approx(
        20.0, abs=3
    )
    assert float(a_star[33] - a_star[32]) > 20  # the step is across ONE pixel (bilinear: ~10)


def test_the_correction_is_off_by_default_and_adds_in_l_when_on():
    off, on = _settings(False), _settings(True)
    thermal = _thermal()
    base = render(_constant_parts(off.layout, 0.4, 0.1), thermal, TINY_INPUT, off)
    corrected = render(_constant_parts(on.layout, 0.4, 0.1, corr=0.15), thermal, TINY_INPUT, on)
    assert torch.allclose(corrected["L_grid"], base["L_grid"])  # the grid is untouched
    assert torch.allclose(
        corrected["lab"][:, :1], 100 * (base["L_grid"] + 0.15).clamp(0, 1), atol=1e-3
    )
    zero = render(_constant_parts(on.layout, 0.4, 0.1, corr=0.0), thermal, TINY_INPUT, on)
    assert torch.allclose(zero["lab"], base["lab"], atol=1e-3)


def test_parts_that_do_not_match_the_layout_are_rejected():
    off = _settings(False)
    with_corr = _constant_parts(tiny_layout(True), 0.4, 0.1, corr=0.1)
    with pytest.raises(ValueError, match="corr"):
        render(with_corr, _thermal(), TINY_INPUT, off)
    on = _settings(True)
    with pytest.raises(ValueError, match="corr"):
        render(_constant_parts(off.layout, 0.4, 0.1), _thermal(), TINY_INPUT, on)


def test_a_smaller_size_downsamples_the_thermal_first_and_a_larger_one_is_an_error():
    settings, thermal = _settings(), _thermal(batch=2)
    parts = _constant_parts(settings.layout, 0.4, 0.1, batch=2)
    small = render(parts, thermal, TINY_RENDER, settings)
    direct = render(parts, area_downsample(thermal, TINY_RENDER), TINY_RENDER, settings)
    assert small["rgb"].shape == (2, 3, *TINY_RENDER)
    assert torch.equal(small["rgb"], direct["rgb"])
    with pytest.raises(ValueError, match="size"):
        render(parts, thermal, (128, 160), settings)
    with pytest.raises(ValueError, match="colour|color"):
        render(parts, thermal, (8, 8), settings)  # smaller than the colour map


def test_render_is_differentiable_in_the_grid_and_the_colour_map():
    settings = _settings(True)
    parts = {
        k: v.clone().requires_grad_()
        for k, v in _constant_parts(settings.layout, 0.4, 0.1, a=5.0, corr=0.05).items()
    }
    out = render(parts, _thermal(), TINY_RENDER, settings)
    (out["lab"].mean() + out["rgb"].mean()).backward()
    for name, value in parts.items():
        assert value.grad is not None and torch.isfinite(value.grad).all(), name
        assert value.grad.abs().sum() > 0, name


def test_settings_come_from_the_config():
    cfg = load_configs()
    settings = RenderSettings.from_config(cfg)
    assert settings.bgu_grid == (cfg["color"]["bgu"]["bins"], *cfg["color"]["bgu"]["grid"])
    assert settings.layout.dim == 48 and settings.layout.correction is False
