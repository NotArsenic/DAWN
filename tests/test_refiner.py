"""Tests for the LocalRefiner model architecture, bounded residuals, and receptive field."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import lab_to_linear  # noqa: E402
from dawn.data.resample import area_downsample  # noqa: E402
from dawn.flow.render import RenderSettings, render  # noqa: E402
from dawn.models.refiner import LocalRefiner, apply_refiner  # noqa: E402
from tests.flow_helpers import TINY_INPUT, TINY_RENDER, tiny_layout, wake_up  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
B, H, W = 2, 12, 16


def _trained(std: float = 0.2, seed: int = 0, **kwargs) -> LocalRefiner:
    """A refiner whose output head is no longer zero: "one optimizer step happened"."""
    return wake_up(LocalRefiner(**kwargs), std=std, seed=seed)


def _inputs(batch=B, size=(H, W), seed=0, scale=1.0):
    """Generate synthetic lightness and thermal tensor inputs."""
    generator = torch.Generator().manual_seed(seed)
    lightness = torch.rand(batch, 1, *size, generator=generator)
    thermal = torch.rand(batch, 1, *size, generator=generator)
    return lightness * scale, thermal * scale


def _lab(batch=B, size=(H, W), seed=5):
    """Generate synthetic CIELAB tensor with values in standard gamut."""
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(batch, 3, *size, generator=generator)
    return torch.cat([100 * values[:, :1], 60 * values[:, 1:] - 30], dim=1)


# --- the model -----------------------------------------------------------------------------


def test_the_refined_lightness_has_the_shape_and_dtype_of_the_input():
    lightness, thermal = _inputs()
    model = _trained()
    for out in (model(lightness, thermal), model.residual(lightness, thermal)):
        assert out.shape == (B, 1, H, W) and out.dtype == torch.float32


def test_the_residual_is_bounded_by_the_residual_scale_whatever_the_input():
    """Verify residual outputs remain strictly bounded by residual_scale parameter."""
    model = _trained(std=3.0, seed=1)
    lightness, thermal = _inputs(scale=100.0, seed=2)
    residual = model.residual(lightness, thermal)
    assert residual.abs().max() <= model.residual_scale
    assert residual.abs().max() > 0.99 * model.residual_scale, "tanh never saturated here"


def test_the_refined_lightness_stays_within_the_residual_scale_of_its_input():
    model = _trained(std=3.0, seed=1)
    lightness, thermal = _inputs(seed=3)  # L in its [0, 1] domain, so the clamp cannot bite
    delta = (model(lightness, thermal) - lightness).abs()
    assert delta.max() <= model.residual_scale + 1e-6  # float32 round-off of (L + d) - L
    assert delta.max() > 0.99 * model.residual_scale, "tanh never saturated here"


def test_a_relaxed_bound_is_the_only_thing_that_widens_the_residual():
    model = _trained(std=3.0, seed=1, residual_scale=0.4)
    residual = model.residual(*_inputs(scale=100.0, seed=2))
    assert 0.39 < residual.abs().max() <= 0.4


@pytest.mark.parametrize("level", [0.0, 1.0])
def test_the_output_is_clamped_to_the_unit_range(level):
    """Verify refined lightness outputs are clamped to unit interval [0, 1]."""
    model = _trained(std=3.0, seed=4)
    lightness = torch.full((B, 1, H, W), level)
    _, thermal = _inputs(seed=5)
    out = model(lightness, thermal)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.equal(out, lightness), "nothing was clamped, so this proves nothing"


def test_forward_is_the_clamped_sum_of_the_lightness_and_the_residual():
    """Verify forward method matches clamped sum of lightness and predicted residual."""
    model = _trained(std=2.0, seed=6)
    lightness, thermal = _inputs(seed=7)
    expected = (lightness + model.residual(lightness, thermal)).clamp(0.0, 1.0)
    assert torch.equal(model(lightness, thermal), expected)


def test_an_untrained_refiner_is_the_exact_identity():
    """Verify zero-initialized refiner acts as an exact identity operation."""
    lightness, thermal = _inputs()
    model = LocalRefiner()
    assert torch.equal(model.residual(lightness, thermal), torch.zeros_like(lightness))
    assert torch.equal(model(lightness, thermal), lightness)


def test_the_residual_takes_gradient_in_both_inputs_and_in_the_weights():
    model = _trained(std=1.0, seed=8)
    lightness, thermal = (t.requires_grad_() for t in _inputs(seed=9))
    model.residual(lightness, thermal).square().mean().backward()
    for name, tensor in (("lightness", lightness), ("thermal", thermal)):
        assert tensor.grad is not None and tensor.grad.abs().sum() > 0, name
    assert all(p.grad is not None for p in model.parameters())


# --- the receptive field --------------------------------------------------------------------


def test_the_receptive_field_is_nine_pixels_at_the_plans_defaults():
    assert LocalRefiner().receptive_field == 9  # 1 + 2 * (5 - 1): the plan's "9x9 field"
    for blocks, kernel in ((1, 5), (3, 5), (2, 3), (4, 7)):
        model = LocalRefiner(num_blocks=blocks, kernel_size=kernel)
        assert model.receptive_field == 1 + blocks * (kernel - 1)


@pytest.mark.parametrize("field", ["lightness", "thermal"])
def test_one_changed_input_pixel_moves_exactly_the_nine_by_nine_window_around_it(field):
    """Verify perturbations strictly propagate within the receptive field window."""
    model = _trained(std=0.2, seed=10)
    size = 25
    inputs = {
        "lightness": torch.full((1, 1, size, size), 0.5),  # mid-range: nothing is clamped
        "thermal": _inputs(1, (size, size), seed=11)[1],
    }
    base = model(**inputs)
    poked = dict(inputs)
    poked[field] = inputs[field].clone()
    poked[field][0, 0, 12, 12] += 0.3
    with torch.no_grad():
        diff = (model(**poked) - base)[0, 0]

    window = torch.zeros(size, size, dtype=torch.bool)
    radius = model.receptive_field // 2
    window[12 - radius : 12 + radius + 1, 12 - radius : 12 + radius + 1] = True
    assert torch.equal(diff != 0, window)
    neighbours = diff.abs().clone()
    neighbours[12, 12] = 0.0  # the poked pixel also moves through the identity term
    assert neighbours.max() > 1e-6, "the net is saturated, so this window proves nothing"


# --- the config -------------------------------------------------------------------------------


def test_from_config_reads_the_whole_config_or_the_bare_section():
    section = {"hidden_channels": 16, "num_blocks": 3, "kernel_size": 7, "residual_scale": 0.25}
    whole = {"refiner": section, "optim": {"lr": 1e-3}}
    for cfg in (whole, section):
        model = LocalRefiner.from_config(cfg)
        assert (model.hidden_channels, model.num_blocks) == (16, 3)
        assert (model.kernel_size, model.residual_scale) == (7, 0.25)
        assert model.receptive_field == 19


def test_from_config_falls_back_to_the_init_defaults_key_by_key():
    default = LocalRefiner()
    for cfg in ({}, {"refiner": {}}):
        model = LocalRefiner.from_config(cfg)
        assert model.num_params() == default.num_params()
        assert model.receptive_field == default.receptive_field == 9
        assert model.residual_scale == default.residual_scale == 0.1
    partial = LocalRefiner.from_config({"refiner": {"residual_scale": 0.3}})
    assert partial.residual_scale == 0.3 and partial.hidden_channels == 24


def test_from_config_builds_the_shipped_refiner_yaml():
    """Verify refiner instantiation from refiner.yaml configuration."""
    omegaconf = pytest.importorskip("omegaconf")
    path = CONFIGS / "refiner.yaml"
    if not path.exists():
        pytest.skip("configs/refiner.yaml does not exist yet")
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path), resolve=True)
    model = LocalRefiner.from_config(cfg)
    assert model.receptive_field >= 3 and 0 < model.residual_scale <= 1
    assert model(*_inputs()).shape == (B, 1, H, W)


def test_the_parameter_count_stays_in_the_low_thousands():
    """Verify model parameter count stays within lightweight bound."""
    model = LocalRefiner()
    assert 2000 < model.num_params() < 5000, model.num_params()


# --- apply_refiner ------------------------------------------------------------------------------


def test_no_refiner_is_an_exact_identity_pass():
    lab, thermal = _lab(), _inputs()[1]
    out = apply_refiner(lab, thermal, None)
    assert set(out) == {"lab", "rgb", "residual"}
    assert torch.equal(out["lab"], lab)
    assert torch.equal(out["residual"], torch.zeros(B, 1, H, W))
    assert torch.equal(out["rgb"], lab_to_linear(lab).clamp(0, 1))


def test_apply_refiner_moves_the_lightness_and_leaves_a_and_b_bit_identical():
    lab, thermal = _lab(), _inputs()[1]
    out = apply_refiner(lab, thermal, _trained(std=2.0, seed=12))
    assert torch.equal(out["lab"][:, 1:], lab[:, 1:])
    assert not torch.allclose(out["lab"][:, :1], lab[:, :1])
    expected = 100 * (lab[:, :1] / 100 + out["residual"]).clamp(0, 1)
    assert torch.allclose(out["lab"][:, :1], expected, atol=1e-5)
    assert out["residual"].abs().max() <= 0.1  # still L = L*/100 units, so at most 10 L*
    assert torch.equal(out["rgb"], lab_to_linear(out["lab"]).clamp(0, 1))


def test_apply_refiner_area_downsamples_a_larger_thermal():
    """Verify apply_refiner downsamples higher-resolution thermal conditioning."""
    lab = _lab()
    big = _inputs(size=(4 * H, 4 * W), seed=13)[1]
    model = _trained(std=1.0, seed=14)
    out = apply_refiner(lab, big, model)
    direct = apply_refiner(lab, area_downsample(big, (H, W)), model)
    assert out["lab"].shape == (B, 3, H, W) and out["residual"].shape == (B, 1, H, W)
    assert torch.equal(out["lab"], direct["lab"])
    assert torch.equal(out["residual"], direct["residual"])


def test_apply_refiner_runs_on_what_the_one_render_path_returned():
    """Verify apply_refiner integrates with pipeline render output."""
    settings = RenderSettings(layout=tiny_layout())
    layout = settings.layout
    generator = torch.Generator().manual_seed(15)
    parts = {
        "grid": torch.rand(1, 2, layout.num_bins, *layout.grid, generator=generator),
        "ab": torch.randn(1, 2, *layout.color_size, generator=generator) * 10,
    }
    thermal = torch.rand(1, 1, *TINY_INPUT, generator=generator)
    rendered = render(parts, thermal, TINY_RENDER, settings)
    out = apply_refiner(rendered["lab"], thermal, _trained(std=1.0, seed=16))
    assert out["lab"].shape == rendered["lab"].shape == (1, 3, *TINY_RENDER)
    assert torch.equal(out["lab"][:, 1:], rendered["lab"][:, 1:])
    assert out["residual"].abs().max() > 0 and out["rgb"].min() >= 0 and out["rgb"].max() <= 1
