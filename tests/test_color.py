"""
dawn/data/color.py: sRGB <-> linear (docs/conventions.md section 1), luminance, color temp.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import (
    COLOR_TEMP_MAX,
    COLOR_TEMP_MIN,
    REFERENCE_COLOR_TEMP,
    color_temp_to_gains,
    color_temp_to_illuminant_rgb,
    linear_to_srgb,
    luminance,
    srgb_gamma_slope,
    srgb_to_linear,
)

SRGB_BREAKPOINT = 0.04045  # in sRGB units
LINEAR_BREAKPOINT = 0.0031308  # in linear units


def test_srgb_roundtrip():
    x = torch.rand(2, 3, 16, 16)
    assert torch.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


@pytest.mark.parametrize(
    "first, second", [(srgb_to_linear, linear_to_srgb), (linear_to_srgb, srgb_to_linear)]
)
def test_roundtrip_error_is_below_1e5_over_the_whole_unit_interval(first, second):
    x = torch.linspace(0.0, 1.0, 10001)
    assert (second(first(x)) - x).abs().max() < 1e-5


def test_srgb_half_is_about_0_2140_linear():
    assert srgb_to_linear(torch.tensor(0.5)).item() == pytest.approx(0.2140, abs=1e-4)
    assert linear_to_srgb(torch.tensor(0.2140)).item() == pytest.approx(0.5, abs=1e-3)


def test_toe_is_the_linear_segment():
    assert srgb_to_linear(torch.tensor(0.02)).item() == pytest.approx(0.02 / 12.92, rel=1e-6)
    assert linear_to_srgb(torch.tensor(0.002)).item() == pytest.approx(0.002 * 12.92, rel=1e-6)


@pytest.mark.parametrize(
    "fn, breakpoint_", [(srgb_to_linear, SRGB_BREAKPOINT), (linear_to_srgb, LINEAR_BREAKPOINT)]
)
def test_no_jump_at_the_breakpoint_from_either_side(fn, breakpoint_):
    eps = 1e-9
    below = fn(torch.tensor(breakpoint_ - eps, dtype=torch.float64))
    at = fn(torch.tensor(breakpoint_, dtype=torch.float64))
    above = fn(torch.tensor(breakpoint_ + eps, dtype=torch.float64))
    assert (at - below).abs() < 1e-6
    assert (above - at).abs() < 1e-6


@pytest.mark.parametrize("fn", [srgb_to_linear, linear_to_srgb])
def test_out_of_range_inputs_are_clamped_by_default(fn):
    x = torch.tensor([-0.1, 1.1])
    assert torch.equal(fn(x), torch.tensor([0.0, 1.0]))


def test_clamp_false_continues_the_curves_and_stays_finite():
    x = torch.tensor([-0.1, 1.1], requires_grad=True)
    lin = srgb_to_linear(x, clamp=False)
    assert lin[0].item() == pytest.approx(-0.1 / 12.92, rel=1e-6)  # the toe segment, extended
    assert lin[1].item() > 1.0
    enc = linear_to_srgb(x, clamp=False)
    assert enc[0].item() == pytest.approx(-0.1 * 12.92, rel=1e-6)
    assert enc[1].item() > 1.0
    (lin.sum() + enc.sum()).backward()
    assert torch.isfinite(lin).all() and torch.isfinite(enc).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize("clamp", [True, False])
@pytest.mark.parametrize("fn", [srgb_to_linear, linear_to_srgb])
def test_no_nan_values_or_gradients_for_inputs_in_minus_0_1_to_1_1(fn, clamp):

    special = torch.tensor([0.0, LINEAR_BREAKPOINT, SRGB_BREAKPOINT, 1.0])
    x = torch.cat([torch.linspace(-0.1, 1.1, 1201), special]).requires_grad_()
    y = fn(x, clamp=clamp)
    y.sum().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()


def test_linear_to_srgb_gradient_is_the_gamma_slope():
    # Guards the NaN fix: clamping the power branch must not zero or bend the real gradient.
    x = torch.tensor([0.001, 0.01, 0.2, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
    linear_to_srgb(x).sum().backward()
    assert torch.allclose(x.grad, srgb_gamma_slope(x.detach()), rtol=1e-6)


@pytest.mark.parametrize("shape", [(3, 8, 8), (2, 3, 8, 8)])
@pytest.mark.parametrize("fn", [srgb_to_linear, linear_to_srgb])
def test_conversions_keep_shape_and_float32(fn, shape):
    y = fn(torch.rand(shape))
    assert y.shape == shape
    assert y.dtype == torch.float32


def test_gamma_slope_matches_finite_difference():
    x = torch.tensor([0.01, 0.05, 0.2, 0.5, 0.9], dtype=torch.float64)
    eps = 1e-6
    numeric = (linear_to_srgb(x + eps) - linear_to_srgb(x - eps)) / (2 * eps)
    assert torch.allclose(srgb_gamma_slope(x), numeric, rtol=1e-4)


#  luminance
@pytest.mark.parametrize("channel, weight", [(0, 0.2126), (1, 0.7152), (2, 0.0722)])
def test_luminance_uses_rec709_weights(channel, weight):
    img = torch.zeros(3, 2, 2)
    img[channel] = 1.0
    assert torch.allclose(luminance(img), torch.full((1, 2, 2), weight))


def test_luminance_of_white_is_one():
    assert torch.allclose(luminance(torch.ones(3, 4, 4)), torch.ones(1, 4, 4), atol=1e-6)


@pytest.mark.parametrize("shape, expected", [((3, 5, 7), (1, 5, 7)), ((2, 3, 5, 7), (2, 1, 5, 7))])
def test_luminance_keeps_the_channel_dimension(shape, expected):
    assert luminance(torch.rand(shape)).shape == expected


def test_batched_luminance_matches_per_image():
    batch = torch.rand(4, 3, 6, 6)
    out = luminance(batch)
    for i in range(4):
        assert torch.allclose(out[i], luminance(batch[i]))


@pytest.mark.parametrize("shape", [(1, 4, 4), (4, 4, 4), (2, 1, 4, 4)])
def test_luminance_rejects_non_rgb_input(shape):
    # A 1-channel input would otherwise broadcast against the 3 weights and silently sum them.
    with pytest.raises(ValueError, match="3"):
        luminance(torch.rand(shape))


#  color temperature
# Linear sRGB (D65) -> XYZ.
SRGB_TO_XYZ = torch.tensor(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=torch.float64,
)

# CIE 1931 chromaticity of a blackbody (Planckian locus); Kim et al. 2002 fits these to ~5e-4.
PLANCKIAN_LOCUS = {
    2000: (0.5267, 0.4133),
    3000: (0.4369, 0.4041),
    4000: (0.3805, 0.3768),
    5000: (0.3451, 0.3516),
    6500: (0.3135, 0.3237),
    10000: (0.2807, 0.2884),
}

VALID_TEMPS = list(range(int(COLOR_TEMP_MIN), int(COLOR_TEMP_MAX) + 1, 50))


@pytest.mark.parametrize("kelvin, xy", PLANCKIAN_LOCUS.items())
def test_illuminant_chromaticity_is_on_the_planckian_locus(kelvin, xy):
    xyz = SRGB_TO_XYZ @ color_temp_to_illuminant_rgb(kelvin).double()
    x, y = (xyz[:2] / xyz.sum()).tolist()
    assert x == pytest.approx(xy[0], abs=1e-3)
    assert y == pytest.approx(xy[1], abs=1e-3)


@pytest.mark.parametrize("kelvin", [2000, 3000, 4500, 6500, 10000])
def test_illuminant_has_unit_luminance(kelvin):
    rgb = color_temp_to_illuminant_rgb(kelvin)
    assert luminance(rgb.view(3, 1, 1)).item() == pytest.approx(1.0, abs=1e-3)


def test_illuminant_at_2000k_is_red_with_barely_any_blue():
    r, g, b = color_temp_to_illuminant_rgb(2000).tolist()
    assert r > g > b > 0
    assert b < 0.05


def test_illuminant_is_float32_with_three_channels():
    rgb = color_temp_to_illuminant_rgb(4000)
    assert rgb.shape == (3,)
    assert rgb.dtype == torch.float32


def test_illuminant_channels_are_floored_at_1e_4_across_the_valid_range():
    # Below 1,904 K the blue primary goes negative (out of sRGB gamut) without the floor.
    rgbs = torch.stack([color_temp_to_illuminant_rgb(k) for k in VALID_TEMPS])
    assert torch.isfinite(rgbs).all()
    assert (rgbs >= 1e-4).all()


@pytest.mark.parametrize("fn", [color_temp_to_illuminant_rgb, color_temp_to_gains])
@pytest.mark.parametrize("kelvin", [1600.0, 1699.0, 12001.0, 20000.0, 0.0, -5.0, float("nan")])
def test_color_temp_outside_the_valid_range_is_rejected(fn, kelvin):
    with pytest.raises(ValueError, match="color_temp"):
        fn(kelvin)


@pytest.mark.parametrize("fn", [color_temp_to_illuminant_rgb, color_temp_to_gains])
@pytest.mark.parametrize("kelvin", [COLOR_TEMP_MIN, COLOR_TEMP_MAX])
def test_valid_range_endpoints_are_accepted(fn, kelvin):
    out = fn(kelvin)
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_gains_at_the_reference_temperature_are_exactly_one():
    assert torch.equal(color_temp_to_gains(REFERENCE_COLOR_TEMP), torch.ones(3))


@pytest.mark.parametrize("kelvin", VALID_TEMPS[::10])
def test_gains_keep_green_at_one_so_brightness_is_untouched(kelvin):
    assert color_temp_to_gains(kelvin)[1].item() == 1.0


def test_gains_at_2000k_are_several_times_red_and_a_few_percent_blue():
    r, g, b = color_temp_to_gains(2000).tolist()
    assert r > g > b
    assert r >= 3.0 * g  # "several times green"
    assert 0.0 < b <= 0.05 * g  # "a few percent of green"


def test_gains_above_the_reference_shift_toward_blue():
    r, g, b = color_temp_to_gains(10000).tolist()
    assert r < g < b


def test_red_to_blue_ratio_increases_steadily_as_temperature_drops():
    ratios = []
    for kelvin in reversed(VALID_TEMPS):  # 12,000 K down to 1,700 K
        r, _, b = color_temp_to_gains(kelvin).tolist()
        ratios.append(r / b)
    assert all(later > earlier for earlier, later in zip(ratios, ratios[1:]))


def test_gains_are_float32_with_three_channels():
    gains = color_temp_to_gains(3000)
    assert gains.shape == (3,)
    assert gains.dtype == torch.float32
