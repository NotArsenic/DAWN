"""Tests for sRGB and linear RGB colour conversions, luminance, and CIELAB transformations."""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import (  # noqa: E402
    lab_to_linear,
    linear_to_lab,
    linear_to_srgb,
    linear_to_srgb8,
    luminance,
    srgb8_to_linear,
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


# --- CIELAB (linear sRGB primaries, D65 white) -------------------------------------------------

LAB_RED = (53.2408, 80.0925, 67.2032)  # CIE reference value of linear sRGB (1, 0, 0), D65


def _flat(rgb):
    return torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1)


def test_lab_of_white_is_100_0_0():
    lab = linear_to_lab(torch.ones(3, 2, 2))
    assert torch.allclose(lab, _flat([100.0, 0.0, 0.0]).expand(3, 2, 2), atol=1e-3)


def test_lab_of_black_is_zero():
    assert torch.allclose(linear_to_lab(torch.zeros(3, 2, 2)), torch.zeros(3, 2, 2), atol=1e-5)


def test_lab_of_linear_red_matches_the_reference_value():
    lab = linear_to_lab(_flat([1.0, 0.0, 0.0]))
    assert lab.flatten().tolist() == pytest.approx(LAB_RED, abs=0.05)


def test_lab_of_18_percent_grey_is_about_l_49_5_with_no_chroma():
    lab = linear_to_lab(torch.full((3, 1, 1), 0.18))
    assert lab[0].item() == pytest.approx(49.5, abs=0.1)
    assert lab[1:].abs().max().item() < 1e-3


def test_lab_uses_the_cie_linear_segment_below_the_threshold():
    # Y <= (6/29)^3: L* = (29/3)^3 * Y = 903.2963 * Y, not the cube root (which would give 10.0).
    lab = linear_to_lab(torch.full((3, 1, 1), 0.001))
    assert lab[0].item() == pytest.approx(903.2963 * 0.001, rel=1e-3)


def test_lab_cube_root_and_linear_segments_meet():
    y = (6 / 29) ** 3
    below = linear_to_lab(torch.full((3, 1, 1), y * (1 - 1e-4)))[0].item()
    above = linear_to_lab(torch.full((3, 1, 1), y * (1 + 1e-4)))[0].item()
    assert abs(above - below) < 1e-2


def test_lab_round_trip_over_the_unit_cube():
    rgb = torch.rand(2, 3, 16, 16)
    assert torch.allclose(lab_to_linear(linear_to_lab(rgb)), rgb, atol=1e-4)


def test_lab_round_trip_through_the_linear_segment():
    rgb = torch.rand(3, 16, 16) * 0.01  # almost black: the segment below (6/29)^3
    assert torch.allclose(lab_to_linear(linear_to_lab(rgb)), rgb, atol=1e-5)


def test_l_star_stays_within_0_100_on_the_unit_cube():
    lab = linear_to_lab(torch.rand(4, 3, 32, 32))
    assert lab[:, 0].min() >= 0.0
    assert lab[:, 0].max() <= 100.0 + 1e-3


def test_lab_gradients_are_finite_including_at_black_and_white():
    rgb = torch.cat([torch.zeros(3, 1, 1), torch.ones(3, 1, 1), torch.rand(3, 1, 8)], dim=-1)
    rgb.requires_grad_()
    linear_to_lab(rgb).sum().backward()
    assert torch.isfinite(rgb.grad).all()

    lab = torch.cat([torch.zeros(3, 1, 1), _flat([100.0, 0.0, 0.0])], dim=-1).requires_grad_()
    lab_to_linear(lab).sum().backward()
    assert torch.isfinite(lab.grad).all()


@pytest.mark.parametrize("shape", [(3, 8, 8), (2, 3, 8, 8)])
@pytest.mark.parametrize("fn", [linear_to_lab, lab_to_linear])
def test_lab_conversions_keep_shape_and_float32(fn, shape):
    out = fn(torch.rand(shape))
    assert out.shape == shape
    assert out.dtype == torch.float32


def test_lab_conversions_work_per_image_in_a_batch():
    batch = torch.rand(3, 3, 6, 6)
    out = linear_to_lab(batch)
    for i in range(3):
        assert torch.allclose(out[i], linear_to_lab(batch[i]), atol=1e-4)


@pytest.mark.parametrize("fn", [linear_to_lab, lab_to_linear])
def test_lab_conversions_reject_non_three_channel_input(fn):
    with pytest.raises(ValueError, match="3"):
        fn(torch.rand(1, 4, 4))


# --- 8-bit sRGB (file boundary) ----------------------------------------------------------------


def test_srgb8_to_linear_is_exactly_srgb_to_linear_of_code_over_255():
    codes = torch.arange(256, dtype=torch.uint8)
    out = srgb8_to_linear(codes)
    assert out.dtype == torch.float32
    assert torch.equal(out, srgb_to_linear(codes.float() / 255))


def test_srgb8_to_linear_keeps_the_image_shape():
    img = torch.randint(0, 256, (2, 3, 5, 7), dtype=torch.uint8)
    assert srgb8_to_linear(img).shape == (2, 3, 5, 7)


def test_every_8bit_code_survives_a_round_trip_through_linear():
    codes = torch.arange(256, dtype=torch.uint8).view(1, 16, 16).expand(3, 16, 16)
    assert torch.equal(linear_to_srgb8(srgb8_to_linear(codes)), codes)


def test_linear_to_srgb8_clamps_and_returns_uint8():
    out = linear_to_srgb8(torch.tensor([-0.5, 0.0, 1.0, 2.0]))
    assert out.dtype == torch.uint8
    assert out.tolist() == [0, 0, 255, 255]


def test_srgb8_to_linear_rejects_float_input():
    # A float image here almost always means it was already divided by 255 (or linearized).
    with pytest.raises(TypeError, match="uint8"):
        srgb8_to_linear(torch.rand(3, 4, 4))
