"""Tests for thermal guidance computation and spatial filtering."""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.guidance import compute_guidance  # noqa: E402


@pytest.mark.parametrize("value", [0.0, 0.05, 0.5, 1.0])
def test_constant_thermal_gives_constant_guidance_including_borders(value):
    # Zero padding in the blur would darken the borders and fail this.
    thermal = torch.full((1, 1, 32, 32), value)
    guidance = compute_guidance(thermal)
    assert guidance.shape == (1, 1, 32, 32)
    assert torch.allclose(guidance, torch.full_like(guidance, value), atol=1e-6)


def test_there_is_no_log_and_no_stretch():
    ramp = torch.linspace(0.2, 0.6, 64).view(1, 1, 1, 64).expand(1, 1, 16, 64).contiguous()
    guidance = compute_guidance(ramp)
    # A box blur of a linear ramp is the ramp itself away from the borders.
    assert torch.allclose(guidance[..., 4:-4], ramp[..., 4:-4], atol=1e-6)
    assert guidance.min() >= 0.19 and guidance.max() <= 0.61  # not rescaled to span [0, 1]


def test_range_is_clamped_to_the_unit_interval():
    guidance = compute_guidance(
        torch.stack([torch.full((8, 8), -0.5), torch.full((8, 8), 1.5)]).unsqueeze(1)
    )
    assert guidance[0].max() == 0.0 and guidance[1].min() == 1.0


def test_blur_is_a_5x5_box_with_replicate_padding():
    thermal = torch.zeros(1, 1, 9, 9)
    thermal[..., 4, 4] = 1.0  # one white pixel spreads to 1/25 over its 5x5 window
    guidance = compute_guidance(thermal)[0, 0]
    lit = guidance > 0
    assert lit.sum() == 25 and lit[2:7, 2:7].all()
    assert torch.allclose(guidance[4, 4], torch.tensor(1 / 25), atol=1e-6)
    corner = torch.zeros(1, 1, 9, 9)
    corner[..., 0, 0] = 1.0  # replicate padding folds the missing taps back onto the corner
    assert torch.allclose(compute_guidance(corner)[0, 0, 0, 0], torch.tensor(9 / 25), atol=1e-6)


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (1, 8, 8), (2, 1)])
def test_only_one_channel_thermal_is_accepted(shape):
    with pytest.raises(ValueError, match="thermal"):
        compute_guidance(torch.rand(shape))
