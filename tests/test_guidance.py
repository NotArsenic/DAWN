"""docs/conventions.md section 4: guidance definition."""

import math

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.guidance import compute_guidance  # noqa: E402


@pytest.mark.parametrize("value", [1e-3, 0.05, 0.5])
def test_constant_image_gives_constant_guidance_including_borders(value):
    # Zero padding in the blur would darken the borders and fail this.
    image = torch.full((1, 3, 32, 32), value)
    expected = (math.log10(value) + 4.0) / 4.0
    guidance = compute_guidance(image)
    assert guidance.shape == (1, 1, 32, 32)
    assert torch.allclose(guidance, torch.full_like(guidance, expected), atol=1e-5)


def test_range_and_clamps():
    assert compute_guidance(torch.zeros(1, 3, 8, 8)).max() == 0.0
    assert torch.allclose(compute_guidance(torch.ones(1, 3, 8, 8)), torch.ones(1, 1, 8, 8))


def test_monotonic_in_brightness():
    levels = torch.tensor([1e-4, 1e-3, 1e-2, 1e-1, 1.0]).view(-1, 1, 1, 1).expand(-1, 3, 8, 8)
    centre = compute_guidance(levels)[:, 0, 4, 4]
    assert (centre[1:] > centre[:-1]).all()


def test_blur_is_a_5x5_box():
    image = torch.zeros(1, 3, 9, 9)
    image[:, :, 4, 4] = 1.0  # one white pixel: luminance 1.0 spreads to 1/25 over its 5x5 window
    guidance = compute_guidance(image)[0, 0]
    lit = guidance > 0
    assert lit.sum() == 25
    assert lit[2:7, 2:7].all()
    assert torch.allclose(guidance[4, 4], torch.tensor((math.log10(1 / 25) + 4.0) / 4.0), atol=1e-5)
