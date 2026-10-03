"""docs/conventions.md section 4a: every downsample is area averaging."""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.resample import area_downsample, fit_long_side  # noqa: E402


def _supersampled_reference(img, size):
    """Brute-force area average: repeat each pixel k times so every footprint is whole pixels."""
    (h_in, w_in), (h_out, w_out) = img.shape[-2:], size
    k = 2
    assert (h_in * k) % h_out == 0 and (w_in * k) % w_out == 0, "pick sizes that align at k=2"
    fine = img.repeat_interleave(k, dim=-2).repeat_interleave(k, dim=-1)
    bh, bw = h_in * k // h_out, w_in * k // w_out
    return fine.unflatten(-1, (w_out, bw)).unflatten(-3, (h_out, bh)).mean(dim=(-3, -1))


def test_integer_factor_is_the_block_mean():
    x = torch.rand(2, 3, 12, 16)
    expected = x.unflatten(-1, (4, 4)).unflatten(-3, (3, 4)).mean(dim=(-3, -1))
    assert torch.allclose(area_downsample(x, (3, 4)), expected, atol=1e-6)


def test_non_integer_factor_weights_straddling_pixels_by_their_overlap():
    x = torch.rand(3, 10, 9, dtype=torch.float64)  # factors 2.5 and 1.5
    assert torch.allclose(area_downsample(x, (4, 6)), _supersampled_reference(x, (4, 6)))


def test_mean_is_preserved():
    x = torch.rand(3, 150, 200)
    assert area_downsample(x, (37, 51)).mean().item() == pytest.approx(x.mean().item(), abs=1e-6)


def test_constant_image_stays_constant():
    out = area_downsample(torch.full((3, 97, 131), 0.3), (13, 29))
    assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-6)


def test_same_size_is_the_identity():
    x = torch.rand(3, 8, 9)
    assert torch.equal(area_downsample(x, (8, 9)), x)


def test_batch_and_single_image_agree():
    batch = torch.rand(4, 3, 30, 40)
    out = area_downsample(batch, (7, 11))
    assert out.shape == (4, 3, 7, 11)
    for i in range(4):
        assert torch.allclose(out[i], area_downsample(batch[i], (7, 11)))


def test_keeps_dtype_and_is_differentiable():
    x = torch.rand(3, 20, 20, requires_grad=True)
    out = area_downsample(x, (6, 6))
    assert out.dtype == torch.float32
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and (x.grad > 0).all()  # every input pixel contributes


def test_rejects_integer_images():
    # Decode uint8 files to linear float first (color.srgb8_to_linear); averaging codes is wrong.
    with pytest.raises(TypeError):
        area_downsample(torch.zeros(3, 8, 8, dtype=torch.uint8), (4, 4))


@pytest.mark.parametrize("size", [(21, 20), (20, 21), (0, 5)])
def test_rejects_upsampling_and_empty_sizes(size):
    with pytest.raises(ValueError):
        area_downsample(torch.rand(3, 20, 20), size)


@pytest.mark.parametrize(
    "hw, long_side, expected",
    [
        ((1500, 2000), 512, (384, 512)),  # landscape 4:3
        ((2000, 1500), 512, (512, 384)),  # portrait
        ((540, 960), 512, (288, 512)),  # 16:9
        ((788, 1400), 512, (288, 512)),  # rounds 288.18
        ((300, 400), 512, (300, 400)),  # already small: never upsample
    ],
)
def test_fit_long_side(hw, long_side, expected):
    assert fit_long_side(*hw, long_side) == expected
