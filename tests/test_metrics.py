"""Validation metrics: masked L* PSNR, CIEDE2000 and Hasler-Suesstrunk colorfulness."""

import math

import pytest

torch = pytest.importorskip("torch")

from dawn.eval.metrics import ciede2000, colorfulness, masked_psnr  # noqa: E402


def _random_lab(shape, seed):
    g = torch.Generator().manual_seed(seed)
    lab = torch.empty(*shape, dtype=torch.float64)
    lab[..., 0, :, :] = torch.rand(lab[..., 0, :, :].shape, generator=g, dtype=torch.float64) * 100
    lab[..., 1:, :, :] = (
        torch.randn(lab[..., 1:, :, :].shape, generator=g, dtype=torch.float64) * 40
    )
    return lab


def test_ciede2000_matches_skimage_on_random_lab_pairs():
    skimage_color = pytest.importorskip("skimage.color")
    one, two = _random_lab((2, 3, 24, 24), 0), _random_lab((2, 3, 24, 24), 1)
    ours = ciede2000(one, two)
    assert ours.shape == (2, 24, 24)
    theirs = skimage_color.deltaE_ciede2000(
        one.permute(0, 2, 3, 1).numpy(), two.permute(0, 2, 3, 1).numpy()
    )
    assert torch.allclose(ours, torch.from_numpy(theirs), atol=1e-6)


def test_ciede2000_matches_skimage_on_the_sharma_hard_cases():
    """Hue wrap-around, near-zero chroma and the 180 degree case are where implementations drift."""
    skimage_color = pytest.importorskip("skimage.color")
    pairs = [
        ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485)),
        ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485)),
        ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485)),
        ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485)),
        ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0)),
        ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0)),
        ((50.0, 2.5, 0.0), (50.0, 2.5, 0.0)),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387)),
    ]
    one = torch.tensor([p[0] for p in pairs], dtype=torch.float64).t().reshape(1, 3, len(pairs), 1)
    two = torch.tensor([p[1] for p in pairs], dtype=torch.float64).t().reshape(1, 3, len(pairs), 1)
    ours = ciede2000(one, two)[0, :, 0]
    theirs = skimage_color.deltaE_ciede2000(
        torch.tensor([p[0] for p in pairs]).numpy().astype("float64"),
        torch.tensor([p[1] for p in pairs]).numpy().astype("float64"),
    )
    assert torch.allclose(ours, torch.from_numpy(theirs), atol=1e-4)


def test_ciede2000_is_zero_for_identical_colours_and_symmetric_in_float32():
    one, two = _random_lab((1, 3, 8, 8), 2).float(), _random_lab((1, 3, 8, 8), 3).float()
    assert ciede2000(one, one).abs().max() < 1e-4
    assert torch.allclose(ciede2000(one, two), ciede2000(two, one), atol=1e-3)
    assert ciede2000(one, two).dtype == torch.float32


def test_masked_psnr_ignores_pixels_outside_the_mask_and_follows_the_formula():
    target = torch.rand(2, 1, 8, 8) * 100
    pred = target.clone()
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4] = 1.0
    pred[:, :, :4] += 2.0  # a constant error of 2 L* units under the mask
    pred[:, :, 4:] += 50.0  # a huge error outside it
    psnr = masked_psnr(pred, target, mask, peak=100.0)
    assert psnr.shape == (2,)
    assert torch.allclose(psnr, torch.full((2,), 10 * math.log10(100.0**2 / 4.0)), atol=1e-4)


def test_masked_psnr_with_fractional_weights_and_an_empty_mask():
    target = torch.zeros(1, 1, 4, 4)
    pred = torch.zeros(1, 1, 4, 4)
    pred[..., :2] = 10.0
    weights = torch.ones(1, 1, 4, 4)
    weights[..., :2] = 0.5  # the erroneous half counts half as much
    expected_mse = (0.5 * 8 * 100.0) / (0.5 * 8 + 8)
    assert torch.allclose(
        masked_psnr(pred, target, weights, 100.0),
        torch.tensor([10 * math.log10(100.0**2 / expected_mse)]),
        atol=1e-4,
    )
    assert torch.isfinite(masked_psnr(pred, target, torch.zeros(1, 1, 4, 4), 100.0)).all()


def test_masked_psnr_of_identical_images_is_large_and_finite():
    image = torch.rand(1, 3, 4, 4) * 100
    assert 100 < float(masked_psnr(image, image, torch.ones(1, 1, 4, 4), 100.0)) < 1e4


def test_colorfulness_of_a_grey_image_is_zero():
    grey = torch.rand(2, 1, 8, 8).expand(2, 3, 8, 8) * 255
    assert torch.allclose(colorfulness(grey), torch.zeros(2), atol=1e-4)


def test_colorfulness_matches_the_hasler_suesstrunk_formula():
    image = torch.rand(1, 3, 16, 16, dtype=torch.float64) * 255
    r, g, b = image[0]
    rg, yb = r - g, 0.5 * (r + g) - b
    expected = math.sqrt(
        rg.std(unbiased=False) ** 2 + yb.std(unbiased=False) ** 2
    ) + 0.3 * math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    assert float(colorfulness(image)[0]) == pytest.approx(expected, rel=1e-6)
    red = torch.zeros(1, 3, 4, 4)
    red[:, 0] = 255.0
    assert float(colorfulness(red)[0]) == pytest.approx(0.3 * math.hypot(255, 127.5), rel=1e-5)


def test_colorfulness_rejects_non_rgb():
    with pytest.raises(ValueError, match="RGB"):
        colorfulness(torch.rand(1, 1, 4, 4))
