"""Tests for image evaluation metrics including masked PSNR, SSIM, CIEDE2000, and colorfulness."""

import math

import pytest

torch = pytest.importorskip("torch")

from dawn.eval.metrics import (  # noqa: E402
    ciede2000,
    colorfulness,
    masked_mean,
    masked_psnr,
    masked_ssim,
)


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
    """Verify CIEDE2000 edge cases including hue wrap-around and low chroma match reference."""
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


def test_masked_mean_is_a_weighted_mean_per_image_and_zero_without_weight():
    value = torch.arange(8.0).reshape(2, 1, 2, 2)
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
    out = masked_mean(value, mask)
    assert torch.allclose(out, torch.tensor([1.0, 0.0]))  # (0 + 2) / 2 ; an empty mask is 0
    half = masked_mean(torch.tensor([[[0.0, 4.0]]]), torch.tensor([[[[1.0, 0.5]]]]))
    assert torch.allclose(half, torch.tensor([2.0 / 1.5]))
    assert masked_mean(torch.ones(2, 4, 4), torch.ones(2, 1, 4, 4)).shape == (2,)


# --- SSIM, LPIPS and the batch aggregator (the plan's evaluation table) -------------------------


def _lab(batch=2, size=(32, 40), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.cat(
        [
            torch.rand(batch, 1, *size, generator=g) * 100,
            torch.rand(batch, 2, *size, generator=g) * 60 - 30,
        ],
        dim=1,
    )


def test_masked_ssim_matches_skimage_on_the_interior():
    """Verify interior masked SSIM matches skimage reference implementation."""
    sk = pytest.importorskip("skimage.metrics")
    torch.manual_seed(0)
    pad = 11 // 2
    for noise in (4, 12):
        x = torch.rand(1, 1, 96, 96) * 100
        y = (x + torch.randn(1, 1, 96, 96) * noise).clamp(0, 100)
        interior = torch.zeros(1, 1, 96, 96)
        interior[..., pad:-pad, pad:-pad] = 1.0
        mine = float(masked_ssim(x, y, interior, peak=100.0))
        theirs = sk.structural_similarity(
            x[0, 0].numpy(),
            y[0, 0].numpy(),
            data_range=100.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        assert mine == pytest.approx(theirs, abs=1e-6)


def test_masked_ssim_is_one_for_identical_and_zero_for_an_empty_mask():
    x = torch.rand(2, 1, 64, 64) * 100
    assert torch.allclose(
        masked_ssim(x, x.clone(), torch.ones(2, 1, 64, 64), 100.0), torch.ones(2), atol=1e-5
    )
    empty = masked_ssim(x, x * 0.5, torch.zeros(2, 1, 64, 64), 100.0)
    assert torch.all(torch.isfinite(empty)) and torch.all(empty == 0)


def test_masked_ssim_falls_as_the_images_diverge():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64) * 100
    m = torch.ones(1, 1, 64, 64)
    scores = [
        float(masked_ssim(x, (x + torch.randn_like(x) * n).clamp(0, 100), m, 100.0))
        for n in (2, 8, 20)
    ]
    assert scores == sorted(scores, reverse=True), scores


def test_masked_ssim_only_scores_the_masked_region():
    """Verify pixels excluded by the mask do not affect SSIM scores outside window boundary."""
    torch.manual_seed(0)
    x = torch.rand(1, 1, 64, 64) * 100
    y = x.clone()
    mask = torch.zeros(1, 1, 64, 64)
    mask[..., :24, :24] = 1.0
    y[..., 40:, 40:] = 0.0  # far from the masked region, outside any window that touches it
    assert float(masked_ssim(x, y, mask, 100.0)) == pytest.approx(1.0, abs=1e-5)


def test_masked_ssim_rejects_an_image_smaller_than_its_window():
    with pytest.raises(ValueError, match="smaller than the 11x11"):
        masked_ssim(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), torch.ones(1, 1, 8, 8))


def test_lpips_converts_to_srgb_before_scoring():
    """Verify LPIPS distance metric converts inputs to display-referred sRGB."""
    from dawn.data.color import linear_to_srgb
    from dawn.eval.metrics import lpips_distance

    seen = {}

    def fake(a, b):
        seen["a"], seen["b"] = a, b
        return torch.zeros(a.shape[0], 1, 1, 1)

    pred, target = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    out = lpips_distance(pred, target, fake)
    assert out.shape == (2,)
    assert torch.allclose(seen["a"], linear_to_srgb(pred) * 2 - 1, atol=1e-6)
    assert seen["a"].min() >= -1.0 and seen["a"].max() <= 1.0


def test_evaluate_batch_returns_the_plans_table_per_image():
    from dawn.eval.metrics import evaluate_batch

    lab_t, lab_p = _lab(seed=0), _lab(seed=1)
    rgb_t, rgb_p = torch.rand(2, 3, 32, 40), torch.rand(2, 3, 32, 40)
    mask = torch.ones(2, 1, 32, 40)
    out = evaluate_batch(lab_p, lab_t, rgb_p, rgb_t, mask)
    assert set(out) == {"psnr", "ssim", "ciede2000", "colorfulness", "colorfulness_target"}
    for k, v in out.items():
        assert v.shape == (2,), k
        assert torch.all(torch.isfinite(v)), k


def test_evaluate_batch_adds_vehicle_and_lpips_keys_only_when_asked():
    from dawn.eval.metrics import evaluate_batch

    lab_t, lab_p = _lab(seed=0), _lab(seed=1)
    rgb_t, rgb_p = torch.rand(2, 3, 32, 40), torch.rand(2, 3, 32, 40)
    mask = torch.ones(2, 1, 32, 40)
    veh = torch.zeros(2, 1, 32, 40)
    veh[..., 8:24, 8:32] = 1.0
    out = evaluate_batch(
        lab_p,
        lab_t,
        rgb_p,
        rgb_t,
        mask,
        vehicle_mask=veh,
        lpips_fn=lambda a, b: torch.zeros(a.shape[0]),
    )
    assert {"lpips", "psnr_vehicles", "ssim_vehicles", "ciede2000_vehicles"} <= set(out)
    assert out["psnr_vehicles"].shape == (2,)


def test_the_vehicle_region_is_intersected_with_the_pixel_mask():
    """Verify vehicle region is intersected with valid pixel mask during evaluation."""
    from dawn.eval.metrics import evaluate_batch

    lab_t = _lab(batch=1, seed=0)
    lab_p = lab_t.clone()
    lab_p[..., 0, :, 20:] = 0.0  # wreck the right half
    rgb = torch.rand(1, 3, 32, 40)
    veh = torch.ones(1, 1, 32, 40)
    keep_left = torch.zeros(1, 1, 32, 40)
    keep_left[..., :20] = 1.0
    out = evaluate_batch(lab_p, lab_t, rgb, rgb, keep_left, vehicle_mask=veh)
    assert float(out["ciede2000_vehicles"]) == pytest.approx(0.0, abs=1e-4)


def test_evaluate_batch_rejects_mismatched_lab_and_rgb_sizes():
    from dawn.eval.metrics import evaluate_batch

    with pytest.raises(ValueError, match="same image in two spaces"):
        evaluate_batch(
            _lab(size=(32, 40)),
            _lab(size=(32, 40)),
            torch.rand(2, 3, 16, 20),
            torch.rand(2, 3, 16, 20),
            torch.ones(2, 1, 32, 40),
        )
