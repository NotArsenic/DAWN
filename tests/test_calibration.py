"""The noise calibration: does it recover a sensor and an ISP it has never been told about?

Every test here synthesizes frames with a known `(a_raw, b_raw)` and a known ISP kernel and
asks the estimator to find them again, because there is no other way to check a measurement
whose whole job is to measure something unobservable. `calibration.synthetic_frame` makes the
frames, and makes them deliberately awkward: smooth-but-not-flat backgrounds and textured
regions much larger than one patch (see its docstring).

Tolerances are loose on purpose. They are what the method actually delivers, measured across
scene types in `scripts/calibrate_noise.py --self-test`, not what would be nice.
"""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data.calibration import (  # noqa: E402
    CHANNELS,
    MAX_LAG,
    PATCH,
    PatchAccumulator,
    _isp_params,
    _surface_projector,
    analyze,
    calibrate,
    fit_variance_line,
    kernel_autocorrelation,
    projected_autocorrelation,
    projected_variance_bias,
    synthetic_frame,
)
from dawn.data.camera import isp_kernel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

ISP_CFG = {"denoise_radius": 2, "sharpen_sigma": 1.0, "sharpen_radius": 2}
TRUE_SIGMA, TRUE_AMOUNT = 0.80, 0.60
TRUE_A, TRUE_B = 2.0e-4, 1.6e-7
SIZE = 512
# A 512-px frame holds 1,024 patches per channel, of which a fifteenth survive the flatness
# gate, so a handful of frames cannot fill 64-patch bins. The real calibration runs on
# thousands of 2000x1500 frames and keeps the default.
MIN_BIN = 16


@pytest.fixture(scope="module")
def kernel():
    """The ISP the synthetic camera applies, and that the estimator has to find again."""
    return isp_kernel(_isp_params(TRUE_SIGMA, TRUE_AMOUNT, ISP_CFG))


def frames(kernel, count=6, seed=23, level=0.06, size=SIZE, **kwargs):
    generator = torch.Generator().manual_seed(seed)
    return [
        synthetic_frame(
            size, size, generator, kernel, level=level, a_raw=TRUE_A, b_raw=TRUE_B, **kwargs
        )
        for _ in range(count)
    ]


@pytest.fixture(scope="module")
def fitted(kernel):
    """One calibration of a mildly textured dark scene, reused by several tests."""
    return calibrate(
        frames(kernel, count=8, texture_fraction=0.35),
        ISP_CFG,
        min_bin_patches=MIN_BIN,
        reference_gain=1.0,
    )


# --- the two measurements ------------------------------------------------------------------


def test_recovers_the_shot_noise_gain(fitted, kernel):
    """`a` is the quantity everything else is calibrated against; it has to be close."""
    expected = TRUE_A * float(kernel.pow(2).sum())
    assert fitted.effective_a == pytest.approx(expected, rel=0.12)
    assert fitted.noise_a == pytest.approx(TRUE_A, rel=0.12)
    assert fitted.full_well == pytest.approx(1.0 / TRUE_A, rel=0.12)


def test_recovers_the_read_noise_where_it_is_identifiable(kernel):
    """`b` is the intercept of a line, so it needs a near-black region to be measured at all."""
    dark = calibrate(
        frames(kernel, count=8, level=0.012, texture_fraction=0.35),
        ISP_CFG,
        min_bin_patches=MIN_BIN,
        min_code=0,
    )
    assert dark.darkest_bin < 0.02
    assert dark.read_noise_share > 0.05  # the fit genuinely reaches down to the read floor
    assert dark.noise_b == pytest.approx(TRUE_B, rel=0.5)
    assert dark.read_noise_sigma == pytest.approx(math.sqrt(TRUE_B) / TRUE_A, rel=0.3)


def test_crushed_blacks_exclusion_keeps_a_unbiased(kernel):
    """Synthetic case with crushed blacks (values clipped at 0 then 8-bit sRGB).
    Excluding patches near code 0 (min_code=3) keeps effective_a unbiased."""
    crushed_frames = [
        synthetic_frame(
            512,
            512,
            torch.Generator().manual_seed(23 + i),
            kernel,
            level=level,
            a_raw=TRUE_A,
            b_raw=TRUE_B,
            texture_fraction=0.5,
            crush_blacks=True,
        )
        for i, level in enumerate([0.005, 0.02, 0.06, 0.15] * 2)
    ]
    fit = calibrate(crushed_frames, ISP_CFG, min_bin_patches=MIN_BIN, min_code=3)
    expected_a = TRUE_A * float(kernel.pow(2).sum())
    assert fit.effective_a == pytest.approx(expected_a, rel=0.20)


def test_a_bright_scene_says_so_rather_than_inventing_a_read_floor(kernel):
    """No near-black patch means no information about `b`. The diagnostic has to show it."""
    bright = calibrate(
        frames(kernel, count=8, level=0.7, texture_fraction=0.35),
        ISP_CFG,
        min_bin_patches=MIN_BIN,
    )
    assert bright.darkest_bin > 0.05
    assert bright.read_noise_share < 0.05
    assert bright.effective_a == pytest.approx(
        TRUE_A * float(isp_kernel(_isp_params(TRUE_SIGMA, TRUE_AMOUNT, ISP_CFG)).pow(2).sum()),
        rel=0.15,
    )


def test_recovers_the_isp_denoise_strength(fitted, kernel):
    """The autocorrelation fit, which is what converts effective `(a, b)` into sensor units."""
    assert fitted.denoise_sigma == pytest.approx(TRUE_SIGMA, rel=0.10)
    assert fitted.sharpen_amount == pytest.approx(TRUE_AMOUNT, rel=0.35)
    assert fitted.isp_var_scale == pytest.approx(float(kernel.pow(2).sum()), rel=0.10)
    assert fitted.autocorr_rms < 0.02


def test_measured_autocorrelation_is_positive_at_lag_one(fitted):
    """A denoising ISP correlates neighbours. White sensor noise alone would sit near zero."""
    assert len(fitted.autocorr) == MAX_LAG
    assert fitted.autocorr[0] > 0.3
    assert fitted.autocorr[0] > fitted.autocorr[1]


def test_both_fits_are_physical(fitted):
    """A variance cannot be negative and neither can a gain, whatever the data look like."""
    assert fitted.effective_a > 0 and fitted.effective_b >= 0
    assert min(fitted.per_channel_a) > 0 and min(fitted.per_channel_b) >= 0
    assert fitted.r_squared > 0.9
    assert fitted.num_bins_used >= 4


# --- the projection correction ---------------------------------------------------------------


def test_projected_autocorrelation_matches_a_monte_carlo_measurement(kernel):
    """The model is compared through the same detrend as the data, so check it against data.

    Filter white noise with the known kernel, detrend 16x16 patches of it exactly as the
    accumulator does, and measure the autocorrelation. `projected_autocorrelation` predicts
    that number in closed form; the two must agree to Monte-Carlo precision.
    """
    import torch.nn.functional as F

    generator = torch.Generator().manual_seed(5)
    noise = torch.randn(1, 1, 1024, 1024, generator=generator)
    radius = kernel.shape[0] // 2
    field = F.conv2d(F.pad(noise, (radius,) * 4, mode="reflect"), kernel.view(1, 1, *kernel.shape))

    tiles = field[0].unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH).reshape(-1, PATCH * PATCH)
    resid = tiles @ _surface_projector(PATCH, tiles.device)
    resid = resid.view(-1, PATCH, PATCH)
    measured = torch.zeros(MAX_LAG + 1, MAX_LAG + 1)
    for dy in range(MAX_LAG + 1):
        for dx in range(MAX_LAG + 1):
            overlap = resid[:, : PATCH - dy, : PATCH - dx] * resid[:, dy:, dx:]
            measured[dy, dx] = overlap.mean()
    measured = measured / measured[0, 0]

    predicted = projected_autocorrelation(kernel, PATCH)
    assert torch.allclose(predicted, measured, atol=0.02)
    # And it is a real correction, not a no-op: the raw model is noticeably different.
    raw = kernel_autocorrelation(kernel)
    assert abs(float(predicted[0, 1]) - float(raw[0, 1])) > 0.02


def test_projected_variance_bias_is_what_the_detrend_actually_costs(kernel):
    """Removing 6 of 256 modes takes more than 6/256 of a correlated field's variance."""
    import torch.nn.functional as F

    generator = torch.Generator().manual_seed(6)
    noise = torch.randn(1, 1, 1024, 1024, generator=generator)
    radius = kernel.shape[0] // 2
    field = F.conv2d(F.pad(noise, (radius,) * 4, mode="reflect"), kernel.view(1, 1, *kernel.shape))
    tiles = field[0].unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH).reshape(-1, PATCH * PATCH)
    resid = tiles @ _surface_projector(PATCH, tiles.device)

    dof = PATCH * PATCH - 6
    measured = float(resid.pow(2).sum(dim=1).mean() / dof) / float(field.var())
    assert projected_variance_bias(kernel, PATCH) == pytest.approx(measured, rel=0.05)
    assert projected_variance_bias(kernel, PATCH) < 1.0  # the detrend only ever removes variance


# --- the line fit --------------------------------------------------------------------------


def test_line_fit_ignores_bins_that_are_only_texture():
    """Contamination is one-sided: texture adds variance, never removes it.

    Least squares through all of these bins would be dragged up by the three bad ones. The
    lower-envelope fit has to come back with the line the clean bins lie on.
    """
    means = torch.linspace(0.01, 0.5, 20)
    variances = 3.0e-4 * means + 2.0e-7
    variances[[7, 13, 18]] *= 25.0  # bins that happened to contain only rooftops
    counts = torch.full((20,), 500.0)

    a, b, r_squared, kept = fit_variance_line(means, variances, counts)
    assert a == pytest.approx(3.0e-4, rel=0.05)
    assert b == pytest.approx(2.0e-7, abs=5e-8)
    assert r_squared > 0.99
    assert not bool(kept[[7, 13, 18]].any())


def test_line_fit_never_returns_a_negative_gain_or_variance():
    """Noise that falls with brightness is impossible, so the fit must refuse to describe it."""
    means = torch.linspace(0.01, 0.5, 12)
    variances = 1.0e-4 - 1.5e-4 * means
    a, b, _, _ = fit_variance_line(means, variances, torch.full((12,), 500.0))
    assert a >= 0.0 and b >= 0.0


def test_line_fit_weights_bins_by_how_many_patches_they_hold():
    """One bin built from four patches must not outvote a bin built from a thousand."""
    means = torch.tensor([0.02, 0.10, 0.20, 0.30, 0.40])
    variances = 3.0e-4 * means + 2.0e-7
    counts = torch.tensor([800.0, 800.0, 800.0, 800.0, 800.0])
    clean, *_ = fit_variance_line(means, variances, counts)

    variances[4] *= 1.4
    counts[4] = 4.0
    noisy, *_ = fit_variance_line(means, variances, counts.clone())
    assert noisy == pytest.approx(clean, rel=0.10)


# --- accumulation --------------------------------------------------------------------------


def test_merging_accumulators_equals_one_pass(kernel):
    """`calibrate_paths` measures frames in separate processes, so the merge must be exact."""
    images = frames(kernel, count=4, texture_fraction=0.35)
    single = PatchAccumulator()
    for image in images:
        single.add_image(image)

    halves = []
    for chunk in (images[:2], images[2:]):
        part = PatchAccumulator()
        for image in chunk:
            part.add_image(image)
        halves.append(part)
    halves[0].merge(halves[1])
    merged = halves[0]

    assert merged.num_images == single.num_images
    assert merged.num_patches_seen == single.num_patches_seen
    for name in ("count", "sum_mean", "sum_var", "corr_num", "corr_den"):
        assert torch.allclose(getattr(merged, name), getattr(single, name), atol=1e-6)

    lums = [0.05] * 4
    merged_fit = analyze(merged, lums, ISP_CFG, min_bin_patches=MIN_BIN)
    single_fit = analyze(single, lums, ISP_CFG, min_bin_patches=MIN_BIN)
    assert merged_fit.effective_a == single_fit.effective_a


def test_patches_are_binned_on_sqrt_brightness(kernel):
    """Equal-width bins would pile the whole dark end, the part that pins `b`, into one bin."""
    accumulator = PatchAccumulator(num_bins=40)
    accumulator.add_image(frames(kernel, count=1, level=0.04)[0])
    means, _, counts = accumulator.bins(min_bin_patches=1)
    populated = means[counts > 0]
    assert populated.numel() >= 4
    # A scene topping out near 0.04 would occupy under two of 40 equal-width bins.
    assert float(populated.max()) < 0.1


def test_flat_patch_selection_is_deterministic(kernel):
    """Same frames in, same numbers out: a calibration has to be reproducible to be checkable."""
    first = calibrate(frames(kernel, count=4), ISP_CFG, min_bin_patches=MIN_BIN)
    second = calibrate(frames(kernel, count=4), ISP_CFG, min_bin_patches=MIN_BIN)
    assert first.effective_a == second.effective_a
    assert first.effective_b == second.effective_b
    assert first.denoise_sigma == second.denoise_sigma
    assert first.num_patches == second.num_patches


def test_per_channel_fits_agree_with_the_pooled_one(fitted):
    """The synthetic sensor is the same in all three channels, so the three fits must match."""
    assert len(fitted.per_channel_a) == CHANNELS
    for value in fitted.per_channel_a:
        assert value == pytest.approx(fitted.effective_a, rel=0.25)


def test_empty_input_is_an_error_not_a_zero():
    """Silently returning a default calibration would quietly mis-expose every training image."""
    with pytest.raises(ValueError):
        calibrate([], ISP_CFG)
    with pytest.raises(ValueError):
        analyze(PatchAccumulator(), [], ISP_CFG)


# --- the file the script writes --------------------------------------------------------------


def _calibrate_noise_script():
    """`scripts/calibrate_noise.py` as a module. It is a script, not a package member."""
    import importlib.util

    path = ROOT / "scripts" / "calibrate_noise.py"
    spec = importlib.util.spec_from_file_location("script_calibrate_noise", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_written_yaml_is_what_camera_py_reads_back(fitted, tmp_path):
    """The loop has to close: whatever the script writes, the simulator has to be able to use.

    Writing a calibration the camera cannot parse, or that drops a key it needs, would not
    show up until a training run produced nonsense, so it is checked here instead.
    """
    from dawn.data.camera import sample_camera_params

    script = _calibrate_noise_script()
    template = omegaconf.OmegaConf.to_container(
        omegaconf.OmegaConf.load(ROOT / "configs" / "camera_calib.yaml")
    )
    text = script._format_calibration(fitted, template, {"fitted": "2026-01-01", "frames": 8})
    path = tmp_path / "camera_calib.yaml"
    path.write_text(text, encoding="utf-8", newline="\n")

    written = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path))
    assert written["source"] == "measured"
    assert set(template) <= set(written)  # nothing the old file had went missing
    assert set(template["isp"]) == set(written["isp"])
    assert written["measured"]["noise_a"] == pytest.approx(fitted.noise_a, rel=1e-3)
    assert written["measured"]["autocorr"][0] == pytest.approx(fitted.autocorr[0], abs=1e-4)

    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(ROOT / "configs" / "data.yaml"))
    params = sample_camera_params(cfg["degradation"]["camera"], written, "moonlit")
    assert params.a_raw == pytest.approx(fitted.noise_a, rel=1e-3)
    assert params.denoise_sigma == pytest.approx(fitted.denoise_sigma, rel=1e-3)
    assert params.ae_target == pytest.approx(fitted.exposure_target, rel=1e-3)
