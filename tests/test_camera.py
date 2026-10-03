"""The sensor and ISP: photon statistics, auto exposure, the noise budget, determinism."""

import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data.camera import (  # noqa: E402
    apply_camera,
    auto_exposure_gain,
    isp_kernel,
    jpeg_roundtrip,
    sample_camera_params,
)
from dawn.data.seeding import generator_for, seed_from_image_id  # noqa: E402
from tests.helpers import skip_until_implemented  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(ROOT / "configs" / name))


@pytest.fixture(scope="module")
def cfg():
    return _load("data.yaml")["degradation"]["camera"]


@pytest.fixture(scope="module")
def calib():
    return _load("camera_calib.yaml")


def _params(cfg, calib, seed=0, **overrides):
    generator = torch.Generator().manual_seed(seed)
    params = sample_camera_params(cfg, calib, "moonlit", None, None, generator)
    for key, value in overrides.items():
        setattr(params, key, value)
    if "read_noise" in overrides:  # keep b consistent with sigma_r
        params.b_raw = overrides["read_noise"] ** 2 * params.a_raw**2
    return params


# --- the reported (a, b) must be the noise that was actually produced ----------------------

@skip_until_implemented
@pytest.mark.parametrize("signal", [0.1, 0.4, 0.8])
def test_measured_variance_matches_the_reported_a_and_b(cfg, calib, signal):
    params = _params(cfg, calib, seed=1)
    patch = torch.full((3, 48, 48), signal / (params.n_white * params.a_raw))
    generator = torch.Generator().manual_seed(100)
    reference = apply_camera(patch, params, noise=False, jpeg=False)
    stack = torch.stack(
        [apply_camera(patch, params, noise=True, jpeg=False, generator=generator).night
         for _ in range(500)]
    )
    predicted = (
        reference.noise_a.view(3, 1, 1) * reference.night
        + reference.noise_b.view(3, 1, 1)
        + reference.quant_var
    ) / reference.n
    ratio = float(stack.var(dim=0, unbiased=True).mean() / predicted.mean())
    assert abs(ratio - 1.0) < 0.05, f"measured / predicted = {ratio:.4f}"


@skip_until_implemented
def test_binning_divides_the_variance_by_n(cfg, calib):
    params = _params(cfg, calib, seed=2)
    patch = torch.full((3, 48, 48), 0.5 / (params.n_white * params.a_raw))
    variances = {}
    for n in (1.0, 16.0):
        generator = torch.Generator().manual_seed(200)
        stack = torch.stack(
            [apply_camera(patch, params, n=n, noise=True, jpeg=False, generator=generator).night
             for _ in range(500)]
        )
        variances[n] = float(stack.var(dim=0, unbiased=True).mean())
    ratio = variances[1.0] / variances[16.0]
    assert abs(ratio / 16.0 - 1.0) < 0.05, f"var(n=1) / var(n=16) = {ratio:.2f}, expected 16"


@skip_until_implemented
def test_the_reported_a_and_b_stay_at_native_resolution(cfg, calib):
    """`n` changes the noise that is drawn, never the numbers that are reported."""
    params = _params(cfg, calib, seed=3)
    patch = torch.full((3, 16, 16), 0.3)
    one = apply_camera(patch, params, n=1.0, noise=False)
    many = apply_camera(patch, params, n=46.0, noise=False)
    assert torch.equal(one.noise_a, many.noise_a)
    assert torch.equal(one.noise_b, many.noise_b)
    assert one.n == 1.0 and many.n == 46.0


@skip_until_implemented
def test_shot_noise_dominates_in_the_light_and_read_noise_in_the_dark(cfg, calib):
    params = _params(cfg, calib, seed=4, read_noise=3.0)
    a, b = float(params.a_raw), float(params.b_raw)
    assert a * 0.5 > 10 * b, "shot noise should dominate a half-scale signal"
    assert b > a * 1e-5, "read noise should dominate a near-black signal"


# --- auto exposure --------------------------------------------------------------------------

@skip_until_implemented
def test_auto_exposure_drives_the_image_to_the_calibrated_target(cfg, calib):
    params = _params(cfg, calib, seed=5)
    target = float(calib["exposure"]["target"])
    raw = torch.full((3, 32, 32), target / 4.0)  # needs a 4x push, well inside the cap
    gain, capped = auto_exposure_gain(raw, params)
    assert not capped
    assert gain == pytest.approx(4.0, rel=1e-3)


@skip_until_implemented
def test_a_capped_gain_leaves_the_image_dark(cfg, calib):
    params = _params(cfg, calib, seed=6)
    target = float(calib["exposure"]["target"])
    cap = float(calib["exposure"]["gain_max"])
    raw = torch.full((3, 32, 32), target / (10 * cap))
    gain, capped = auto_exposure_gain(raw, params)
    assert capped and gain == pytest.approx(cap)
    # And the render really does come out ten times below the target, not rescued elsewhere.
    result = apply_camera(raw / (params.n_white * params.a_raw), params, noise=False, jpeg=False)
    assert float(result.night[1].median()) < target / 5


@skip_until_implemented
def test_fewer_photons_mean_a_bigger_effective_a(cfg, calib):
    """Darkness and noise are the same knob: a dim scene needs gain, and gain amplifies a."""
    bright = _params(cfg, calib, seed=7, n_white=1000.0)
    dim = _params(cfg, calib, seed=7, n_white=20.0)
    scene = torch.full((3, 32, 32), 0.2)
    assert apply_camera(scene, dim, noise=False).noise_a.mean() > (
        apply_camera(scene, bright, noise=False).noise_a.mean()
    )


# --- quantization, white balance, ISP ------------------------------------------------------

@skip_until_implemented
def test_the_output_lands_on_the_twelve_bit_lattice(cfg, calib):
    params = _params(cfg, calib, seed=8)
    result = apply_camera(torch.rand(3, 32, 32) * 0.3, params, noise=False, jpeg=False)
    levels = 2 ** int(cfg["quant_bits"]) - 1
    assert torch.allclose(result.night * levels, (result.night * levels).round(), atol=1e-3)
    assert result.quant_var == pytest.approx((1.0 / levels) ** 2 / 12.0)


@skip_until_implemented
def test_partial_white_balance_only_partly_removes_the_cast(cfg, calib):
    from dawn.data.color import color_temp_to_gains

    ambient = color_temp_to_gains(2000.0)  # strongly warm
    generator = torch.Generator().manual_seed(9)
    params = sample_camera_params(cfg, calib, "streetlit", None, ambient, generator)
    assert float(params.wb_gains[1]) == pytest.approx(1.0)
    corrected = ambient * params.wb_gains
    # Less cast than the raw illuminant, but not neutral.
    assert 1.0 < float(corrected[0] / corrected[2]) < float(ambient[0] / ambient[2])


@skip_until_implemented
def test_the_isp_kernel_has_unit_dc_gain(cfg, calib):
    kernel = isp_kernel(_params(cfg, calib, seed=10))
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < float(kernel.pow(2).sum()) < 1.0
    assert float(kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]) > 0


@skip_until_implemented
def test_the_native_path_scales_the_noise_budget_by_the_squared_weights(cfg, calib):
    params = _params(cfg, calib, seed=11)
    scene = torch.full((3, 32, 32), 0.3)
    plain = apply_camera(scene, params, noise=False, jpeg=False)
    filtered = apply_camera(scene, params, noise=False, jpeg=False, native_path=True)
    expected = float(isp_kernel(params).pow(2).sum())
    assert filtered.isp_var_scale == pytest.approx(expected, rel=1e-5)
    assert float(filtered.quant_var / plain.quant_var) == pytest.approx(expected, rel=1e-5)
    assert torch.allclose(filtered.noise_a / plain.noise_a, torch.full((3,), expected), rtol=1e-5)


@skip_until_implemented
def test_the_native_path_really_does_reduce_the_noise_it_predicts(cfg, calib):
    params = _params(cfg, calib, seed=12)
    patch = torch.full((3, 64, 64), 0.4 / (params.n_white * params.a_raw))
    measured = {}
    for native in (False, True):
        generator = torch.Generator().manual_seed(300)
        stack = torch.stack(
            [apply_camera(patch, params, noise=True, jpeg=False, native_path=native,
                          generator=generator).night for _ in range(300)]
        )
        # Trim the border, where replicate padding correlates the samples.
        measured[native] = float(stack[:, :, 8:-8, 8:-8].var(dim=0, unbiased=True).mean())
    expected = float(isp_kernel(params).pow(2).sum())
    assert measured[True] / measured[False] == pytest.approx(expected, rel=0.1)


@skip_until_implemented
def test_jpeg_roundtrip_stays_in_range_and_close_to_the_input():
    # A smooth frame, because JPEG is built for those. Per-pixel noise is the one input a DCT
    # codec genuinely cannot carry, so it would measure the test's input, not the round trip.
    ramp = torch.linspace(0, 0.5, 64)
    image = ramp.view(1, 1, 64) * torch.ones(3, 64, 1)
    image = image + 0.05 * torch.sin(torch.linspace(0, 9.0, 64)).view(1, 64, 1)
    image = image.clamp(0.0, 1.0)
    out = jpeg_roundtrip(image, 90)
    assert out.shape == image.shape and out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert float((out - image).abs().mean()) < 0.02


# --- sampling and determinism ----------------------------------------------------------------

@skip_until_implemented
@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_darkness_levels_pick_their_configured_band(cfg, calib, level):
    lo, hi = (float(v) for v in cfg["n_white_range"]["moonlit"])
    q_lo, q_hi = (float(v) for v in cfg["darkness_levels"][level])
    band = (
        math.exp(math.log(lo) + q_lo * (math.log(hi) - math.log(lo))),
        math.exp(math.log(lo) + q_hi * (math.log(hi) - math.log(lo))),
    )
    generator = torch.Generator().manual_seed(13)
    draws = [
        sample_camera_params(cfg, calib, "moonlit", level, None, generator).n_white
        for _ in range(100)
    ]
    assert band[0] - 1e-6 <= min(draws) and max(draws) <= band[1] + 1e-6


@skip_until_implemented
def test_read_noise_stays_in_its_configured_range(cfg, calib):
    lo, hi = (float(v) for v in cfg["read_noise_range"])
    generator = torch.Generator().manual_seed(14)
    draws = [sample_camera_params(cfg, calib, "moonlit", None, None, generator).read_noise
             for _ in range(200)]
    assert lo <= min(draws) and max(draws) <= hi


@skip_until_implemented
def test_sample_camera_params_rejects_unknown_modes_and_levels(cfg, calib):
    with pytest.raises(ValueError):
        sample_camera_params(cfg, calib, "daylight")
    with pytest.raises(ValueError):
        sample_camera_params(cfg, calib, "moonlit", "pitch-black")


@skip_until_implemented
def test_apply_camera_validates_its_inputs(cfg, calib):
    params = _params(cfg, calib, seed=15)
    with pytest.raises(ValueError):
        apply_camera(torch.rand(1, 3, 8, 8), params)
    with pytest.raises(ValueError):
        apply_camera(torch.rand(3, 8, 8), params, n=0.0)


def test_image_ids_seed_independently_and_reproducibly():
    assert seed_from_image_id("abc") == seed_from_image_id("abc")
    assert seed_from_image_id("abc") != seed_from_image_id("abd")
    assert seed_from_image_id("abc", "easy") != seed_from_image_id("abc", "hard")
    assert 0 <= seed_from_image_id("abc") < 2**63
    # Neighbouring IDs must not give neighbouring streams.
    first = torch.rand(4, generator=generator_for("0000001_00000_d_0000001"))
    second = torch.rand(4, generator=generator_for("0000001_00000_d_0000002"))
    assert not torch.allclose(first, second, atol=0.05)
    assert torch.equal(first, torch.rand(4, generator=generator_for("0000001_00000_d_0000001")))
