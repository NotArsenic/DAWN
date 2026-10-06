"""The degradation contract, end to end.

Every requirement the pipeline has to meet numerically lives here: the noise budget it reports
has to be the noise it actually produced, binning has to reduce variance the way averaging
does, and a seeded render has to be reproducible. Skips while lighting.py / camera.py are stubs.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data import degrade  # noqa: E402
from dawn.data.camera import apply_camera  # noqa: E402
from dawn.data.lighting import MODES, apply_lighting  # noqa: E402
from tests.helpers import skip_until_implemented  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_CONFIG = ROOT / "configs" / "data.yaml"
CALIB_CONFIG = ROOT / "configs" / "camera_calib.yaml"


def _load(path):
    return omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(path))


@pytest.fixture(scope="module")
def cfg():
    return _load(DATA_CONFIG)["degradation"]


@pytest.fixture(scope="module")
def calib():
    return _load(CALIB_CONFIG)


@pytest.fixture
def day():
    generator = torch.Generator().manual_seed(7)
    return torch.rand(3, 256, 256, generator=generator) * 0.4 + 0.05


@skip_until_implemented
def test_degradation_returns_a_matched_pair_and_its_noise_budget(cfg, calib, day):
    result = degrade.degrade_image(
        day, cfg, calib, image_id="0000287_00759_d_0000759", level="medium",
        native_size=(2000, 1500),
    )
    for image in (result.night, result.night_clean, result.day):
        assert image.shape == (3, 256, 256)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
    assert result.noise_a.shape == result.noise_b.shape == (3,)
    assert (result.noise_a > 0).all() and (result.noise_b > 0).all()
    assert result.quant_var > 0
    assert result.n == pytest.approx(2000 * 1500 / 256**2)
    assert result.params.mode in MODES
    assert result.night_clean.mean() < day.mean()
    assert not torch.equal(result.night, result.night_clean)


@skip_until_implemented
def test_noise_off_makes_night_identical_to_night_clean(cfg, calib, day):
    result = degrade.degrade_image(
        day, cfg, calib, image_id="0000001_00000_d_0000001", level="hard", noise=False
    )
    assert torch.equal(result.night, result.night_clean)


@skip_until_implemented
def test_the_same_image_id_gives_the_same_night_image(cfg, calib, day):
    kwargs = dict(image_id="0000002_00001_d_0000002", level="easy", native_size=(1920, 1080))
    first = degrade.degrade_image(day, cfg, calib, **kwargs)
    second = degrade.degrade_image(day, cfg, calib, **kwargs)
    assert torch.equal(first.night, second.night)
    assert torch.equal(first.night_clean, second.night_clean)
    assert first.params.camera.n_white == second.params.camera.n_white

    other = degrade.degrade_image(day, cfg, calib, **{**kwargs, "image_id": "0000003_x"})
    assert not torch.equal(first.night, other.night)
    # A different darkness level is a different draw for the same image.
    harder = degrade.degrade_image(day, cfg, calib, **{**kwargs, "level": "hard"})
    assert not torch.equal(first.night, harder.night)


@skip_until_implemented
def test_training_draws_are_fresh_every_call(cfg, calib, day):
    torch.manual_seed(0)
    first = degrade.degrade_image(day, cfg, calib, mode="moonlit")
    second = degrade.degrade_image(day, cfg, calib, mode="moonlit")
    assert first.params.camera.n_white != second.params.camera.n_white


@skip_until_implemented
@pytest.mark.parametrize("level", degrade.DARKNESS_LEVELS)
def test_every_darkness_level_renders(cfg, calib, day, level):
    result = degrade.degrade_image(day, cfg, calib, image_id="0000004_x", level=level)
    assert torch.isfinite(result.night).all()


@skip_until_implemented
def test_harder_levels_are_darker_or_noisier(cfg, calib, day):
    out = {
        level: degrade.degrade_image(day, cfg, calib, image_id="0000005_x", level=level,
                                     mode="moonlit")
        for level in degrade.DARKNESS_LEVELS
    }
    targets = [out[level].params.camera.target for level in ("hard", "medium", "easy")]
    assert targets[0] < targets[1] < targets[2]


@skip_until_implemented
def test_flow_path_always_outputs_256_squared(cfg, calib):
    for shape in ((3, 512, 512), (3, 400, 640), (3, 1080, 1920)):
        result = degrade.degrade_image(
            torch.rand(shape) * 0.3, cfg, calib, image_id="0000006_x", level="easy"
        )
        assert result.night.shape == (3, 256, 256)
        assert result.day.shape == (3, 256, 256)


@skip_until_implemented
def test_native_path_keeps_its_resolution_and_uses_n_equal_to_one(cfg, calib):
    crop = torch.rand(3, 192, 192) * 0.3
    result = degrade.degrade_image(
        crop, cfg, calib, image_id="0000007_x", level="medium",
        native_size=(2000, 1500), native_path=True,
    )
    assert result.night.shape == (3, 192, 192)
    assert result.n == 1.0


# --- the noise budget has to be the truth, not a label -------------------------------------

def _flat_patch_variance(cfg, calib, signal, n, trials, seed=11):
    """Render a flat grey patch `trials` times; return (measured var, predicted var, clean)."""
    generator = torch.Generator().manual_seed(seed)
    day = torch.full((3, 64, 64), signal)
    params = degrade.sample_degradation_params(
        64, 64, cfg, calib, mode="moonlit", level="medium", n=n, generator=generator
    )
    # Flatten the ambient and drop the colour cast, so the only spread left is the sensor's.
    params.lighting.illum_field = torch.ones_like(params.lighting.illum_field)
    params.lighting.ambient_gains = torch.ones(3)
    params.camera.wb_gains = torch.ones(3)

    radiance = apply_lighting(day, params.lighting)
    clean = degrade.apply_degradation(day, params, noise=False)
    stack = torch.stack(
        [
            apply_camera(
                radiance, params.camera, n, noise=True, jpeg=False, generator=generator
            ).night
            for _ in range(trials)
        ]
    )
    return stack.var(dim=0, unbiased=True), clean.variance(clean.night_clean), clean


@skip_until_implemented
@pytest.mark.parametrize("signal", [0.3, 0.7])
def test_flat_patch_variance_matches_the_reported_a_and_b(cfg, calib, signal):
    measured, predicted, _ = _flat_patch_variance(cfg, calib, signal, n=1.0, trials=400)
    ratio = float(measured.mean() / predicted.mean())
    assert abs(ratio - 1.0) < 0.05, f"measured / predicted variance = {ratio:.4f}"


@skip_until_implemented
def test_binning_sixteen_native_pixels_divides_the_variance_by_sixteen(cfg, calib):
    one, _, _ = _flat_patch_variance(cfg, calib, 0.5, n=1.0, trials=400, seed=3)
    sixteen, _, _ = _flat_patch_variance(cfg, calib, 0.5, n=16.0, trials=400, seed=3)
    ratio = float(one.mean() / sixteen.mean())
    assert abs(ratio / 16.0 - 1.0) < 0.05, f"var(n=1) / var(n=16) = {ratio:.3f}, expected 16"


@skip_until_implemented
def test_monte_carlo_colour_variance_matches_formula(cfg, calib):
    from dawn.data.camera import apply_camera

    generator = torch.Generator().manual_seed(100)

    day = torch.zeros((3, 64, 64))
    day[0] = 0.5
    day[1] = 0.2
    day[2] = 0.1

    params = degrade.sample_degradation_params(
        64, 64, cfg, calib, mode="moonlit", level="medium", n=1.0, generator=generator
    )

    params.lighting.illum_field = torch.ones_like(params.lighting.illum_field)

    radiance = apply_lighting(day, params.lighting)
    clean_result = apply_camera(radiance, params.camera, params.n, noise=False, native_path=True, jpeg=False)

    # We need S and M, quant_var, etc from clean_result for formula
    # Or we can just build a DegradationResult manually to call variance()
    # It's easier to just compute variance() manually or use a dummy DegradationResult

    res = degrade.DegradationResult(
        night=clean_result.night,
        night_clean=clean_result.night,
        day=day,
        noise_a=clean_result.noise_a,
        noise_b=clean_result.noise_b,
        quant_var=clean_result.quant_var,
        n=clean_result.n,
        S=clean_result.S,
        M=clean_result.M,
        params=params,
        gain=clean_result.gain,
        gain_capped=clean_result.gain_capped,
        saturated=clean_result.saturated,
    )

    trials = 400
    stack = torch.stack([
        apply_camera(radiance, params.camera, params.n, noise=True, native_path=True, jpeg=False, generator=generator).night
        for _ in range(trials)
    ])

    measured = stack[:, :, 16:-16, 16:-16].var(dim=0, unbiased=True).mean(dim=(1, 2))
    predicted = res.variance(res.night_clean)[:, 16:-16, 16:-16].mean(dim=(1, 2))

    assert torch.allclose(measured, predicted, rtol=0.10)
