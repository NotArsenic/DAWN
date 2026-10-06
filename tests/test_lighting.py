"""The scene illumination model: ambient level, colour, lamp falloff, source-image choice."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data.color import color_temp_to_gains  # noqa: E402
from dawn.data.lighting import (  # noqa: E402
    MODES,
    apply_lighting,
    choose_mode,
    emission_field,
    illumination_field,
    overcast_score,
    sample_lighting_params,
)
from tests.helpers import skip_until_implemented  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


@pytest.fixture(scope="module")
def cfg():
    return omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(CONFIG))["degradation"][
        "lighting"
    ]


@skip_until_implemented
@pytest.mark.parametrize("mode", MODES)
def test_ambient_sits_at_relative_level_one(cfg, mode):
    """A white surface under ambient alone reads 1.0: that is what "relative level 1" means."""
    generator = torch.Generator().manual_seed(0)
    params = sample_lighting_params(32, 48, cfg, mode, generator)
    params.light_xy = torch.zeros(0, 2)  # drop any lamps; ambient only
    params.emitters = []  # drop any emitters; ambient only
    white = torch.ones(3, 32, 48)
    lit = apply_lighting(white, params)
    # Green is the reference channel of `color_temp_to_gains`, so it is exactly the level.
    assert float(lit[1].mean()) == pytest.approx(1.0, rel=1e-5)


@skip_until_implemented
def test_the_illumination_field_is_smooth_and_averages_to_one(cfg):
    generator = torch.Generator().manual_seed(1)
    params = sample_lighting_params(64, 64, cfg, "moonlit", generator)
    field = params.illum_field
    assert float(field.mean()) == pytest.approx(1.0, rel=1e-5)
    assert field.min() > 0
    # Smooth: neighbouring pixels differ by far less than the field's own spread.
    step = (field[:, 1:] - field[:, :-1]).abs().max()
    assert float(step) < 0.1 * float(field.max() - field.min())


@skip_until_implemented
@pytest.mark.parametrize("mode,lo,hi", [("moonlit", 3800, 4300), ("starlight", 3800, 4050)])
def test_ambient_colour_temperature_stays_in_its_configured_band(cfg, mode, lo, hi):
    generator = torch.Generator().manual_seed(2)
    temps = [sample_lighting_params(8, 8, cfg, mode, generator).ambient_temp for _ in range(50)]
    assert lo <= min(temps) and max(temps) <= hi


@skip_until_implemented
def test_starlight_is_redder_than_moonlight(cfg):
    generator = torch.Generator().manual_seed(3)
    def mean_temp(mode):
        return sum(
            sample_lighting_params(8, 8, cfg, mode, generator).ambient_temp for _ in range(200)
        ) / 200
    assert mean_temp("starlight") < mean_temp("moonlit")


@skip_until_implemented
def test_warm_ambient_tints_the_scene_red(cfg):
    generator = torch.Generator().manual_seed(4)
    params = sample_lighting_params(16, 16, cfg, "moonlit", generator)
    lit = apply_lighting(torch.ones(3, 16, 16), params)
    assert lit[0].mean() > lit[1].mean() > lit[2].mean()
    assert torch.allclose(params.ambient_gains, color_temp_to_gains(params.ambient_temp))
    assert float(params.ambient_gains[1]) == pytest.approx(1.0)


@skip_until_implemented
def test_only_streetlit_places_lamps(cfg):
    generator = torch.Generator().manual_seed(5)
    for mode in ("moonlit", "starlight"):
        for _ in range(10):
            assert sample_lighting_params(16, 16, cfg, mode, generator).num_lights == 0
    counts = {sample_lighting_params(16, 16, cfg, "streetlit", generator).num_lights
              for _ in range(100)}
    assert max(counts) <= cfg["num_lights_range"][1]
    assert min(counts) >= cfg["num_lights_range"][0]
    assert len(counts) > 1, "the lamp count should actually vary"


@skip_until_implemented
def test_lamp_illuminance_follows_the_inverse_square_cosine_law(cfg):
    """E(r) = peak * h^3 / (h^2 + r^2)^1.5, measured off the rendered field itself."""
    generator = torch.Generator().manual_seed(6)
    params = sample_lighting_params(256, 256, cfg, "streetlit", generator)
    height, peak = 0.25, 20.0
    params.light_xy = torch.tensor([[0.5, 0.5]])
    params.light_height = torch.tensor([height])
    params.light_peak = torch.tensor([peak])
    params.light_temp = torch.tensor([6500.0])
    params.light_gains = torch.ones(1, 3)
    params.illum_field = torch.ones(256, 256)
    params.ambient_gains = torch.ones(3)

    field = illumination_field(params)[1]  # green, where every gain is 1
    centre = field[128, 128] - 1.0  # subtract the ambient
    assert float(centre) == pytest.approx(peak, rel=0.02)
    for offset in (16, 48, 96):
        r = offset / 256.0
        measured = float(field[128, 128 + offset] - 1.0)
        expected = peak * height**3 / (height**2 + r**2) ** 1.5
        assert measured == pytest.approx(expected, rel=0.02), f"at r = {r:.3f}"


@skip_until_implemented
def test_lamps_are_far_brighter_than_the_ambient(cfg):
    generator = torch.Generator().manual_seed(7)
    for _ in range(30):
        params = sample_lighting_params(16, 16, cfg, "streetlit", generator)
        if params.num_lights:
            assert params.light_peak.min() >= cfg["light_peak_range"][0]
            assert params.light_peak.max() <= cfg["light_peak_range"][1]


@skip_until_implemented
def test_lamp_colour_is_sodium_or_led(cfg):
    generator = torch.Generator().manual_seed(8)
    temps = torch.cat(
        [sample_lighting_params(16, 16, cfg, "streetlit", generator).light_temp
         for _ in range(60)]
    )
    sodium_lo, sodium_hi = cfg["sodium_temp_range"]
    led_lo, led_hi = cfg["led_temp_range"]
    in_band = ((temps >= sodium_lo) & (temps <= sodium_hi)) | (
        (temps >= led_lo) & (temps <= led_hi)
    )
    assert bool(in_band.all())
    assert bool((temps <= sodium_hi).any()) and bool((temps >= led_lo).any())


# --- choosing source images ----------------------------------------------------------------

def _flat_image():
    return torch.full((3, 64, 64), 0.4) + torch.randn(3, 64, 64) * 0.01


def _contrasty_image():
    image = torch.full((3, 64, 64), 0.01)
    image[:, :32] = 0.95  # hard sun / shadow split
    return image


@skip_until_implemented
def test_overcast_score_separates_flat_from_contrasty():
    assert overcast_score(_flat_image()) < 2.0
    assert overcast_score(_contrasty_image()) > 20.0


@skip_until_implemented
def test_a_contrasty_source_is_never_streetlit_or_starlit(cfg):
    generator = torch.Generator().manual_seed(9)
    image = _contrasty_image()
    chosen = {choose_mode(image, cfg, generator) for _ in range(200)}
    assert chosen == {"moonlit"}


@skip_until_implemented
def test_a_flat_source_can_be_any_mode(cfg):
    generator = torch.Generator().manual_seed(10)
    image = _flat_image()
    chosen = {choose_mode(image, cfg, generator) for _ in range(300)}
    assert chosen == set(MODES)


@skip_until_implemented
def test_apply_lighting_rejects_a_size_mismatch(cfg):
    params = sample_lighting_params(16, 16, cfg, "moonlit")
    with pytest.raises(ValueError):
        apply_lighting(torch.rand(3, 32, 32), params)
    with pytest.raises(ValueError):
        apply_lighting(torch.rand(1, 3, 16, 16), params)


@skip_until_implemented
def test_sample_lighting_params_rejects_an_unknown_mode(cfg):
    with pytest.raises(ValueError):
        sample_lighting_params(8, 8, cfg, "daylight")


# --- emitters & streetlit lamps -------------------------------------------------------------


@skip_until_implemented
def test_streetlit_always_has_lamps(cfg):
    """Streetlit must always have at least 1 lamp, and lamp peaks dominate ambient."""
    generator = torch.Generator().manual_seed(42)
    for _ in range(50):
        params = sample_lighting_params(32, 48, cfg, "streetlit", generator)
        assert params.num_lights >= 1
        assert params.light_peak.min() >= cfg["light_peak_range"][0]
        assert params.light_peak.max() <= cfg["light_peak_range"][1]


@skip_until_implemented
def test_emission_non_negative_and_absent_from_day(cfg):
    """Emission field is non-negative and is deliberately absent from the day reflectance."""
    generator = torch.Generator().manual_seed(43)
    for mode in MODES:
        params = sample_lighting_params(64, 64, cfg, mode, generator)
        em = emission_field(params)
        assert em.min() >= 0.0

        day = torch.rand(3, 64, 64)
        day_clone = day.clone()
        radiance = apply_lighting(day, params)
        # Day input tensor is never modified in-place
        assert torch.equal(day, day_clone)
        # Scene radiance is reflectance * illumination + emission
        illum = illumination_field(params)
        diff = radiance - (day * illum)
        assert diff.min() >= -1e-6
        if params.num_emitters > 0:
            assert diff.max() > 0.0


@skip_until_implemented
def test_emitter_counts_per_mode_respected(cfg):
    """Sampled counts of windows, headlights, taillights, signs respect configured ranges."""
    generator = torch.Generator().manual_seed(44)
    for mode in MODES:
        counts_cfg = cfg.get("emitter_counts", {}).get(mode, {})
        for _ in range(30):
            params = sample_lighting_params(64, 64, cfg, mode, generator)
            for key in ("windows", "headlights", "taillights", "signs"):
                lo, hi = counts_cfg.get(key, (0, 0))
                actual = params.emitter_counts.get(key, 0)
                assert lo <= actual <= hi
        if mode == "starlight":
            assert params.num_emitters == 0


@skip_until_implemented
def test_emitter_determinism(cfg):
    """RNG seed reproducibility: same generator seed produces identical emitter parameters."""
    for mode in MODES:
        g1 = torch.Generator().manual_seed(101)
        g2 = torch.Generator().manual_seed(101)
        p1 = sample_lighting_params(64, 64, cfg, mode, generator=g1)
        p2 = sample_lighting_params(64, 64, cfg, mode, generator=g2)
        assert p1.num_emitters == p2.num_emitters
        assert p1.emitter_counts == p2.emitter_counts
        for e1, e2 in zip(p1.emitters, p2.emitters, strict=True):
            assert e1.shape == e2.shape
            assert e1.emitter_type == e2.emitter_type
            assert e1.x == e2.x and e1.y == e2.y
            assert e1.w == e2.w and e1.h == e2.h
            assert e1.core_peak == e2.core_peak
            assert e1.halo_peak == e2.halo_peak
            assert e1.halo_sigma == e2.halo_sigma
            assert torch.equal(e1.color, e2.color)
        em1 = emission_field(p1)
        em2 = emission_field(p2)
        assert torch.equal(em1, em2)


@skip_until_implemented
def test_emitter_energy_resolution_independence(cfg):
    """Area coverage preserves total emitted energy per unit area across resolutions within ~5%."""
    generator = torch.Generator().manual_seed(202)
    params = sample_lighting_params(256, 256, cfg, "streetlit", generator)
    assert params.num_emitters > 0

    em256 = emission_field(params, height=256, width=256)
    em1024 = emission_field(params, height=1024, width=1024)

    energy_256 = float(em256.mean())
    energy_1024 = float(em1024.mean())
    rel_diff = abs(energy_256 - energy_1024) / energy_1024
    assert rel_diff < 0.05, f"Energy mismatch: {rel_diff * 100:.2f}% (exceeds 5%)"

