"""Degradation contract: night_clean, night, (a, b), all linear RGB in [0, 1].

Skips while lighting.py / camera.py are still stubs.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.data import degrade  # noqa: E402
from tests.helpers import skip_until_implemented  # noqa: E402

DATA_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


@skip_until_implemented
def test_apply_degradation_returns_night_clean_night_and_noise_params():
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(DATA_CONFIG))["degradation"]
    day = torch.rand(3, 64, 64)
    params = degrade.sample_degradation_params(64, 64, cfg)
    night_clean, night, (a, b) = degrade.apply_degradation(day, params)

    for image in (night_clean, night):
        assert image.shape == (3, 64, 64)
        assert image.dtype == torch.float32
        assert image.min() >= 0.0 and image.max() <= 1.0
    assert a > 0 and b > 0
    assert night_clean.mean() < day.mean()
    assert not torch.equal(night, night_clean)
