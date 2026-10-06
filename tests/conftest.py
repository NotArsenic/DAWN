"""Shared pytest setup."""

import pytest


@pytest.fixture(autouse=True)
def _seed():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def fake_dronevehicle(tmp_path_factory):
    """Raw fake DroneVehicle tree (tests/fake_dronevehicle.py): a FakeTree with ground truth.

    Read-only for tests: the pipeline stages write elsewhere (`fake_dv_config`).
    """
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    pytest.importorskip("scipy")
    from tests.fake_dronevehicle import make_fake_dronevehicle

    return make_fake_dronevehicle(tmp_path_factory.mktemp("dronevehicle") / "raw")
