"""Shared pytest setup."""

import pytest


@pytest.fixture(autouse=True)
def _seed():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
