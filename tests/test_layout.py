"""docs/conventions.md section 2: coefficient order. The token layout is `test_tokens.py`."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dawn.grid import layout  # noqa: E402


def test_coefficient_order():
    assert (layout.GAIN, layout.OFFSET, layout.NUM_COEFFS) == (0, 1, 2)


def test_output_channel_coefficients_are_gain_then_offset():
    assert layout.coeff_channels(0) == (0, 1)
    assert layout.coeff_channels(3) == (6, 7)


def test_grid_code_never_imports_from_flow():
    """The layout flows one way: flow/tokens.py imports grid/layout.py, never the reverse."""
    grid_dir = Path(layout.__file__).parent
    for path in grid_dir.glob("*.py"):
        assert "dawn.flow" not in path.read_text(), path.name
