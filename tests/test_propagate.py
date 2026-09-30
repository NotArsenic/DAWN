"""docs/conventions.md sections 2.2 and 6: coefficient order and the noise model."""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.layout import IDENTITY_COEFFS  # noqa: E402
from dawn.uncertainty.propagate import propagate_variance, sensor_variance  # noqa: E402


def _coeffs(values, batch=2, size=4):
    return torch.tensor(values, dtype=torch.float32).view(1, 12, 1, 1).expand(batch, 12, size, size)


def test_identity_keeps_variance():
    sigma = torch.rand(2, 3, 4, 4)
    assert torch.allclose(propagate_variance(_coeffs(IDENTITY_COEFFS), sigma), sigma)


def test_matrix_row_is_the_output_channel_and_offsets_are_ignored():
    values = [0.0] * 12
    values[1] = 2.0  # M[0, 1]: input G feeds output R
    values[9:12] = [5.0, 6.0, 7.0]
    sigma = torch.rand(2, 3, 4, 4)
    out = propagate_variance(_coeffs(values), sigma)
    assert torch.allclose(out[:, 0], 4.0 * sigma[:, 1])
    assert torch.count_nonzero(out[:, 1:]) == 0


def test_sensor_variance_is_a_x_plus_b_with_b_a_variance():
    x = torch.rand(2, 3, 4, 4)
    a, b = torch.tensor([1e-3, 2e-3]), torch.tensor([1e-5, 4e-5])
    expected = a.view(2, 1, 1, 1) * x + b.view(2, 1, 1, 1)
    assert torch.allclose(sensor_variance(x, a, b), expected)
