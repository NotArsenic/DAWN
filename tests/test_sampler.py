"""The ODE samplers: NFE accounting, the un-standardization, and the K-sample spread.

NFE is the number of velocity evaluations. Euler spends one per step and midpoint two, so a
2-NFE midpoint run is ONE step and costs the same as two Euler steps. Getting that wrong would
make every "at 2 NFE" comparison in the paper a lie, so it is pinned here by counting calls.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow.sampler import (  # noqa: E402
    euler_sample,
    integrate,
    midpoint_sample,
    nfe_for_steps,
    render_spread,
    sample_grid,
    steps_for_nfe,
)
from dawn.grid.layout import IDENTITY_COEFFS  # noqa: E402
from dawn.grid.normalize import standardize, unstandardize  # noqa: E402

B, K, GY, GX = 2, 8, 16, 16


class CountingVelocity:
    """Records every call, and returns a constant field so the integral has a known answer."""

    def __init__(self, value=1.0):
        self.value = value
        self.times: list[float] = []

    def __call__(self, z_t, t):
        self.times.append(float(t[0]))
        return torch.full_like(z_t, self.value)

    @property
    def nfe(self) -> int:
        return len(self.times)


@pytest.fixture
def z_0():
    return torch.zeros(B, 12, K, GY, GX)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_euler_spends_one_evaluation_per_step(z_0, steps):
    fn = CountingVelocity()
    euler_sample(fn, z_0, num_steps=steps)
    assert fn.nfe == steps == nfe_for_steps("euler", steps)


@pytest.mark.parametrize("steps", [1, 2])
def test_midpoint_spends_two_evaluations_per_step(z_0, steps):
    fn = CountingVelocity()
    midpoint_sample(fn, z_0, num_steps=steps)
    assert fn.nfe == 2 * steps == nfe_for_steps("midpoint", steps)


def test_midpoint_at_two_nfe_is_a_single_step(z_0):
    assert steps_for_nfe("midpoint", 2) == 1
    assert steps_for_nfe("euler", 2) == 2
    fn = CountingVelocity()
    integrate(fn, z_0, method="midpoint", nfe=2)
    assert fn.nfe == 2
    assert fn.times == [0.0, 0.5], fn.times


def test_euler_at_two_nfe_is_two_steps(z_0):
    fn = CountingVelocity()
    integrate(fn, z_0, method="euler", nfe=2)
    assert fn.times == [0.0, 0.5]


def test_midpoint_cannot_run_at_an_odd_nfe():
    with pytest.raises(ValueError, match="NFE=3"):
        steps_for_nfe("midpoint", 3)
    with pytest.raises(ValueError):
        steps_for_nfe("euler", 0)
    with pytest.raises(ValueError, match="method"):
        steps_for_nfe("heun", 2)


@pytest.mark.parametrize("method,nfe", [("euler", 1), ("euler", 2), ("euler", 4),
                                        ("midpoint", 2), ("midpoint", 4)])
def test_a_constant_unit_field_integrates_to_one(z_0, method, nfe):
    """Both solvers are exact on dz/dt = 1, whatever the step count."""
    out = integrate(CountingVelocity(1.0), z_0, method=method, nfe=nfe)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_samplers_integrate_from_noise_to_data(z_0):
    """t runs 0 -> 1 and the step is +dt*v, so the field is evaluated at increasing times."""
    fn = CountingVelocity()
    euler_sample(fn, z_0, num_steps=4)
    assert fn.times == sorted(fn.times) and fn.times[0] == 0.0 and fn.times[-1] < 1.0


# --- the full entry point ---------------------------------------------------------------


class ConstantGridModel(torch.nn.Module):
    """A velocity model whose integral lands exactly on a chosen standardized grid."""

    def __init__(self, target):
        super().__init__()
        self.register_buffer("target", target)
        self.calls = 0

    def forward(self, z_t, t, features, attn=None, thumb=None, bias=None):
        self.calls += 1
        return (self.target.to(z_t.device) - z_t) / (1.0 - t.view(-1, 1, 1, 1, 1)).clamp(min=1e-3)


def test_sample_grid_returns_an_unstandardized_grid():
    mean = torch.randn(12, K)
    std = torch.rand(12, K) + 0.5
    want_raw = torch.randn(1, 12, K, GY, GX).expand(B, 12, K, GY, GX).contiguous()
    model = ConstantGridModel(standardize(want_raw, mean, std))

    out = sample_grid(model, torch.randn(B, 256, 384), mean, std, nfe=1)
    assert out.shape == (1, B, 12, K, GY, GX)
    assert torch.allclose(out[0], want_raw, atol=1e-4)


def test_sample_grid_runs_the_velocity_net_nfe_times_and_the_encoder_never():
    model = ConstantGridModel(torch.zeros(B, 12, K, GY, GX))
    sample_grid(model, torch.randn(B, 256, 384), torch.zeros(12, K), torch.ones(12, K),
                method="midpoint", nfe=2)
    assert model.calls == 2


def test_k_samples_are_batched_into_one_pass():
    """K members cost K times the batch, not K times the number of velocity calls."""
    model = ConstantGridModel(torch.zeros(1, 12, K, GY, GX))
    out = sample_grid(model, torch.randn(B, 256, 384), torch.zeros(12, K), torch.ones(12, K),
                      nfe=2, num_samples=5)
    assert out.shape == (5, B, 12, K, GY, GX)
    assert model.calls == 2


class ConstantFieldModel(torch.nn.Module):
    """dz/dt = 1, so the result is z_0 + 1 and depends on the noise draw."""

    def forward(self, z_t, t, features, attn=None, thumb=None, bias=None):
        return torch.ones_like(z_t)


def test_the_sample_is_reproducible_from_its_generator():
    model = ConstantFieldModel()
    feats = torch.randn(B, 256, 384)
    args = (model, feats, torch.zeros(12, K), torch.ones(12, K))
    first = sample_grid(*args, nfe=2, generator=torch.Generator().manual_seed(7))
    second = sample_grid(*args, nfe=2, generator=torch.Generator().manual_seed(7))
    other = sample_grid(*args, nfe=2, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_standardize_round_trip_through_the_sampler_is_exact():
    mean, std = torch.randn(12, K), torch.rand(12, K) + 0.1
    raw = torch.randn(B, 12, K, GY, GX)
    assert torch.allclose(unstandardize(standardize(raw, mean, std), mean, std), raw, atol=1e-5)


def test_render_spread_is_zero_for_one_sample_and_positive_for_disagreement():
    image = torch.rand(B, 3, 32, 32) * 0.3
    identity = torch.tensor(IDENTITY_COEFFS).view(1, 1, 12, 1, 1, 1).expand(
        1, B, 12, K, GY, GX
    ).contiguous()
    mean, spread = render_spread(identity, image)
    assert mean.shape == image.shape == spread.shape
    assert torch.allclose(mean, image, atol=1e-6)
    assert float(spread.max()) == 0.0

    pair = torch.cat([identity, identity * 1.5], dim=0)
    mean, spread = render_spread(pair, image)
    assert float(spread.max()) > 0.0


def test_render_spread_rejects_a_grid_without_the_sample_axis():
    with pytest.raises(ValueError):
        render_spread(torch.zeros(B, 12, K, GY, GX), torch.rand(B, 3, 8, 8))
