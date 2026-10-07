"""The ODE samplers: NFE accounting, the one un-standardization, and seeded noise.

NFE is the number of velocity evaluations. Euler spends one per step and midpoint two, so a 2-NFE
midpoint run is ONE step and costs the same as two Euler steps. Getting that wrong would make
every "at 2 NFE" comparison a lie, so it is pinned here by counting calls.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow import sampler as sampler_mod  # noqa: E402
from dawn.flow.sampler import (  # noqa: E402
    euler_sample,
    integrate,
    midpoint_sample,
    nfe_for_steps,
    noise_for_ids,
    sample_parts,
    steps_for_nfe,
)
from dawn.grid.normalize import standardize  # noqa: E402
from tests.flow_helpers import tiny_layout  # noqa: E402

B = 2


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
    return torch.zeros(B, 20, 48)


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


@pytest.mark.parametrize(
    "method,nfe", [("euler", 1), ("euler", 2), ("euler", 4), ("midpoint", 2), ("midpoint", 4)]
)
def test_a_constant_unit_field_integrates_to_one(z_0, method, nfe):
    """Both solvers are exact on dz/dt = 1, whatever the step count."""
    out = integrate(CountingVelocity(1.0), z_0, method=method, nfe=nfe)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_samplers_integrate_from_noise_to_data(z_0):
    """t runs 0 -> 1 and the step is +dt*v, so the field is evaluated at increasing times."""
    fn = CountingVelocity()
    euler_sample(fn, z_0, num_steps=4)
    assert fn.times == sorted(fn.times) and fn.times[0] == 0.0 and fn.times[-1] < 1.0


# --- the full entry point ---------------------------------------------------------------------


class ConstantTokensModel(torch.nn.Module):
    """A velocity model whose integral lands exactly on a chosen standardized token tensor."""

    def __init__(self, target):
        super().__init__()
        self.register_buffer("target", target)
        self.calls = 0
        self.batch_sizes = []

    def forward(self, z_t, t, features, attn=None, thumb=None, bias=None):
        self.calls += 1
        self.batch_sizes.append(z_t.shape[0])
        return (self.target.to(z_t.device) - z_t) / (1.0 - t.view(-1, 1, 1)).clamp(min=1e-3)


class ConstantFieldModel(torch.nn.Module):
    """dz/dt = 1, so the result is z_0 + 1 and depends on the noise draw."""

    def forward(self, z_t, t, features, attn=None, thumb=None, bias=None):
        return torch.ones_like(z_t)


def _stats(layout):
    return torch.randn(layout.dim), torch.rand(layout.dim) + 0.5


@pytest.mark.parametrize("correction", [False, True])
def test_sample_parts_returns_raw_parts_with_a_sample_axis(correction):
    layout = tiny_layout(correction)
    mean, std = _stats(layout)
    raw = torch.randn(1, layout.num_tokens, layout.dim).expand(B, -1, -1).contiguous()
    model = ConstantTokensModel(standardize(raw, mean, std))
    feats = torch.randn(B, layout.num_tokens, 384)
    out = sample_parts(model, feats, layout, mean, std, nfe=1)
    expected = layout.unpack(raw)
    assert set(out) == set(expected)
    for name, value in expected.items():
        assert out[name].shape == (1, *value.shape), name
        assert torch.allclose(out[name][0], value, atol=1e-4), name


def test_sample_parts_unstandardizes_exactly_once(monkeypatch):
    layout = tiny_layout()
    calls = []
    real = sampler_mod.unstandardize

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(sampler_mod, "unstandardize", spy)
    sample_parts(
        ConstantFieldModel(),
        torch.randn(B, layout.num_tokens, 384),
        layout,
        torch.zeros(layout.dim),
        torch.ones(layout.dim),
        method="midpoint",
        nfe=4,
        num_samples=3,
    )
    assert len(calls) == 1


def test_the_velocity_net_runs_nfe_times_and_the_encoder_never():
    layout = tiny_layout()
    model = ConstantTokensModel(torch.zeros(B, layout.num_tokens, layout.dim))
    sample_parts(
        model,
        torch.randn(B, layout.num_tokens, 384),
        layout,
        torch.zeros(layout.dim),
        torch.ones(layout.dim),
        method="midpoint",
        nfe=2,
    )
    assert model.calls == 2


def test_k_samples_are_batched_into_one_pass():
    """K members cost K times the batch, not K times the number of velocity calls."""
    layout = tiny_layout()
    model = ConstantTokensModel(torch.zeros(1, layout.num_tokens, layout.dim))
    out = sample_parts(
        model,
        torch.randn(B, layout.num_tokens, 384),
        layout,
        torch.zeros(layout.dim),
        torch.ones(layout.dim),
        nfe=2,
        num_samples=5,
    )
    assert out["grid"].shape[:2] == (5, B) and out["ab"].shape[:2] == (5, B)
    assert model.calls == 2 and model.batch_sizes == [5 * B, 5 * B]


def test_the_sample_is_reproducible_from_its_generator():
    layout = tiny_layout()
    feats = torch.randn(B, layout.num_tokens, 384)
    args = (ConstantFieldModel(), feats, layout, torch.zeros(layout.dim), torch.ones(layout.dim))
    first = sample_parts(*args, nfe=2, generator=torch.Generator().manual_seed(7))
    second = sample_parts(*args, nfe=2, generator=torch.Generator().manual_seed(7))
    other = sample_parts(*args, nfe=2, generator=torch.Generator().manual_seed(8))
    assert torch.equal(first["ab"], second["ab"]) and not torch.equal(first["ab"], other["ab"])


def test_the_noise_of_an_image_comes_from_its_id_and_not_from_its_batch():
    first = noise_for_ids(["train_000001", "train_000002"], 20, 48)
    swapped = noise_for_ids(["train_000002", "train_000001"], 20, 48)
    alone = noise_for_ids(["train_000002"], 20, 48)
    assert first.shape == (2, 20, 48)
    assert torch.equal(first[0], swapped[1]) and torch.equal(first[1], swapped[0])
    assert torch.equal(first[1], alone[0])
    assert not torch.equal(first[0], first[1])
    from dawn.data.seeding import generator_for

    manual = torch.randn(20, 48, generator=generator_for("train_000001", "val_z0"))
    assert torch.equal(first[0], manual)


def test_sample_parts_with_image_ids_is_independent_of_the_batch():
    layout = tiny_layout()
    g = torch.Generator().manual_seed(0)
    feats = torch.randn(2, layout.num_tokens, 384, generator=g)
    args = dict(layout=layout, mean=torch.zeros(layout.dim), std=torch.ones(layout.dim), nfe=2)
    both = sample_parts(ConstantFieldModel(), feats, image_ids=["a_1", "b_2"], **args)
    one = sample_parts(ConstantFieldModel(), feats[1:], image_ids=["b_2"], **args)
    assert torch.equal(both["ab"][0, 1], one["ab"][0, 0])
    members = sample_parts(
        ConstantFieldModel(), feats, image_ids=["a_1", "b_2"], num_samples=2, **args
    )
    assert torch.equal(members["ab"][0], both["ab"][0])  # member 0 is the single-sample noise
    assert not torch.equal(members["ab"][0], members["ab"][1])


def test_the_uncertainty_helper_of_the_old_regime_is_gone():
    """Sample-spread uncertainty is the inference phase's (dawn/uncertainty)."""
    assert not hasattr(sampler_mod, "render_spread")
