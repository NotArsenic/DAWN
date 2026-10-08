"""Loading a trained checkpoint for inference (dawn/flow/inference.py).

The rule this pins is docs/conventions.md section 5.1: the stats come out of the CHECKPOINT. A
run that read `grid.stats_cache` instead would unstandardize with whatever the cache happens to
hold today, which is a silent, plausible-looking wrongness rather than a crash.

Stage 2 is a separate run with a separate checkpoint, so `load_refiner` is tested separately
too, including that the two loaders refuse each other's files: a Stage 1 checkpoint has no
`refiner` in it, and handing one over has to say so rather than build a random refiner.
"""

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("omegaconf")
pytest.importorskip("peft")
pytest.importorskip("timm")

from dawn.flow.inference import load_checkpoint, load_refiner  # noqa: E402
from dawn.flow.targets import FitSettings, compute_token_stats, prepare_batch  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import FlowTrainer  # noqa: E402
from dawn.models.refiner import LocalRefiner  # noqa: E402
from tests.flow_helpers import TINY_INPUT, synthetic_batch, tiny_cfg, wake_up  # noqa: E402

# Nothing like the defaults (24, 2, 5, 0.1), so a load that ignored `refiner_config` and built
# a default refiner would be caught by the shapes rather than pass quietly.
REFINER_CFG = {
    "hidden_channels": 4,
    "num_blocks": 1,
    "kernel_size": 3,
    "residual_scale": 0.05,
}


def _stats(cfg, batches):
    prepared = [prepare_batch(b, TINY_INPUT, cfg["pairs"]) for b in batches]
    return compute_token_stats(prepared, TokenLayout.from_config(cfg), FitSettings.from_config(cfg))


def _saved(tmp_path, correction=False, steps=1, name="ckpt.pt"):
    """A trainer that has taken a step, and the checkpoint it wrote."""
    torch.manual_seed(0)
    cfg = tiny_cfg(correction)
    batch = synthetic_batch(2, seed=0, fallback=[False, True])
    trainer = FlowTrainer.from_config(cfg, _stats(cfg, [batch]), "cpu")
    for _ in range(steps):
        trainer.train_step(batch)
    return trainer, trainer.save_checkpoint(tmp_path / name)


@pytest.fixture(params=[False, True], ids=["correction_off", "correction_on"])
def correction(request):
    return request.param


def test_round_trip_gives_back_the_layout_stats_and_sizes(tmp_path, correction):
    trainer, path = _saved(tmp_path, correction)
    bundle = load_checkpoint(path, "cpu", use_ema=False)

    assert bundle.layout.descriptor() == trainer.layout.descriptor()
    assert bundle.layout.correction is correction
    assert bundle.input_size == TINY_INPUT
    assert bundle.render_size == tuple(trainer.render_size)
    assert bundle.step == trainer.step
    mean, std = trainer.layout.stats_vector(trainer.stats)
    assert torch.equal(bundle.mean, mean) and torch.equal(bundle.std, std)
    assert bundle.mean.shape == (bundle.layout.dim,)


def test_the_stats_come_from_the_checkpoint_not_the_cache(tmp_path):
    """Two checkpoints with different stats must load as different stats, same config."""
    trainer, path = _saved(tmp_path)
    bumped = trainer.state_dict()
    bumped["stats"] = {
        name: {kind: value + 1.0 for kind, value in seg.items()}
        for name, seg in trainer.stats.items()
    }
    torch.save(bumped, tmp_path / "bumped.pt")

    first = load_checkpoint(path, "cpu", use_ema=False)
    second = load_checkpoint(tmp_path / "bumped.pt", "cpu", use_ema=False)
    assert torch.allclose(second.mean, first.mean + 1.0)
    assert not torch.equal(first.mean, second.mean)


def test_weights_match_the_trainer_and_the_models_are_frozen(tmp_path):
    trainer, path = _saved(tmp_path)
    bundle = load_checkpoint(path, "cpu", use_ema=False)
    live = trainer.velocity.state_dict()
    for name, value in bundle.velocity.state_dict().items():
        assert torch.equal(value, live[name]), name
    assert not bundle.velocity.training and not bundle.encoder.training
    assert not any(p.requires_grad for p in bundle.velocity.parameters())
    assert not any(p.requires_grad for p in bundle.encoder.parameters())


def test_use_ema_applies_the_shadow_and_changes_the_weights(tmp_path):
    """After a step the EMA shadow is not the live weights, so the two loads must differ."""
    trainer, path = _saved(tmp_path, steps=2)
    live = load_checkpoint(path, "cpu", use_ema=False)
    ema = load_checkpoint(path, "cpu", use_ema=True)
    assert ema.ema and not live.ema

    shadow = trainer.ema.shadow
    differs = False
    for name, param in ema.velocity.named_parameters():
        key = f"velocity.{name}"
        if key in shadow:
            assert torch.allclose(param, shadow[key].to(param.dtype), atol=1e-6), name
            differs |= not torch.equal(param, live.velocity.state_dict()[name])
    assert differs, "the EMA shadow should not equal the live weights after two steps"


def test_a_checkpoint_without_an_ema_shadow_says_so(tmp_path):
    trainer, path = _saved(tmp_path)
    state = trainer.state_dict()
    state["ema"] = {"decay": 0.999, "shadow": {}}
    torch.save(state, tmp_path / "no_ema.pt")
    with pytest.raises(ValueError, match="no EMA shadow"):
        load_checkpoint(tmp_path / "no_ema.pt", "cpu", use_ema=True)
    assert load_checkpoint(tmp_path / "no_ema.pt", "cpu", use_ema=False).step == trainer.step


def test_a_file_that_is_not_a_dawn_checkpoint_is_a_clear_error(tmp_path):
    torch.save({"velocity": {}, "something": 1}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a Dawn flow checkpoint"):
        load_checkpoint(tmp_path / "other.pt", "cpu")
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        load_checkpoint(tmp_path / "missing.pt", "cpu")


def test_the_bundle_renders_through_sample_with_spread(tmp_path, correction):
    """End to end on the loaded bundle: encode, K samples, maps, at the checkpoint's own size."""
    from dawn.uncertainty.spread import sample_with_spread

    _, path = _saved(tmp_path, correction)
    bundle = load_checkpoint(path, "cpu", use_ema=True)
    wake_up(bundle.velocity, std=0.1, seed=3)  # random init would give a near-constant field
    thermal = torch.rand(1, 1, *bundle.input_size) * 0.8 + 0.1

    encoded = bundle.encoder.encode(thermal, return_attn=True)
    out = sample_with_spread(
        bundle.velocity,
        encoded,
        bundle.layout,
        bundle.mean,
        bundle.std,
        thermal,
        bundle.input_size,
        bundle.settings,
        num_samples=3,
        nfe=2,
        image_ids=["frame"],
    )
    assert out.sample_rgb.shape == (1, 3, *bundle.input_size)
    assert out.l_std.shape == (1, 1, *bundle.input_size)
    assert torch.all(torch.isfinite(out.l_std)) and torch.all(torch.isfinite(out.ab_std))
    assert out.rgb.min() >= 0.0 and out.rgb.max() <= 1.0


# --- stage 2: the local refiner ---------------------------------------------------------------


def _refiner(**updates) -> LocalRefiner:
    """A tiny refiner whose zero-initialized head has been perturbed: "a step happened"."""
    return wake_up(LocalRefiner(**{**REFINER_CFG, **updates}), std=0.2, seed=4)


def _shadow(refiner: LocalRefiner, offset: float = 1.0, prefix: str = "refiner.") -> dict:
    """The EMA shadow the Stage 2 trainer writes: float32, keyed `refiner.<param>`.

    `offset` is what separates the average from the live weights here, so a load that forgot to
    apply the shadow cannot pass by accident.
    """
    return {
        f"{prefix}{name}": param.detach().clone().float() + offset
        for name, param in refiner.named_parameters()
    }


def _save_refiner(path, refiner: LocalRefiner, shadow=None, step: int = 11):
    """The pinned Stage 2 checkpoint, built here rather than by running the Stage 2 trainer.

    Synthesized on purpose: this file tests the loader, and depending on `scripts/train_refiner.py`
    would make a loader bug and a trainer bug fail the same test.
    """
    torch.save(
        {
            "step": step,
            "config": {"refiner": dict(REFINER_CFG), "optim": {"ema_decay": 0.999}},
            "refiner": refiner.state_dict(),
            "refiner_config": dict(REFINER_CFG),
            "ema": {
                "decay": 0.999,
                "warmup": True,
                "num_updates": 20,
                "shadow": {} if shadow is None else shadow,
            },
            "optimizer": {},
            "scheduler": {},
        },
        path,
    )
    return path


def test_load_refiner_round_trips_the_model_and_freezes_it(tmp_path):
    refiner = _refiner()
    path = _save_refiner(tmp_path / "refiner.pt", refiner, _shadow(refiner))
    loaded = load_refiner(path, "cpu", use_ema=False)

    assert loaded.hidden_channels == REFINER_CFG["hidden_channels"]
    assert loaded.num_blocks == REFINER_CFG["num_blocks"]
    assert loaded.kernel_size == REFINER_CFG["kernel_size"]
    assert loaded.residual_scale == REFINER_CFG["residual_scale"]

    live = refiner.state_dict()
    assert set(loaded.state_dict()) == set(live)
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, live[name]), name
    assert not loaded.training
    assert not any(p.requires_grad for p in loaded.parameters())

    lightness, thermal = torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8)
    with torch.no_grad():
        delta = refiner.residual(lightness, thermal)
        assert torch.equal(loaded.residual(lightness, thermal), delta)


def test_the_refiner_ema_shadow_is_applied_only_when_asked(tmp_path):
    refiner = _refiner()
    path = _save_refiner(tmp_path / "ema.pt", refiner, _shadow(refiner, offset=1.0))
    averaged = dict(load_refiner(path, "cpu", use_ema=True).named_parameters())
    live = dict(load_refiner(path, "cpu", use_ema=False).named_parameters())

    for name, param in refiner.named_parameters():
        assert torch.equal(live[name], param), name
        assert torch.allclose(averaged[name], param + 1.0, atol=1e-6), name


def test_a_refiner_checkpoint_without_a_usable_shadow_says_so(tmp_path):
    """Asking for the average and silently getting the live weights is the bad outcome."""
    refiner = _refiner()
    empty = _save_refiner(tmp_path / "no_ema.pt", refiner, shadow={})
    with pytest.raises(ValueError, match="no EMA shadow"):
        load_refiner(empty, "cpu", use_ema=True)
    assert load_refiner(empty, "cpu", use_ema=False).num_params() == refiner.num_params()

    # A shadow that is there, but under another module's prefix: nothing of the refiner is in it.
    wrong = _save_refiner(tmp_path / "wrong.pt", refiner, _shadow(refiner, prefix="velocity."))
    with pytest.raises(ValueError, match="matched none of the refiner"):
        load_refiner(wrong, "cpu", use_ema=True)


def test_the_two_stages_refuse_each_others_checkpoints(tmp_path):
    refiner = _refiner()
    stage_2 = _save_refiner(tmp_path / "refiner.pt", refiner, _shadow(refiner))
    _, stage_1 = _saved(tmp_path)

    with pytest.raises(ValueError, match="not a Dawn refiner checkpoint"):
        load_refiner(stage_1, "cpu")
    with pytest.raises(ValueError, match="not a Dawn flow checkpoint"):
        load_checkpoint(stage_2, "cpu")
    with pytest.raises(FileNotFoundError, match="no checkpoint at"):
        load_refiner(tmp_path / "missing.pt", "cpu")
