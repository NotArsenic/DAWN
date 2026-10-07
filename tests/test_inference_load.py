"""Loading a trained checkpoint for inference (dawn/flow/inference.py).

The rule this pins is docs/conventions.md section 5.1: the stats come out of the CHECKPOINT. A
run that read `grid.stats_cache` instead would unstandardize with whatever the cache happens to
hold today, which is a silent, plausible-looking wrongness rather than a crash.
"""

import pytest

torch = pytest.importorskip("torch")

pytest.importorskip("omegaconf")
pytest.importorskip("peft")
pytest.importorskip("timm")

from dawn.flow.inference import load_checkpoint  # noqa: E402
from dawn.flow.targets import FitSettings, compute_token_stats, prepare_batch  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import FlowTrainer  # noqa: E402
from tests.flow_helpers import TINY_INPUT, synthetic_batch, tiny_cfg, wake_up  # noqa: E402


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
