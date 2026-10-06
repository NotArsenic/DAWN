"""Stage 1 end to end: shapes through the whole chain, determinism, resume, and an overfit.

PARKED: this tests the old night-regime trainer, which imported the deleted degradation glue
(`dawn.flow.dataset`) and `configs/camera_calib.yaml`. It is skipped until the flow phase
rebuilds the trainer on `AlignedPairs`; keep it as the reference for what to cover again.

Everything runs on the CPU with `configs/debug.yaml`. The encoder is a two-block randomly
initialized DINOv3 (`tests/flow_helpers.py`), so nothing is downloaded; the velocity net, the
loss, the grid fitting and the samplers are the real ones.
"""

import os

import pytest

pytest.skip(
    "parked: old night-regime trainer, rebuilt in the flow phase", allow_module_level=True
)

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")

from dawn.flow.dataset import (  # noqa: E402
    NightPairs,
    SyntheticDay,
    build_night_loaders,
    collate_night_pairs,
)
from dawn.flow.trainer import EMA, FlowTrainer, split_param_groups  # noqa: E402
from dawn.grid.fit import fit_grid  # noqa: E402
from dawn.grid.guidance import compute_guidance  # noqa: E402
from dawn.grid.normalize import compute_grid_stats, standardize, unstandardize  # noqa: E402
from dawn.grid.slice import apply_affine, slice_grid  # noqa: E402
from dawn.models.noise_head import NoiseHead  # noqa: E402
from dawn.models.velocity import VelocityTransformer  # noqa: E402
from tests.flow_helpers import load_configs, night_batch, tiny_encoder  # noqa: E402

RUN_SLOW = bool(os.environ.get("DAWN_SLOW_TESTS"))


@pytest.fixture(scope="module")
def configs():
    return load_configs("debug.yaml")


@pytest.fixture(scope="module")
def batch(configs):
    cfg, calib = configs
    return night_batch(2, cfg, calib)


@pytest.fixture(scope="module")
def stats(batch):
    return compute_grid_stats(
        [{"night_clean": batch["night_clean"][i], "day": batch["day"][i]} for i in range(2)],
        num_samples=2,
    )


def make_trainer(cfg, stats, seed=0, **overrides):
    torch.manual_seed(seed)
    cfg = {**cfg, **overrides}
    encoder = tiny_encoder("timm", depth=2)
    return FlowTrainer(
        cfg,
        encoder,
        VelocityTransformer.from_config(cfg["velocity"], cfg["grid"]),
        NoiseHead(dino_dim=encoder.embed_dim),
        stats,
        "cpu",
    )


# --- the data side ------------------------------------------------------------------------


def test_night_pairs_returns_the_documented_contract(configs):
    cfg, calib = configs
    pairs = NightPairs(SyntheticDay(num_images=2, size=320), cfg["degradation"], calib,
                       level="medium")
    item = pairs[0]
    for key in ("night", "night_clean", "day"):
        assert item[key].shape == (3, 256, 256)
        assert item[key].dtype == torch.float32
        assert 0.0 <= float(item[key].min()) and float(item[key].max()) <= 1.0
    assert item["noise_params"]["a"].shape == (3,)
    assert item["night_clean"].mean() < item["day"].mean()
    assert not torch.equal(item["night"], item["night_clean"])


def test_the_fixed_val_level_makes_a_sample_reproducible(configs):
    cfg, calib = configs
    pairs = NightPairs(SyntheticDay(num_images=1), cfg["degradation"], calib, level="medium")
    assert torch.equal(pairs[0]["night"], pairs[0]["night"])
    fresh = NightPairs(SyntheticDay(num_images=1), cfg["degradation"], calib, level=None)
    torch.manual_seed(0)
    first = fresh[0]["night"]
    assert not torch.equal(first, fresh[0]["night"])


def test_collate_keeps_the_noise_params_addressable(configs):
    cfg, calib = configs
    pairs = NightPairs(SyntheticDay(num_images=3), cfg["degradation"], calib, level="easy")
    collated = collate_night_pairs([pairs[i] for i in range(3)])
    assert collated["night"].shape == (3, 3, 256, 256)
    assert collated["noise_params"]["a"].shape == (3, 3)
    assert collated["noise_params"]["n"].shape == (3,)
    assert len(collated["image_id"]) == 3


def test_build_night_loaders_follows_the_debug_overlay(configs):
    cfg, calib = configs
    train, val = build_night_loaders(cfg, calib)
    assert len(train.dataset) == cfg["loader"]["synthetic_images"]
    assert len(val.dataset) == min(cfg["logging"]["num_val_samples"],
                                   cfg["loader"]["synthetic_images"])
    assert train.batch_size == cfg["optim"]["batch_size"]
    assert {item["image_id"] for item in train.dataset.day_source} .isdisjoint(
        {item["image_id"] for item in val.dataset.day_source}
    ), "val must not be the training images"


# --- shapes through the whole chain --------------------------------------------------------


def test_shapes_through_the_whole_chain(configs, batch, stats):
    """night -> encoder -> velocity -> grid -> sliced -> applied, every shape checked."""
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    night, night_clean, day = batch["night"], batch["night_clean"], batch["day"]
    assert night.shape == (2, 3, 256, 256)

    z_1, node_weights, guidance = trainer.fit_targets(night_clean, day)
    assert z_1.shape == (2, 12, 8, 16, 16)
    assert node_weights.shape == (2, 8, 16, 16)
    assert guidance.shape == (2, 1, 256, 256)

    cond = trainer.encoder.encode(night)
    assert cond["patch_tokens"].shape == (2, 256, 384)
    assert cond["attn_bias"].shape == (2, 1, 256, 256)
    assert cond["thumb"].shape == (2, 3, 16, 16)

    t = torch.rand(2)
    z_t = torch.randn_like(z_1)
    v = trainer.velocity(z_t, t, cond["patch_tokens"], None, cond["thumb"], cond["attn_bias"])
    assert v.shape == z_1.shape

    applied = trainer.render(z_t + (1 - t.view(-1, 1, 1, 1, 1)) * v, night_clean, guidance)
    assert applied.shape == (2, 3, 256, 256)
    assert float(applied.detach().min()) >= 0.0 and float(applied.detach().max()) <= 1.0
    assert trainer.noise_head(cond["cls"]).shape == (2, 2)


def test_the_standardized_target_renders_back_to_the_oracle(configs, batch, stats):
    """standardize -> unstandardize is exact, so z_1 renders exactly like the fitted grid."""
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    night_clean, day = batch["night_clean"], batch["day"]
    guidance = compute_guidance(night_clean)
    raw, _ = fit_grid(night_clean, day, guidance)
    z_1 = standardize(raw, trainer.mean, trainer.std)
    assert torch.allclose(unstandardize(z_1, trainer.mean, trainer.std), raw, atol=1e-4)
    assert torch.allclose(
        trainer.render(z_1, night_clean, guidance),
        apply_affine(slice_grid(raw, guidance), night_clean),
        atol=1e-5,
    )


def test_a_train_step_reports_every_loss_and_moves_the_weights(configs, batch, stats):
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    before = trainer.velocity.input_proj.weight.detach().clone()
    out = trainer.train_step(batch)
    for key in ("total", "velocity", "apply", "noise", "align", "grad_norm"):
        assert key in out and out[key] == out[key], key  # present and not NaN
    assert out["velocity"] > 0
    assert trainer.step == 1
    assert not torch.equal(before, trainer.velocity.input_proj.weight)


def test_the_conditioning_is_expanded_not_re_encoded(configs, batch, stats, monkeypatch):
    """`flow.timesteps_per_image` velocity samples must cost exactly one encoder pass."""
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    calls = {"encode": 0, "velocity": 0}
    real_encode, real_velocity = trainer.encoder.encode, trainer.velocity.forward

    def counted_encode(*a, **k):
        calls["encode"] += 1
        return real_encode(*a, **k)

    def counted_velocity(*a, **k):
        calls["velocity"] += 1
        assert a[0].shape[0] == trainer.timesteps_per_image * batch["night"].shape[0]
        return real_velocity(*a, **k)

    monkeypatch.setattr(trainer.encoder, "encode", counted_encode)
    monkeypatch.setattr(trainer.velocity, "forward", counted_velocity)
    trainer.train_step(batch)
    assert calls == {"encode": 1, "velocity": 1}


def test_the_same_seed_gives_an_identical_step(configs, batch, stats):
    cfg, _ = configs
    results = []
    for _ in range(2):
        trainer = make_trainer(cfg, stats, seed=5)
        torch.manual_seed(123)
        results.append(
            (trainer.train_step(batch), trainer.velocity.output_proj.weight.detach().clone())
        )
    assert results[0][0] == results[1][0]
    assert torch.equal(results[0][1], results[1][1])

    trainer = make_trainer(cfg, stats, seed=5)
    torch.manual_seed(124)
    assert trainer.train_step(batch) != results[0][0]


def test_warmup_detaches_the_conditioning_so_lora_only_sees_align(configs, batch, stats):
    """Before `flow.warmup_steps` the encoder must get gradient from L_align alone."""
    cfg, _ = configs
    flow = {**cfg["flow"], "warmup_steps": 1, "align_weight": 0.0}
    trainer = make_trainer(cfg, stats, flow=flow)
    trainer.train_step(batch)  # step 0: detached, and with align switched off
    lora = [p for n, p in trainer.encoder.named_parameters() if p.requires_grad]
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in lora)

    trainer.train_step(batch)  # step 1: past the warmup, the velocity net feeds back
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in lora)


def test_align_alone_trains_lora_during_warmup(configs, batch, stats):
    cfg, _ = configs
    trainer = make_trainer(cfg, stats, flow={**cfg["flow"], "warmup_steps": 10})
    trainer.train_step(batch)
    lora = [p for n, p in trainer.encoder.named_parameters() if p.requires_grad]
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0.0 for p in lora)


# --- optimizer, EMA, checkpoints ------------------------------------------------------------


def test_param_groups_separate_lora_and_skip_decay_on_gates_and_norms(configs, stats):
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    groups = split_param_groups(
        trainer.encoder, trainer.velocity, trainer.noise_head,
        lr=2e-4, lora_lr=1e-4, weight_decay=0.01,
    )
    by_name = {group["name"]: group for group in groups}
    assert by_name["lora_True"]["lr"] == 1e-4
    assert all(group["lr"] == 2e-4 for name, group in by_name.items() if name.startswith("main"))
    assert by_name["main_False"]["weight_decay"] == 0.0

    gates = {id(block.dino_gate) for block in trainer.velocity.blocks}
    decayed = {id(p) for name, group in by_name.items() if group["weight_decay"] > 0
               for p in group["params"]}
    assert not (gates & decayed), "the attention gates must not be weight-decayed"
    assert id(trainer.velocity.pos_embed) not in decayed
    biases = {id(p) for n, p in trainer.velocity.named_parameters() if n.endswith("bias")}
    assert not (biases & decayed)


def test_the_lr_schedule_warms_up_then_decays():
    from dawn.flow.trainer import cosine_warmup_schedule

    param = torch.nn.Parameter(torch.zeros(1))
    param.grad = torch.zeros(1)
    optimizer = torch.optim.AdamW([param], lr=1.0)
    scheduler = cosine_warmup_schedule(optimizer, warmup_steps=10, total_steps=100)
    seen = []
    for _ in range(100):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert seen[0] == pytest.approx(0.1)
    assert seen[9] == pytest.approx(1.0)
    assert seen[:10] == sorted(seen[:10])
    assert seen[10:] == sorted(seen[10:], reverse=True)
    assert seen[-1] < 0.01


def test_ema_tracks_the_weights_and_can_be_swapped_in():
    module = torch.nn.Linear(4, 4)
    ema = EMA(dict(module.named_parameters()), decay=0.5)
    start = module.weight.detach().clone()
    with torch.no_grad():
        module.weight.add_(1.0)
    ema.update()
    assert torch.allclose(ema.shadow["weight"], start + 0.5)
    with ema.applied():
        assert torch.allclose(module.weight, start + 0.5)
    assert torch.allclose(module.weight, start + 1.0)  # restored


def test_a_checkpoint_round_trips_and_carries_its_stats(configs, batch, stats, tmp_path):
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "flow.pt")
    state = torch.load(path, weights_only=False)
    assert state["step"] == 1
    assert state["config"]["grid"]["spatial_size"] == 16
    assert torch.allclose(state["grid_stats"]["mean"], stats["mean"])
    assert state["lora"] and all("lora_" in key for key in state["lora"])

    # Same seed, so the FROZEN backbone matches: a checkpoint stores the LoRA weights only and
    # expects the pretrained encoder to be reloaded from encoder.model_ids.
    resumed = make_trainer(cfg, stats, seed=0)
    with torch.no_grad():
        resumed.velocity.input_proj.weight.add_(1.0)
        resumed.noise_head.mlp[0].weight.add_(1.0)
    assert not torch.equal(resumed.velocity.input_proj.weight, trainer.velocity.input_proj.weight)
    resumed.load_state_dict(state)
    assert resumed.step == 1
    assert torch.equal(resumed.velocity.input_proj.weight, trainer.velocity.input_proj.weight)

    torch.manual_seed(7)
    first = trainer.train_step(batch)
    torch.manual_seed(7)
    assert resumed.train_step(batch) == first


def test_validate_reports_both_samplers_against_the_oracle(configs, batch, stats):
    cfg, _ = configs
    trainer = make_trainer(cfg, stats)
    scores = trainer.validate([batch], nfe=2)
    assert set(scores) == {"psnr/oracle", "psnr/euler", "psnr/midpoint", "psnr/euler_night",
                           "psnr/gap_to_oracle"}
    assert scores["psnr/oracle"] > 25, "the fitted grid should be well above 25 dB"
    assert scores["psnr/gap_to_oracle"] == pytest.approx(
        scores["psnr/oracle"] - scores["psnr/euler"]
    )


def test_param_counts_are_in_the_expected_range(configs, stats):
    cfg, _ = configs
    counts = make_trainer(cfg, stats).param_counts()
    assert 6e6 < counts["velocity"][0] < 9e6
    assert counts["encoder"][0] == sum(
        p.numel() for p in make_trainer(cfg, stats).encoder.parameters() if p.requires_grad
    )
    assert counts["total"][0] == sum(v[0] for k, v in counts.items() if k != "total")


# --- the overfit sanity check ----------------------------------------------------------------


def _overfit(num_images: int, steps: int, lr: float = 3e-4):
    """Train on one fixed batch and return `(psnr curve, oracle psnr)`."""
    cfg, calib = load_configs("debug.yaml")
    cfg = {
        **cfg,
        "flow": {**cfg["flow"], "num_train_steps": steps, "warmup_steps": 0,
                 "timesteps_per_image": 4},
        "optim": {**cfg["optim"], "lr": lr, "lr_warmup_steps": max(1, steps // 20)},
    }
    torch.manual_seed(0)
    batch = night_batch(num_images, cfg, calib, level="medium")
    stats = compute_grid_stats(
        [{"night_clean": batch["night_clean"][i], "day": batch["day"][i]}
         for i in range(num_images)],
        num_samples=num_images,
    )
    trainer = make_trainer(cfg, stats)
    curve = [(0, trainer.validate([batch], nfe=2))]
    for step in range(steps):
        trainer.train_step(batch)
        if (step + 1) % max(1, steps // 6) == 0:
            curve.append((step + 1, trainer.validate([batch], nfe=2)))
    return curve, curve[0][1]["psnr/oracle"]


def test_the_model_overfits_a_handful_of_images_toward_the_oracle():
    """The fast version of the spec's 100-image overfit: does the chain learn at all?

    A randomly initialized two-block encoder on the CPU cannot reach the oracle in a few
    hundred steps, so this asserts the direction and the size of the move, not the gate.
    `DAWN_SLOW_TESTS=1` runs the 100-image version below.
    """
    curve, oracle = _overfit(num_images=2, steps=240)
    start, end = curve[0][1]["psnr/euler"], curve[-1][1]["psnr/euler"]
    print("\n  step   euler  midpoint   oracle")
    for step, scores in curve:
        print(f"  {step:5d}  {scores['psnr/euler']:6.2f}  {scores['psnr/midpoint']:8.2f}"
              f"  {oracle:7.2f}")
    assert end > start + 4.0, f"render PSNR only moved {start:.2f} -> {end:.2f} dB"
    assert end > oracle - 12.0, f"render PSNR {end:.2f} dB vs oracle {oracle:.2f} dB"


@pytest.mark.skipif(not RUN_SLOW, reason="set DAWN_SLOW_TESTS=1 (tens of minutes on CPU)")
def test_the_model_overfits_one_hundred_images_toward_the_oracle():
    curve, oracle = _overfit(num_images=100, steps=3000)
    for step, scores in curve:
        print(f"  {step:5d}  {scores['psnr/euler']:6.2f}  {oracle:7.2f}")
    assert curve[-1][1]["psnr/euler"] > oracle - 6.0
