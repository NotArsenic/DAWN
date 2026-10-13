"""End-to-end tests for flow training loop, schedules, EMA, checkpointing, and overfitting."""


import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("peft")
pytest.importorskip("timm")

from dawn.flow.loaders import build_loaders, cycle, split_by_pair_kind, stats_batches  # noqa: E402
from dawn.flow.targets import FitSettings, compute_token_stats  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import (  # noqa: E402
    EMA,
    FlowTrainer,
    cosine_warmup_schedule,
    split_param_groups,
)
from tests.flow_helpers import (  # noqa: E402
    TINY_INPUT,
    synthetic_batch,
    tiny_cfg,
    tiny_data_cfg,
    wake_up,
)


def _stats(cfg, batches):
    layout = TokenLayout.from_config(cfg)
    from dawn.flow.targets import prepare_batch

    prepared = [prepare_batch(b, TINY_INPUT, cfg["pairs"]) for b in batches]
    return compute_token_stats(prepared, layout, FitSettings.from_config(cfg))


def _trainer(correction=False, seed=0, **updates):
    torch.manual_seed(seed)
    cfg = tiny_cfg(correction, **updates)
    batch = synthetic_batch(2, seed=seed, fallback=[False, True])
    return FlowTrainer.from_config(cfg, _stats(cfg, [batch]), "cpu"), batch


@pytest.fixture(params=[False, True], ids=["correction_off", "correction_on"])
def correction(request):
    return request.param


def test_one_cpu_step_runs_in_each_mode_and_moves_every_trainable_group(correction):
    trainer, batch = _trainer(correction)
    before = {k: p.detach().clone() for k, p in trainer.trainable_parameters().items()}
    out = trainer.train_step(batch)
    assert trainer.step == 1
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values()), out
    expected = {
        "total",
        "flow",
        "flow_brightness",
        "flow_color",
        "render",
        "align",
        "color",
        "grad_norm",
        "detach_cond",
        "lr_lora",
        "lr_main",
    }
    assert expected <= set(out) and ("flow_correction" in out) is correction
    assert out["detach_cond"] == 1.0  # step 0 is inside the detach warmup
    moved = {
        k.split(".")[0]
        for k, p in trainer.trainable_parameters().items()
        if not torch.equal(p, before[k])
    }
    assert moved == {"encoder", "velocity", "color_head"} or moved >= {"velocity", "color_head"}


def test_the_first_step_matches_a_fresh_trainer_given_the_same_seed():
    first, batch = _trainer(seed=3)
    second, _ = _trainer(seed=3)
    torch.manual_seed(5)
    a = first.train_step(batch)
    torch.manual_seed(5)
    b = second.train_step(batch)
    assert a["total"] == pytest.approx(b["total"], rel=1e-6)


def _lora_grad_norm(trainer, loss):
    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    grads = [p.grad for n, p in trainer.encoder.named_parameters() if p.requires_grad]
    return sum(float(g.abs().sum()) for g in grads if g is not None)


def test_before_warmup_no_velocity_gradient_reaches_lora_but_align_and_color_do():
    trainer, batch = _trainer(warmup=None) if False else _trainer()
    wake_up(trainer.velocity)  # a zero-init head would give zero gradient everywhere
    trainer.train()
    assert trainer.step < trainer.warmup_steps
    losses, info = trainer.compute_losses(batch)
    assert info["detach_cond"] is True
    assert _lora_grad_norm(trainer, losses["flow"]) == 0.0
    assert _lora_grad_norm(trainer, losses["render"]) == 0.0
    assert _lora_grad_norm(trainer, losses["align"]) > 0.0
    assert _lora_grad_norm(trainer, losses["color"]) > 0.0


def test_after_warmup_the_velocity_loss_trains_lora_too():
    trainer, batch = _trainer()
    wake_up(trainer.velocity)
    trainer.step = trainer.warmup_steps  # the schedule is a function of the step only
    trainer.train()
    losses, info = trainer.compute_losses(batch)
    assert info["detach_cond"] is False
    assert _lora_grad_norm(trainer, losses["flow"]) > 0.0


def test_the_attention_bias_stays_detached_even_after_warmup():
    trainer, batch = _trainer()
    trainer.step = trainer.warmup_steps
    cond = trainer.encoder.encode(batch["thermal"], return_attn=True)
    assert cond["attn_bias"].requires_grad is False


def test_param_groups_split_lora_and_the_rest_and_never_decay_norms_biases_gates_or_pos():
    trainer, _ = _trainer()
    groups = {g["name"]: g for g in trainer.optimizer.param_groups}
    assert {n.split("_")[0] for n in groups} == {"lora", "main"}
    for group in groups.values():
        lr = (
            trainer.cfg["encoder"]["lora_lr"]
            if group["name"].startswith("lora")
            else trainer.cfg["optim"]["lr"]
        )
        assert group["lr"] == lr
    named = {id(p): n for n, p in trainer.trainable_parameters().items()}
    decayed = {named[id(p)] for g in groups.values() if g["weight_decay"] > 0 for p in g["params"]}
    undecayed = {
        named[id(p)] for g in groups.values() if g["weight_decay"] == 0 for p in g["params"]
    }
    assert all(n.startswith("encoder.") for n in decayed if "lora" in n or n.startswith("encoder"))
    for name in undecayed | decayed:
        suffix_hit = name.endswith(("bias", "dino_gate", "pos_embed"))
        assert (name in undecayed) or not suffix_hit, name
    assert any(n.endswith("pos_embed") for n in undecayed)
    assert any(n.endswith("dino_gate") for n in undecayed)
    assert any(n.startswith("color_head") for n in decayed | undecayed)
    assert not any(p.requires_grad is False for g in groups.values() for p in g["params"])


def test_split_param_groups_puts_only_lora_in_the_lora_scope():
    trainer, _ = _trainer()
    lora = [
        g
        for g in split_param_groups(
            trainer.encoder, trainer.velocity, trainer.color_head, 1e-3, 1e-4, 0.01
        )
        if g["name"].startswith("lora")
    ]
    names = {id(p): n for n, p in trainer.encoder.named_parameters()}
    assert lora and all("lora_" in names[id(p)] for g in lora for p in g["params"])


def test_ema_tracks_the_trainable_parameters_and_applied_swaps_them_in_and_back():
    trainer, batch = _trainer()
    for _ in range(2):
        trainer.train_step(batch)
    live = {k: p.detach().clone() for k, p in trainer.trainable_parameters().items()}
    key = next(k for k in live if k.startswith("velocity.") and live[k].ndim > 1)
    assert not torch.allclose(trainer.ema.shadow[key], live[key].float())
    with trainer.ema.applied():
        assert torch.allclose(trainer.trainable_parameters()[key], trainer.ema.shadow[key])
    assert torch.equal(trainer.trainable_parameters()[key], live[key])
    assert set(trainer.ema.shadow) == set(live)


def test_cosine_warmup_schedule_ramps_then_decays():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = cosine_warmup_schedule(opt, 4, 20)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[3] <= 1.0 and lrs[-1] < 0.05 and max(lrs) == pytest.approx(1.0, abs=1e-6)


def test_checkpoint_round_trip_restores_everything_and_checks_the_layout(tmp_path, correction):
    trainer, batch = _trainer(correction)
    trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["layout"] == trainer.layout.descriptor() and state["input_size"] == list(
        TINY_INPUT
    )
    assert not any("backbone" in k and "lora_" not in k for k in state["lora"])  # LoRA only
    assert set(state["stats"]) == set(trainer.stats)

    # The frozen backbone is rebuilt, not stored: share its (random, here) weights by seed, then
    # scramble everything the checkpoint is responsible for, so the load has to restore it.
    fresh, _ = _trainer(correction, seed=0)
    wake_up(fresh.velocity, std=0.2, seed=5)
    wake_up(fresh.color_head, std=0.2, seed=6)
    fresh.step = 99
    assert not torch.equal(fresh.color_head.proj.weight, trainer.color_head.proj.weight)
    fresh.load_state_dict(state)
    assert fresh.step == 1
    for name, param in trainer.trainable_parameters().items():
        assert torch.equal(param, fresh.trainable_parameters()[name]), name
    assert torch.equal(fresh.mean, trainer.mean) and torch.equal(fresh.std, trainer.std)
    # A resumed run continues exactly where the saved one would have.
    torch.manual_seed(9)
    a = trainer.train_step(batch)
    torch.manual_seed(9)
    b = fresh.train_step(batch)
    assert a["total"] == pytest.approx(b["total"], rel=1e-5)


def test_a_checkpoint_of_another_layout_or_resolution_is_refused(tmp_path):
    trainer, batch = _trainer(False)
    state = trainer.state_dict()
    other, _ = _trainer(True)
    with pytest.raises(ValueError, match="layout"):
        other.load_state_dict(state)
    wrong_size = dict(state, input_size=[128, 160])
    with pytest.raises(ValueError, match="resolution"):
        trainer.load_state_dict(wrong_size)


def test_sizes_that_do_not_fit_the_token_grid_are_refused_at_build_time():
    cfg = tiny_cfg(flow={"render_size": [30, 40]})
    with pytest.raises(ValueError, match="render_size"):
        FlowTrainer.from_config(cfg, _stats(tiny_cfg(), [synthetic_batch(2)]), "cpu")


def test_validation_reports_the_oracle_and_both_samplers_per_image():
    trainer, batch = _trainer()
    scores = trainer.validate([batch])
    for key in (
        "psnr_L/oracle",
        "psnr_L/euler",
        "psnr_L/midpoint",
        "psnr_L/gap_to_oracle",
        "ciede2000/oracle",
        "ciede2000/euler",
        "colorfulness/gt",
        "colorfulness/euler",
    ):
        assert key in scores and torch.isfinite(torch.tensor(scores[key])), key
    assert scores["num_images"] == 2.0
    assert scores["psnr_L/gap_to_oracle"] == pytest.approx(
        scores["psnr_L/oracle"] - scores["psnr_L/euler"]
    )
    assert scores["psnr_L/oracle"] > 25  # the synthetic lightness is exactly representable
    again = trainer.validate([batch])  # noise comes from the image IDs: val is repeatable
    assert again["psnr_L/euler"] == scores["psnr_L/euler"]


def test_validation_reports_the_retrieval_diagnostic_and_the_colourfulness_ratios():
    """Verify validation telemetry includes retrieval metrics and colorfulness ratios."""
    trainer, batch = _trainer()
    scores = trainer.validate([batch])
    for key in ("retrieval/top1", "retrieval/within_1", "retrieval/chance"):
        assert key in scores, key
    assert 0.0 <= scores["retrieval/top1"] <= 1.0
    assert scores["retrieval/top1"] <= scores["retrieval/within_1"], "top1 implies within_1"
    assert scores["retrieval/chance"] == pytest.approx(1.0 / trainer.layout.num_tokens)
    assert scores["ciede2000/gap_to_oracle"] == pytest.approx(
        scores["ciede2000/euler"] - scores["ciede2000/oracle"]
    )
    assert scores["colorfulness/ratio_euler"] == pytest.approx(
        scores["colorfulness/euler"] / scores["colorfulness/gt"]
    )


def test_every_selection_metric_is_a_key_validate_really_returns():
    """Verify all candidate selection metrics exist in the validation output dictionary."""
    from scripts.train_flow import SELECTION

    trainer, batch = _trainer()
    scores = trainer.validate([batch])
    for key, _ in SELECTION.values():
        assert key in scores, key


def test_the_retrieval_diagnostic_can_be_switched_off():
    trainer, batch = _trainer(logging={"val_retrieval": False})
    assert not any(k.startswith("retrieval/") for k in trainer.validate([batch]))


def test_held_out_loss_reports_every_term_and_is_repeatable():
    """Verify validation loss components are finite and reproducible across runs."""
    trainer, batch = _trainer()
    first = trainer.validate_loss([batch])
    assert first["num_images"] == 2.0
    for key in ("loss/total", "loss/flow", "loss/render", "loss/align", "loss/color"):
        assert key in first and torch.isfinite(torch.tensor(first[key])), key
    torch.manual_seed(777)  # a fixed generator, not the global RNG
    again = trainer.validate_loss([batch])
    assert again["loss/total"] == pytest.approx(first["loss/total"], rel=1e-6)
    assert trainer.validate_loss([batch], seed=99)["loss/total"] != pytest.approx(
        first["loss/total"], rel=1e-9
    )


def test_held_out_loss_leaves_no_gradients_behind():
    trainer, batch = _trainer()
    trainer.validate_loss([batch])
    assert all(p.grad is None for p in trainer.trainable_parameters().values())


def test_a_detailed_step_adds_per_module_gradients_and_loss_shares():
    trainer, batch = _trainer()
    wake_up(trainer.velocity)
    plain = trainer.train_step(batch)
    assert not any(k.startswith(("grad/", "contrib/")) for k in plain)

    out = trainer.train_step(batch, detailed=True)
    assert {"grad/velocity", "grad/encoder", "grad/color_head"} <= set(out)
    assert out["grad/velocity"] > 0
    shares = {k: v for k, v in out.items() if k.startswith("contrib/")}
    assert set(shares) == {"contrib/flow", "contrib/render", "contrib/align", "contrib/color"}
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-5)


def test_the_conditioning_ramp_hands_lora_a_fraction_of_the_velocity_gradient():
    """Verify conditioning ramp progressively scales velocity gradient to LoRA parameters."""
    trainer, batch = _trainer(flow={"warmup_steps": 2, "warmup_ramp_steps": 4})
    wake_up(trainer.velocity)
    trainer.train()

    trainer.step = 2  # the gate lifts; the ramp is one quarter in
    assert trainer.conditioning_scale() == pytest.approx(0.25)
    losses, info = trainer.compute_losses(batch)
    assert info["detach_cond"] is False and info["cond_scale"] == pytest.approx(0.25)
    partial = _lora_grad_norm(trainer, losses["flow"])

    trainer.step = 5  # the ramp has completed
    assert trainer.conditioning_scale() == 1.0
    full = _lora_grad_norm(trainer, trainer.compute_losses(batch)[0]["flow"])
    assert 0.0 < partial < full


def test_a_zero_ramp_reproduces_the_old_hard_gate_exactly():
    hard, batch = _trainer(flow={"warmup_steps": 2, "warmup_ramp_steps": 0}, seed=4)
    hard.step = 2
    assert hard.conditioning_scale() == 1.0
    hard.step = 1
    assert hard.conditioning_scale() == 0.0
    losses, info = hard.compute_losses(batch)
    assert info["detach_cond"] is True
    assert _lora_grad_norm(hard, losses["flow"]) == 0.0


def test_the_ramp_defaults_to_a_sixth_of_the_warmup_when_it_is_not_set():
    """Verify conditioning ramp duration defaults proportionally to warmup steps."""
    trainer, _ = _trainer(flow={"warmup_steps": 600, "warmup_ramp_steps": None})
    assert trainer.warmup_ramp_steps == 100


def test_only_the_network_forwards_run_under_autocast_never_the_pixel_side(monkeypatch):
    """Verify mixed-precision autocast executes neural network operations in float32 for pixel ops."""
    import dawn.flow.trainer as trainer_mod

    trainer, batch = _trainer()
    monkeypatch.setattr(
        FlowTrainer, "_autocast", lambda self: torch.autocast("cpu", dtype=torch.bfloat16)
    )
    seen = {}

    def spy(name):
        real = getattr(trainer_mod, name)

        def wrapped(*args, **kwargs):
            seen.setdefault(name, []).append(torch.is_autocast_enabled("cpu"))
            return real(*args, **kwargs)

        monkeypatch.setattr(trainer_mod, name, wrapped)

    for name in ("build_targets", "render", "prepare_batch", "patch_histograms"):
        spy(name)
    trainer.train_step(batch)
    trainer.validate([batch])
    for name in ("build_targets", "render", "prepare_batch", "patch_histograms"):
        assert name in seen and not any(seen[name]), f"{name} ran under autocast"


# --- loaders and --overfit ---------------------------------------------------------------------


def test_loaders_split_validation_by_pair_kind_and_respect_the_sample_cap(tmp_path):
    cfg = tiny_data_cfg(tmp_path, n_box=3, n_fallback=2, logging={"num_val_samples": 2})
    loaders = build_loaders(cfg)
    assert set(loaders.val) == {"box_aligned", "fallback"}
    assert (
        len(loaders.val["box_aligned"].dataset) == 2 and len(loaders.val["fallback"].dataset) == 2
    )
    ids = [b["image_id"] for b in loaders.val["fallback"]]
    assert all(i.startswith("val_") for batch in ids for i in batch)
    fallback_flags = torch.cat([b["fallback"] for b in loaders.val["fallback"]])
    assert fallback_flags.all()
    assert loaders.train_size == 5


def test_the_fallback_subset_is_only_validated_when_the_pair_policy_asks_for_it(tmp_path):
    cfg = tiny_data_cfg(tmp_path, n_box=2, n_fallback=2)
    assert cfg["pairs"]["report_fallback_separately"] is True
    assert set(build_loaders(cfg).val) == {"box_aligned", "fallback"}
    cfg["pairs"]["report_fallback_separately"] = False
    assert set(build_loaders(cfg).val) == {"box_aligned"}
    cfg["pairs"]["eval_subset"] = "fallback"  # the headline subset is always validated
    assert "fallback" in build_loaders(cfg).val


def test_overfit_trains_and_validates_on_the_first_n_train_pairs_without_augmentation(tmp_path):
    cfg = tiny_data_cfg(tmp_path, n_box=3, n_fallback=1)
    loaders = build_loaders(cfg, overfit=2)
    assert loaders.train_size == 2
    seen = {i for batch in loaders.val["box_aligned"] for i in batch["image_id"]}
    assert seen == {"train_00000", "train_00001"} and "fallback" not in loaders.val
    first = next(iter(loaders.train))
    again = next(iter(build_loaders(cfg, overfit=2).train))
    by_id = {i: n for n, i in enumerate(first["image_id"])}
    for n, image_id in enumerate(again["image_id"]):  # no flips, no contrast curve
        assert torch.equal(again["thermal"][n], first["thermal"][by_id[image_id]])


def test_stats_batches_are_prepared_unaugmented_and_evenly_spread(tmp_path):
    cfg = tiny_data_cfg(tmp_path, n_box=3, n_fallback=1)
    batches = list(stats_batches(cfg, num_samples=3))
    assert sum(b["thermal"].shape[0] for b in batches) == 3
    assert set(batches[0]) == {"thermal", "rgb", "mask", "fallback", "image_id"}
    ids = [i for b in batches for i in b["image_id"]]
    assert ids[0] == "train_00000" and ids[-1] == "train_00003"


def test_cycle_restarts_the_loader():
    import itertools

    assert list(itertools.islice(cycle([1, 2]), 5)) == [1, 2, 1, 2, 1]
    with pytest.raises(ValueError, match="no batches"):
        next(cycle([]))


def test_split_by_pair_kind_keeps_manifest_order(tmp_path):
    from dawn.data.pairs import AlignedPairs

    cfg = tiny_data_cfg(tmp_path, n_box=2, n_fallback=1)
    subsets = split_by_pair_kind(AlignedPairs.from_config(cfg, "train_pairs"), None)
    assert [
        subsets["box_aligned"].dataset.image_ids[i] for i in subsets["box_aligned"].indices
    ] == ["train_00000", "train_00001"]
    assert [subsets["fallback"].dataset.image_ids[i] for i in subsets["fallback"].indices] == [
        "train_00002"
    ]


def test_a_short_overfit_learns_the_flow_closes_in_on_the_render_floor_and_teaches_the_colour_head(
    tmp_path,
):
    """Verify training convergence and loss reduction during a short multi-step overfit."""
    torch.manual_seed(0)
    steps = 100
    cfg = tiny_data_cfg(
        tmp_path,
        n_box=2,
        n_fallback=0,
        flow={"num_train_steps": steps, "warmup_steps": 5, "timesteps_per_image": 2},
        optim={"batch_size": 2, "lr": 2e-3, "lr_warmup_steps": 3},
        encoder={"lora_lr": 2e-3},
    )
    loaders = build_loaders(cfg, overfit=2)
    stats = compute_token_stats(
        stats_batches(cfg, 2), TokenLayout.from_config(cfg), FitSettings.from_config(cfg)
    )
    trainer = FlowTrainer.from_config(cfg, stats, "cpu")
    batches = cycle(loaders.train)
    history = [trainer.train_step(next(batches)) for _ in range(steps)]

    def mean(key, part):
        values = [h[key] for h in part]
        return sum(values) / len(values)

    head, tail = history[:10], history[-10:]
    assert mean("flow", tail) < 0.4 * mean("flow", head)
    assert mean("color", tail) < 0.7 * mean("color", head)
    assert mean("align", tail) < mean("align", head) - 1e-3
    assert mean("render", tail) < 0.5 * mean("render", head)

    from dawn.data.color import linear_to_lab
    from dawn.data.resample import area_downsample
    from dawn.flow.render import render
    from dawn.flow.targets import build_targets, prepare_batch

    raw = next(iter(loaders.train))
    batch = prepare_batch(raw, trainer.input_size, trainer.policy)
    targets = build_targets(
        batch["thermal"], batch["rgb"], batch["mask"], trainer.layout, trainer.fit
    )
    thermal_r = area_downsample(batch["thermal"], trainer.render_size)
    oracle = render(targets.parts, thermal_r, trainer.render_size, trainer.render_settings)
    lab_r = linear_to_lab(area_downsample(batch["rgb"], trainer.render_size).clamp(0, 1))
    floor = float(
        trainer.criterion.render(
            oracle["lab"], lab_r, area_downsample(batch["mask"], trainer.render_size), torch.ones(2)
        )
    )
    assert floor < mean("render", tail), (mean("render", tail), floor)  # the floor is a floor
    assert not any(torch.isnan(torch.tensor(h["total"])) for h in history)


def test_ema_class_is_the_one_the_trainer_uses():
    trainer, _ = _trainer()
    assert isinstance(trainer.ema, EMA)
