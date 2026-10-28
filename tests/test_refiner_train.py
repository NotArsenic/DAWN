"""End-to-end tests for Stage 2 refiner training, loss functions, crops, validation, and checkpoints."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
# `scripts.train_refiner` imports `dawn.flow.trainer`, and `dawn.flow.inference` imports the
# encoder, so both stages' import graphs pull peft and timm in even though stage 2 uses neither.
pytest.importorskip("peft")
pytest.importorskip("timm")

from dawn.data.color import lab_to_linear  # noqa: E402
from dawn.eval.metrics import masked_mean, masked_psnr  # noqa: E402
from dawn.flow.inference import load_refiner  # noqa: E402
from dawn.flow.loaders import load_config  # noqa: E402
from dawn.flow.targets import build_targets, prepare_batch  # noqa: E402
from dawn.models.refiner import LocalRefiner  # noqa: E402
from scripts.train_refiner import (  # noqa: E402
    WATCH_SCOPES,
    RefinerTrainer,
    crop_batch,
    crop_windows,
    improvement_verdict,
)
from tests.flow_helpers import CONFIGS, TINY_INPUT, synthetic_batch, tiny_cfg, wake_up  # noqa: E402

PINNED_KEYS = {
    "step",
    "config",
    "refiner",
    "refiner_config",
    "ema",
    "optimizer",
    "scheduler",
}
TRAIN_PAIRS = 3662


def tiny_refiner_cfg(**updates) -> dict:
    """Generate minimal test configuration for refiner model."""
    cfg = tiny_cfg()
    cfg["refiner"] = {
        "hidden_channels": 8,
        "num_blocks": 2,
        "kernel_size": 5,
        "residual_scale": 0.1,
        "crop_size": 16,
        "residual_penalty": 0.01,
        "num_train_steps": 20,
    }
    cfg["optim"].update(lr=5.0e-3, lr_warmup_steps=2, weight_decay=0.0)
    for section, values in updates.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


def _trainer(seed: int = 0, awake: float | None = None, **updates):
    """Instantiate a refiner trainer and test batch."""
    torch.manual_seed(seed)
    trainer = RefinerTrainer.from_config(tiny_refiner_cfg(**updates), "cpu", seed=seed)
    if awake is not None:
        wake_up(trainer.refiner, std=awake, seed=seed + 1)
    return trainer, synthetic_batch(2, seed=seed)


def rolled_batch(batch: int = 2, seed: int = 0, shift: int = 3) -> dict:
    """Generate synthetic batch with spatially shifted lightness channel."""
    item = synthetic_batch(batch, seed=seed)
    lightness = 100 * (0.25 + 0.45 * torch.roll(item["thermal"], shifts=shift, dims=-1))
    flat = torch.zeros_like(lightness)
    item["rgb"] = lab_to_linear(torch.cat([lightness, flat + 5, flat - 3], dim=1)).clamp(0, 1)
    item["rgb"] = item["rgb"].float()
    return item


# --- one step ---------------------------------------------------------------------------------


def test_one_cpu_step_runs_and_moves_every_parameter():
    trainer, batch = _trainer(awake=0.2)
    before = {name: p.detach().clone() for name, p in trainer.refiner.named_parameters()}
    out = trainer.train_step(batch)

    assert trainer.step == 1
    assert {"total", "l1", "residual_abs", "residual_abs_Lstar", "grad_norm", "lr"} <= set(out)
    assert all(torch.isfinite(torch.tensor(v)) for v in out.values()), out
    moved = [n for n, p in trainer.refiner.named_parameters() if not torch.equal(p, before[n])]
    assert moved and len(moved) == len(before), moved


def test_a_fresh_zero_init_refiner_still_moves_its_output_head():
    """Verify initial training step updates zero-initialized output projection head."""
    trainer, batch = _trainer()
    before = {name: p.detach().clone() for name, p in trainer.refiner.named_parameters()}
    trainer.train_step(batch)
    moved = {n for n, p in trainer.refiner.named_parameters() if not torch.equal(p, before[n])}
    head = {n for n in before if n.startswith(f"net.{len(trainer.refiner.net) - 1}.")}
    assert head and moved == head, (moved, head)


def test_the_total_is_the_l1_plus_the_configured_weight_times_the_residual():
    """Verify total loss computes weighted sum of L1 reconstruction and residual penalty."""
    for penalty in (0.0, 0.01, 2.0):
        trainer, batch = _trainer(awake=0.5, refiner={"residual_penalty": penalty})
        terms = {name: float(value.detach()) for name, value in
                 trainer.compute_losses(batch).items()}
        assert terms["total"] == pytest.approx(
            terms["l1"] + penalty * terms["residual_abs"], rel=1e-6
        )
        assert terms["residual_abs"] > 0.0  # otherwise the weight is untested


def test_the_step_log_reports_the_residual_in_l_star_units_too():
    trainer, batch = _trainer(awake=0.5)
    out = trainer.train_step(batch)
    assert out["residual_abs_Lstar"] == pytest.approx(100.0 * out["residual_abs"])
    assert 0.0 < out["residual_abs_Lstar"] <= 100.0 * trainer.refiner.residual_scale


def test_a_fully_masked_batch_gives_zero_not_nan():
    """Verify fully masked batches produce finite zero losses without NaNs."""
    trainer, batch = _trainer(awake=0.5)
    batch["valid"] = torch.zeros_like(batch["valid"])  # the pixel mask is `valid & ~moving`
    losses = trainer.compute_losses(batch)
    for name, value in losses.items():
        assert torch.isfinite(value) and float(value.detach()) == 0.0, name
    losses["total"].backward()  # and the zero still backpropagates
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in
               trainer.refiner.parameters())


def test_a_bigger_residual_penalty_leaves_a_smaller_residual():
    """Verify residual regularization penalty constrains learned residual magnitude."""
    batch = rolled_batch(2, seed=0, shift=3)
    tails = {}
    for penalty in (0.0, 0.01, 5.0):
        trainer, _ = _trainer(seed=0, refiner={"residual_penalty": penalty})
        history = [trainer.train_step(batch) for _ in range(40)]
        tails[penalty] = sum(h["residual_abs"] for h in history[-5:]) / 5
    assert tails[0.0] > 0.0, "the L1 term never wanted a residual, so nothing was penalised"
    assert tails[5.0] < 0.1 * tails[0.0], tails
    assert tails[5.0] < tails[0.01] <= tails[0.0]  # the plan's 0.01 is a light touch, not a gate


# --- the crops --------------------------------------------------------------------------------


def test_crop_windows_are_square_inside_the_frame_and_one_per_image():
    windows = crop_windows(TINY_INPUT, 16, [f"train_{i:05d}" for i in range(7)])
    assert len(windows) == 7
    for top, left, height, width in windows:
        assert (height, width) == (16, 16)
        assert 0 <= top <= TINY_INPUT[0] - height and 0 <= left <= TINY_INPUT[1] - width


def test_a_frame_smaller_than_the_crop_falls_back_to_the_whole_frame_per_axis():
    """Verify crop window generator handles input dimensions equal to or smaller than crop size."""
    assert crop_windows((8, 10), 16, ["train_00000"]) == [(0, 0, 8, 10)]
    windows = crop_windows((256, 320), 256, ["train_00000", "train_00001"])
    for top, left, height, width in windows:
        assert (top, height, width) == (0, 256, 256) and 0 <= left <= 64


def test_crop_offsets_are_per_image_without_a_generator_and_fresh_with_one():
    """Verify crop offsets are deterministic per image without a generator and stochastic with one."""
    ids = [f"train_{i:05d}" for i in range(6)]
    assert crop_windows(TINY_INPUT, 16, ids) == crop_windows(TINY_INPUT, 16, ids)
    assert crop_windows(TINY_INPUT, 16, ids) != crop_windows(TINY_INPUT, 16, list(reversed(ids)))
    generator = torch.Generator().manual_seed(0)
    draws = [crop_windows(TINY_INPUT, 16, ids, generator) for _ in range(3)]
    assert draws[0] != draws[1] and draws[1] != draws[2]
    again = torch.Generator().manual_seed(0)
    assert [crop_windows(TINY_INPUT, 16, ids, again) for _ in range(3)] == draws


def test_crop_batch_takes_the_same_window_from_every_tensor():
    """Verify identical spatial windows are extracted across all batch tensors."""
    height, width = TINY_INPUT
    rows = torch.arange(height).view(1, 1, height, 1).expand(3, 1, height, width)
    columns = torch.arange(width).view(1, 1, 1, width).expand(3, 1, height, width)
    coded = (1000 * rows + columns).float()
    windows = crop_windows(TINY_INPUT, 16, ["a", "b", "c"], torch.Generator().manual_seed(1))
    out = crop_batch({name: coded.clone() for name in ("thermal", "L_grid", "L_true", "mask")},
                     windows)

    assert {k: tuple(v.shape) for k, v in out.items()} == {
        name: (3, 1, 16, 16) for name in ("thermal", "L_grid", "L_true", "mask")
    }
    for index, (top, left, _, _) in enumerate(windows):
        expected = coded[index, :, top : top + 16, left : left + 16]
        for name, value in out.items():
            assert torch.equal(value[index], expected), (name, index)


def test_crop_batch_refuses_a_batch_that_does_not_match_the_windows():
    with pytest.raises(ValueError, match="crop windows"):
        crop_batch({"thermal": torch.zeros(2, 1, 8, 8)}, crop_windows((8, 8), 4, ["a"]))


def test_a_training_step_crops_all_four_tensors_to_the_crop_size(monkeypatch):
    import scripts.train_refiner as module

    seen = {}
    real = module.crop_batch

    def spy(tensors, windows):
        seen["keys"], seen["windows"] = set(tensors), list(windows)
        seen["sizes"] = {tuple(v.shape[-2:]) for v in tensors.values()}
        return real(tensors, windows)

    monkeypatch.setattr(module, "crop_batch", spy)
    trainer, batch = _trainer(awake=0.2)
    out = trainer.compute_losses(batch)

    assert seen["keys"] == {"thermal", "L_grid", "L_true", "mask"}
    assert seen["sizes"] == {TINY_INPUT}  # the grid was fit on the FULL frame, then cropped
    assert len(seen["windows"]) == 2 and all(w[2:] == (16, 16) for w in seen["windows"])
    assert torch.isfinite(out["total"])


def test_a_crop_larger_than_the_frame_trains_on_the_whole_frame_instead_of_raising(monkeypatch):
    import scripts.train_refiner as module

    sizes = []
    real = module.crop_batch
    monkeypatch.setattr(
        module,
        "crop_batch",
        lambda tensors, windows: sizes.append(windows[0][2:]) or real(tensors, windows),
    )
    trainer, batch = _trainer(awake=0.2, refiner={"crop_size": 256})
    trainer.train_step(batch)
    assert sizes == [TINY_INPUT]


# --- validation: the gate ----------------------------------------------------------------------


def test_validation_reports_the_grid_the_refined_image_and_their_difference():
    trainer, batch = _trainer(awake=0.3)
    scores = trainer.validate([batch])
    for key in ("psnr_L/grid", "psnr_L/refined", "psnr_L/improvement",
                "residual/abs_mean_Lstar"):
        assert key in scores and torch.isfinite(torch.tensor(scores[key])), key
    assert scores["num_images"] == 2.0
    assert scores["psnr_L/improvement"] == pytest.approx(
        scores["psnr_L/refined"] - scores["psnr_L/grid"]
    )
    assert 0.0 < scores["residual/abs_mean_Lstar"] <= 100.0 * trainer.refiner.residual_scale


def test_an_untrained_refiner_scores_exactly_zero_improvement():
    """Verify zero-initialized refiner yields zero improvement over base grid."""
    trainer, batch = _trainer()
    scores = trainer.validate([batch])
    assert scores["psnr_L/improvement"] == pytest.approx(0.0, abs=1e-6)
    assert scores["psnr_L/refined"] == pytest.approx(scores["psnr_L/grid"])
    assert scores["residual/abs_mean_Lstar"] == 0.0


def test_validation_runs_on_the_full_frame_and_ignores_the_crop_size():
    """Verify validation evaluates on full frames independent of training crop size."""
    small, batch = _trainer(seed=1, awake=0.3, refiner={"crop_size": 8})
    big, _ = _trainer(seed=1, awake=0.3, refiner={"crop_size": 256})
    assert small.validate([batch]) == big.validate([batch])


def test_the_validation_psnr_is_the_masked_one_on_l_star_with_a_peak_of_100():
    """Verify validation PSNR matches masked L* PSNR calculation with peak 100."""
    trainer, batch = _trainer(awake=0.3)
    prepared = prepare_batch(batch, trainer.input_size, trainer.policy)
    targets = build_targets(
        prepared["thermal"], prepared["rgb"], prepared["mask"], trainer.layout, trainer.fit
    )
    mask, l_true = prepared["mask"], targets.lab[:, :1]
    with torch.no_grad():
        residual = trainer.refiner.residual(targets.L_grid, prepared["thermal"])
    refined = (targets.L_grid + residual).clamp(0, 1)
    scores = trainer.validate([batch])
    assert scores["psnr_L/grid"] == pytest.approx(
        float(masked_psnr(100 * targets.L_grid, l_true, mask, 100.0).mean()), rel=1e-5
    )
    assert scores["psnr_L/refined"] == pytest.approx(
        float(masked_psnr(100 * refined, l_true, mask, 100.0).mean()), rel=1e-5
    )
    assert scores["residual/abs_mean_Lstar"] == pytest.approx(
        float(100 * masked_mean(residual.abs(), mask).mean()), rel=1e-5
    )


def test_a_batch_with_no_mask_weight_is_skipped_rather_than_scored():
    trainer, batch = _trainer(awake=0.3)
    batch["valid"] = torch.zeros_like(batch["valid"])
    scores = trainer.validate([batch])
    assert scores == {"num_images": 0.0}
    assert "no verdict" in improvement_verdict(scores)


def test_the_verdict_says_in_words_whether_the_refiner_earns_its_place():
    earns = {"psnr_L/improvement": 0.42, "residual/abs_mean_Lstar": 1.5}
    assert "EARNS" in improvement_verdict(earns) and "+0.420" in improvement_verdict(earns)
    for gain in (0.0, -0.3):
        verdict = improvement_verdict({"psnr_L/improvement": gain,
                                       "residual/abs_mean_Lstar": 1.5})
        assert "does NOT earn" in verdict and "cut it" in verdict


# --- checkpoints -------------------------------------------------------------------------------


def test_the_checkpoint_holds_exactly_the_pinned_keys(tmp_path):
    """Verify refiner checkpoint contains expected dictionary keys."""
    trainer, batch = _trainer(awake=0.2)
    trainer.train_step(batch)
    state = torch.load(trainer.save_checkpoint(tmp_path / "ckpt.pt"), weights_only=False)

    assert set(state) == PINNED_KEYS
    assert state["step"] == 1
    assert state["refiner_config"] == trainer.cfg["refiner"]
    assert isinstance(state["config"], dict) and state["config"]["refiner"]["crop_size"] == 16
    assert LocalRefiner.from_config(state["refiner_config"]).receptive_field == 9


def test_checkpoint_round_trip_restores_the_weights_the_step_and_the_schedule(tmp_path):
    trainer, batch = _trainer(awake=0.2)
    for _ in range(3):
        trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "ckpt.pt")

    fresh, _ = _trainer(seed=0)
    wake_up(fresh.refiner, std=0.4, seed=99)  # scramble everything the load must restore
    fresh.step = 77
    assert not torch.equal(fresh.refiner.net[0].weight, trainer.refiner.net[0].weight)

    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    assert fresh.step == trainer.step == 3
    for name, param in trainer.refiner.named_parameters():
        assert torch.equal(param, dict(fresh.refiner.named_parameters())[name]), name
    assert fresh.scheduler.get_last_lr() == trainer.scheduler.get_last_lr()
    assert fresh.ema.num_updates == trainer.ema.num_updates
    for name, shadow in trainer.ema.shadow.items():
        assert torch.equal(shadow, fresh.ema.shadow[name]), name
    # A resumed run continues where the saved one would have: same crop, same loss.
    with torch.no_grad():
        a, b = trainer.compute_losses(batch), fresh.compute_losses(batch)
    assert float(a["total"]) == pytest.approx(float(b["total"]), rel=1e-6)


def test_a_checkpoint_with_another_residual_bound_is_refused(tmp_path):
    """Verify checkpoint loading validates matching residual scale parameter."""
    trainer, batch = _trainer(awake=0.2)
    trainer.train_step(batch)
    state = trainer.state_dict()
    other, _ = _trainer(refiner={"residual_scale": 0.4})
    with pytest.raises(ValueError, match="residual_scale"):
        other.load_state_dict(state)


# --- the shipped config --------------------------------------------------------------------------


def _shipped() -> dict:
    return load_config(
        str(CONFIGS / "flow.yaml"), str(CONFIGS / "data.yaml"), [str(CONFIGS / "refiner.yaml")]
    )


def test_refiner_yaml_merges_on_flow_and_data_and_builds_a_working_refiner():
    cfg = _shipped()
    section = cfg["refiner"]
    assert set(section) >= {
        "hidden_channels",
        "num_blocks",
        "kernel_size",
        "residual_scale",
        "crop_size",
        "residual_penalty",
        "num_train_steps",
    }
    assert (section["hidden_channels"], section["num_blocks"]) == (24, 2)
    assert (section["kernel_size"], section["residual_scale"]) == (5, 0.1)
    assert section["crop_size"] == 256 and section["residual_penalty"] == 0.01

    model = LocalRefiner.from_config(cfg)
    assert model.receptive_field == 9 and 2000 < model.num_params() < 5000
    lightness = torch.rand(2, 1, 32, 40)
    assert model(lightness, torch.rand(2, 1, 32, 40)).shape == lightness.shape
    # It still inherits stage 1's resolution, grid fit and pair policy, unchanged.
    assert cfg["encoder"]["input_size"] == [512, 640] and cfg["grid"]["refine_steps"] == 1
    assert cfg["pairs"]["eval_subset"] == "box_aligned"


def test_the_shipped_budget_is_the_one_its_comment_claims_in_epochs():
    """Verify training budget step count corresponds to expected epoch coverage."""
    cfg = _shipped()
    steps, batch = cfg["refiner"]["num_train_steps"], cfg["optim"]["batch_size"]
    assert (steps, batch) == (8000, 16)
    assert steps * batch / TRAIN_PAIRS == pytest.approx(35.0, abs=0.5)
    log = cfg["logging"]
    assert steps / log["val_every"] >= 20  # enough gate readings to see when it turned
    assert log["save_every"] < steps and log["log_every"] < log["val_every"]
    assert cfg["optim"]["lr_warmup_steps"] < steps // 10


def test_the_shipped_logging_goes_to_the_stage_two_workspace_with_a_valid_watch_scope():
    log = _shipped()["logging"]
    assert log["wandb_project"] == "dawn-refiner"  # stage 1 is `dawn`
    assert log["wandb_entity"] is None  # inherited from flow.yaml, not repeated
    assert log["watch_scope"] in WATCH_SCOPES and log["watch_scope"] != "none"
    assert WATCH_SCOPES[log["watch_scope"]] == ("refiner",)


def test_the_ema_shadow_is_keyed_the_way_the_inference_loader_reads_it(tmp_path):
    """Verify EMA shadow key structure matches inference loader naming convention."""
    trainer, batch = _trainer(awake=0.2)
    for _ in range(3):
        trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path / "refiner_best.pt")

    # The prefix is the convention `_apply_ema` resolves: `f"{prefix}.{name}"` under `refiner`.
    live = dict(trainer.refiner.named_parameters())
    assert set(trainer.ema.shadow) == {f"refiner.{name}" for name in live}

    loaded = load_refiner(path, "cpu", use_ema=True)  # the default path: must not raise
    moved = 0
    for name, param in loaded.named_parameters():
        assert torch.equal(param, trainer.ema.shadow[f"refiner.{name}"]), name
        moved += int(not torch.equal(param, live[name]))
    assert moved, "the shadow equals the live weights, so this proves nothing about use_ema"
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())

    raw = load_refiner(path, "cpu", use_ema=False)  # and the averaged weights are not the raw ones
    for name, param in raw.named_parameters():
        assert torch.equal(param, live[name]), name


def _tiny_tree_override(tmp_path):
    """A one-file override that points the SHIPPED refiner.yaml at a tiny aligned tree.

    Only keys that already exist in flow.yaml / data.yaml, exactly as a real overlay must.
    """
    from omegaconf import OmegaConf

    from tests.flow_helpers import write_tiny_tree

    path = tmp_path / "tiny.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "dronevehicle": {**write_tiny_tree(tmp_path / "tree"), "max_pairs": None},
                "encoder": {"input_size": list(TINY_INPUT)},
                "optim": {"batch_size": 2, "precision": "fp32", "lr_warmup_steps": 1},
                "loader": {"num_workers": 0, "pin_memory": False, "val_batch_size": 2},
                "logging": {
                    "wandb_mode": "disabled",
                    "log_every": 1,
                    "val_every": 2,
                    "save_every": 2,
                    "num_val_samples": 2,
                },
            }
        ),
        path,
    )
    return path


def test_main_runs_the_loop_and_writes_the_pinned_checkpoint_names(tmp_path, monkeypatch):
    """Verify refiner training script execution, checkpoint generation, and resume functionality."""
    import sys

    from scripts.train_refiner import main

    override, out = _tiny_tree_override(tmp_path), tmp_path / "out"
    argv = [
        "train_refiner.py",
        "--override", str(override),
        "--device", "cpu",
        "--max-steps", "4",
        "--out-dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()

    assert {p.name for p in out.glob("refiner_*.pt")} == {
        "refiner_best.pt",
        "refiner_last.pt",
        "refiner_000002.pt",
        "refiner_000004.pt",
    }
    state = torch.load(out / "refiner_last.pt", map_location="cpu", weights_only=False)
    assert set(state) == PINNED_KEYS and state["step"] == 4
    assert (out / "run_config.json").exists() and (out / "train_log.jsonl").exists()
    kinds = {line.split('"kind": "')[1].split('"')[0]
             for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()}
    assert "train" in kinds and any(k.startswith("val_") for k in kinds)
    assert (out / "metrics" / "train.csv").exists()

    monkeypatch.setattr(sys, "argv", argv + ["--resume", str(out / "refiner_last.pt")])
    main()  # already at the budget: it resumes at step 4 and writes refiner_last.pt again
    assert torch.load(out / "refiner_last.pt", map_location="cpu", weights_only=False)["step"] == 4


def test_the_shipped_config_drives_the_trainer():
    """Verify refiner trainer instantiation from shipped configuration."""
    cfg = _shipped()
    trainer = RefinerTrainer.from_config(cfg, "cpu")
    assert trainer.crop_size == 256 and trainer.num_train_steps == 8000
    assert trainer.residual_penalty == 0.01 and trainer.refiner.num_params() == 2545
    assert all(hasattr(trainer, name) for name in WATCH_SCOPES["refiner"])
    # fp32 and no weight decay are stage 2's deliberate departures from flow.yaml.
    assert cfg["optim"]["precision"] == "fp32" and cfg["optim"]["weight_decay"] == 0.0
    assert all(g["weight_decay"] == 0.0 for g in trainer.optimizer.param_groups)
    # The LR warms up over `lr_warmup_steps` and then decays: `cosine_warmup_schedule` directly.
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(cfg["optim"]["lr"] / 200)
