"""The telemetry added for the first real run: diagnostics, held-out loss and the local mirrors.

Three groups, in the order a run uses them:

1. **Schedule smoothing** -- the EMA decay warmup and the conditioning ramp. Both are strict
   generalizations: with their knobs off they must reproduce the old behaviour exactly, which is
   what makes a baseline comparable.
2. **Premise diagnostics** -- `patch_retrieval` and `edge_quantile_mask`. These exist to measure
   claims made about the model rather than act on them (`docs/proposals_review.md`), so they are
   tested against cases whose answer is known by construction.
3. **The local record** -- `CsvLog` / `RunLog`. W&B owns the live curves; these files are what a
   plot is rebuilt from, so the thing to pin is that they hold *the same keys* W&B got.

Everything here runs on the CPU with tiny tensors and never reads the real dataset.
"""

import json

import pytest

torch = pytest.importorskip("torch")

from dawn.eval.metrics import edge_quantile_mask, sobel_magnitude  # noqa: E402
from dawn.flow.trainer import EMA, budget_summary, patch_retrieval  # noqa: E402

# --- 1. schedule smoothing ----------------------------------------------------------------------


def _ema(decay=0.999, warmup=True):
    param = torch.nn.Parameter(torch.zeros(4))
    return EMA({"p": param}, decay=decay, warmup=warmup), param


def test_ema_without_warmup_is_the_old_fixed_decay():
    ema, param = _ema(warmup=False)
    with torch.no_grad():
        param.fill_(1.0)
    ema.update()
    assert ema.current_decay() == 0.999
    assert ema.shadow["p"][0] == pytest.approx(1e-3)


def test_ema_warmup_kills_the_initialization_in_a_handful_of_updates():
    """The point of the warmup: at `val_every: 500` a fixed 0.999 would validate 61% noise."""
    warm, warm_param = _ema()
    fixed, fixed_param = _ema(warmup=False)
    for _ in range(10):
        with torch.no_grad():
            warm_param.fill_(1.0)
            fixed_param.fill_(1.0)
        warm.update()
        fixed.update()
    assert warm.shadow["p"][0] > 0.99, "warmed EMA has tracked the weights"
    assert fixed.shadow["p"][0] < 0.02, "fixed decay is still almost entirely the initialization"


def test_ema_warmup_only_ever_lowers_the_decay_and_converges_to_the_configured_one():
    ema, _ = _ema(decay=0.999)
    seen = []
    for _ in range(20000):
        seen.append(ema.current_decay())
        ema.update()
    assert max(seen) <= 0.999 + 1e-12, "never exceeds the configured decay"
    assert seen == sorted(seen), "ramps monotonically from below"
    assert seen[-1] == pytest.approx(0.999)


def test_a_checkpoint_without_an_update_count_does_not_restart_the_warmup():
    """Resuming a converged run must not re-ramp the decay and let fresh weights swamp it."""
    ema, _ = _ema()
    old_style = {"decay": 0.999, "shadow": ema.shadow}  # written before the warmup existed
    ema.load_state_dict(old_style)
    assert ema.current_decay() == pytest.approx(0.999)


def test_ema_state_round_trips_the_update_count():
    ema, param = _ema()
    for _ in range(5):
        with torch.no_grad():
            param.add_(1.0)
        ema.update()
    restored, _ = _ema()
    restored.load_state_dict(ema.state_dict())
    assert restored.num_updates == 5
    assert restored.current_decay() == ema.current_decay()


@pytest.mark.parametrize(
    "step, warmup, ramp, expected",
    [
        (0, 100, 50, 0.0),  # inside the detach gate
        (99, 100, 50, 0.0),  # the last detached step
        (100, 100, 50, 0.02),  # the gate lifts: the ramp starts, it does not jump
        (124, 100, 50, 0.5),  # halfway
        (149, 100, 50, 1.0),  # the ramp completes
        (500, 100, 50, 1.0),
        (100, 100, 0, 1.0),  # ramp 0 == the old hard gate exactly
        (99, 100, 0, 0.0),
    ],
)
def test_the_conditioning_ramp_eases_the_gate_in(step, warmup, ramp, expected):
    from dawn.flow.trainer import FlowTrainer

    trainer = FlowTrainer.__new__(FlowTrainer)  # the schedule is a function of three ints only
    trainer.step, trainer.warmup_steps, trainer.warmup_ramp_steps = step, warmup, ramp
    assert trainer.conditioning_scale() == pytest.approx(expected)


# --- 2. premise diagnostics ---------------------------------------------------------------------


def test_patch_retrieval_is_perfect_when_the_two_encoders_agree():
    tokens = torch.randn(2, 12, 8)
    out = patch_retrieval(tokens, tokens.clone(), (3, 4))
    assert torch.allclose(out["top1"], torch.ones(2))
    assert torch.allclose(out["within_1"], torch.ones(2))


def test_patch_retrieval_separates_a_one_patch_shift_from_real_confusion():
    """The distinction the metric exists for: misaligned labels vs features that do not localize.

    A frozen target shifted one column to the left retrieves the *neighbour*, so `top1`
    collapses while `within_1` stays high. Shuffled targets lose both.
    """
    grid = (4, 5)
    tokens = torch.randn(1, grid[0] * grid[1], 16)
    shifted = tokens.view(1, *grid, 16).roll(shifts=1, dims=2).reshape(1, -1, 16)
    shift = patch_retrieval(tokens, shifted, grid)
    assert shift["top1"].item() == pytest.approx(0.0)
    # Every patch but the column that wrapped around the frame edge retrieves its neighbour.
    assert shift["within_1"].item() == pytest.approx(1.0 - 1.0 / grid[1])

    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(grid[0] * grid[1], generator=generator)
    confused = patch_retrieval(tokens, tokens[:, order], grid)
    assert confused["within_1"].item() < 0.5


def test_patch_retrieval_ignores_masked_patches():
    tokens = torch.randn(1, 6, 4)
    broken = tokens.clone()
    broken[:, 3:] = torch.randn(1, 3, 4)  # the last three patches cannot retrieve themselves
    weight = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    assert patch_retrieval(tokens, broken, (2, 3), weight)["top1"].item() == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [(torch.randn(1, 6, 4), (3, 3)), (torch.randn(1, 5, 4), (2, 3))])
def test_patch_retrieval_rejects_a_grid_that_does_not_match(bad):
    tokens, grid = bad
    with pytest.raises(ValueError):
        patch_retrieval(tokens, tokens, grid)


def test_sobel_finds_a_step_edge_and_ignores_a_flat_field():
    flat = torch.full((1, 1, 8, 8), 0.4)
    assert float(sobel_magnitude(flat).abs().max()) == pytest.approx(0.0, abs=1e-6)
    step = torch.zeros(1, 1, 8, 8)
    step[..., 4:] = 1.0
    magnitude = sobel_magnitude(step)
    assert float(magnitude[0, 0, :, 3:5].min()) > 0
    assert float(magnitude[0, 0, :, 0].max()) == pytest.approx(0.0, abs=1e-6)


def test_edge_mask_keeps_about_the_requested_share_of_the_kept_pixels():
    torch.manual_seed(0)
    image = torch.rand(3, 1, 32, 32)
    mask = torch.ones(3, 1, 32, 32)
    edge = edge_quantile_mask(image, mask, quantile=0.9)
    share = edge.flatten(1).mean(dim=1)
    assert ((share > 0.05) & (share < 0.2)).all(), share


def test_edge_mask_never_marks_a_pixel_the_pixel_mask_dropped():
    torch.manual_seed(0)
    image = torch.rand(1, 1, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., :8] = 1.0
    edge = edge_quantile_mask(image, mask, 0.9)
    assert float(edge[..., 8:].sum()) == 0.0
    assert float(edge[..., :8].sum()) > 0.0


def test_edge_mask_of_an_entirely_masked_image_is_empty_not_wrong():
    image = torch.rand(1, 1, 8, 8)
    edge = edge_quantile_mask(image, torch.zeros(1, 1, 8, 8), 0.9)
    assert float(edge.sum()) == 0.0


def test_edge_mask_concentrates_on_the_boundary_of_a_two_tone_frame():
    """The case the oracle's `*_edge` metric is for: a sharp boundary in an otherwise flat frame.

    More than 90% of this frame is flat, so the 0.9 quantile of gradient magnitude is exactly
    zero. Thresholding on `>=` alone would mark every pixel -- the opposite of an edge mask.
    """
    image = torch.zeros(1, 1, 24, 24)
    image[..., 12:] = 1.0
    edge = edge_quantile_mask(image, torch.ones(1, 1, 24, 24), 0.9)
    assert 0 < float(edge.sum()) < 0.25 * edge.numel()
    assert float(edge[..., 10:14].sum()) == float(edge.sum()), "confined to the boundary"


def test_an_entirely_flat_frame_has_no_edges_rather_than_all_of_them():
    edge = edge_quantile_mask(torch.full((1, 1, 16, 16), 0.3), torch.ones(1, 1, 16, 16), 0.9)
    assert float(edge.sum()) == 0.0


# --- 3. budget arithmetic and the local record ----------------------------------------------------


def _cfg(batch=16, steps=20000, warmup=3000, val_every=500, save_every=5000, timesteps=4):
    return {
        "flow": {
            "num_train_steps": steps,
            "warmup_steps": warmup,
            "timesteps_per_image": timesteps,
        },
        "optim": {"batch_size": batch},
        "logging": {"val_every": val_every, "save_every": save_every},
    }


def test_budget_summary_reports_the_real_configuration_in_epochs():
    out = budget_summary(_cfg(), 3662)
    assert out["steps_per_epoch"] == pytest.approx(228.9, abs=0.1)
    assert out["epochs"] == pytest.approx(87.4, abs=0.1)
    assert out["detach_epochs"] == pytest.approx(13.1, abs=0.1)
    assert out["val_points"] == 40.0, "val_every 500 over 20K steps"
    assert out["samples_per_step"] == 64.0


def test_budget_summary_exposes_the_rescaling_trap():
    """Batch 128 at the same step count: 29 steps an epoch, and the detach never lifts."""
    out = budget_summary(_cfg(batch=128, steps=2900), 3662)
    assert out["steps_per_epoch"] == pytest.approx(28.6, abs=0.1)
    assert out["epochs"] < 102
    assert out["detach_epochs"] > out["epochs"], "the conditioning stays detached the whole run"


def test_csv_log_writes_one_file_per_kind_with_a_stable_header(tmp_path):
    from scripts.train_flow import CsvLog

    log = CsvLog(tmp_path)
    log.write(100, "train", {"loss/total": 1.0})
    log.write(200, "train", {"loss/total": 0.5})
    log.write(200, "val_box_aligned", {"val_box_aligned/psnr_L/euler": 21.0})
    lines = (tmp_path / "train.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "step,loss/total"
    assert lines[1:] == ["100,1.0", "200,0.5"]
    assert (tmp_path / "val_box_aligned.csv").exists()


def test_a_column_that_appears_late_lands_in_the_header_retroactively(tmp_path):
    """`beta1` and `peak_vram_gb` do not exist on every row; the file must stay readable."""
    from scripts.train_flow import CsvLog

    log = CsvLog(tmp_path)
    log.write(1, "train", {"a": 1.0})
    log.write(2, "train", {"a": 2.0, "b": 9.0})
    lines = (tmp_path / "train.csv").read_text(encoding="utf-8").splitlines()
    assert lines == ["step,a,b", "1,1.0,", "2,2.0,9.0"]


def test_the_local_record_holds_exactly_the_keys_wandb_got(tmp_path):
    """The whole point of `RunLog`: the mirrors cannot drift from the live curves."""
    from scripts.train_flow import RunLog

    cfg = {"logging": {"wandb_mode": "disabled", "metrics_csv": True}}
    run = RunLog(cfg, tmp_path)
    values = {"loss/total": 1.5, "grad/velocity": 0.25, "contrib/flow": 0.8}
    run.log("train", 300, values)
    run.close()

    record = json.loads((tmp_path / "train_log.jsonl").read_text(encoding="utf-8").strip())
    assert record == {"step": 300, "kind": "train", **values}
    header = (tmp_path / "metrics" / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[1:] == list(values)


def test_metrics_csv_can_be_switched_off(tmp_path):
    from scripts.train_flow import RunLog

    run = RunLog({"logging": {"wandb_mode": "disabled", "metrics_csv": False}}, tmp_path)
    run.log("train", 1, {"loss/total": 1.0})
    run.close()
    assert not (tmp_path / "metrics").exists()
    assert (tmp_path / "train_log.jsonl").exists(), "the JSONL mirror is not optional"


def test_the_run_config_is_written_beside_the_metrics(tmp_path):
    from scripts.train_flow import write_run_config

    path = write_run_config({"flow": {"num_train_steps": 20000}}, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["flow"]["num_train_steps"] == 20000


def test_every_selection_metric_names_a_key_validate_actually_returns():
    """A typo here would surface as a KeyError at the first validation of an overnight run."""
    from dawn.flow.trainer import FlowTrainer
    from scripts.train_flow import SELECTION

    assert set(SELECTION) == {"ciede2000", "psnr_L"}
    documented = FlowTrainer.validate.__doc__
    for key, _ in SELECTION.values():
        assert key.split("/")[0] in documented
