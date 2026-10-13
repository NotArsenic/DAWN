"""Tests for learning rate and momentum scheduling policies."""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow.trainer import (  # noqa: E402
    MomentumCycler,
    build_schedule,
    cosine_warmup_schedule,
    one_cycle_schedule,
)

TOTAL = 200


def _optimizer(lrs=(2e-4, 1e-4)):
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr, "name": f"g{i}"}
        for i, lr in enumerate(lrs)
    ]
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def _trace(optimizer, scheduler, steps=TOTAL, cycler=None):
    out = []
    for _ in range(steps):
        out.append([group["lr"] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
        if cycler is not None:
            cycler.step()
    return out


def test_one_cycle_ramps_to_the_configured_lr_then_anneals_to_nearly_zero():
    opt = _optimizer()
    lrs = [row[0] for row in _trace(opt, one_cycle_schedule(opt, TOTAL, pct_start=0.25))]
    peak = max(lrs)
    assert peak == pytest.approx(2e-4, rel=1e-3), "the group's lr is the PEAK, never exceeded"
    assert lrs.index(peak) == pytest.approx(0.25 * TOTAL, abs=2), "peaks at pct_start"
    assert lrs[0] == pytest.approx(2e-4 / 25, rel=1e-6), "starts at lr / div_factor"
    assert lrs[-1] < 2e-4 / 1000, "ends near zero"
    assert lrs[:50] == sorted(lrs[:50]), "monotone up to the peak"
    assert lrs[50:] == sorted(lrs[50:], reverse=True), "monotone down after it"


def test_one_cycle_keeps_the_ratio_between_param_groups():
    """Verify constant learning rate ratio between parameter groups throughout training."""
    opt = _optimizer((2e-4, 1e-4))
    for fast, slow in _trace(opt, one_cycle_schedule(opt, TOTAL)):
        assert slow == pytest.approx(fast / 2, rel=1e-6)


def test_one_cycle_has_a_much_longer_ramp_than_the_cosine_warmup():
    """Verify OneCycle schedule has a longer warmup phase than cosine warmup."""
    cos_opt, one_opt = _optimizer(), _optimizer()
    cos = [row[0] for row in _trace(cos_opt, cosine_warmup_schedule(cos_opt, 10, TOTAL))]
    one = [row[0] for row in _trace(one_opt, one_cycle_schedule(one_opt, TOTAL, pct_start=0.2))]
    assert cos.index(max(cos)) < one.index(max(one))
    assert one.index(max(one)) == pytest.approx(0.2 * TOTAL, abs=2)


def test_stepping_past_the_budget_clamps_instead_of_raising():
    """Verify scheduler clamps learning rate instead of raising past total budget."""
    opt = _optimizer()
    scheduler = one_cycle_schedule(opt, TOTAL)
    _trace(opt, scheduler, steps=TOTAL + 25)
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-4 / 1e4, rel=1e-6)


@pytest.mark.parametrize("pct_start, div", [(0.0, 25.0), (1.0, 25.0), (0.2, 0.0), (0.2, -1.0)])
def test_bad_one_cycle_settings_are_clear_errors(pct_start, div):
    with pytest.raises(ValueError):
        one_cycle_schedule(_optimizer(), TOTAL, pct_start=pct_start, div_factor=div)


def test_momentum_cycles_opposite_the_learning_rate():
    opt = _optimizer()
    scheduler = one_cycle_schedule(opt, TOTAL, pct_start=0.25)
    cycler = MomentumCycler(opt, TOTAL, pct_start=0.25)
    betas, lrs = [], []
    for _ in range(TOTAL):
        betas.append(opt.param_groups[0]["betas"][0])
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        scheduler.step()
        cycler.step()
    assert betas[0] == pytest.approx(0.95)
    assert min(betas) == pytest.approx(0.85, abs=1e-6)
    assert betas[-1] == pytest.approx(0.95, abs=1e-3)
    assert betas.index(min(betas)) == pytest.approx(lrs.index(max(lrs)), abs=2), "troughs at peak"
    assert all(0.85 - 1e-9 <= b <= 0.95 + 1e-9 for b in betas)


def test_momentum_leaves_beta2_alone():
    opt = _optimizer()
    cycler = MomentumCycler(opt, TOTAL)
    for _ in range(50):
        cycler.step()
    assert opt.param_groups[0]["betas"][1] == 0.999


def test_a_disabled_cycler_never_touches_the_betas():
    opt = _optimizer()
    cycler = MomentumCycler(opt, TOTAL, enabled=False)
    for _ in range(50):
        cycler.step()
    assert opt.param_groups[0]["betas"] == (0.9, 0.999)


def test_momentum_state_round_trips():
    opt = _optimizer()
    cycler = MomentumCycler(opt, TOTAL, pct_start=0.25)
    for _ in range(60):
        cycler.step()
    fresh = MomentumCycler(_optimizer(), TOTAL, pct_start=0.25)
    fresh.load_state_dict(cycler.state_dict())
    assert fresh.step_count == 60
    assert fresh.optimizer.param_groups[0]["betas"][0] == pytest.approx(
        opt.param_groups[0]["betas"][0]
    )


def _cfg(schedule, **one_cycle):
    return {
        "optim": {
            "schedule": schedule,
            "lr_warmup_steps": 10,
            "one_cycle": {"pct_start": 0.2, **one_cycle},
        },
        "flow": {"num_train_steps": TOTAL},
    }


def test_build_schedule_picks_the_configured_one():
    opt = _optimizer()
    scheduler, cycler = build_schedule(opt, _cfg("cosine"))
    assert not cycler.enabled, "the plain cosine does not cycle momentum"
    lrs = [row[0] for row in _trace(opt, scheduler, cycler=cycler)]
    assert lrs.index(max(lrs)) == pytest.approx(10, abs=1), "cosine peaks at lr_warmup_steps"

    opt = _optimizer()
    scheduler, cycler = build_schedule(opt, _cfg("one_cycle"))
    assert cycler.enabled
    lrs = [row[0] for row in _trace(opt, scheduler, cycler=cycler)]
    assert lrs.index(max(lrs)) == pytest.approx(0.2 * TOTAL, abs=2)


def test_momentum_can_be_switched_off_for_one_cycle():
    _, cycler = build_schedule(_optimizer(), _cfg("one_cycle", cycle_momentum=False))
    assert not cycler.enabled


def test_an_unknown_schedule_name_is_a_clear_error():
    with pytest.raises(ValueError, match="cosine.*one_cycle"):
        build_schedule(_optimizer(), _cfg("triangular"))


def test_a_budget_near_its_own_warmup_is_warned_about(caplog):
    """Verify warning is emitted when step budget is close to or below warmup duration."""
    import logging

    pytest.importorskip("omegaconf")
    pytest.importorskip("peft")
    pytest.importorskip("timm")
    from dawn.flow.targets import FitSettings, compute_token_stats, prepare_batch
    from dawn.flow.tokens import TokenLayout
    from dawn.flow.trainer import FlowTrainer
    from tests.flow_helpers import TINY_INPUT, synthetic_batch, tiny_cfg

    def build(num_train_steps, warmup_steps):
        torch.manual_seed(0)
        cfg = tiny_cfg()
        cfg["flow"].update(num_train_steps=num_train_steps, warmup_steps=warmup_steps)
        batch = synthetic_batch(2, seed=0, fallback=[False, True])
        stats = compute_token_stats(
            [prepare_batch(batch, TINY_INPUT, cfg["pairs"])],
            TokenLayout.from_config(cfg),
            FitSettings.from_config(cfg),
        )
        return FlowTrainer.from_config(cfg, stats, "cpu")

    with caplog.at_level(logging.WARNING):
        build(num_train_steps=20, warmup_steps=30)  # the budget ends before the detach lifts
    assert any("NEVER lifts" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        build(num_train_steps=50, warmup_steps=30)  # most of the run still detached
    assert any("under half the run trains jointly" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        build(num_train_steps=100, warmup_steps=15)  # healthy: no warning
    assert not any("warmup_steps" in r.message for r in caplog.records)
