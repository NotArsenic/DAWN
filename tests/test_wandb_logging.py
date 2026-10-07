"""Weights & Biases plumbing in scripts/train_flow.py.

wandb is an optional dependency and is not installed in CI, so a fake module stands in. These
tests exist because `WandbRun` is otherwise only exercised by a real training run: a typo in it
surfaces at the start of an overnight job, not here.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_flow import WandbRun  # noqa: E402


class FakeRun:
    def __init__(self):
        self.logged, self.finished = [], False

    def log(self, values, step=None):
        self.logged.append((step, values))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    """A stand-in `wandb` module that records init/watch calls."""
    module = types.ModuleType("wandb")
    module.calls = {"init": [], "watch": []}

    def init(**kwargs):
        module.calls["init"].append(kwargs)
        return FakeRun()

    def watch(models, **kwargs):
        module.calls["watch"].append((models, kwargs))

    module.init, module.watch = init, watch
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def _cfg(mode="online", scope="velocity", freq=500):
    return {
        "logging": {
            "wandb_project": "dawn",
            "wandb_mode": mode,
            "watch_scope": scope,
            "watch_freq": freq,
        }
    }


class _Trainer:
    """Stands in for FlowTrainer: watch() only ever reads these three attributes."""

    def __init__(self):
        self.velocity, self.encoder, self.color_head = object(), object(), object()


def test_a_disabled_run_never_imports_or_calls_wandb(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)  # importing it would raise
    run = WandbRun(_cfg(mode="disabled"))
    assert run.run is None
    run.watch(_Trainer(), _cfg(mode="disabled"))
    run.log({"loss": 1.0}, step=1)
    run.finish()  # all no-ops, no exception


def test_a_missing_wandb_degrades_to_console_logging(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    run = WandbRun(_cfg(mode="online"))
    assert run.run is None, "an uninstalled wandb must not stop a run"
    run.log({"loss": 1.0}, step=1)


def test_init_passes_the_project_and_mode(fake_wandb):
    run = WandbRun(_cfg(mode="offline"))
    assert run.run is not None
    assert fake_wandb.calls["init"][0]["project"] == "dawn"
    assert fake_wandb.calls["init"][0]["mode"] == "offline"


def test_log_and_finish_reach_the_run(fake_wandb):
    run = WandbRun(_cfg())
    run.log({"loss/total": 0.5}, step=7)
    assert run.run.logged == [(7, {"loss/total": 0.5})]
    run.finish()
    assert run.run.finished


def test_watch_velocity_hooks_only_the_velocity_net(fake_wandb):
    trainer, cfg = _Trainer(), _cfg(scope="velocity", freq=250)
    run = WandbRun(cfg)
    run.watch(trainer, cfg)

    models, kwargs = fake_wandb.calls["watch"][0]
    assert models == [trainer.velocity], "the frozen DINO backbone must stay unhooked by default"
    assert kwargs == {"log": "all", "log_freq": 250}


def test_watch_all_adds_the_encoder_and_the_colour_head(fake_wandb):
    trainer, cfg = _Trainer(), _cfg(scope="all")
    run = WandbRun(cfg)
    run.watch(trainer, cfg)

    models, _ = fake_wandb.calls["watch"][0]
    assert models == [trainer.velocity, trainer.encoder, trainer.color_head]


def test_watch_none_calls_nothing(fake_wandb):
    cfg = _cfg(scope="none")
    WandbRun(cfg).watch(_Trainer(), cfg)
    assert fake_wandb.calls["watch"] == []


def test_a_bad_scope_is_caught_even_when_wandb_is_off(fake_wandb):
    """It must fail at the start of the run, not silently log nothing for 20K steps."""
    for mode in ("online", "disabled"):
        cfg = _cfg(mode=mode, scope="trainable")  # the old name, now invalid
        with pytest.raises(ValueError, match="watch_scope must be one of"):
            WandbRun(cfg).watch(_Trainer(), cfg)


def test_the_shipped_config_uses_a_scope_that_exists():
    from dawn.flow.loaders import load_config

    pytest.importorskip("omegaconf")
    cfg = load_config("configs/flow.yaml", "configs/data.yaml", [])
    assert cfg["logging"]["watch_scope"] in WandbRun.SCOPES
    assert int(cfg["logging"]["watch_freq"]) > 0
