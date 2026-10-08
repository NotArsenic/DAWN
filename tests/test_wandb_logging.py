"""The run-logging plumbing of `dawn.runlog`: Weights & Biases, and the local mirrors it fronts.

wandb is an optional dependency and is not installed in CI, so a fake module stands in. These
tests exist because `WandbRun` is otherwise only exercised by a real training run: a typo in it
surfaces at the start of an overnight job, not here. The same goes for the two settings that
decide *where* a run lands -- `wandb_project` and `wandb_entity` -- whose failure mode is a run
that trains perfectly well into the wrong workspace.

`dawn.runlog` is shared by both training stages, so nothing here imports torch except the one
test that the stage-1 script still re-exports the names older code imports from it.
"""

import json
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.runlog import WATCH_SCOPES, RunLog, WandbRun  # noqa: E402


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


def _cfg(mode="online", scope="velocity", freq=500, **logging_extra):
    """The `logging` section `WandbRun` reads. `wandb_entity` is absent unless asked for."""
    cfg = {
        "logging": {
            "wandb_project": "dawn",
            "wandb_mode": mode,
            "watch_scope": scope,
            "watch_freq": freq,
        }
    }
    cfg["logging"].update(logging_extra)
    return cfg


class _Trainer:
    """Stands in for FlowTrainer: the stage-1 scopes only ever read these three attributes."""

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


# --- where the run lands: project and entity -----------------------------------------------------


def test_the_entity_is_forwarded_when_one_is_configured(fake_wandb):
    WandbRun(_cfg(wandb_entity="my-team"))
    assert fake_wandb.calls["init"][0]["entity"] == "my-team"


@pytest.mark.parametrize("cfg", [_cfg(), _cfg(wandb_entity=None), _cfg(wandb_entity="")])
def test_no_entity_means_the_kwarg_is_absent_not_none(cfg, fake_wandb):
    """`entity=None` is not the same as not passing it: the account default must be left to win.

    Someone without access to the configured entity has to be able to run the script unchanged,
    which only works if wandb never sees the keyword at all.
    """
    WandbRun(cfg)
    assert "entity" not in fake_wandb.calls["init"][0]


def test_the_init_line_names_the_project_and_the_entity(fake_wandb, caplog):
    """A run that went to the wrong workspace must be visible in the log, not only in a browser."""
    with caplog.at_level(logging.INFO, logger="dawn.runlog"):
        WandbRun(_cfg(mode="offline", wandb_entity="my-team"))
    assert "my-team" in caplog.text and "dawn" in caplog.text and "offline" in caplog.text


def test_the_init_line_says_so_when_the_account_default_is_used(fake_wandb, caplog):
    with caplog.at_level(logging.INFO, logger="dawn.runlog"):
        WandbRun(_cfg())
    assert "account default" in caplog.text


# --- wandb.watch and its scopes ------------------------------------------------------------------


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


def test_another_stage_supplies_its_own_scope_map(fake_wandb):
    """Stage 2's trainer holds none of stage 1's nets, so the scope map travels with the caller."""

    class _RefinerTrainer:
        def __init__(self):
            self.refiner = object()

    scopes = {"none": (), "refiner": ("refiner",)}
    trainer, cfg = _RefinerTrainer(), _cfg(scope="refiner")
    WandbRun(cfg).watch(trainer, cfg, scopes)
    assert fake_wandb.calls["watch"][0][0] == [trainer.refiner]
    # Stage 1's map is still the default, and stage 2's scope is not in it.
    with pytest.raises(ValueError, match="watch_scope must be one of"):
        WandbRun(cfg).watch(trainer, cfg)


def test_a_scope_naming_a_module_the_trainer_lacks_fails_at_the_start(fake_wandb):
    """The wrong stage's map, caught where it is cheap rather than as a bare AttributeError."""
    cfg = _cfg(scope="velocity")
    with pytest.raises(ValueError, match="does not have"):
        WandbRun(cfg).watch(object(), cfg)


# --- the local mirrors ---------------------------------------------------------------------------


def test_a_run_log_round_trip_writes_the_jsonl_and_the_per_kind_csv(tmp_path):
    """The point of `RunLog`: all three records hold the keys the caller handed it, verbatim."""
    run = RunLog({"logging": {"wandb_mode": "disabled", "metrics_csv": True}}, tmp_path)
    values = {"loss/total": 1.5, "grad/velocity": 0.25, "contrib/flow": 0.8}
    run.log("train", 300, values)
    run.log("val_box_aligned", 300, {"val_box_aligned/ciede2000/euler": 4.0})
    run.close()

    record = json.loads((tmp_path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record == {"step": 300, "kind": "train", **values}
    header = (tmp_path / "metrics" / "train.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ["step", *values]
    assert (tmp_path / "metrics" / "val_box_aligned.csv").exists(), "one CSV per kind of record"


# --- the shipped config and the stage-1 script ---------------------------------------------------


def test_the_shipped_config_uses_a_scope_that_exists():
    from dawn.flow.loaders import load_config

    pytest.importorskip("omegaconf")
    cfg = load_config("configs/flow.yaml", "configs/data.yaml", [])
    assert cfg["logging"]["watch_scope"] in WATCH_SCOPES
    assert int(cfg["logging"]["watch_freq"]) > 0


def test_the_shipped_config_names_the_stage_1_project_and_the_entity():
    from dawn.flow.loaders import load_config

    pytest.importorskip("omegaconf")
    logging_cfg = load_config("configs/flow.yaml", "configs/data.yaml", [])["logging"]
    assert logging_cfg["wandb_project"] == "dawn", "stage 2 is the one that logs to dawn-refiner"
    assert logging_cfg["wandb_entity"] == "my-team"


def test_the_stage_1_script_still_re_exports_the_names():
    """`from scripts.train_flow import RunLog` is what the older tests and notes use."""
    pytest.importorskip("torch")
    from dawn import runlog
    from scripts import train_flow

    for name in ("CsvLog", "JsonlLog", "RunLog", "WandbRun", "write_run_config"):
        assert getattr(train_flow, name) is getattr(runlog, name), name
