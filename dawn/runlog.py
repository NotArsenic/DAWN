"""Where a training run's numbers go: Weights & Biases, a JSONL mirror and a CSV per kind.

Both training stages record the same way, so this lives in the package rather than in either
script: stage 1 (`scripts/train_flow.py`) and stage 2 (`scripts/train_refiner.py`) each build one
`RunLog` and every metric they produce goes through its single `log` call. That single call site
is the point. Before it existed the JSONL held the raw loss dict while W&B got the prefixed one
plus the attention gates, so the two quietly disagreed about what a run had recorded and the
local files could not be used to rebuild the W&B plots.

W&B is optional in three independent ways, because telemetry must never be what kills a run: the
package may not be installed, `logging.wandb_mode` may be `disabled`, and `logging.wandb_entity`
may be null (the account's own default entity is then used, so a collaborator without access to
the team's can still train). Which project and entity were actually used is logged at INFO on
init -- a run that went to the wrong workspace should be visible in the log, not only in a
browser hours later.

The one thing that genuinely differs between the stages is which modules `wandb.watch` may hook,
since the two trainers hold different nets. So the scope map is the caller's to supply;
`WATCH_SCOPES` is stage 1's and the default.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

# `logging.watch_scope` -> the trainer attributes that scope hooks. Stage 1's map (`FlowTrainer`),
# and the default of `watch`; stage 2's trainer holds none of these modules and passes its own.
WATCH_SCOPES: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "velocity": ("velocity",),
    "all": ("velocity", "encoder", "color_head"),
}


class WandbRun:
    """wandb if it is installed and `logging.wandb_mode` is not 'disabled'; else a no-op.

    Reads `logging.wandb_project`, `logging.wandb_entity` and `logging.wandb_mode`. The entity is
    the W&B team (or user) the run belongs to; **null, empty or absent means the kwarg is not
    passed to `wandb.init` at all**, which is not the same as passing `entity=None` to every
    wandb version, and which is what lets someone without access to the configured entity run
    anyway. The project and entity actually used go into the log at INFO.
    """

    def __init__(self, cfg):
        self.run = None
        log_cfg = cfg["logging"]
        mode = log_cfg["wandb_mode"]
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError:
            logger.warning("wandb is not installed; logging to the console only")
            return
        project = log_cfg["wandb_project"]
        entity = log_cfg.get("wandb_entity") or None  # null, "" and absent all mean "omit it"
        kwargs = {"project": project, "mode": mode, "config": cfg}
        if entity is not None:
            kwargs["entity"] = entity
        self.run = wandb.init(**kwargs)
        logger.info(
            "wandb: project %s, entity %s, mode %s", project, entity or "<account default>", mode
        )

    def log(self, values: dict, step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def watch(self, trainer, cfg, scopes: Mapping[str, tuple[str, ...]] = WATCH_SCOPES) -> None:
        """Weight and gradient histograms through `wandb.watch`'s hooks.

        `wandb.watch` takes whole modules, not parameters, so the scope is the unit of choice.
        Stage 1's scopes (`WATCH_SCOPES`, the default):

            velocity  the 7M-parameter net that learns the field. The default.
            all       plus the encoder and the colour head. The encoder is 29M parameters of
                      FROZEN DINO backbone around ~0.5M of LoRA, and watching it means hooking
                      all of them; use it to debug LoRA, not for a normal run.
            none      off.

        `watch_freq` throttles the logging, not the hooks: those run on every backward pass
        whatever it says, which is why the scope matters more than the frequency.

        Args:
            trainer: the trainer holding the modules; a scope names its attributes.
            cfg: the merged config, read for `logging.watch_scope` and `logging.watch_freq`.
            scopes: scope name -> the trainer attributes it hooks. A stage whose trainer holds
                other nets passes its own map; keep a `none` entry in it.

        Raises:
            ValueError: `logging.watch_scope` is not a key of `scopes`, or the scope names an
                attribute `trainer` does not have (the map belongs to another stage). Both are
                checked before the wandb_mode test, so a typo fails at the start of the run
                instead of silently logging nothing for 20K steps -- debug and probe runs have
                wandb disabled, and that is exactly where a typo gets introduced.
        """
        scope = str(cfg["logging"].get("watch_scope", "none"))
        if scope not in scopes:
            raise ValueError(f"logging.watch_scope must be one of {sorted(scopes)}, got {scope!r}")
        missing = [name for name in scopes[scope] if not hasattr(trainer, name)]
        if missing:
            raise ValueError(
                f"watch_scope {scope!r} names {missing}, which {type(trainer).__name__} does not "
                f"have: pass this stage's own scope map to watch()"
            )
        if self.run is None or scope == "none":
            return
        import wandb

        freq = int(cfg["logging"].get("watch_freq", 500))
        wandb.watch([getattr(trainer, name) for name in scopes[scope]], log="all", log_freq=freq)
        logger.info("  wandb.watch: %s, histograms every %d steps", scope, freq)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


class JsonlLog:
    """One JSON object per line under the output directory: curves without wandb."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, step: int, kind: str, values: dict) -> None:
        self.handle.write(json.dumps({"step": step, "kind": kind, **values}) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class CsvLog:
    """One CSV per kind of record, rewritten whole on every append.

    W&B owns the live curves; these files exist so every plot can be rebuilt from disk months
    later, by someone without the W&B run, in a spreadsheet. One file per kind keeps the columns
    stable and dense -- a single wide file would be mostly empty, because train rows arrive every
    `log_every` steps and validation rows every `val_every`.

    Rewriting the whole file each time (rather than appending) costs nothing at this scale -- a
    20K-step run writes about 200 train rows and 40 of each validation kind -- and buys two
    things: a column that only appears later (`beta1`, `peak_vram_gb`) lands in the header
    retroactively, and the file on disk is always complete, even if the run is killed.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows: dict[str, list[dict]] = {}
        self.columns: dict[str, list[str]] = {}

    def write(self, step: int, kind: str, values: dict) -> None:
        row = {"step": step, **{k: v for k, v in values.items()}}
        rows = self.rows.setdefault(kind, [])
        columns = self.columns.setdefault(kind, [])
        rows.append(row)
        columns.extend(key for key in row if key not in columns)
        path = self.directory / f"{kind}.csv"
        partial = path.with_suffix(".partial")
        with partial.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(partial, path)


class RunLog:
    """The one place a metric is recorded: W&B, a JSONL mirror and a CSV per kind.

    Callers pass the final W&B key names, so the three records hold identical keys by
    construction and the local files really can rebuild the W&B plots. Before this existed the
    JSONL held the raw loss dict while W&B got the prefixed one plus the attention gates, so the
    two quietly disagreed about what a run had recorded.
    """

    def __init__(self, cfg, out_dir: Path):
        self.wandb = WandbRun(cfg)
        self.jsonl = JsonlLog(out_dir / "train_log.jsonl")
        self.csv = CsvLog(out_dir / "metrics") if cfg["logging"].get("metrics_csv", True) else None

    def log(self, kind: str, step: int, values: dict) -> None:
        self.wandb.log(values, step=step)
        self.jsonl.write(step, kind, values)
        if self.csv is not None:
            self.csv.write(step, kind, values)

    def watch(self, trainer, cfg, scopes: Mapping[str, tuple[str, ...]] = WATCH_SCOPES) -> None:
        """`WandbRun.watch`; `scopes` defaults to stage 1's, so stage 2 passes its own map."""
        self.wandb.watch(trainer, cfg, scopes)

    def close(self) -> None:
        self.wandb.finish()
        self.jsonl.close()


def write_run_config(cfg: dict, out_dir: Path) -> Path:
    """The fully merged config beside the metrics, so a plot can be traced to its settings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")
    return path
