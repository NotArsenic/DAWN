"""Experiment tracking and metric logging to Weights & Biases, JSONL, and CSV."""

from __future__ import annotations

import csv
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

WATCH_SCOPES: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "velocity": ("velocity",),
    "all": ("velocity", "encoder", "color_head"),
}


class WandbRun:
    """Weights & Biases run wrapper handling optional initialization and logging."""

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
        entity = log_cfg.get("wandb_entity") or None
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
        """Hook model modules for gradient and parameter histogram logging in Weights & Biases.

        Args:
            trainer: Trainer instance containing model attributes.
            cfg: Configuration dictionary with logging settings.
            scopes: Mapping of scope names to attribute names to watch.

        Raises:
            ValueError: If watch_scope is invalid or named attributes are missing.
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
    """Appends structured metrics records as JSON lines."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, step: int, kind: str, values: dict) -> None:
        self.handle.write(json.dumps({"step": step, "kind": kind, **values}) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class CsvLog:
    """Writes per-metric-kind records to separate CSV files."""

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
    """Unified logger dispatching metrics to Weights & Biases, JSONL, and CSV destinations."""

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
        """Hook model modules for histogram logging."""
        self.wandb.watch(trainer, cfg, scopes)

    def close(self) -> None:
        self.wandb.finish()
        self.jsonl.close()


def write_run_config(cfg: dict, out_dir: Path) -> Path:
    """Write merged experiment configuration to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_config.json"
    path.write_text(json.dumps(cfg, indent=2, default=str) + "\n", encoding="utf-8")
    return path
