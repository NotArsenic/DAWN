"""The training campaign as data: build the plan, run it, resume it, read the results.

`scripts/run_pipeline.py` is a thin CLI over this module, so the parts worth testing -- what the
plan is, what counts as finished, whether a gate passed, how a stage resumes -- are testable
without a GPU and without training anything.

The shape of a campaign (`configs/experiments.yaml`):

    gates       cheap checks, in order, each a gate for the next. smoke -> oracle -> probe ->
                overfit. None of the expensive runs starts until these pass.
    runs        stage 1: `baseline` first and alone, then one ablation per open decision, each
                exactly one knob off the baseline.
    refiner     stage 2, trained off the fitted grid so it does not depend on which run won.
    evaluation  the winning checkpoint on the test split.

Three properties the whole thing is built around:

1. **Resumable.** A stage that finished is skipped; a training run with a partial `flow_last.pt`
   is restarted with `--resume`. An overnight sequence that dies on stage 5 keeps stages 1-4. The
   record is `pipeline_state.json`, written after every stage.
2. **Gates really gate.** A failed check stops the campaign and prints the configured `on_fail`,
   which says what to do rather than only what broke. The plan's own decision rules live there:
   24 dB on the oracle, the overfit run approaching its oracle, the refiner earning its place.
3. **The cost is known before it is spent.** `gpu_probe` measures seconds/step, and every later
   estimate is rescaled from it, so "6 runs x 8 hours" stops being a guess made in a config file.

Nothing here imports torch: it is subprocess orchestration and CSV reading, so it stays importable
and testable anywhere.
"""

from __future__ import annotations

import json
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

STATE_FILE = "pipeline_state.json"
# Stage kinds, in the order a campaign runs them.
GATE, TRAIN, REFINER, EVAL = "gate", "train", "refiner", "eval"


@dataclass(frozen=True)
class Stage:
    """One step of the campaign: a command, where it writes, and how to tell it finished.

    Args:
        name: Unique; also the directory under the campaign root and the `--only` selector.
        kind: `gate`, `train`, `refiner` or `eval`.
        command: argv WITHOUT the interpreter; `scripts/...` paths are resolved against the repo.
        out_dir: Where the stage writes, or None for a stage that writes outside the campaign
            root (the oracle and the smoke test do).
        done: Files that must all exist for the stage to count as finished.
        resumable_from: A checkpoint that, if present without `done`, restarts with `--resume`.
        tests: One line: the question this stage answers. Printed in `--list`.
        check: How to judge the result (`kind` plus its own keys), or None for exit-code only.
        hours: Rough cost, rescaled once the probe has measured seconds/step.
        ablates: For a stage-1 run, the single config key it moves off the baseline.
    """

    name: str
    kind: str
    command: tuple[str, ...]
    out_dir: Path | None = None
    done: tuple[Path, ...] = ()
    resumable_from: Path | None = None
    tests: str = ""
    check: Mapping | None = None
    hours: float = 0.0
    ablates: str = ""

    def is_done(self) -> bool:
        """Every `done` file exists. A stage with no `done` files never counts as finished."""
        return bool(self.done) and all(path.exists() for path in self.done)

    def can_resume(self) -> bool:
        return self.resumable_from is not None and self.resumable_from.exists()

    def argv(self, python: str = sys.executable) -> list[str]:
        """The full command, with `--resume` appended when a partial checkpoint is there."""
        argv = [python, *self.command]
        if not self.is_done() and self.can_resume():
            argv += ["--resume", str(self.resumable_from)]
        return argv

    def shell(self, python: str = sys.executable) -> str:
        return " ".join(shlex.quote(part) for part in self.argv(python))


@dataclass
class StageResult:
    name: str
    status: str  # ok | skipped | failed | gate_failed
    seconds: float = 0.0
    returncode: int = 0
    detail: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("ok", "skipped")


# --- building the plan ---------------------------------------------------------------------------


def _train_command(
    out_dir: Path, overrides: Sequence[str], device: str, extra: Sequence[str] = ()
) -> tuple[str, ...]:
    command = ["scripts/train_flow.py", "--config", "configs/flow.yaml"]
    if overrides:
        command += ["--override", *overrides]
    command += ["--device", device, "--out-dir", str(out_dir), *extra]
    return tuple(command)


def build_plan(
    cfg: Mapping,
    root: Path | None = None,
    include_ablations: bool = True,
    include_gates: bool = True,
) -> list[Stage]:
    """Every stage of the campaign, in run order.

    The order is the dependency: gates, then `baseline`, then the ablations, then stage 2, then
    evaluation. Nothing runs in parallel -- there is one GPU, and two runs sharing it would make
    both their seconds/step meaningless.
    """
    root = Path(root or cfg.get("root", "runs"))
    device = str(cfg.get("device", "cuda"))
    common = [str(p) for p in cfg.get("common_overrides") or []]
    stages: list[Stage] = []

    if include_gates:
        for gate in cfg.get("gates") or []:
            out_dir = root / gate["name"]
            command = list(gate["command"])
            # A gate that trains gets the campaign's shared overrides and its own directory.
            if command[0].endswith("train_flow.py"):
                command = _append_train_options(command, common, device, out_dir)
            stages.append(
                Stage(
                    name=gate["name"],
                    kind=GATE,
                    command=tuple(command),
                    out_dir=out_dir,
                    done=(out_dir / "flow_last.pt",) if "train_flow.py" in command[0] else (),
                    tests=" ".join(str(gate.get("tests", "")).split()),
                    check=gate.get("check"),
                    hours=float(gate.get("hours", 0.0)),
                )
            )

    for run in cfg.get("runs") or []:
        if not include_ablations and run["name"] != "baseline":
            continue
        out_dir = root / run["name"]
        stages.append(
            Stage(
                name=run["name"],
                kind=TRAIN,
                command=_train_command(out_dir, common + list(run.get("overrides") or []), device),
                out_dir=out_dir,
                done=(out_dir / "flow_last.pt",),
                resumable_from=out_dir / "flow_last.pt",
                tests=" ".join(str(run.get("tests", "")).split()),
                hours=float(run.get("hours", 0.0)),
                ablates=str(run.get("ablates", "")),
            )
        )

    refiner = cfg.get("refiner")
    if refiner:
        out_dir = root / refiner["name"]
        command = ["scripts/train_refiner.py", "--device", device, "--out-dir", str(out_dir)]
        if common:
            command += ["--override", *common]
        stages.append(
            Stage(
                name=refiner["name"],
                kind=REFINER,
                command=tuple(command),
                out_dir=out_dir,
                done=(out_dir / "refiner_last.pt",),
                resumable_from=out_dir / "refiner_last.pt",
                tests=" ".join(str(refiner.get("tests", "")).split()),
                check=refiner.get("check"),
                hours=float(refiner.get("hours", 0.0)),
            )
        )

    evaluation = cfg.get("evaluation")
    if evaluation:
        out_dir = root / "evaluation"
        stages.append(
            Stage(
                name="evaluation",
                kind=EVAL,
                command=(),  # filled in at run time: it needs the winning checkpoint
                out_dir=out_dir,
                done=(out_dir / "metrics.json",),
                tests="The winning stage-1 checkpoint on the test split.",
                hours=float(evaluation.get("hours", 0.0)),
            )
        )
    return stages


def _append_train_options(
    command: Sequence[str], common: Sequence[str], device: str, out_dir: Path
) -> list[str]:
    """Merge the campaign's shared overrides into a gate's own `--override` list."""
    command = list(command)
    if "--config" not in command:
        command = [command[0], "--config", "configs/flow.yaml", *command[1:]]
    if common:
        if "--override" in command:
            command.insert(command.index("--override") + 1, *common)
        else:
            command += ["--override", *common]
    return command + ["--device", device, "--out-dir", str(out_dir)]


def evaluation_command(
    checkpoint: Path, out_dir: Path, cfg: Mapping, device: str
) -> tuple[str, ...]:
    """`evaluate.py` on the winning checkpoint. Built late, because the winner is not known
    until every stage-1 run has a validation curve."""
    evaluation = cfg.get("evaluation") or {}
    return (
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--split",
        str(evaluation.get("split", "test_pairs")),
        "--device",
        device,
        "--output",
        str(out_dir),
    )


# --- reading what a run produced -----------------------------------------------------------------


def read_csv_rows(path: Path) -> list[dict]:
    """A `metrics/<kind>.csv` as a list of dicts, numbers parsed, blanks dropped."""
    import csv

    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {}
            for key, value in raw.items():
                if value in (None, ""):
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
        return rows


def run_metrics(out_dir: Path, subset: str = "box_aligned") -> dict:
    """The last validation row of a stage-1 run, keys stripped of their `val_<subset>/` prefix.

    The last row, not the best: `flow_best.pt` already records the best, and a campaign table
    comparing runs wants where each one ended up, with the best shown beside it.
    """
    prefix = f"val_{subset}/"
    rows = read_csv_rows(out_dir / "metrics" / f"val_{subset}.csv")
    if not rows:
        return {}
    last = rows[-1]
    out = {k[len(prefix) :]: v for k, v in last.items() if k.startswith(prefix)}
    out["step"] = last.get("step", 0)
    # The best CIEDE2000 and L* PSNR seen at any validation, which is what flow_best.pt holds.
    for metric, better in (("ciede2000/euler", min), ("psnr_L/euler", max)):
        seen = [r[prefix + metric] for r in rows if prefix + metric in r]
        if seen:
            out[f"best_{metric}"] = better(seen)
    return out


def median_sec_per_step(out_dir: Path) -> float | None:
    """Median `loss/sec_per_step` from a run's train CSV, skipping the first few warm-up steps."""
    rows = read_csv_rows(out_dir / "metrics" / "train.csv")
    values = [r["loss/sec_per_step"] for r in rows if "loss/sec_per_step" in r]
    values = values[2:] or values
    return statistics.median(values) if values else None


def rescale_hours(stages: Sequence[Stage], measured: float, reference: float) -> dict[str, float]:
    """Each stage's estimated hours, scaled by how the measured step time compares with the
    estimate the config was written against. Gates keep their own numbers: they are short and
    dominated by data loading, not by the step time."""
    if not measured or not reference or measured <= 0:
        return {s.name: s.hours for s in stages}
    factor = measured / reference
    return {s.name: (s.hours if s.kind == GATE else s.hours * factor) for s in stages}


# --- checks --------------------------------------------------------------------------------------


def check_stage(stage: Stage, root: Path) -> tuple[bool, str]:
    """Judge a finished stage against its configured `check`. `(passed, detail)`.

    A stage with no check passes on its exit code alone, which the caller has already seen.
    """
    check = stage.check
    if not check:
        return True, ""
    kind = str(check.get("kind", ""))
    if kind == "oracle_psnr":
        return _check_oracle(check, root)
    if kind == "val_metric":
        return _check_val_metric(stage, check)
    return True, f"unknown check kind {kind!r}, skipped"


def _check_oracle(check: Mapping, root: Path) -> tuple[bool, str]:
    rows_path = Path(str(check.get("rows", "outputs/oracle/rows_val.jsonl")))
    if not rows_path.is_absolute():
        rows_path = root.parent / rows_path if root.name else rows_path
    if not rows_path.exists():
        rows_path = Path(str(check.get("rows", "outputs/oracle/rows_val.jsonl")))
    if not rows_path.exists():
        return False, f"no oracle rows at {rows_path}"
    metric = str(check.get("metric", "psnr_grid_r1"))
    values = []
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if metric in row:
                values.append(float(row[metric]))
    if not values:
        return False, f"{rows_path} has no {metric}"
    mean = sum(values) / len(values)
    floor = float(check.get("min", 0.0))
    return mean >= floor, f"{metric} {mean:.2f} dB over {len(values)} pairs (needs >= {floor})"


def _check_val_metric(stage: Stage, check: Mapping) -> tuple[bool, str]:
    if stage.out_dir is None:
        return True, ""
    scores = run_metrics(stage.out_dir)
    if not scores:
        # A refiner run writes its own subset file; fall back to whatever validation CSV exists.
        for path in sorted((stage.out_dir / "metrics").glob("val*.csv")):
            rows = read_csv_rows(path)
            if rows:
                scores = {k.split("/", 1)[-1]: v for k, v in rows[-1].items() if "/" in k}
                scores.update({k: v for k, v in rows[-1].items() if "/" not in k})
                break
    metric = str(check.get("metric", ""))
    value = scores.get(metric)
    if value is None:
        return False, f"no {metric} in {stage.out_dir}/metrics"
    reference = check.get("relative_to")
    if reference:
        target = scores.get(str(reference))
        if target is None:
            return False, f"no {reference} to compare {metric} against"
        fraction = float(check.get("min_fraction", 0.0))
        passed = value >= target * fraction
        return passed, f"{metric} {value:.3f} vs {reference} {target:.3f} (needs >= {fraction:.0%})"
    floor = float(check.get("min", 0.0))
    return value >= floor, f"{metric} {value:.4f} (needs >= {floor})"


# --- the comparison table ------------------------------------------------------------------------


def summary_table(cfg: Mapping, root: Path, subset: str = "box_aligned") -> str:
    """A markdown table of every stage-1 run that has a validation curve.

    This is the payoff of the campaign: one table, read down the `CIEDE2000` column, with the
    knob each row moved named beside it so a difference can be attributed to something.
    """
    rows = []
    for run in cfg.get("runs") or []:
        scores = run_metrics(root / run["name"], subset)
        if not scores:
            continue
        rows.append((run["name"], str(run.get("ablates", "") or "the reference"), scores))
    if not rows:
        return "No finished stage-1 run has a validation curve yet."

    header = (
        "| run | knob | step | CIEDE2000 | best | L* PSNR | best | gap | colourfulness |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    baseline = next((s for n, _, s in rows if n == "baseline"), None)
    for name, knob, s in rows:
        delta = ""
        if baseline and name != "baseline" and "ciede2000/euler" in s:
            diff = s["ciede2000/euler"] - baseline.get("ciede2000/euler", s["ciede2000/euler"])
            delta = f" ({diff:+.3f})"
        lines.append(
            f"| `{name}` | {knob} | {int(s.get('step', 0))} "
            f"| {s.get('ciede2000/euler', float('nan')):.3f}{delta} "
            f"| {s.get('best_ciede2000/euler', float('nan')):.3f} "
            f"| {s.get('psnr_L/euler', float('nan')):.2f} "
            f"| {s.get('best_psnr_L/euler', float('nan')):.2f} "
            f"| {s.get('psnr_L/gap_to_oracle', float('nan')):.2f} "
            f"| {s.get('colorfulness/ratio_euler', float('nan')):.2f} |"
        )
    lines += [
        "",
        f"CIEDE2000 is lower-is-better and is what `flow_best.pt` follows; the bracketed number "
        f"is the difference from `baseline`. Colourfulness is a ratio of the ground truth's, so "
        f"1.00 is as vivid as the real photo. Subset: `{subset}`.",
    ]
    return "\n".join(lines)


def pick_winner(cfg: Mapping, root: Path, subset: str = "box_aligned") -> tuple[str, Path] | None:
    """The run with the best `evaluation.select_by`, and its `flow_best.pt`.

    Ablations that move the token layout (the correction) are still eligible: a checkpoint is
    self-describing, so evaluation and inference pick the layout up from it.
    """
    metric = str((cfg.get("evaluation") or {}).get("select_by", "ciede2000"))
    key = {"ciede2000": "best_ciede2000/euler", "psnr_L": "best_psnr_L/euler"}.get(metric)
    if key is None:
        return None
    higher_is_better = metric == "psnr_L"
    best: tuple[float, str] | None = None
    for run in cfg.get("runs") or []:
        scores = run_metrics(root / run["name"], subset)
        checkpoint = root / run["name"] / "flow_best.pt"
        if key not in scores or not checkpoint.exists():
            continue
        value = scores[key] * (1.0 if higher_is_better else -1.0)
        if best is None or value > best[0]:
            best = (value, run["name"])
    if best is None:
        return None
    return best[1], root / best[1] / "flow_best.pt"


# --- state ---------------------------------------------------------------------------------------


def load_state(root: Path) -> dict:
    path = root / STATE_FILE
    if not path.exists():
        return {"stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(root: Path, state: Mapping) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / STATE_FILE).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def record(state: dict, result: StageResult) -> dict:
    state.setdefault("stages", {})[result.name] = {
        "status": result.status,
        "seconds": round(result.seconds, 1),
        "returncode": result.returncode,
        "detail": result.detail,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return state


# --- running -------------------------------------------------------------------------------------


def run_stage(
    stage: Stage,
    repo: Path,
    log_dir: Path,
    python: str = sys.executable,
    echo: bool = True,
) -> StageResult:
    """Run one stage, streaming its output to the console and to `log_dir/<name>.log`.

    Streaming to both is deliberate: an overnight campaign is watched through `tail -f`, and the
    log is the record of what happened when nobody was watching.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.name}.log"
    argv = stage.argv(python)
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {stage.name} :: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{stage.shell()}\n")
        log.flush()
        process = subprocess.Popen(
            argv,
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        code = process.wait()
    seconds = time.perf_counter() - started
    if code != 0:
        return StageResult(stage.name, "failed", seconds, code, f"exit {code}, see {log_path}")
    return StageResult(stage.name, "ok", seconds, 0)


def plan_table(stages: Iterable[Stage], hours: Mapping[str, float] | None = None) -> str:
    """The campaign as a table: what runs, whether it is already done, and what it costs."""
    stages = list(stages)
    hours = hours or {s.name: s.hours for s in stages}
    lines = ["| # | stage | kind | state | ~hours | what it answers |", "|---|---|---|---|---|---|"]
    for index, stage in enumerate(stages, 1):
        if stage.is_done():
            state = "done"
        elif stage.can_resume():
            state = "**resume**"
        else:
            state = "to run"
        knob = f"`{stage.ablates}` — " if stage.ablates else ""
        lines.append(
            f"| {index} | `{stage.name}` | {stage.kind} | {state} | {hours.get(stage.name, 0):.1f} "
            f"| {knob}{stage.tests} |"
        )
    remaining = sum(hours.get(s.name, 0) for s in stages if not s.is_done())
    left = sum(1 for s in stages if not s.is_done())
    lines += ["", f"**{remaining:.1f} GPU-hours remaining** over {left} stages."]
    return "\n".join(lines)


__all__ = [
    "EVAL",
    "GATE",
    "REFINER",
    "TRAIN",
    "Stage",
    "StageResult",
    "build_plan",
    "check_stage",
    "evaluation_command",
    "load_state",
    "median_sec_per_step",
    "pick_winner",
    "plan_table",
    "read_csv_rows",
    "record",
    "rescale_hours",
    "run_metrics",
    "run_stage",
    "save_state",
    "summary_table",
]
