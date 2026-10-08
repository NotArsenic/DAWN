"""The training campaign as data: build the plan, run it, resume it, read the results.

`scripts/run_pipeline.py` is a thin CLI over this module, so the parts worth testing -- what the
plan is, what counts as finished, whether a gate passed, how a stage resumes -- are testable
without a GPU and without training anything.

The shape of a campaign (`configs/experiments.yaml`):

    gates       cheap checks, in order, each a gate for the next. smoke -> oracle -> probe ->
                overfit. None of the expensive runs starts until these pass.
    runs        stage 1. One run by default: the model that gets shipped.
    ablations   PARKED. One extra run per open decision, each exactly one knob off the
                baseline. Opt in with `--ablations`; they buy understanding of the method, not
                a better model, and they are five times the GPU time of everything else.
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
    steps: int = 0  # optimizer steps this stage really runs; 0 when it does not train

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


def stage_one_runs(cfg: Mapping, include_ablations: bool = True) -> list[Mapping]:
    """The stage-1 runs: always `runs`, plus `ablations` when they are wanted.

    Reading results always wants both -- a summary written after someone ran `--ablations` must
    still show those rows -- so `summary_table` passes True while `build_plan` passes whatever
    the caller asked for.

    Raises:
        ValueError: a name appears in both lists. `Stage.name` is the directory, the `--only`
            selector and the key in `pipeline_state.json`, so a duplicate silently gives two
            stages one output directory. The natural way in is un-parking an ablation by
            copying it into `runs:` instead of moving it.
    """
    runs = list(cfg.get("runs") or [])
    if include_ablations:
        runs += list(cfg.get("ablations") or [])
    seen = [run["name"] for run in runs]
    if len(seen) != len(set(seen)):
        repeated = sorted({n for n in seen if seen.count(n) > 1})
        raise ValueError(f"experiments.yaml names {repeated} in both runs: and ablations:")
    return runs


def command_overrides(command: Sequence[str]) -> list[str]:
    """The `--override` values of a built command, so a gate's own overlay is not lost.

    `configs/overfit_gpu.yaml` sets `num_train_steps: 3000`; costing that gate from the
    campaign's overrides alone resolved it to the baseline's 20,000.
    """
    command = list(command)
    if "--override" not in command:
        return []
    out = []
    for part in command[command.index("--override") + 1 :]:
        if part.startswith("-"):
            break
        out.append(part)
    return out


def _trains(command: Sequence[str]) -> bool:
    """Whether a command runs a training loop at all (`smoke_dino` and the oracle do not)."""
    return any(c.endswith(("train_flow.py", "train_refiner.py")) for c in command)


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
    include_ablations: bool = False,
    include_gates: bool = True,
    repo: Path | None = None,
    extra_overrides: Sequence[str] = (),
) -> list[Stage]:
    """Every stage of the campaign, in run order.

    The order is the dependency: gates, then `baseline`, then the ablations, then stage 2, then
    evaluation. Nothing runs in parallel -- there is one GPU, and two runs sharing it would make
    both their seconds/step meaningless.
    """
    root = Path(root or cfg.get("root", "runs"))
    repo = Path(repo or Path(__file__).resolve().parents[1])
    device = str(cfg.get("device", "cuda"))
    # `extra_overrides` go last so they win: that is how `--extra-override configs/fast.yaml`
    # puts back a key configs/gpu_5090.yaml changed.
    base_overrides = [str(p) for p in cfg.get("common_overrides") or []]
    common = base_overrides + [str(p) for p in extra_overrides]
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
                    steps=(
                        resolve_steps(repo, command_overrides(command), command)
                        if _trains(command)
                        else 0
                    ),
                )
            )

    for run in stage_one_runs(cfg, include_ablations):
        out_dir = root / run["name"]
        overrides = common + list(run.get("overrides") or [])
        stages.append(
            Stage(
                name=run["name"],
                kind=TRAIN,
                command=_train_command(out_dir, overrides, device),
                out_dir=out_dir,
                done=(out_dir / "flow_last.pt",),
                resumable_from=out_dir / "flow_last.pt",
                tests=" ".join(str(run.get("tests", "")).split()),
                hours=float(run.get("hours", 0.0)),
                ablates=str(run.get("ablates", "")),
                steps=resolve_steps(repo, overrides, ("train_flow.py",)),
            )
        )

    refiner = cfg.get("refiner")
    if refiner:
        out_dir = root / refiner["name"]
        # `common_overrides` only, NOT `extra_overrides`. `train_refiner.py` merges its
        # `--override` files AFTER configs/refiner.yaml, so a stage-1 budget overlay like
        # configs/fast.yaml would silently rewrite stage 2's LR warmup and checkpoint spacing
        # while leaving its own step count alone -- on the one stage whose entire output is a
        # go/no-go number. Hardware overlays belong in `common_overrides` and are fine here.
        command = ["scripts/train_refiner.py", "--device", device, "--out-dir", str(out_dir)]
        if base_overrides:
            command += ["--override", *base_overrides]
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
                done=(out_dir / "summary.json",),  # what scripts/evaluate.py writes
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
            # Slice assignment, not `insert(i, *common)`: that only ever worked while
            # `common_overrides` held exactly one entry, and silently raised on the second.
            # The campaign's overrides go FIRST so the gate's own overlay still wins.
            at = command.index("--override") + 1
            command[at:at] = list(common)
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


def resolve_steps(repo: Path, overrides: Sequence[str], command: Sequence[str]) -> int:
    """How many optimizer steps a stage will really run, from its own merged config.

    Read rather than assumed, because `--extra-override configs/fast.yaml` changes
    `flow.num_train_steps` and a cost table that still quoted the full budget would be worse
    than no table. `--max-steps` on the command line wins, as it does in the trainer.
    """
    command = list(command)
    if "--max-steps" in command:
        return int(command[command.index("--max-steps") + 1])
    try:
        from omegaconf import OmegaConf

        merged = OmegaConf.merge(
            OmegaConf.load(repo / "configs/flow.yaml"), OmegaConf.load(repo / "configs/data.yaml")
        )
        for path in overrides:
            merged = OmegaConf.merge(merged, OmegaConf.load(repo / path))
        section = "refiner" if any("train_refiner" in c for c in command) else "flow"
        return int(OmegaConf.to_container(merged, resolve=True)[section]["num_train_steps"])
    except Exception:  # a missing or unreadable overlay: fall back to the declared estimate
        return 0


def rescale_hours(stages: Sequence[Stage], measured: float, reference: float) -> dict[str, float]:
    """Each stage's estimated hours, given the step time the probe measured.

    A stage-1 run whose real step count is known is costed directly as `steps x seconds`,
    which is exact once the probe has run and stays right when an overlay changes the budget.
    Gates and the refiner keep their declared estimate whatever the probe measures -- see the
    comment in the loop for why neither shares stage 1's per-step profile.
    """
    # Before the probe has run, the reference step time stands in. It is a single number, so it
    # cannot know that an overlay halved `timesteps_per_image`; the probe measures the real one
    # and every number here becomes exact.
    seconds = measured if measured and measured > 0 else reference
    hours = {}
    for stage in stages:
        # A stage with a real step count is costed directly. Gates keep their declared
        # estimate: two of them do not train at all, and the two that do are short and
        # dominated by data loading. So does the refiner -- it has no DINO pass, just two
        # encoder passes' worth of grid fit against a 2,545-parameter conv on a crop, so its
        # 8,000 steps cost nothing like stage 1's. Evaluation IS the probe's profile (one DINO
        # pass plus NFE velocity passes per test pair), so it scales.
        if stage.steps and stage.kind in (TRAIN, EVAL):
            hours[stage.name] = stage.steps * seconds / 3600.0
        elif stage.kind in (TRAIN, EVAL) and measured and reference:
            hours[stage.name] = stage.hours * measured / reference
        else:
            hours[stage.name] = stage.hours
    return hours


# --- checks --------------------------------------------------------------------------------------


def is_blocking(stage: Stage) -> bool:
    """Whether a failed check should stop the campaign.

    A gate is blocking: it exists to say the rest is not worth running. The refiner's check is
    not -- "the refiner did not earn its place" is a RESULT, and halting on it would throw away
    the evaluation that is the whole point of the night. `check.blocking: false` says so.
    """
    return bool((stage.check or {}).get("blocking", True))


def check_stage(stage: Stage, root: Path) -> tuple[bool, str]:
    """Judge a finished stage against its configured `check`. `(passed, detail)`.

    A stage with no check passes on its exit code alone, which the caller has already seen.
    Judged from the artifacts on disk, never from `pipeline_state.json`, so re-running a
    campaign re-evaluates a gate that failed rather than skipping it as "done".
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
    for run in stage_one_runs(cfg):
        scores = run_metrics(root / run["name"], subset)
        if not scores:
            continue
        rows.append((run["name"], str(run.get("ablates", "") or "the reference"), scores))
    if not rows:
        return "No finished stage-1 run has a validation curve yet."

    header = (
        "| run | knob | step | CIEDE2000 | best | L* PSNR | best | gap | colourfulness |\n"
        "|---|---|---|---|---|---|---|---|---|"
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


def pick_winner(
    cfg: Mapping,
    root: Path,
    subset: str = "box_aligned",
    names: Iterable[str] | None = None,
) -> tuple[str, Path] | None:
    """The run with the best `evaluation.select_by`, and its `flow_best.pt`.

    Two restrictions, both of which cost a night when they are missing:

    `names` scopes the candidates to the runs the campaign actually planned. Without it the
    default essentials campaign would happily evaluate and ship a PARKED ablation left over
    from an earlier `--ablations` run, so the 8 hours of `baseline` nobody had to re-run would
    go unevaluated. `None` means every run in the file, which is what a bare `--summary` wants.

    A run must also have **finished**: `flow_best.pt` is rewritten at every validation, so an
    aborted run has one from step 500 and it can easily hold the best score of a short curve.
    `flow_last.pt` only appears when the loop completes, so that is the test.

    An ablation that moves the token layout (the correction) is still eligible when it is in
    scope: a checkpoint is self-describing, so evaluation picks the layout up from it.
    """
    metric = str((cfg.get("evaluation") or {}).get("select_by", "ciede2000"))
    key = {"ciede2000": "best_ciede2000/euler", "psnr_L": "best_psnr_L/euler"}.get(metric)
    if key is None:
        return None
    higher_is_better = metric == "psnr_L"
    wanted = None if names is None else set(names)
    best: tuple[float, str] | None = None
    for run in stage_one_runs(cfg):
        if wanted is not None and run["name"] not in wanted:
            continue
        scores = run_metrics(root / run["name"], subset)
        checkpoint = root / run["name"] / "flow_best.pt"
        if key not in scores or not checkpoint.exists():
            continue
        if not (root / run["name"] / "flow_last.pt").exists():
            continue  # the run did not finish; its "best" is the best of a truncated curve
        value = scores[key] * (1.0 if higher_is_better else -1.0)
        if best is None or value > best[0]:
            best = (value, run["name"])
    if best is None:
        return None
    return best[1], root / best[1] / "flow_best.pt"


# --- state ---------------------------------------------------------------------------------------


def preflight_data(data_cfg: Mapping, sample: int = 25) -> list[str]:
    """What is missing before any training can start. Empty means the data phase is done.

    This is the cheapest possible check against the most expensive possible mistake: a campaign
    that starts, spends a minute building token stats, and only then dies because the aligned
    PNGs were never written. The manifests are committed to the repo, but
    `data/dronevehicle_aligned/` is gitignored and has to be produced locally
    (`scripts/fetch_dronevehicle.py` then `scripts/prepare_dronevehicle.py all`).

    Args:
        data_cfg: the `dronevehicle` section of data.yaml.
        sample: how many manifest entries to actually look for on disk. Spot-checking beats
            stat-ing 5,385 ids x 5 files every time the pipeline starts.
    """
    manifest_dir = Path(str(data_cfg["manifest_dir"]))
    aligned_dir = Path(str(data_cfg["aligned_dir"]))
    problems: list[str] = []

    manifests = {}
    for name in ("train_pairs", "val_pairs"):
        path = manifest_dir / f"{name}.txt"
        if not path.exists():
            problems.append(f"no {path}: run scripts/prepare_dronevehicle.py all")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        ids = [line.strip() for line in lines if line.strip()]
        if not ids:
            problems.append(f"{path} is empty")
        manifests[name] = ids
    if problems:
        return problems

    if not aligned_dir.is_dir():
        return [
            f"no {aligned_dir}: the manifests are committed but the aligned PNGs are not. "
            "Run scripts/fetch_dronevehicle.py, then scripts/prepare_dronevehicle.py all"
        ]

    kinds = ("rgb", "thermal", "valid", "moving", "vehicles")
    for name, ids in manifests.items():
        step = max(1, len(ids) // max(sample, 1))
        absent = []
        for image_id in ids[::step][:sample]:
            split = image_id.split("_", 1)[0]
            for kind in kinds:
                if not (aligned_dir / split / f"{image_id}_{kind}.png").exists():
                    absent.append(f"{image_id}_{kind}.png")
        if absent:
            problems.append(
                f"{name}: {len(absent)} of {sample * len(kinds)} sampled files missing under "
                f"{aligned_dir} (e.g. {absent[0]}). Re-run scripts/prepare_dronevehicle.py write"
            )
    return problems


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
    "command_overrides",
    "evaluation_command",
    "is_blocking",
    "load_state",
    "median_sec_per_step",
    "pick_winner",
    "plan_table",
    "preflight_data",
    "read_csv_rows",
    "record",
    "resolve_steps",
    "rescale_hours",
    "run_metrics",
    "run_stage",
    "save_state",
    "stage_one_runs",
    "summary_table",
]
