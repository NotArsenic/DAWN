"""The training campaign: the plan it builds, what counts as finished, and whether gates gate.

`scripts/run_pipeline.py` launches days of GPU time unattended, so the failure that matters is
not a crash -- it is a campaign that quietly runs the wrong thing, skips a stage it should have
resumed, or carries on past a gate whose whole job was to stop it. Everything here is about those.

No torch, no GPU, no training: the module is subprocess orchestration and CSV reading, and the
one test that actually executes a stage runs `python -c` rather than anything in `scripts/`.
"""

import json

import pytest

omegaconf = pytest.importorskip("omegaconf")

from dawn.pipeline import (  # noqa: E402
    EVAL,
    GATE,
    REFINER,
    TRAIN,
    Stage,
    StageResult,
    build_plan,
    check_stage,
    evaluation_command,
    load_state,
    median_sec_per_step,
    pick_winner,
    plan_table,
    read_csv_rows,
    record,
    rescale_hours,
    run_metrics,
    run_stage,
    save_state,
    summary_table,
)

CAMPAIGN = {
    "root": "runs",
    "device": "cuda",
    "common_overrides": ["configs/gpu_5090.yaml"],
    "gates": [
        {
            "name": "oracle",
            "hours": 0.2,
            "tests": "the ceiling",
            "command": ["scripts/check_oracle.py", "--device", "cuda"],
            "check": {
                "kind": "oracle_psnr",
                "metric": "psnr_grid_r1",
                "rows": "oracle_rows.jsonl",
                "min": 24.0,
                "on_fail": "turn the correction on",
            },
        },
        {
            "name": "gpu_probe",
            "hours": 0.1,
            "tests": "sec/step",
            "command": ["scripts/train_flow.py", "--override", "configs/gpu_probe.yaml"],
            "measures_sec_per_step": True,
        },
    ],
    "runs": [
        {"name": "baseline", "hours": 8, "tests": "the reference", "overrides": []},
        {
            "name": "one_cycle",
            "hours": 8,
            "ablates": "optim.schedule",
            "tests": "one cycle",
            "overrides": ["configs/abl_one_cycle.yaml"],
        },
    ],
    "refiner": {"name": "refiner", "hours": 1, "tests": "stage 2"},
    "evaluation": {"select_by": "ciede2000", "split": "test_pairs", "hours": 0.3},
}


def _plan(tmp_path, **kwargs):
    return build_plan(CAMPAIGN, root=tmp_path / "runs", **kwargs)


def _write_val_csv(out_dir, rows, subset="box_aligned"):
    """A `metrics/val_<subset>.csv` shaped exactly as `CsvLog` writes one."""
    directory = out_dir / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"val_{subset}/"
    columns = ["step"] + [prefix + k for k in rows[0] if k != "step"]
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(c.replace(prefix, ""), "")) for c in columns))
    (directory / f"val_{subset}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- the plan ------------------------------------------------------------------------------------


def test_the_plan_runs_gates_then_baseline_then_ablations_then_stage_2_then_eval(tmp_path):
    """The order IS the dependency: nothing expensive starts before the cheap checks."""
    stages = _plan(tmp_path)
    assert [s.name for s in stages] == [
        "oracle",
        "gpu_probe",
        "baseline",
        "one_cycle",
        "refiner",
        "evaluation",
    ]
    assert [s.kind for s in stages] == [GATE, GATE, TRAIN, TRAIN, REFINER, EVAL]
    assert [s.name for s in stages].index("baseline") < [s.name for s in stages].index("one_cycle")


def test_every_run_gets_the_common_overrides_and_its_own_directory(tmp_path):
    stages = {s.name: s for s in _plan(tmp_path)}
    for name in ("baseline", "one_cycle"):
        command = stages[name].command
        assert "configs/gpu_5090.yaml" in command, name
        assert command[command.index("--out-dir") + 1].endswith(f"runs/{name}"), name


def test_an_ablation_adds_exactly_one_overlay_to_the_baseline(tmp_path):
    """The campaign's one rule. Two knobs in a run makes its result unattributable."""
    stages = {s.name: s for s in _plan(tmp_path)}
    baseline = set(stages["baseline"].command)
    extra = [c for c in stages["one_cycle"].command if c.endswith(".yaml") and c not in baseline]
    assert extra == ["configs/abl_one_cycle.yaml"]


def test_a_gate_that_trains_still_gets_the_campaigns_overrides(tmp_path):
    """The probe must measure the configuration that will actually run, overlays included."""
    probe = {s.name: s for s in _plan(tmp_path)}["gpu_probe"]
    assert "configs/gpu_5090.yaml" in probe.command and "configs/gpu_probe.yaml" in probe.command
    assert probe.command[probe.command.index("--out-dir") + 1].endswith("runs/gpu_probe")


def test_no_ablations_keeps_the_baseline_and_drops_the_rest(tmp_path):
    names = [s.name for s in _plan(tmp_path, include_ablations=False)]
    assert "baseline" in names and "one_cycle" not in names
    assert "refiner" in names and "evaluation" in names


def test_no_gates_drops_only_the_gates(tmp_path):
    names = [s.name for s in _plan(tmp_path, include_gates=False)]
    assert "oracle" not in names and "gpu_probe" not in names and "baseline" in names


def test_the_evaluation_command_is_empty_until_a_winner_is_known(tmp_path):
    """It cannot be built in advance: which checkpoint wins depends on every run's curve."""
    evaluation = {s.name: s for s in _plan(tmp_path)}["evaluation"]
    assert evaluation.command == ()
    built = evaluation_command(tmp_path / "flow_best.pt", tmp_path / "out", CAMPAIGN, "cuda")
    assert "--checkpoint" in built and "test_pairs" in built


# --- finished, resumable, skipped ---------------------------------------------------------------


def test_a_finished_run_is_skipped_and_a_half_finished_one_is_resumed(tmp_path):
    """The property an overnight campaign lives on: dying at 4 a.m. must not cost
    the runs that already finished."""
    stages = {s.name: s for s in _plan(tmp_path)}
    baseline = stages["baseline"]
    assert not baseline.is_done() and not baseline.can_resume()
    assert "--resume" not in baseline.argv()

    baseline.out_dir.mkdir(parents=True)
    (baseline.out_dir / "flow_last.pt").write_bytes(b"x")
    assert baseline.is_done() and baseline.can_resume()
    # Done wins over resumable: a finished run is not restarted.
    assert "--resume" not in baseline.argv()


def test_a_partial_run_resumes_from_its_own_checkpoint(tmp_path):
    stage = Stage(
        name="partial",
        kind=TRAIN,
        command=("scripts/train_flow.py",),
        out_dir=tmp_path,
        done=(tmp_path / "never.pt",),
        resumable_from=tmp_path / "flow_last.pt",
    )
    (tmp_path / "flow_last.pt").write_bytes(b"x")
    argv = stage.argv()
    assert argv[-2:] == ["--resume", str(tmp_path / "flow_last.pt")]


def test_a_stage_with_no_done_files_never_counts_as_finished(tmp_path):
    """The smoke test writes nothing, so it must re-run rather than be silently skipped."""
    assert not Stage(name="smoke", kind=GATE, command=("x",)).is_done()


# --- gates actually gate -------------------------------------------------------------------------


def _oracle_stage(tmp_path, values):
    (tmp_path / "oracle_rows.jsonl").write_text(
        "\n".join(json.dumps({"id": f"p{i}", "psnr_grid_r1": v}) for i, v in enumerate(values)),
        encoding="utf-8",
    )
    check = dict(CAMPAIGN["gates"][0]["check"], rows=str(tmp_path / "oracle_rows.jsonl"))
    return Stage(name="oracle", kind=GATE, command=("x",), check=check)


def test_the_oracle_gate_passes_above_the_plans_24_db_and_fails_below(tmp_path):
    passed, detail = check_stage(_oracle_stage(tmp_path, [24.6, 25.1, 24.9]), tmp_path)
    assert passed and "24.87" in detail

    failed, detail = check_stage(_oracle_stage(tmp_path, [20.0, 21.0]), tmp_path)
    assert not failed and "20.50" in detail


def test_a_missing_oracle_report_fails_the_gate_rather_than_passing_it(tmp_path):
    check = dict(CAMPAIGN["gates"][0]["check"], rows=str(tmp_path / "absent.jsonl"))
    passed, detail = check_stage(Stage("oracle", GATE, ("x",), check=check), tmp_path)
    assert not passed and "absent.jsonl" in detail


def test_the_overfit_gate_compares_the_run_against_its_own_oracle(tmp_path):
    """Not an absolute dB bar: the question is whether the model approached the ceiling it had."""
    out = tmp_path / "overfit"
    check = {
        "kind": "val_metric",
        "metric": "psnr_L/euler",
        "relative_to": "psnr_L/oracle",
        "min_fraction": 0.85,
    }
    stage = Stage("overfit", GATE, ("x",), out_dir=out, check=check)

    _write_val_csv(out, [{"step": 500, "psnr_L/euler": 24.0, "psnr_L/oracle": 26.0}])
    passed, detail = check_stage(stage, tmp_path)
    assert passed and "85%" in detail

    _write_val_csv(out, [{"step": 500, "psnr_L/euler": 12.0, "psnr_L/oracle": 26.0}])
    assert not check_stage(stage, tmp_path)[0]


def test_the_refiner_gate_is_the_improvement_being_positive(tmp_path):
    out = tmp_path / "refiner"
    stage = Stage(
        "refiner", REFINER, ("x",), out_dir=out,
        check={"kind": "val_metric", "metric": "psnr_L/improvement", "min": 0.0},
    )
    _write_val_csv(out, [{"step": 100, "psnr_L/improvement": 0.4}])
    assert check_stage(stage, tmp_path)[0]
    _write_val_csv(out, [{"step": 100, "psnr_L/improvement": -0.2}])
    assert not check_stage(stage, tmp_path)[0], "a refiner that does not help must not pass"


def test_a_metric_the_run_never_logged_fails_rather_than_defaulting(tmp_path):
    out = tmp_path / "run"
    _write_val_csv(out, [{"step": 1, "psnr_L/euler": 20.0}])
    stage = Stage("r", TRAIN, ("x",), out_dir=out, check={"kind": "val_metric", "metric": "nope"})
    assert not check_stage(stage, tmp_path)[0]


def test_a_stage_with_no_check_passes_on_its_exit_code(tmp_path):
    assert check_stage(Stage("smoke", GATE, ("x",)), tmp_path) == (True, "")


# --- cost ----------------------------------------------------------------------------------------


def test_the_probe_rescales_the_training_estimates_but_not_the_gates(tmp_path):
    """Gates are short and dominated by data loading, so the step time does not scale them."""
    stages = _plan(tmp_path)
    hours = rescale_hours(stages, measured=2.8, reference=1.4)
    assert hours["baseline"] == pytest.approx(16.0), "twice the step time, twice the hours"
    assert hours["oracle"] == pytest.approx(0.2), "a gate keeps its own estimate"


def test_an_unmeasured_probe_leaves_the_estimates_alone(tmp_path):
    stages = _plan(tmp_path)
    assert rescale_hours(stages, 0.0, 1.4)["baseline"] == pytest.approx(8.0)


def test_sec_per_step_comes_from_the_probes_csv_and_ignores_the_warm_up(tmp_path):
    directory = tmp_path / "metrics"
    directory.mkdir(parents=True)
    (directory / "train.csv").write_text(
        "step,loss/sec_per_step\n1,9.0\n2,9.0\n3,1.0\n4,2.0\n5,3.0\n", encoding="utf-8"
    )
    assert median_sec_per_step(tmp_path) == pytest.approx(2.0)
    assert median_sec_per_step(tmp_path / "nowhere") is None


def test_the_plan_table_totals_only_what_is_left_to_run(tmp_path):
    stages = _plan(tmp_path)
    stages[2].out_dir.mkdir(parents=True)
    (stages[2].out_dir / "flow_last.pt").write_bytes(b"x")  # baseline already finished
    table = plan_table(stages, {s.name: s.hours for s in stages})
    assert "| `baseline` | train | done |" in table
    assert "9.6 GPU-hours remaining" in table  # 17.6 total less the finished 8


# --- reading results -----------------------------------------------------------------------------


def test_run_metrics_takes_the_last_row_and_the_best_of_every_row(tmp_path):
    _write_val_csv(
        tmp_path,
        [
            {"step": 500, "ciede2000/euler": 6.0, "psnr_L/euler": 19.0},
            {"step": 1000, "ciede2000/euler": 4.0, "psnr_L/euler": 22.0},
            {"step": 1500, "ciede2000/euler": 5.0, "psnr_L/euler": 21.0},
        ],
    )
    scores = run_metrics(tmp_path)
    assert scores["step"] == 1500 and scores["ciede2000/euler"] == 5.0  # where it ended
    assert scores["best_ciede2000/euler"] == 4.0  # lower is better
    assert scores["best_psnr_L/euler"] == 22.0  # higher is better


def test_the_winner_is_the_best_ciede2000_with_a_checkpoint_on_disk(tmp_path):
    root = tmp_path / "runs"
    for name, delta in (("baseline", 4.0), ("one_cycle", 3.0)):
        _write_val_csv(root / name, [{"step": 1, "ciede2000/euler": delta}])
        (root / name / "flow_best.pt").write_bytes(b"x")
    assert pick_winner(CAMPAIGN, root)[0] == "one_cycle"


def test_a_run_without_a_checkpoint_cannot_win(tmp_path):
    """A curve with no `flow_best.pt` is a crashed run, not a candidate."""
    root = tmp_path / "runs"
    _write_val_csv(root / "baseline", [{"step": 1, "ciede2000/euler": 4.0}])
    (root / "baseline" / "flow_best.pt").write_bytes(b"x")
    _write_val_csv(root / "one_cycle", [{"step": 1, "ciede2000/euler": 1.0}])  # better, no file
    assert pick_winner(CAMPAIGN, root)[0] == "baseline"


def test_there_is_no_winner_before_anything_has_run(tmp_path):
    assert pick_winner(CAMPAIGN, tmp_path / "runs") is None


def test_the_summary_names_each_runs_knob_and_its_difference_from_the_baseline(tmp_path):
    root = tmp_path / "runs"
    for name, delta in (("baseline", 4.0), ("one_cycle", 3.5)):
        _write_val_csv(root / name, [{"step": 20000, "ciede2000/euler": delta}])
    table = summary_table(CAMPAIGN, root)
    assert "`baseline`" in table and "the reference" in table
    assert "optim.schedule" in table, "an ablation row must name the knob it moved"
    assert "-0.500" in table, "and its difference from the baseline"


def test_the_summary_says_so_rather_than_printing_an_empty_table(tmp_path):
    assert "No finished stage-1 run" in summary_table(CAMPAIGN, tmp_path / "runs")


def test_read_csv_rows_parses_numbers_and_drops_the_blanks_a_late_column_leaves(tmp_path):
    path = tmp_path / "m.csv"
    path.write_text("step,a,b\n1,1.5,\n2,2.5,9\n", encoding="utf-8")
    assert read_csv_rows(path) == [{"step": 1.0, "a": 1.5}, {"step": 2.0, "a": 2.5, "b": 9.0}]
    assert read_csv_rows(tmp_path / "absent.csv") == []


# --- state and execution -------------------------------------------------------------------------


def test_state_round_trips_and_an_absent_file_is_an_empty_campaign(tmp_path):
    assert load_state(tmp_path) == {"stages": {}}
    state = record({"stages": {}}, StageResult("baseline", "ok", 12.0, 0, "fine"))
    save_state(tmp_path, state)
    reloaded = load_state(tmp_path)["stages"]["baseline"]
    assert reloaded["status"] == "ok" and reloaded["seconds"] == 12.0 and "finished_at" in reloaded


def test_running_a_stage_captures_its_output_and_its_exit_code(tmp_path):
    """Executes a real subprocess, but `python -c`, never anything that trains."""
    ok = Stage("ok", GATE, ("-c", "print('hello from the stage')"))
    result = run_stage(ok, tmp_path, tmp_path / "logs", echo=False)
    assert result.status == "ok" and result.ok
    assert "hello from the stage" in (tmp_path / "logs" / "ok.log").read_text(encoding="utf-8")

    bad = Stage("bad", GATE, ("-c", "raise SystemExit(3)"))
    failed = run_stage(bad, tmp_path, tmp_path / "logs", echo=False)
    assert failed.status == "failed" and failed.returncode == 3 and not failed.ok


def test_the_shipped_campaign_names_real_configs_and_one_knob_per_ablation():
    """The real configs/experiments.yaml, not a fixture: a typo'd overlay path would otherwise
    only surface hours into an overnight run."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    loaded = omegaconf.OmegaConf.load(repo / "configs/experiments.yaml")
    cfg = omegaconf.OmegaConf.to_container(loaded)
    paths = list(cfg["common_overrides"])
    for run in cfg["runs"]:
        paths += list(run.get("overrides") or [])
    for gate in cfg["gates"]:
        paths += [c for c in gate["command"] if str(c).endswith(".yaml")]
    for path in paths:
        assert (repo / path).exists(), path
    for command in (gate["command"] for gate in cfg["gates"]):
        assert (repo / command[0]).exists(), command[0]

    assert cfg["runs"][0]["name"] == "baseline" and not cfg["runs"][0].get("overrides")
    for run in cfg["runs"][1:]:
        assert len(run["overrides"]) == 1, f"{run['name']} moves more than one knob"
        assert run.get("ablates"), f"{run['name']} does not say which key it ablates"


# --- the data preflight -------------------------------------------------------------------------


def _data_cfg(tmp_path):
    return {
        "manifest_dir": str(tmp_path / "manifests"),
        "aligned_dir": str(tmp_path / "aligned"),
    }


def _manifests(tmp_path, ids=("train_00001", "train_00002")):
    directory = tmp_path / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "train_pairs.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (directory / "val_pairs.txt").write_text("val_00001\n", encoding="utf-8")


def _aligned(tmp_path, ids, kinds=("rgb", "thermal", "valid", "moving", "vehicles")):
    for image_id in ids:
        split = image_id.split("_", 1)[0]
        directory = tmp_path / "aligned" / split
        directory.mkdir(parents=True, exist_ok=True)
        for kind in kinds:
            (directory / f"{image_id}_{kind}.png").write_bytes(b"x")


def test_the_preflight_passes_once_the_data_phase_has_really_run(tmp_path):
    from dawn.pipeline import preflight_data

    _manifests(tmp_path)
    _aligned(tmp_path, ["train_00001", "train_00002", "val_00001"])
    assert preflight_data(_data_cfg(tmp_path)) == []


def test_committed_manifests_without_the_aligned_pngs_are_caught(tmp_path):
    """The exact state of a fresh clone: manifests are in git, the 5,385 PNGs are not."""
    from dawn.pipeline import preflight_data

    _manifests(tmp_path)
    problems = preflight_data(_data_cfg(tmp_path))
    assert len(problems) == 1 and "prepare_dronevehicle.py all" in problems[0]


def test_a_missing_manifest_names_the_command_that_writes_it(tmp_path):
    from dawn.pipeline import preflight_data

    problems = preflight_data(_data_cfg(tmp_path))
    assert any("train_pairs.txt" in p and "prepare_dronevehicle" in p for p in problems)


def test_a_half_written_aligned_tree_is_caught(tmp_path):
    """`write` interrupted partway leaves ids with some of their five PNGs and not others."""
    from dawn.pipeline import preflight_data

    _manifests(tmp_path)
    _aligned(tmp_path, ["train_00001", "train_00002"], kinds=("rgb", "thermal"))
    _aligned(tmp_path, ["val_00001"])
    problems = preflight_data(_data_cfg(tmp_path))
    assert any("train_pairs" in p and "missing" in p for p in problems)


def test_an_empty_manifest_is_not_mistaken_for_a_finished_one(tmp_path):
    from dawn.pipeline import preflight_data

    _manifests(tmp_path, ids=())
    assert any("is empty" in p for p in preflight_data(_data_cfg(tmp_path)))


def test_the_evaluation_stage_looks_for_the_file_evaluate_py_actually_writes(tmp_path):
    """`evaluate.py` writes summary.json. A wrong marker here would silently re-run it forever."""
    import re
    from pathlib import Path

    stage = {s.name: s for s in _plan(tmp_path)}["evaluation"]
    assert [p.name for p in stage.done] == ["summary.json"]
    source = (Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py").read_text("utf-8")
    written = set(re.findall(r'out_dir / "([\w.]+)"', source))
    assert {p.name for p in stage.done} <= written, f"evaluate.py writes {sorted(written)}"
