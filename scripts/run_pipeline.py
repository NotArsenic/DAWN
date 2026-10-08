#!/usr/bin/env python3
"""Run the Dawn training campaign: the gates, the model, stage 2 and evaluation.

One command, start to finish:

    python scripts/run_pipeline.py --list       # what would run, in order, with the cost
    python scripts/run_pipeline.py --dry-run    # the exact commands, run nothing
    python scripts/run_pipeline.py --yes        # run it

What runs is declared in `configs/experiments.yaml` and nowhere else. The order is the dependency:

    smoke_dino -> oracle -> gpu_probe -> overfit      the gates, each a gate for the next
    baseline                                          the model
    refiner                                           stage 2
    evaluation                                        the winning checkpoint on the test split

The five ablations in `ablations:` are PARKED: they answer open decisions about the method, not
about this model, and cost five times the rest put together. `--ablations` runs them all;
`--only correction` runs one.

Three things this buys over running the commands by hand:

**It will not start the expensive runs until the cheap checks pass.** The oracle's 24 dB rule and
the overfit gate's "approach your own oracle" are enforced, not suggested, and a failure prints
what to do about it rather than only what broke.

**It is resumable and idempotent.** A finished stage is skipped. A training run with a partial
`flow_last.pt` restarts with `--resume`. A campaign that dies on stage 5 at 4 a.m. keeps stages 1
to 4 and picks up where it stopped; `pipeline_state.json` is the record.

**It costs what it says.** `gpu_probe` measures seconds/step on the real configuration and the
stage-1 and evaluation estimates are rescaled from it (a gate and the refiner do not share that
per-step profile, so they keep their declared numbers). Nothing launches without `--yes`.

Afterwards, `--summary` prints the comparison table at any point, finished or not.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf  # noqa: E402

from dawn.pipeline import (  # noqa: E402
    EVAL,
    Stage,
    build_plan,
    check_stage,
    evaluation_command,
    is_blocking,
    load_state,
    median_sec_per_step,
    pick_winner,
    plan_table,
    preflight_data,
    record,
    rescale_hours,
    run_stage,
    save_state,
    summary_table,
)
from dawn.progress import format_duration  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pipeline")

# The seconds/step the hours in experiments.yaml were written against. The probe replaces it.
REFERENCE_SEC_PER_STEP = 1.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the training campaign (gates, the model, stage 2, evaluation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments", default="configs/experiments.yaml", help="The campaign declaration"
    )
    parser.add_argument("--root", default=None, help="Where runs land (default: experiments.root)")
    parser.add_argument("--device", default=None, help="Override experiments.device")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="STAGE",
        help="Run only these stages, by name. A parked ablation named here is pulled into the "
        "plan, so `--only correction` works without --ablations. At least one name is "
        "required: a bare --only used to mean 'everything', which with --yes is a two-day job.",
    )
    parser.add_argument("--skip", nargs="*", default=[], metavar="STAGE", help="Skip these stages")
    parser.add_argument(
        "--from", dest="start_from", default=None, metavar="STAGE", help="Start at this stage"
    )
    parser.add_argument(
        "--ablations",
        action="store_true",
        help="Also run the parked ablations: five more stage-1 runs, one knob each. They "
        "answer open decisions about the METHOD; the default campaign ships a model without "
        "them, at about a fifth of the GPU time.",
    )
    parser.add_argument("--no-gates", action="store_true", help="Skip the gates (not advised)")
    parser.add_argument(
        "--extra-override",
        nargs="*",
        default=[],
        metavar="YAML",
        help="Extra overlays appended to experiments.common_overrides for every stage-1 run, "
        "winning over them. `configs/fast.yaml` is the short-budget one. NOT applied to the "
        "refiner, whose own config would lose to it.",
    )
    parser.add_argument("--list", action="store_true", help="Print the plan and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print every command and exit")
    parser.add_argument("--summary", action="store_true", help="Print the results table and exit")
    parser.add_argument(
        "--yes", action="store_true", help="Actually run. Without it this only prints the plan."
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Carry on after a FAILED run. Gate failures always stop: they mean the rest is void.",
    )
    return parser.parse_args()


def _guidance(stage: Stage) -> None:
    """The configured `on_fail`: what to do about it, not only what broke."""
    text = " ".join(str((stage.check or {}).get("on_fail", "")).split())
    if text:
        logger.error("    %s", text)


def select(stages: list[Stage], args: argparse.Namespace) -> list[Stage]:
    """Apply `--only`, `--skip` and `--from`, keeping the declared order."""
    known = {s.name for s in stages}
    named = list(args.only or []) + list(args.skip) + ([args.start_from] if args.start_from else [])
    for name in named:
        if name not in known:
            raise SystemExit(f"unknown stage {name!r}; known: {', '.join(sorted(known))}")
    if args.start_from:
        stages = stages[[s.name for s in stages].index(args.start_from) :]
    if args.only:
        stages = [s for s in stages if s.name in set(args.only)]
    return [s for s in stages if s.name not in set(args.skip)]


def main() -> int:
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / args.experiments), resolve=True)
    if args.device:
        cfg["device"] = args.device
    root = Path(args.root or cfg.get("root", "runs"))
    if not root.is_absolute():
        root = ROOT / root

    if args.summary:
        print(summary_table(cfg, root))
        return 0

    stages = build_plan(
        cfg,
        root=root,
        # `--only correction` has to find `correction`, so naming any stage pulls the parked
        # ablations into the plan before `select()` validates the names against it.
        include_ablations=args.ablations or bool(args.only),
        include_gates=not args.no_gates,
        repo=ROOT,
        extra_overrides=args.extra_override,
    )
    if args.extra_override:
        logger.info("extra overrides on every stage: %s", " ".join(args.extra_override))
    stages = select(stages, args)

    # Rescale the estimates from whatever the probe measured, if it has run.
    measured = median_sec_per_step(root / "gpu_probe")
    hours = rescale_hours(stages, measured or 0.0, REFERENCE_SEC_PER_STEP)
    if measured:
        logger.info(
            "gpu_probe measured %.2f s/step; estimates rescaled by %.2fx",
            measured,
            measured / REFERENCE_SEC_PER_STEP,
        )
    else:
        logger.info(
            "gpu_probe has not run: hours below are the config's estimates at ~%.1f s/step",
            REFERENCE_SEC_PER_STEP,
        )

    print()
    print(plan_table(stages, hours))
    print()

    if args.list:
        return 0
    if args.dry_run:
        for stage in stages:
            if stage.kind == EVAL and not stage.command:
                # Built late on purpose: which checkpoint wins is not known until every stage-1
                # run has a validation curve, so there is no command to print yet.
                winner = pick_winner(cfg, root)
                target = str(winner[1]) if winner else "<the winning run>/flow_best.pt"
                print(f"# evaluation, once the runs above have curves: --checkpoint {target}")
                continue
            marker = "  # done" if stage.is_done() else ""
            print(f"{stage.shell()}{marker}")
        return 0
    if not args.yes:
        logger.warning(
            "Nothing was run. This campaign is %.0f GPU-hours; pass --yes to start it.",
            sum(hours.get(s.name, 0) for s in stages if not s.is_done()),
        )
        return 0

    # The cheapest check against the most expensive mistake: a campaign that starts, builds
    # token stats for a minute, and only then dies because the aligned PNGs were never written.
    data_cfg = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/data.yaml"), resolve=True)
    problems = preflight_data(data_cfg["dronevehicle"])
    if problems:
        logger.error("The data phase is not finished, so nothing below can run:")
        for problem in problems:
            logger.error("  %s", problem)
        return 3

    state = load_state(root)
    log_dir = root / "logs"
    started = time.perf_counter()
    failures: list[str] = []

    planned = {s.name for s in stages}
    winner_file = (root / "evaluation" / "winner.txt") if root else None

    for index, stage in enumerate(stages, 1):
        head = f"[{index}/{len(stages)}] {stage.name}"

        # Evaluation needs the winner, which is not known until the stage-1 runs have curves,
        # so it is resolved BEFORE the is-it-done test: an evaluation of yesterday's winner is
        # stale, not finished.
        expected = None
        if stage.kind == EVAL:
            winner = pick_winner(cfg, root, names=planned)
            if winner is None:
                logger.warning("%s: no FINISHED stage-1 run to evaluate, skipping", head)
                continue
            expected, checkpoint = winner
            logger.info("%s: the winning run is %s (%s)", head, expected, checkpoint)
            stage = Stage(
                **{
                    **stage.__dict__,
                    "command": evaluation_command(
                        checkpoint, stage.out_dir or root, cfg, str(cfg.get("device", "cuda"))
                    ),
                }
            )

        if stage.is_done():
            recorded = None
            if expected is not None and winner_file and winner_file.exists():
                recorded = winner_file.read_text(encoding="utf-8").strip()
            if expected is not None and recorded != expected:
                logger.info(
                    "%s: on disk it evaluated %s, the winner is now %s -- redoing it",
                    head,
                    recorded or "an unrecorded run",
                    expected,
                )
            else:
                # A finished stage is still JUDGED. Its check reads the artifacts on disk, and
                # a gate that failed wrote its checkpoint anyway, so without this one re-run
                # of the same command walks straight past it.
                passed, detail = check_stage(stage, root)
                if not passed and is_blocking(stage):
                    logger.error("%s: already finished, but its CHECK FAILS -- %s", head, detail)
                    _guidance(stage)
                    logger.error("Stopping: a gate decides whether the rest is worth running.")
                    return 2
                logger.info("%s: already finished, skipping", head)
                continue

        verb = "resuming" if stage.can_resume() else "starting"
        logger.info("%s: %s (~%.1f h) -- %s", head, verb, hours.get(stage.name, 0), stage.tests)
        try:
            result = run_stage(stage, ROOT, log_dir)
        except KeyboardInterrupt:
            logger.warning("%s: interrupted. Re-run to resume from here.", head)
            save_state(root, state)
            return 130

        if result.ok:
            passed, detail = check_stage(stage, root)
            result.detail = detail
            if detail:
                logger.info("%s: %s", head, detail)
            if not passed and is_blocking(stage):
                result.status = "gate_failed"
                logger.error("%s: CHECK FAILED -- %s", head, detail)
                _guidance(stage)
            elif not passed:
                # Advisory: "the refiner did not earn its place" is a RESULT. Stopping here
                # would throw away the evaluation that is the point of the night.
                result.status = "check_failed"
                logger.warning("%s: check did not pass -- %s", head, detail)
                _guidance(stage)
            elif expected is not None and stage.out_dir:
                (stage.out_dir / "winner.txt").write_text(expected + "\n", encoding="utf-8")
        state = record(state, result)
        save_state(root, state)

        if result.status == "gate_failed":
            logger.error("Stopping: a gate decides whether the rest is worth running.")
            return 2
        if result.status == "failed":
            failures.append(stage.name)
            logger.error("%s: FAILED (%s)", head, result.detail)
            if not args.keep_going:
                logger.error("Stopping. Fix it and re-run; finished stages will be skipped.")
                return 1
        else:
            logger.info("%s: done in %s", head, format_duration(result.seconds))

    elapsed = format_duration(time.perf_counter() - started)
    print()
    print(summary_table(cfg, root))
    print()
    winner = pick_winner(cfg, root)
    if winner:
        logger.info("Best stage-1 run: %s (%s)", winner[0], winner[1])
    if failures:
        logger.error("Campaign finished in %s with failures: %s", elapsed, ", ".join(failures))
        return 1
    logger.info("Campaign finished in %s. Results above; logs in %s.", elapsed, log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
