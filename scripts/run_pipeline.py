#!/usr/bin/env python3
"""Run the whole Dawn training campaign: gates, the baseline, every ablation, stage 2, evaluation.

One command, start to finish:

    python scripts/run_pipeline.py --list       # what would run, in order, with the cost
    python scripts/run_pipeline.py --dry-run    # the exact commands, run nothing
    python scripts/run_pipeline.py --yes        # run it

What runs is declared in `configs/experiments.yaml` and nowhere else. The order is the dependency:

    smoke_dino -> oracle -> gpu_probe -> overfit      the gates, each a gate for the next
    baseline                                          the reference run
    correction, one_cycle, segment_equal, bgu, ...    one knob off the baseline, each
    refiner                                           stage 2
    evaluation                                        the winning checkpoint on the test split

Three things this buys over running the commands by hand:

**It will not start the expensive runs until the cheap checks pass.** The oracle's 24 dB rule and
the overfit gate's "approach your own oracle" are enforced, not suggested, and a failure prints
what to do about it rather than only what broke.

**It is resumable and idempotent.** A finished stage is skipped. A training run with a partial
`flow_last.pt` restarts with `--resume`. A campaign that dies on stage 5 at 4 a.m. keeps stages 1
to 4 and picks up where it stopped; `pipeline_state.json` is the record.

**It costs what it says.** `gpu_probe` measures seconds/step on the real configuration and every
later estimate is rescaled from it, so the hours in the table stop being a guess written into a
config file. Nothing launches without `--yes`, because the full campaign is days of GPU time.

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
    load_state,
    median_sec_per_step,
    pick_winner,
    plan_table,
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
        description="Run the whole training campaign (gates, baseline, ablations, stage 2, eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiments", default="configs/experiments.yaml", help="The campaign declaration"
    )
    parser.add_argument("--root", default=None, help="Where runs land (default: experiments.root)")
    parser.add_argument("--device", default=None, help="Override experiments.device")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="STAGE",
        help="Run only these stages, by name. Their gates are still enforced if already run.",
    )
    parser.add_argument("--skip", nargs="*", default=[], metavar="STAGE", help="Skip these stages")
    parser.add_argument(
        "--from", dest="start_from", default=None, metavar="STAGE", help="Start at this stage"
    )
    parser.add_argument(
        "--no-ablations",
        action="store_true",
        help="Baseline only: the gates, one stage-1 run, stage 2 and evaluation",
    )
    parser.add_argument("--no-gates", action="store_true", help="Skip the gates (not advised)")
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
        include_ablations=not args.no_ablations,
        include_gates=not args.no_gates,
    )
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

    state = load_state(root)
    log_dir = root / "logs"
    started = time.perf_counter()
    failures: list[str] = []

    for index, stage in enumerate(stages, 1):
        head = f"[{index}/{len(stages)}] {stage.name}"
        if stage.is_done():
            logger.info("%s: already finished, skipping", head)
            continue

        # Evaluation needs the winner, which is not known until the stage-1 runs have curves.
        if stage.kind == EVAL:
            winner = pick_winner(cfg, root)
            if winner is None:
                logger.warning("%s: no finished stage-1 run to evaluate, skipping", head)
                continue
            name, checkpoint = winner
            logger.info("%s: the winning run is %s (%s)", head, name, checkpoint)
            stage = Stage(
                **{
                    **stage.__dict__,
                    "command": evaluation_command(
                        checkpoint, stage.out_dir or root, cfg, str(cfg.get("device", "cuda"))
                    ),
                }
            )

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
            if not passed:
                result.status = "gate_failed"
                logger.error("%s: CHECK FAILED -- %s", head, detail)
                guidance = " ".join(str((stage.check or {}).get("on_fail", "")).split())
                if guidance:
                    logger.error("    %s", guidance)
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
