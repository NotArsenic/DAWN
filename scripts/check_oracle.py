#!/usr/bin/env python3
"""Evaluate brightness and color oracle reconstruction ceilings on aligned pairs."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.data.pairs import AlignedPairs  # noqa: E402
from dawn.flow import oracle  # noqa: E402
from dawn.flow.loaders import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROBE_PAIRS, PROBE_SEED = 120, 0
PROBE_GRID_ONLY_DB = 24.72
PROBE_TOLERANCE_DB = 0.5
SWEEP_BINS, SWEEP_CELLS = (8, 16), ((16, 20), (32, 40), (64, 80))
SWEEP_RIDGE, SWEEP_REFINE = (1e-3, 1e-1), (0, 2)


def probe_ids(dataset) -> set[str]:
    """Sample fixed probe subset of image IDs for validation check."""
    random.seed(PROBE_SEED)
    picks = random.sample(range(len(dataset)), min(PROBE_PAIRS, len(dataset)))
    return {dataset.image_ids[i] for i in picks}


def build_report(args, cfg, rows, train_rows, sweeps, sheets, checks) -> str:
    """Format oracle evaluation metrics and verification results as Markdown."""
    size = cfg["encoder"]["input_size"]
    n_box = sum(not r["fallback"] for r in rows)
    lines = [
        "# Brightness and colour oracle",
        "",
        f"- config: `{args.config}` (+ {', '.join(args.override) or 'no overrides'})",
        f"- resolution (H, W): {size}; grid ridge {cfg['grid']['ridge_alpha']}, "
        f"refine {cfg['grid']['refine_steps']}; BGU {cfg['color']['bgu']}",
        f"- split: `{args.split}`, {len(rows)} pairs "
        f"({n_box} box-aligned, {len(rows) - n_box} fallback)",
        "- masks: `valid & ~moving` (box pairs), `valid & ~vehicles` (fallback pairs)",
        "",
        "## L* PSNR, colour ceiling and assembled oracle",
        "",
        oracle.metrics_table(rows),
        "",
    ]
    subset = [r for r in rows if r.get("probe")]
    if subset:
        lines += [
            f"## The handoff's probe subset ({len(subset)} pairs, seed {PROBE_SEED})",
            "",
            oracle.metrics_table(subset),
            "",
        ]
    if train_rows:
        lines += [
            f"## Train sample ({len(train_rows)} pairs)",
            "",
            oracle.metrics_table(train_rows),
            "",
        ]
    if sweeps:
        table, best, baseline = oracle.sweep_table(sweeps)
        lines += [
            "## BGU grid sweep (colour ceiling, lower is better)",
            "",
            table,
            "",
            f"Best: bins, rows, cols, ridge, refine = {best}" if best else "",
            "",
        ]
    lines += ["## Worst 10% of grid-only scenes", ""]
    lines += [f"- `{p}`" for p in sheets] or ["- (no sheet written)"]
    lines += ["", "## Checks against the handoff", ""] + [f"- {c}" for c in checks] + [""]
    return "\n".join(lines)


def main():
    """Run brightness and color oracle evaluation and output report."""
    parser = argparse.ArgumentParser(description="Brightness and colour oracle check")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Flow config")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--override", nargs="*", default=[], help="Extra YAML files, in order")
    parser.add_argument("--split", type=str, default="val_pairs", help="Manifest to evaluate")
    parser.add_argument("--max-pairs", type=int, default=None, help="Only the first N pairs")
    parser.add_argument(
        "--train-sample", type=int, default=0, help="Also N evenly spaced train pairs"
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu | mps | cuda")
    parser.add_argument("--batch-size", type=int, default=4, help="Pairs per batch")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--sweep-pairs", type=int, default=40, help="Pairs for the BGU sweep")
    parser.add_argument("--no-sweep", action="store_true", help="Skip the BGU grid sweep")
    parser.add_argument("--no-sheet", action="store_true", help="Skip the worst-10%% sheets")
    parser.add_argument("--output", type=str, default="outputs/oracle", help="Output directory")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data_config, args.override)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    logger.info(
        "Oracle check on %s at %s, device %s", args.split, cfg["encoder"]["input_size"], device
    )

    dataset = AlignedPairs.from_config(cfg, args.split)
    indices = list(range(len(dataset)))[: args.max_pairs]
    sweep = (
        ()
        if args.no_sweep
        else tuple(
            (bins, rows, cols, ridge, refine)
            for bins in SWEEP_BINS
            for rows, cols in SWEEP_CELLS
            for ridge in SWEEP_RIDGE
            for refine in SWEEP_REFINE
        )
    )
    rows, sweeps = oracle.run_oracle(
        dataset, indices, cfg, device, args.batch_size, args.workers, sweep, args.sweep_pairs
    )
    probe = probe_ids(dataset)
    for row in rows:
        row["probe"] = row["id"] in probe
    oracle.save_rows(rows, out_dir / "rows_val.jsonl")

    train_rows: list[dict] = []
    if args.train_sample:
        train_set = AlignedPairs.from_config(cfg, "train_pairs")
        count = min(args.train_sample, len(train_set))
        picks = np.unique(np.linspace(0, len(train_set) - 1, count).round().astype(int)).tolist()
        train_rows, _ = oracle.run_oracle(
            train_set, picks, cfg, device, args.batch_size, args.workers
        )
        oracle.save_rows(train_rows, out_dir / "rows_train.jsonl")

    sheets = []
    if not args.no_sheet:
        ids = oracle.worst_ids(rows)
        sheets = oracle.write_worst_sheets(dataset, ids, rows, cfg, out_dir, device)

    checks = []
    subset = [r for r in rows if r.get("probe")]
    if subset:
        got = oracle.summarize(subset, "psnr_grid_r0")["mean"]
        verdict = "OK" if abs(got - PROBE_GRID_ONLY_DB) <= PROBE_TOLERANCE_DB else "STOP AND ASK"
        checks.append(
            f"grid-only refine 0 on the probe subset: {got:.2f} dB vs the handoff's "
            f"{PROBE_GRID_ONLY_DB} dB (tolerance {PROBE_TOLERANCE_DB}): **{verdict}**"
        )
    bgu = oracle.summarize(rows, "ab_err_bgu")["mean"]
    bil = oracle.summarize(rows, "ab_err_bilinear")["mean"]
    checks.append(
        f"colour ceiling over all pairs: bilinear {bil:.3f} vs BGU {bgu:.3f} (configured "
        f"{cfg['color']['bgu']}) mean abs delta a*b*; `color.upsample` is "
        f"**{cfg['color']['upsample']}** (the maintainer chose bilinear at Checkpoint A)"
    )
    # Evaluate color ceiling on structural edge pixels.
    bgu_e = oracle.summarize(rows, "ab_err_bgu_edge")["mean"]
    bil_e = oracle.summarize(rows, "ab_err_bilinear_edge")["mean"]
    quantile = float(cfg["color"].get("edge_quantile", 0.9))
    checks.append(
        f"colour ceiling AT EDGES (top {100 * (1 - quantile):.0f}% of gradient magnitude): "
        f"bilinear {bil_e:.3f} vs BGU {bgu_e:.3f} mean abs delta a*b* -- "
        f"**{'bilinear ahead' if bil_e <= bgu_e else 'BGU ahead at edges'}**"
    )
    de_bil = oracle.summarize(rows, "de_grid_only_bilinear")["mean"]
    de_bgu = oracle.summarize(rows, "de_grid_only_bgu")["mean"]
    checks.append(
        f"assembled oracle CIEDE2000 (grid only): bilinear colour {de_bil:.2f} vs BGU colour "
        f"{de_bgu:.2f}"
    )
    if sweeps:
        _, best, baseline = oracle.sweep_table(sweeps)
        if best and baseline is not None:
            best_mean = float(np.mean(sweeps[best]))
            checks.append(
                f"best swept BGU {best}: {best_mean:.3f} vs bilinear {baseline:.3f} on the sweep "
                f"pairs ({'BGU ahead' if best_mean <= baseline else 'bilinear ahead'})"
            )
    report = build_report(args, cfg, rows, train_rows, sweeps, sheets, checks)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    logger.info("Wrote %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
