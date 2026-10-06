#!/usr/bin/env python3
"""Audit the alignment (plan step 11): look at what was accepted and what was rejected.

Reads the align log written by `prepare_dronevehicle.py align` and writes, under
`<dronevehicle.outputs_dir>/audit/`:

    accepted_NN.jpg   contact sheets of random accepted pairs (default 100)
    rejected_NN.jpg   contact sheets of rejected pairs (default 100), stratified by reason so a
                      rare reason is not drowned by the common ones
    fit_histograms.png  histograms of the median inlier residual, scale, rotation, shift and the
                      edge-check gain, accepted against rejected, with the thresholds marked
    sampled.csv       which pair is on which sheet, and why it was rejected
    fallback_accepted_NN.jpg / fallback_rejected_NN.jpg   the pairs the edge-correlation fallback
                      aligned (no boxes behind them) and the ones it looked at and rejected, on
                      sheets of their own: audit these separately from the box-aligned pairs
    vehicles_dense_01.jpg  the frames whose vehicles mask covers the most (flagged over 30% in
                      vehicles.csv), mask tinted red: check what a dense frame loses
    usable_luminance.png  luminance of the usable pairs (the final manifests) against the day
                      pairs that were rejected later, per split
    usable_darkest.jpg / usable_random.jpg   RGB | thermal of the 24 darkest and of 24 random
                      usable pairs, from the written PNGs: the darkest are where dusk and night
                      frames that passed the day filter hide

On every tile the edges of the warped thermal are drawn in magenta over the RGB: on an aligned
pair they sit on the RGB edges of the same objects. Rejected pairs are shown with their best
attempted fit (the unwarped thermal when no fit was made), so a wrongly rejected good pair is
visible too. Sampling is seeded, so the same seed gives the same sheets.

Usage:
    python scripts/audit_alignment.py
    python scripts/audit_alignment.py --accepted 100 --rejected 100 --seed 1 --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _read_manifest(pairs, layout, split: str) -> list[str]:
    path = layout.manifest(f"{split}_pairs")
    return pairs.read_manifest(path) if path.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the alignment: contact sheets with thermal edges over the RGB"
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--accepted", type=int, default=100, help="Accepted pairs to show")
    parser.add_argument("--rejected", type=int, default=100, help="Rejected pairs to show")
    parser.add_argument(
        "--fallback",
        type=int,
        default=60,
        help="Edge-fallback pairs to show, accepted (and rejected)",
    )
    parser.add_argument(
        "--dense", type=int, default=20, help="Densest vehicles-mask frames to show (red tint)"
    )
    parser.add_argument("--per-sheet", type=int, default=20, help="Tiles per sheet")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count() or 1, help="Processes (default: every core)"
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from dawn.data import align, audit, pairs, report
    from dawn.data import day_filter as df
    from dawn.data import dronevehicle as dv

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    layout = dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])
    acfg = align.AlignConfig.from_config(cfg["align"])
    rows = list(align.read_align_log(layout.log("align")).values())
    out_dir = layout.outputs_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    box_rows = [r for r in rows if r.model != "edge_translation"]
    accepted_all = sorted((r for r in box_rows if r.accepted), key=lambda r: r.image_id)
    accepted = sorted(
        random.Random(f"{args.seed}/accepted").sample(
            accepted_all, min(args.accepted, len(accepted_all))
        ),
        key=lambda r: r.image_id,
    )
    by_reason: dict[str, list] = {}
    for row in sorted((r for r in box_rows if not r.accepted), key=lambda r: r.image_id):
        by_reason.setdefault(row.reason, []).append(row)
    rejected = [row for _, row in report.stratified_sample(by_reason, args.rejected, args.seed)]
    reason_counts = {reason: len(group) for reason, group in sorted(by_reason.items())}
    logger.info(
        f"{len(accepted_all)} accepted, {sum(reason_counts.values())} rejected {reason_counts}; "
        f"showing {len(accepted)} and {len(rejected)}"
    )

    sheets = audit.save_audit_sheets(
        layout, accepted, rejected, out_dir, per_sheet=args.per_sheet, workers=args.workers
    )
    for sheet in sheets:
        logger.info(f"Sheet: {sheet}")
    fb_accepted = sorted(
        (r for r in rows if r.model == "edge_translation"), key=lambda r: r.image_id
    )
    fb_rejected = sorted(
        (r for r in rows if r.fb_tried and not r.accepted), key=lambda r: r.image_id
    )
    fb_sample = {
        "fallback_accepted": random.Random(f"{args.seed}/fb_accepted").sample(
            fb_accepted, min(args.fallback, len(fb_accepted))
        ),
        "fallback_rejected": random.Random(f"{args.seed}/fb_rejected").sample(
            fb_rejected, min(args.fallback, len(fb_rejected))
        ),
    }
    logger.info(
        f"Edge fallback: {len(fb_accepted)} aligned, {len(fb_rejected)} tried and rejected; "
        f"showing {len(fb_sample['fallback_accepted'])} and {len(fb_sample['fallback_rejected'])}"
    )
    sheets += audit.save_audit_sheets(
        layout,
        fb_sample["fallback_accepted"],
        fb_sample["fallback_rejected"],
        out_dir,
        per_sheet=args.per_sheet,
        workers=args.workers,
        prefix="fallback_",
    )
    vehicles_csv = layout.log("vehicles")
    if vehicles_csv.exists():
        with vehicles_csv.open(encoding="utf-8") as f:
            vehicle_rows = list(csv.DictReader(f))
        fractions = {r["image_id"]: float(r["vehicles_fraction"]) for r in vehicle_rows}
        densest = sorted(fractions.items(), key=lambda kv: (-kv[1], kv[0]))[: args.dense]
        over = sum(r["dense"] == "1" for r in vehicle_rows)  # the flag as logged, not re-derived
        sheet = audit.save_vehicle_sheet(
            layout,
            densest,
            out_dir / "vehicles_dense_01.jpg",
            f"the {len(densest)} densest vehicles masks ({over} of {len(fractions)} frames over "
            f"{100 * float(cfg['masks']['vehicles']['max_frame_fraction']):.0f}%); mask in red",
            workers=args.workers,
        )
        logger.info(f"Dense-frame sheet: {sheet}")
    hist = out_dir / "fit_histograms.png"
    audit.plot_fit_histograms(rows, acfg, hist)
    logger.info(f"Histograms: {hist}")

    usable_ids = {i for split in dv.SPLITS for i in _read_manifest(pairs, layout, split)}
    stats = df.read_stats_csv(layout.stats_csv)
    day_log = df.read_day_log(layout.log("day"))
    cutoff = float(cfg["day_filter"]["min_lum_median"])
    counts = audit.plot_usable_luminance(
        stats, day_log, usable_ids, cutoff, out_dir / "usable_luminance.png"
    )
    logger.info(f"Usable pairs {sum(n for n, _ in counts.values())}: {counts} (usable, day pairs)")
    logger.info(f"Luminance plot: {out_dir / 'usable_luminance.png'}")
    lum = {r.image_id: r.lum_median for r in stats if r.image_id in usable_ids}
    for how, name, title in (
        ("darkest", "usable_darkest.jpg", "the 24 darkest usable pairs"),
        ("random", "usable_random.jpg", "24 random usable pairs"),
    ):
        picked = audit.pick_usable(lum, how, 24, args.seed)
        sheet = audit.save_usable_sheet(
            layout,
            picked,
            lum,
            out_dir / name,
            f"{title}: RGB | thermal (warped), darkest first; the model sees these PNGs",
            workers=args.workers,
        )
        logger.info(f"Usable sheet: {sheet}")

    with (out_dir / "sampled.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("sheet,position,image_id,reason\n")
        for name, shown in (
            ("accepted", accepted),
            ("rejected", rejected),
            ("fallback_accepted", fb_sample["fallback_accepted"]),
            ("fallback_rejected", fb_sample["fallback_rejected"]),
        ):
            for index, row in enumerate(shown):
                sheet = f"{name}_{index // args.per_sheet + 1:02d}.jpg"
                f.write(f"{sheet},{index % args.per_sheet + 1},{row.image_id},{row.reason}\n")


if __name__ == "__main__":
    main()
