#!/usr/bin/env python3
"""Prepare VisDrone-DET in three phases; dawn/data/prepare.py does the work.

    stats         Slow, run once. Every raw image of every split: native size, luminance
                  median / mean / 5th / 95th percentile (256-px linear copy), box count, noise
                  level and highlight ratio (native) -> `visdrone.stats_csv`. Every CPU core.
    label-sheets  Labelling sheets, one row per video sequence (darkest, median and brightest
                  frame), and, if it does not exist yet, the hand-label CSV `visdrone.labels_csv`
                  with one blank row per sequence in the same order.
    split         Instant, rerun freely. Stats CSV + hand labels ->
                  <manifest_dir>/<split>/{day,night,dusk}.txt (excluded and unlabelled images
                  are in none), a luminance histogram, a noise-vs-highlight scatter and one
                  contact sheet per manifest, next to the stats CSV.
    resize        512-px copies (long side, aspect kept), area-averaged in linear space, 8-bit
                  sRGB PNG under `visdrone.prepared_dir`. Existing copies are kept, so it resumes.

Paths come from the `visdrone` section of configs/data.yaml.

Usage:
    python scripts/prepare_visdrone.py stats
    python scripts/prepare_visdrone.py label-sheets
    python scripts/prepare_visdrone.py split
    python scripts/prepare_visdrone.py resize
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_stats(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data.prepare import compute_image_stats, run_parallel
    from dawn.data.visdrone import BoxFilter, VisDroneLayout, write_stats_csv

    layout = VisDroneLayout.from_config(cfg["visdrone"])
    box_filter = BoxFilter.from_config(cfg["visdrone"].get("box_filter"))
    jobs = []
    for split in args.splits:
        image_ids = layout.raw_image_ids(split)[: args.limit]
        if not image_ids:
            raise SystemExit(f"No images in {layout.raw_image(split, 'x').parent}")
        logger.info(f"  {split}: {len(image_ids)} images")
        jobs += [(layout, split, image_id, box_filter) for image_id in image_ids]
    rows = run_parallel(compute_image_stats, jobs, args.workers, desc="stats")
    write_stats_csv(layout.stats_csv, rows)
    logger.info(f"Wrote {len(rows)} rows to {layout.stats_csv}")


def run_label_sheets(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data.prepare import save_sequence_sheets, write_labels_template
    from dawn.data.visdrone import (
        VisDroneLayout,
        is_still_collection,
        read_labels,
        read_stats_csv,
        sequence_id,
    )

    layout = VisDroneLayout.from_config(cfg["visdrone"])
    labels_csv = Path(cfg["visdrone"]["labels_csv"])
    stats = [row for row in read_stats_csv(layout.stats_csv).values() if row.split in args.splits]
    out_dir = layout.stats_csv.parent / "labelling"
    pages = save_sequence_sheets(layout, stats, out_dir, args.per_page, args.workers)
    logger.info(f"{len(pages)} sequence sheets in {out_dir}")
    if not labels_csv.exists():
        write_labels_template(stats, labels_csv)
        logger.info(f"Wrote {labels_csv}: fill in the label column (day / night / dusk / exclude)")
        return
    listed = set(read_labels(labels_csv))
    videos = {sequence_id(r.image_id) for r in stats} - listed
    missing = sorted(seq for seq in videos if not is_still_collection(seq))
    logger.info(f"{labels_csv} exists and was left alone; video sequences not in it: {missing}")


def run_split(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data.prepare import (
        plot_feature_scatter,
        plot_log_luminance_histogram,
        resolve_labels,
        save_auto_day_sheet,
        save_contact_sheets,
        write_manifests,
    )
    from dawn.data.splitting import (
        FINAL_SUBSETS,
        auto_day_ids,
        build_final_manifests,
        group_id,
        write_final_manifests,
    )
    from dawn.data.visdrone import VisDroneLayout, read_labels, read_stats_csv

    layout = VisDroneLayout.from_config(cfg["visdrone"])
    labels_csv = Path(cfg["visdrone"]["labels_csv"])
    stats = [row for row in read_stats_csv(layout.stats_csv).values() if row.split in args.splits]
    if labels_csv.exists():
        labels = resolve_labels(stats, read_labels(labels_csv))
    else:
        logger.warning(f"No {labels_csv} yet (run label-sheets): every image is unlabelled")
        labels = resolve_labels(stats, {})

    counts = write_manifests(stats, labels, layout.manifest_dir)
    names = ("day", "night", "dusk", "exclude", "unlabelled")
    logger.info(f"Manifests in {layout.manifest_dir} (only day, night and dusk get one):")
    logger.info("  " + f"{'split':<10}" + "".join(f"{name:>11}" for name in names))
    for split, by_label in counts.items():
        logger.info("  " + f"{split:<10}" + "".join(f"{by_label[name]:>11}" for name in names))

    out_dir = layout.stats_csv.parent
    for name, plot in [
        ("log_luminance_hist.png", plot_log_luminance_histogram),
        ("noise_vs_highlights.png", plot_feature_scatter),
    ]:
        plot(stats, labels, out_dir / name)
        logger.info(f"Plot: {out_dir / name}")
    sheets = save_contact_sheets(
        layout,
        stats,
        labels,
        out_dir / "contact_sheets",
        per_sheet=args.per_sheet,
        seed=args.seed,
        workers=args.workers,
    )
    logger.info(f"Contact sheets: {len(sheets)} in {out_dir / 'contact_sheets'}")

    # The four manifests the rest of the project actually reads. `write_manifests` above is the
    # per-label bookkeeping of phase 2; this is the training split, and it is the step that
    # auto-accepts still collections, regroups train/val by group and checks test-dev shares no
    # group with either (dawn/data/splitting.py).
    if not args.final:
        logger.info("Skipping final manifests (--no-final)")
        return
    final, cutoffs = build_final_manifests(stats, labels, cfg["split"])
    counts = write_final_manifests(final, layout.manifest_dir)
    logger.info(f"Final manifests in {layout.manifest_dir}:")
    for name in FINAL_SUBSETS:
        groups = len({group_id(image_id) for image_id in final[name]})
        logger.info(f"  {name:<10}{counts[name]:>9} images{groups:>7} groups")
    if cutoffs is None:
        logger.info("Auto-day is off: no still collection entered a manifest")
        return
    logger.info(
        f"Auto-day cutoffs: median luminance >= {cutoffs.min_luminance:.4g}, "
        f"highlight ratio <= {cutoffs.max_highlight_ratio:.4g} "
        f"(fit on {cutoffs.num_day} day and {cutoffs.num_other} non-day hand-labelled frames: "
        f"{cutoffs.errors} errors, {100 * cutoffs.recall:.1f}% of day frames kept)"
    )
    accepted = auto_day_ids(stats, cutoffs)
    sheet = save_auto_day_sheet(
        layout,
        stats,
        accepted,
        out_dir / "contact_sheets",
        count=int(cfg["split"]["auto_day"]["contact_sheet"]),
        seed=args.seed,
        workers=args.workers,
    )
    logger.info(f"Auto-day accepted {len(accepted)} still images; spot-check sheet: {sheet}")


def run_resize(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data.prepare import make_lowres_copy, run_parallel
    from dawn.data.visdrone import VisDroneLayout

    layout = VisDroneLayout.from_config(cfg["visdrone"])
    long_side = cfg["visdrone"]["image_size"]
    jobs = [
        (layout, split, image_id, long_side, args.overwrite)
        for split in args.splits
        for image_id in layout.raw_image_ids(split)[: args.limit]
    ]
    run_parallel(make_lowres_copy, jobs, args.workers, desc="resize")
    logger.info(f"{len(jobs)} copies at {long_side} px in {layout.prepared_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare VisDrone-DET: stats, label-sheets, split, resize",
        epilog="Run `<phase> --help` for the options of each phase.",
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--splits", nargs="+", default=None, help="Override visdrone.splits")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Processes (default: every core)")
    phases = parser.add_subparsers(dest="phase", required=True)

    stats = phases.add_parser("stats", help="Phase 1: per-image stats CSV (slow, once)")
    stats.add_argument("--limit", type=int, default=None, help="First N images per split only")
    stats.set_defaults(run=run_stats)

    sheets = phases.add_parser("label-sheets", help="Sequence labelling sheets + label CSV")
    sheets.add_argument("--per-page", type=int, default=50, help="Sequences per sheet")
    sheets.set_defaults(run=run_label_sheets)

    split = phases.add_parser("split", help="Phase 2: labels -> manifests, plots, contact sheets")
    split.add_argument("--per-sheet", type=int, default=50, help="Thumbnails per contact sheet")
    split.add_argument("--seed", type=int, default=0, help="Contact-sheet sampling seed")
    split.add_argument("--no-final", dest="final", action="store_false",
                       help="Per-label manifests and plots only: skip the four training manifests")
    split.set_defaults(run=run_split)

    resize = phases.add_parser("resize", help="Phase 3: 512-px linear-space PNG copies")
    resize.add_argument("--limit", type=int, default=None, help="First N images per split only")
    resize.add_argument("--overwrite", action="store_true", help="Redo existing copies")
    resize.set_defaults(run=run_resize)

    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    args.splits = args.splits or cfg["visdrone"]["splits"]
    logger.info(f"prepare_visdrone {args.phase}: splits {args.splits}, {args.workers} workers")
    args.run(cfg, args)


if __name__ == "__main__":
    main()
