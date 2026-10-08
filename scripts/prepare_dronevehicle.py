#!/usr/bin/env python3
"""Prepare DroneVehicle day thermal / RGB pairs in stages; dawn/data/ does the work.

    stats   Slow, once. Every stem of every split: existence, frame, border and thermal-channel
            checks, RGB luminance stats on the cropped frame, folder and file names from the XML,
            and 64-bit perceptual hashes of both modalities
            -> `dronevehicle.stats_csv` (not committed, regenerable).
    day     Instant, rerun freely. The day filter at `day_filter.min_lum_median`: a luminance
            histogram, contact sheets (random, just above / just below the cutoff, night
            folders), a per-folder day-share table and the per-pair day log
            -> <outputs_dir>, <manifest_dir>/logs/day.csv.
            *** Choose the cutoff by looking at these before running `align`. ***
    align   Box matching, RANSAC similarity (homography fallback), acceptance and the
            independent edge check for the day pairs -> <manifest_dir>/logs/align.csv, with the
            fitted transform of every pair. Pairs the boxes cannot judge (`labels_copied`,
            `few_matches`) get the edge-correlation fallback (`align.fallback`). Incremental: only
            pairs missing from the log are computed, unless the align thresholds changed (then
            all) or `--force`.
            `--include-copied` is a dry run with copied boxes allowed to count as evidence: it
            writes <outputs_dir>/align_include_copied.csv and touches nothing else.
    dedup   Accepted val and test pairs against the train pairs that passed the day filter
            (RGB and thermal hashes both within `dedup.max_hamming`) -> logs/dedup.csv and an
            example sheet just under / over the threshold.
    write   Warp, mask and write the PNGs of the surviving pairs under `aligned_dir`, the three
            manifests <manifest_dir>/{train,val,test}_pairs.txt, <manifest_dir>/fallback_pairs.txt
            (the ones aligned by edge correlation instead of boxes), logs/vehicles.csv (the share
            of each frame its vehicles mask covers, dense frames flagged, and how many vehicles
            took each branch of the per-vehicle rule) and logs/summary.md. The vehicles masks
            are recomputed on every run (they are deterministic), the other PNGs only when missing.
    vehicle-dilation   Read-only. The displacement of the vehicles that moved between the two
            exposures, over the box-aligned pairs, and its 95th percentile: how far the vehicles
            mask extends a vehicle that may have moved. Run after `align`; put the result in
            `masks.vehicles.extend_px`.
    all     stats, day, align, dedup, write, with no pause. Look at `day` first on a new dataset.

Paths and thresholds come from configs/data.yaml. Every stage skips work already on disk unless
`--force`, except `day`, `dedup` and the summary, which are cheap and always recomputed.

Usage:
    python scripts/prepare_dronevehicle.py stats --workers 8
    python scripts/prepare_dronevehicle.py day
    python scripts/prepare_dronevehicle.py align
    python scripts/prepare_dronevehicle.py dedup
    python scripts/prepare_dronevehicle.py write
"""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _setup(cfg: dict, args: argparse.Namespace):
    from dawn.data import align
    from dawn.data import dronevehicle as dv

    layout = dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])
    return layout, align.AlignConfig.from_config(cfg["align"])


def _split_of(image_id: str) -> str:
    return image_id.split("_", 1)[0]


def _stems(layout, split: str, limit: int | None) -> list[str]:
    from dawn.data import dronevehicle as dv

    listing = dv.list_pairs(layout, split)
    stems = sorted({*listing.paired, *listing.rgb_only, *listing.thermal_only})
    logger.info(
        f"  {split}: {len(stems)} stems = {len(listing.paired)} paired + "
        f"{len(listing.rgb_only)} RGB only + {len(listing.thermal_only)} thermal only"
    )
    return stems[:limit]


def run_stats(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data import day_filter as df
    from dawn.data.sheets import run_parallel

    layout, _ = _setup(cfg, args)
    if layout.stats_csv.exists() and not args.force:
        logger.info(f"{layout.stats_csv} exists: skipping stats (--force to redo)")
        return
    jobs = []
    for split in args.splits:
        stems = _stems(layout, split, args.limit)
        if not stems:
            raise SystemExit(f"No images under {layout.rgb_image(split, 'x').parent}")
        jobs += [(layout, split, stem) for stem in stems]
    rows = run_parallel(df.compute_pair_stats, jobs, args.workers, desc="stats")
    df.write_stats_csv(layout.stats_csv, rows)
    logger.info(f"Wrote {len(rows)} rows to {layout.stats_csv}")
    for split in args.splits:
        status = Counter(r.status for r in rows if r.split == split)
        borders = sum(1 for r in rows if r.split == split and (r.border_rgb or r.border_thermal))
        channels = sum(1 for r in rows if r.split == split and not r.thermal_channels_ok)
        logger.info(
            f"  {split}: {dict(status)}, bad borders {borders}, thermal channels {channels}"
        )


def run_day(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data import day_filter as df

    layout, _ = _setup(cfg, args)
    rows = [r for r in df.read_stats_csv(layout.stats_csv) if r.split in args.splits]
    section = cfg["day_filter"]
    cutoff = float(section["min_lum_median"])
    out_dir = layout.outputs_dir

    df.write_day_log(layout.log("day"), rows, cutoff)
    logger.info(f"Day log: {layout.log('day')}")
    hist = out_dir / "day_histogram.png"
    df.plot_log_luminance_histogram(rows, cutoff, hist, float(section["histogram_bin_log10"]))
    logger.info(f"Histogram: {hist}")
    if not args.no_sheets:
        sheets = df.save_day_sheets(
            layout,
            rows,
            cutoff,
            out_dir / "day_sheets",
            count=int(section["sheet_count"]),
            band_log10=float(section["band_log10"]),
            seed=args.seed,
            workers=args.workers,
        )
        for sheet in sheets:
            logger.info(f"Sheet: {sheet}")

    table = df.folder_table(rows, cutoff)
    folders_csv = out_dir / "day_folders.csv"
    folders_csv.parent.mkdir(parents=True, exist_ok=True)
    with folders_csv.open("w", encoding="utf-8", newline="") as f:
        f.write("folder,frames,day,day_share,median_log10_lum,splits\n")
        for t in table:
            f.write(
                f'"{t["folder"]}",{t["frames"]},{t["day"]},{t["day_share"]:.3f},'
                f"{t['median_log10_lum']:.3f},{t['splits']}\n"
            )
    mixed = [t for t in table if 0 < t["day_share"] < 1]
    logger.info(f"Folders: {len(table)} ({len(mixed)} mixed day/not-day): {folders_csv}")

    cutoffs = sorted({round(cutoff * f, 6) for f in (0.5, 0.7, 1.0, 1.4, 2.0)})
    counts = df.counts_at_cutoffs(rows, cutoffs)
    splits = list(next(iter(counts.values())))
    logger.info("Day pairs per split at neighbouring cutoffs (min_lum_median):")
    logger.info("  " + f"{'cutoff':>10}" + "".join(f"{s:>9}" for s in splits))
    for cut in cutoffs:
        mark = "  <- config" if cut == round(cutoff, 6) else ""
        logger.info("  " + f"{cut:>10g}" + "".join(f"{counts[cut][s]:>9}" for s in splits) + mark)
    reasons = Counter(df.day_reason(r, cutoff) for r in rows)
    logger.info(f"Day verdicts: {dict(reasons)}")


def run_align(cfg: dict, args: argparse.Namespace) -> None:
    import dataclasses

    from dawn.data import align
    from dawn.data import day_filter as df
    from dawn.data.sheets import run_parallel

    layout, acfg = _setup(cfg, args)
    day_log = df.read_day_log(layout.log("day"))
    ids = sorted(i for i, (is_day, _) in day_log.items() if is_day and _split_of(i) in args.splits)
    if args.include_copied:
        acfg = dataclasses.replace(acfg, exclude_copied_boxes=False)
        target = layout.outputs_dir / "align_include_copied.csv"
        existing, fingerprint_file = {}, None
        logger.info("Dry run: copied boxes count as evidence. The committed log is not touched.")
    else:
        target = layout.log("align")
        fingerprint_file = layout.stats_csv.parent / "align.fingerprint"
        existing = {}
        fingerprint = align.config_fingerprint(acfg)
        if target.exists() and not args.force:
            if fingerprint_file.exists() and fingerprint_file.read_text().strip() == fingerprint:
                existing = align.read_align_log(target)
            else:
                logger.info("Align thresholds changed (or no fingerprint): recomputing every pair")
    todo = [i for i in ids if i not in existing]
    logger.info(
        f"Align: {len(ids)} day pairs, {len(ids) - len(todo)} already logged, {len(todo)} to do"
    )
    jobs = [(layout, acfg, *_stem_split(i)) for i in todo]
    new = run_parallel(align.align_pair, jobs, args.workers, desc="align")
    rows = {**{i: existing[i] for i in ids if i in existing}, **{r.image_id: r for r in new}}
    ordered = [rows[i] for i in sorted(rows, key=_sort_key)]
    align.write_align_log(target, ordered)
    if fingerprint_file is not None:
        fingerprint_file.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_file.write_text(align.config_fingerprint(acfg) + "\n")
    logger.info(f"Align log: {target}")
    for split in args.splits:
        reasons = Counter(r.reason for r in ordered if _split_of(r.image_id) == split)
        logger.info(f"  {split}: {dict(sorted(reasons.items(), key=lambda kv: -kv[1]))}")
    by_edges = sum(r.model == "edge_translation" for r in ordered)
    logger.info(
        f"  match gate: {acfg.max_match_dist_px:g} px; homography fallback used on "
        f"{sum(r.model == 'homography' for r in ordered)} pairs; edge-correlation fallback aligned "
        f"{by_edges} of {sum(r.fb_tried for r in ordered)} tried"
    )


def _stem_split(image_id: str) -> tuple[str, str]:
    from dawn.data import dronevehicle as dv

    return dv.parse_image_id(image_id)


def _sort_key(image_id: str) -> tuple[int, str]:
    from dawn.data import dronevehicle as dv

    split = _split_of(image_id)
    return (dv.SPLITS.index(split) if split in dv.SPLITS else len(dv.SPLITS), image_id)


def run_dedup(cfg: dict, args: argparse.Namespace) -> None:
    import numpy as np

    from dawn.data import align, dedup
    from dawn.data import day_filter as df
    from dawn.data.audit import save_dedup_sheet

    layout, _ = _setup(cfg, args)
    stats = {r.image_id: r for r in df.read_stats_csv(layout.stats_csv)}
    day_log = df.read_day_log(layout.log("day"))
    aligned = align.read_align_log(layout.log("align"))
    section = cfg["dedup"]
    max_hamming = int(section["max_hamming"])
    # Train pairs that passed the day filter: independent of every alignment threshold, so
    # loosening alignment later cannot let a duplicate of an aligned-and-rejected train pair in.
    references = [
        stats[i] for i, (is_day, _) in sorted(day_log.items()) if is_day and _split_of(i) == "train"
    ]
    candidates = [
        stats[i]
        for i in sorted(aligned, key=_sort_key)
        if aligned[i].accepted and _split_of(i) in ("val", "test") and _split_of(i) in args.splits
    ]
    rows = dedup.cross_split_duplicates(candidates, references, max_hamming)
    dedup.write_dedup_log(layout.log("dedup"), rows)
    dropped = {
        s: sum(1 for r in rows if r.near_duplicate and _split_of(r.image_id) == s)
        for s in ("val", "test")
    }
    logger.info(
        f"Dedup: {len(candidates)} accepted val/test pairs against {len(references)} train day "
        f"pairs; dropped at max_hamming={max_hamming}: {dropped}"
    )
    for split in ("val", "test"):
        distances = [r.distance for r in rows if _split_of(r.image_id) == split]
        if distances:
            counts = dedup.counts_at(np.array(distances), list(section["report_hamming"]))
            logger.info(f"  {split}: pairs with a train pair within k bits: {counts}")
    sheet = save_dedup_sheet(
        layout,
        rows,
        max_hamming,
        layout.outputs_dir / "dedup_examples.jpg",
        count=int(section["sheet_count"]),
        workers=args.workers,
    )
    logger.info(f"Example sheet: {sheet}")


def run_write(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data import align, dedup, pairs, report
    from dawn.data import day_filter as df
    from dawn.data.sheets import run_parallel

    layout, acfg = _setup(cfg, args)
    stats = [r for r in df.read_stats_csv(layout.stats_csv) if r.split in args.splits]
    day_log = df.read_day_log(layout.log("day"))
    aligned = align.read_align_log(layout.log("align"))
    dedup_path = layout.log("dedup")
    dedups = dedup.read_dedup_log(dedup_path) if dedup_path.exists() else {}
    needs_dedup = [i for i, r in aligned.items() if r.accepted and _split_of(i) in ("val", "test")]
    if needs_dedup and not dedup_path.exists():
        raise SystemExit(
            "Run `dedup` before `write`: accepted val/test pairs have not been checked"
        )

    final = report.final_reasons(stats, day_log, aligned, dedups)
    keep = sorted((i for i, reason in final.items() if reason == "ok"), key=_sort_key)
    logger.info(f"Writing {len(keep)} pairs to {layout.aligned_dir}")
    masks = pairs.VehicleMaskConfig.from_config(cfg["masks"]["vehicles"])
    jobs = [(layout, acfg, aligned[i], args.force) for i in keep]
    written = run_parallel(pairs.write_aligned_pair, jobs, args.workers, desc="write")
    logger.info(f"  {sum(written)} written, {len(written) - sum(written)} already complete")
    for split in args.splits:
        ids = [i for i in keep if _split_of(i) == split]
        pairs.write_manifest(layout.manifest(f"{split}_pairs"), ids)
        logger.info(f"  {layout.manifest(f'{split}_pairs')}: {len(ids)} pairs")
    logger.info(
        f"Vehicles mask: margin {masks.dilate_px:g} px, extended {masks.extend_px:g} px along the "
        "long axis where a vehicle may have moved"
    )
    results = run_parallel(
        pairs.write_vehicles,
        [(layout, acfg, masks, aligned[i]) for i in keep],
        args.workers,
        desc="vehicles",
    )
    fractions = {i: r.fraction for i, r in zip(keep, results)}
    branches = {i: r.branches for i, r in zip(keep, results)}
    dense = report.write_vehicle_log(
        layout.log("vehicles"), fractions, masks.max_frame_fraction, branches
    )
    mean = 100 * sum(fractions.values()) / max(len(fractions), 1)
    logger.info(
        f"  vehicles mask covers {mean:.1f}% of a frame on average; {len(dense)} of {len(keep)} "
        f"frames over {100 * masks.max_frame_fraction:.0f}% flagged in {layout.log('vehicles')}"
    )
    totals = {
        n: sum(getattr(b, n) for b in branches.values())
        for n in ("matched", "static", "moved", "unsure", "thermal_only")
    }
    logger.info(f"  vehicles per branch: {totals}")
    fallback = report.fallback_ids(final, aligned)
    pairs.write_manifest(layout.manifest("fallback_pairs"), fallback)
    logger.info(
        f"  {layout.manifest('fallback_pairs')}: {len(fallback)} of them aligned by edge "
        "correlation (no boxes, no moving mask)"
    )
    summary = layout.log("summary").with_suffix(".md")
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        report.build_summary(
            stats, day_log, aligned, dedups, cfg, fractions, masks.max_frame_fraction, branches
        ),
        encoding="utf-8",
    )
    logger.info(f"Summary: {summary}")


def run_vehicle_dilation(cfg: dict, args: argparse.Namespace) -> None:
    from dawn.data import align
    from dawn.data.sheets import run_parallel

    layout, acfg = _setup(cfg, args)
    rows = [r for r in align.read_align_log(layout.log("align")).values() if r.accepted]
    box_rows = [r for r in rows if r.model in ("similarity", "homography")]
    jobs = [(layout, acfg, row) for row in box_rows]
    per_pair = run_parallel(
        align.pair_moving_displacements, jobs, args.workers, desc="displacement"
    )
    values = [d for ds in per_pair for d in ds]
    summary = align.summarize_displacements(values)
    logger.info(
        f"Moving vehicles in {len(box_rows)} box-aligned pairs: {summary['n']} vehicles in "
        f"{sum(1 for ds in per_pair if ds)} pairs"
    )
    if summary["n"]:
        logger.info(
            "  displacement px: median %.1f, p90 %.1f, p95 %.1f, p99 %.1f, max %.1f"
            % tuple(summary[k] for k in ("p50", "p90", "p95", "p99", "max"))
        )
    logger.info(
        f"  suggested masks.vehicles.extend_px = {summary['suggested_extend_px']} (the p95); "
        f"configured: {cfg['masks']['vehicles']['extend_px']}"
    )


def run_all(cfg: dict, args: argparse.Namespace) -> None:
    for stage in (run_stats, run_day, run_align, run_dedup, run_write):
        stage(cfg, args)


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    common.add_argument("--splits", nargs="+", default=None, help="Override dronevehicle.splits")
    common.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help=f"Processes (default: {default_workers()}, the cores this process may actually use)",
    )
    common.add_argument("--force", action="store_true", help="Redo work already on disk")
    return common


def default_workers(cap: int = 16) -> int:
    """Processes to fork by default: the cores this process may actually use, capped.

    NOT `os.cpu_count()`. Inside a container that reports the HOST's core count -- 100+ on a
    rented GPU box -- so the default forked ~100 workers, each importing torch, and the OOM
    killer took one and the pool died with `BrokenProcessPool`. `sched_getaffinity` respects
    the cgroup's CPU set, and the cap keeps memory sane on a machine that really does have
    that many cores: these jobs are JPEG decode and PNG encode, which saturate well before 16.
    """
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        available = os.cpu_count() or 1
    return max(1, min(cap, available))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DroneVehicle day thermal/RGB pairs: stats, day, align, dedup, write",
        epilog="Run `<stage> --help` for the options of each stage.",
    )
    stages = parser.add_subparsers(dest="stage", required=True)
    common = _common()

    stats = stages.add_parser("stats", parents=[common], help="Per-stem stats CSV (slow, once)")
    stats.add_argument("--limit", type=int, default=None, help="First N stems per split only")
    stats.set_defaults(run=run_stats)

    day = stages.add_parser("day", parents=[common], help="Day filter: histogram, sheets, log")
    day.add_argument("--seed", type=int, default=0, help="Contact-sheet sampling seed")
    day.add_argument("--no-sheets", action="store_true", help="Skip the contact sheets (faster)")
    day.set_defaults(run=run_day)

    aligning = stages.add_parser("align", parents=[common], help="Match, fit, accept, edge check")
    aligning.add_argument(
        "--include-copied",
        action="store_true",
        help="Dry run: let copied boxes count as evidence; writes only to outputs_dir",
    )
    aligning.set_defaults(run=run_align)

    stages.add_parser(
        "dedup", parents=[common], help="Drop val/test pairs that near-duplicate a train pair"
    ).set_defaults(run=run_dedup)
    stages.add_parser(
        "vehicle-dilation",
        parents=[common],
        help="Measure moving-vehicle displacement; suggest the vehicles-mask extension",
    ).set_defaults(run=run_vehicle_dilation)
    stages.add_parser(
        "write", parents=[common], help="Aligned PNGs, the three manifests, summary.md"
    ).set_defaults(run=run_write)

    everything = stages.add_parser("all", parents=[common], help="stats, day, align, dedup, write")
    everything.add_argument("--limit", type=int, default=None, help="First N stems per split only")
    everything.add_argument("--seed", type=int, default=0, help="Contact-sheet sampling seed")
    everything.add_argument("--no-sheets", action="store_true", help="Skip the day sheets")
    everything.set_defaults(run=run_all)

    args = parser.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    args.splits = args.splits or cfg["dronevehicle"]["splits"]
    for name, default in (
        ("limit", None),
        ("seed", 0),
        ("no_sheets", False),
        ("include_copied", False),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)
    logger.info(f"prepare_dronevehicle {args.stage}: splits {args.splits}, {args.workers} workers")
    args.run(cfg, args)


if __name__ == "__main__":
    main()
