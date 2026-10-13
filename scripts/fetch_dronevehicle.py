#!/usr/bin/env python3
"""Download and unpack the DroneVehicle dataset archives for preprocessing."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf  # noqa: E402

from dawn.data.fetch import (  # noqa: E402
    ARRANGE_MODES,
    REPO_ID,
    SPLITS,
    arrange_split,
    discover_split_dirs,
    download_archives,
    safe_extract,
    verify,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fetch")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and arrange DroneVehicle for the data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-config", default="configs/data.yaml", help="For the target root")
    parser.add_argument("--root", default=None, help="Override dronevehicle.root")
    parser.add_argument(
        "--archives",
        default=None,
        help="Directory of zips you already downloaded. Default: fetch them from the Hub.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Where archives are downloaded and unpacked (default: <root>/_archives). Kept "
        "separate from the arranged tree so a re-extract never half-overwrites it.",
    )
    parser.add_argument("--splits", nargs="*", default=list(SPLITS), help="Which splits")
    parser.add_argument("--mode", choices=ARRANGE_MODES, default="symlink", help="How to arrange")
    parser.add_argument("--repo", default=REPO_ID, help="Hub dataset id")
    parser.add_argument("--force-extract", action="store_true", help="Unpack again")
    parser.add_argument("--verify", action="store_true", help="Only check what is on disk")
    parser.add_argument("--sample", type=int, default=5, help="Frames per split to open")
    return parser.parse_args()


def main() -> int:
    """Run dataset download, extraction, and verification workflow."""
    args = parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(ROOT / args.data_config), resolve=True)
    root = Path(args.root or cfg["dronevehicle"]["root"])
    if not root.is_absolute():
        root = ROOT / root

    if not args.verify:
        work = Path(args.work_dir) if args.work_dir else root / "_archives"
        if not work.is_absolute():
            work = ROOT / work

        if args.archives:
            source = Path(args.archives)
            archives = {s: source / f"{s}.zip" for s in args.splits}
            archives = {s: p for s, p in archives.items() if p.exists()}
            missing = sorted(set(args.splits) - set(archives))
            if missing:
                logger.error("no %s in %s", ", ".join(f"{m}.zip" for m in missing), source)
                return 1
        else:
            logger.info("About 14 GB. Already-cached archives are not re-downloaded.")
            archives = download_archives(work / "zips", args.repo, args.splits)
            if not archives:
                logger.error("nothing downloaded from %s", args.repo)
                return 1

        for split, archive in archives.items():
            tree = safe_extract(archive, work / "extracted", force=args.force_extract)
            found = discover_split_dirs(tree, split)
            if len(found) < 4:
                logger.error(
                    "%s: found only %s under %s. The archive is laid out differently than any "
                    "mirror seen so far; point --archives at the right zips, or add the folder "
                    "names to dawn/data/fetch.py FOLDER_SUFFIXES.",
                    split,
                    ", ".join(sorted(found)) or "nothing",
                    tree,
                )
                return 1
            for action in arrange_split(found, root, split, args.mode):
                logger.info("  %s", action)

    problems = verify(root, args.splits, args.sample)
    if problems:
        logger.error("%d problem(s) with %s:", len(problems), root)
        for problem in problems:
            logger.error("  %s", problem)
        return 1

    logger.info("%s looks right for every split: %s", root, ", ".join(args.splits))
    logger.info("Next: python scripts/prepare_dronevehicle.py all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
