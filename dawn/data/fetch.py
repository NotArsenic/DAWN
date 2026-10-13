"""Download, unpack, and arrange the DroneVehicle dataset."""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "McCheng/DroneVehicle"
SPLITS = ("train", "val", "test")

FOLDER_SUFFIXES = ("img", "imgr", "label", "labelr")

MIN_FILES_PER_FOLDER = 100
MAX_BAD_MEMBERS = 50
MAX_UNPAIRED_FRACTION = 0.01
ARRANGE_MODES = ("symlink", "copy", "move")


def folder_names(split: str) -> tuple[str, ...]:
    """Return the expected subdirectory names for a split."""
    return tuple(f"{split}{suffix}" for suffix in FOLDER_SUFFIXES)


def download_archives(
    dest: Path, repo_id: str = REPO_ID, splits: Sequence[str] = SPLITS
) -> dict[str, Path]:
    """Download split zip archives from Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    dest.mkdir(parents=True, exist_ok=True)
    patterns = [f"{split}.zip" for split in splits]
    logger.info("downloading %s (%s) -> %s", repo_id, ", ".join(patterns), dest)
    local = Path(
        snapshot_download(
            repo_id=repo_id, repo_type="dataset", allow_patterns=patterns, local_dir=dest
        )
    )
    found = {}
    for split in splits:
        path = local / f"{split}.zip"
        if path.exists():
            found[split] = path
        else:
            logger.warning("%s.zip is not in %s", split, repo_id)
    return found


def safe_members(archive: zipfile.ZipFile, dest: Path) -> list[zipfile.ZipInfo]:
    """Validate archive members to prevent path traversal outside the destination directory."""
    dest = dest.resolve()
    members = []
    for member in archive.infolist():
        target = (dest / member.filename).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"{archive.filename}: member {member.filename!r} escapes {dest}")
        members.append(member)
    return members


def safe_extract(
    zip_path: Path, dest: Path, force: bool = False, max_bad: int = MAX_BAD_MEMBERS
) -> Path:
    """Safely extract an archive into destination directory."""
    out = dest / zip_path.stem
    marker = out / ".extracted"
    if marker.exists() and not force:
        logger.info("%s already extracted at %s", zip_path.name, out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    logger.info("extracting %s -> %s", zip_path.name, out)
    bad: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in safe_members(archive, out):
            try:
                archive.extract(member, out)
            except (zipfile.BadZipFile, OSError) as error:
                partial = out / member.filename
                partial.unlink(missing_ok=True)
                bad.append(member.filename)
                logger.warning("  skipping %s: %s", member.filename, error)
                if len(bad) > max_bad:
                    raise zipfile.BadZipFile(
                        f"{zip_path.name}: {len(bad)} members failed to extract, the last "
                        f"{member.filename!r}. That is archive-level corruption, not one bad "
                        f"file: delete {zip_path} and re-run to download it again."
                    ) from error
    if bad:
        logger.warning(
            "%s: %d of its files were corrupt and were skipped; the pairs they belong to will "
            "drop out at the alignment stage",
            zip_path.name,
            len(bad),
        )
    marker.write_text("ok\n" + "".join(f"skipped {name}\n" for name in bad), encoding="utf-8")
    return out


def discover_split_dirs(tree: Path, split: str) -> dict[str, Path]:
    """Locate split subdirectories within an extracted directory tree."""
    wanted = {name.lower(): name for name in folder_names(split)}
    best: dict[str, tuple[int, Path]] = {}
    for current, directories, _ in os.walk(tree):
        directories[:] = [d for d in directories if d != "__MACOSX"]
        for directory in directories:
            key = directory.lower()
            if key not in wanted:
                continue
            path = Path(current) / directory
            depth = len(path.relative_to(tree).parts)
            if key not in best or depth < best[key][0]:
                best[key] = (depth, path)
    return {wanted[key]: path for key, (_, path) in best.items()}


def arrange_split(
    found: Mapping[str, Path], root: Path, split: str, mode: str = "symlink"
) -> list[str]:
    """Organize split folders into the target directory structure."""
    if mode not in ARRANGE_MODES:
        raise ValueError(f"mode must be one of {ARRANGE_MODES}, got {mode!r}")
    missing = [name for name in folder_names(split) if name not in found]
    if missing:
        raise ValueError(f"{split}: no {', '.join(missing)} anywhere in the extracted archive")

    target_dir = root / split
    target_dir.mkdir(parents=True, exist_ok=True)
    actions = []
    for name in folder_names(split):
        source, target = found[name].resolve(), target_dir / name
        if target.is_symlink() and target.resolve() == source:
            actions.append(f"{split}/{name}: already linked")
            continue
        if target.exists() and not target.is_symlink():
            actions.append(f"{split}/{name}: already there, left alone")
            continue
        if target.is_symlink():
            target.unlink()
        if mode == "symlink":
            target.symlink_to(source, target_is_directory=True)
        elif mode == "copy":
            shutil.copytree(source, target)
        else:
            shutil.move(str(source), str(target))
        actions.append(f"{split}/{name}: {mode} <- {source}")
    return actions


def verify_split(root: Path, split: str, sample: int = 5) -> list[str]:
    """Verify that split files and folders are present and structurally valid."""
    problems: list[str] = []
    counts: dict[str, set[str]] = {}
    for name in folder_names(split):
        directory = root / split / name
        if not directory.is_dir():
            problems.append(f"{split}/{name} is missing")
            continue
        suffix = ".xml" if name.endswith("label") or name.endswith("labelr") else ".jpg"
        stems = {p.stem for p in directory.iterdir() if p.suffix.lower() == suffix}
        counts[name] = stems
        if len(stems) < MIN_FILES_PER_FOLDER:
            problems.append(f"{split}/{name} holds only {len(stems)} {suffix} files")
    if len(counts) != len(folder_names(split)):
        return problems

    shared = set.intersection(*counts.values())
    for name, stems in counts.items():
        extra = len(stems - shared)
        if not extra:
            continue
        share = extra / max(len(stems), 1)
        message = f"{split}/{name} has {extra} stems the other three do not ({share:.1%})"
        if share > MAX_UNPAIRED_FRACTION:
            problems.append(message)
        else:
            logger.warning("%s -- tolerated; those pairs drop out at the day stage", message)
    if not shared:
        problems.append(f"{split}: no stem appears in all four folders")
        return problems

    problems += _check_frames(root, split, sorted(shared)[:sample])
    return problems


def _check_frames(root: Path, split: str, stems: Iterable[str]) -> list[str]:
    """Verify image dimensions for a sample of image stems."""
    from PIL import Image

    problems = []
    for stem in stems:
        for name in (f"{split}img", f"{split}imgr"):
            path = root / split / name / f"{stem}.jpg"
            try:
                with Image.open(path) as image:
                    size = image.size
            except Exception as error:
                problems.append(f"{split}/{name}/{stem}.jpg cannot be read: {error}")
                continue
            if size != (840, 712):
                problems.append(
                    f"{split}/{name}/{stem}.jpg is {size[0]}x{size[1]}, expected 840x712 "
                    "(the border is cropped to 640x512; configs/data.yaml frame_size)"
                )
    return problems


def verify(root: Path, splits: Sequence[str] = SPLITS, sample: int = 5) -> list[str]:
    """Verify all dataset splits and return a list of any detected issues."""
    return [problem for split in splits for problem in verify_split(root, split, sample)]


__all__ = [
    "ARRANGE_MODES",
    "FOLDER_SUFFIXES",
    "MAX_BAD_MEMBERS",
    "MAX_UNPAIRED_FRACTION",
    "MIN_FILES_PER_FOLDER",
    "REPO_ID",
    "SPLITS",
    "arrange_split",
    "discover_split_dirs",
    "download_archives",
    "folder_names",
    "safe_extract",
    "safe_members",
    "verify",
    "verify_split",
]
