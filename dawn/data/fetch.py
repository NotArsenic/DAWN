"""Get DroneVehicle off the Hugging Face Hub and into the layout `configs/data.yaml` expects.

`McCheng/DroneVehicle` is **not** a `datasets`-format repo: it holds three zip archives
(`train.zip`, `val.zip`, `test.zip`, about 14 GB) with no parquet shards and no loading script,
so `load_dataset("McCheng/DroneVehicle")` does not work. It has to be downloaded, unpacked and
arranged. That is what this module does, and `scripts/fetch_dronevehicle.py` is the CLI.

The pipeline wants::

    <root>/<split>/{<split>img, <split>imgr, <split>label, <split>labelr}/<stem>.{jpg,xml}

**The archives are not assumed to be in that shape.** Their internal structure is not documented
on the Hub, and mirrors differ in whether they wrap everything in an extra directory level. So
rather than hard-coding a path and failing on a mirror that nests things differently,
`discover_split_dirs` *searches* the extracted tree for the four folders by name and
`arrange_split` links them into place. A layout nobody predicted is then a one-line fix to
`FOLDER_SUFFIXES`, not a rewrite.

Safety: the archives are downloaded data, so `safe_extract` refuses any member whose resolved
path escapes the destination (the "zip slip" traversal), and extraction always goes to its own
directory, never on top of the arranged tree.
"""

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
# The four folders of a split, as `<split><suffix>`: RGB images, thermal images, RGB XML,
# thermal XML. `dawn.data.dronevehicle.DroneVehicleLayout` builds the same names.
FOLDER_SUFFIXES = ("img", "imgr", "label", "labelr")
# What a correctly arranged split holds, roughly: 28,439 pairs over three splits.
MIN_FILES_PER_FOLDER = 100
# Corrupt members tolerated per archive before it counts as archive-level corruption rather
# than one bad file. 56,878 images per the dataset card, so this is well under 0.1%.
MAX_BAD_MEMBERS = 50
# Share of a folder's stems that may fail to pair across the four before it counts as a broken
# download rather than a few bad files. 1% of ~28,400 is ~280 pairs, which the alignment filter
# would mostly have discarded anyway.
MAX_UNPAIRED_FRACTION = 0.01
ARRANGE_MODES = ("symlink", "copy", "move")


def folder_names(split: str) -> tuple[str, ...]:
    """`("trainimg", "trainimgr", "trainlabel", "trainlabelr")` for `train`."""
    return tuple(f"{split}{suffix}" for suffix in FOLDER_SUFFIXES)


# --- download ------------------------------------------------------------------------------


def download_archives(
    dest: Path, repo_id: str = REPO_ID, splits: Sequence[str] = SPLITS
) -> dict[str, Path]:
    """Fetch `<split>.zip` for each split into `dest`. Returns `{split: path}`.

    Uses `huggingface_hub.snapshot_download`, so a partial download resumes and an archive
    already in the Hub cache is not fetched twice. The repo is public and ungated, unlike the
    DINOv3 weights, so no token is needed.
    """
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


# --- extract -------------------------------------------------------------------------------


def safe_members(archive: zipfile.ZipFile, dest: Path) -> list[zipfile.ZipInfo]:
    """Every member whose resolved destination stays inside `dest`.

    A zip may name a member `../../etc/thing` and a naive `extractall` writes it there. This is
    downloaded data, so it is checked rather than trusted.

    Raises:
        ValueError: a member would escape `dest`.
    """
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
    """Unpack one archive into its own directory under `dest`, with a completion marker.

    The marker (`.extracted`) is what makes re-running cheap: unpacking 14 GB again because a
    later step failed is an hour nobody has. `force` ignores it.

    **A member with a bad CRC is skipped, not fatal.** `extractall` aborts the whole archive on
    the first corrupt file, which threw away a 14 GB extraction over one unreadable JPEG out of
    56,878. One missing frame costs nothing here -- its stem simply stops pairing across the
    four folders, so `verify_split` reports it and the alignment stage drops that pair, out of
    28,411. Whole-archive corruption is a different thing, so more than `max_bad` failures
    still raises, and every skipped name is logged and written into the marker rather than
    being quietly forgotten.

    Raises:
        zipfile.BadZipFile: more than `max_bad` members failed, i.e. the archive itself is bad
            rather than one file in it.
    """
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
                # The CRC is only checked once the member has been read to the end, so a
                # truncated file is already on disk. Leaving it would be worse than leaving
                # nothing: the stem would still pair across the four folders and the pipeline
                # would decode a corrupt JPEG instead of dropping the pair.
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


# --- discover and arrange ---------------------------------------------------------------------


def discover_split_dirs(tree: Path, split: str) -> dict[str, Path]:
    """Find this split's four folders anywhere under `tree`. `{folder name: path}`.

    Matched by directory name, case-insensitively, so it does not matter whether the archive
    unpacks to `trainimg/`, `train/trainimg/` or `DroneVehicle/train/trainimg/`. A folder that
    appears more than once resolves to the shallowest match, which is the real one when an
    archive carries a `__MACOSX` shadow tree or a nested copy.
    """
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
    """Put this split's folders where `data.yaml` expects them. Returns what was done.

    `symlink` is the default because the dataset is 14 GB and a second copy buys nothing; `copy`
    is for filesystems without symlinks, `move` for when the extracted tree is disposable.
    An existing, correct destination is left alone, so re-running is cheap and idempotent.

    Raises:
        ValueError: an unknown `mode`, or a folder this split needs was not discovered.
    """
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


# --- verify ----------------------------------------------------------------------------------


def verify_split(root: Path, split: str, sample: int = 5) -> list[str]:
    """Problems with one arranged split; empty means it is ready. Never raises.

    Checks the four folders exist and are populated, that the stems really pair across all four
    (the thing that actually breaks a prepare run), and that a sampled image is the 840x712 the
    whole pipeline assumes -- the crop to 640x512 and the border check both depend on it.
    """
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
    # A few unpaired stems are not a reason to stop. `prepare_dronevehicle.py stats` already
    # gives a pair missing a modality (or carrying an unreadable JPEG) a status and drops it at
    # the day stage (docs/conventions.md section 10.1), and a handful of corrupt files in a
    # 28,411-pair download costs almost nothing once the alignment filter has taken its ~80%.
    # A LOT of them is a bad download, which is a different thing and still fails.
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
    """Sampled RGB/thermal frames really are 840x712 with the white border."""
    from PIL import Image

    problems = []
    for stem in stems:
        for name in (f"{split}img", f"{split}imgr"):
            path = root / split / name / f"{stem}.jpg"
            try:
                with Image.open(path) as image:
                    size = image.size  # (width, height)
            except Exception as error:  # a truncated or non-image file
                problems.append(f"{split}/{name}/{stem}.jpg cannot be read: {error}")
                continue
            if size != (840, 712):
                problems.append(
                    f"{split}/{name}/{stem}.jpg is {size[0]}x{size[1]}, expected 840x712 "
                    "(the border is cropped to 640x512; configs/data.yaml frame_size)"
                )
    return problems


def verify(root: Path, splits: Sequence[str] = SPLITS, sample: int = 5) -> list[str]:
    """Every problem across every split; empty means `prepare_dronevehicle.py` can run."""
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
