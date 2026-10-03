"""VisDrone-DET: on-disk layout, file formats, box filtering and the dataset class.

Layout (paths from the `visdrone` section of configs/data.yaml, see `VisDroneLayout`):

    <root>/VisDrone2019-DET-<split>/images/<image_id>.jpg        raw download
    <root>/VisDrone2019-DET-<split>/annotations/<image_id>.txt
    <stats_csv>                                          prepare_visdrone.py stats   (phase 1)
    <labels_csv>                                         hand labels, committed
    <manifest_dir>/<split>/{day,night,dusk}.txt          prepare_visdrone.py split   (phase 2)
    <prepared_dir>/<split>/<image_id>.png                prepare_visdrone.py resize  (phase 3)

Image IDs look like `0000287_00759_d_0000759`. The first field is the sequence: frames of one
video share it, so labels are given per sequence. The exception is the 49 groups whose ID starts
with 99999: each is a collection of stills (6,497 images, day and night interleaved), so those
images are labelled one by one.

The dataset only loads images: no degradation happens here. Degradation is a separate transform
in the dataloader workers, and it ends with fixed-size 256x256 tensors so batches can mix
VisDrone's aspect ratios. Every image tensor is linear RGB, float32, [0, 1]
(docs/conventions.md section 1): files are decoded and linearized here, sRGB never leaves this
module.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dawn.data.color import srgb8_to_linear

SPLITS = ("train", "val", "test-dev")
LABELS = ("day", "night", "dusk", "exclude")  # hand labels
SUBSETS = ("day", "night", "dusk")  # labels that get a manifest; `exclude` never does
STILLS_PREFIX = "99999"  # sequence IDs of still collections, not videos
Mode = Literal["lowres", "native_crops", "eval"]
MODES = ("lowres", "native_crops", "eval")


@dataclass(frozen=True)
class VisDroneLayout:
    """Where every VisDrone file lives (module docstring)."""

    root: Path  # raw download
    stats_csv: Path
    manifest_dir: Path
    prepared_dir: Path

    @classmethod
    def from_config(cls, cfg: Mapping) -> VisDroneLayout:
        """Build from the `visdrone` section of data.yaml."""
        return cls(
            root=Path(cfg["root"]),
            stats_csv=Path(cfg["stats_csv"]),
            manifest_dir=Path(cfg["manifest_dir"]),
            prepared_dir=Path(cfg["prepared_dir"]),
        )

    def raw_image(self, split: str, image_id: str) -> Path:
        return self._raw_split_dir(split) / "images" / f"{image_id}.jpg"

    def raw_annotation(self, split: str, image_id: str) -> Path:
        return self._raw_split_dir(split) / "annotations" / f"{image_id}.txt"

    def lowres_image(self, split: str, image_id: str) -> Path:
        return self.prepared_dir / split / f"{image_id}.png"

    def manifest(self, split: str, subset: str) -> Path:
        return manifest_path(self.manifest_dir, split, subset)

    def raw_image_ids(self, split: str) -> list[str]:
        """Sorted IDs (file stems) of every raw image in `split`."""
        return sorted(path.stem for path in (self._raw_split_dir(split) / "images").glob("*.jpg"))

    def _raw_split_dir(self, split: str) -> Path:
        return self.root / f"VisDrone2019-DET-{split}"


def manifest_path(manifest_dir: str | Path, split: str, subset: str) -> Path:
    """`<manifest_dir>/<split>/<subset>.txt`."""
    return Path(manifest_dir) / split / f"{subset}.txt"


# --- file formats ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageStats:
    """One row of the stats CSV (phase 1). Luminance is linear Rec. 709 in [0, 1], measured on
    the linearized image area-downsampled to 256 px on the long side."""

    image_id: str
    split: str
    native_width: int
    native_height: int
    lum_median: float
    lum_mean: float
    lum_p05: float
    lum_p95: float
    num_boxes: int  # after the label rules of BoxFilter (category and score), before min_size
    noise: float  # prepare.noise_level, native resolution: ~sqrt(sensor gain), NaN if unmeasurable
    highlight_ratio: float  # prepare.highlight_ratio, native resolution: p99.5 / median


_STATS_TYPES = {"str": str, "int": int, "float": float}


def write_stats_csv(path: str | Path, stats: Iterable[ImageStats]) -> None:
    """Write the stats CSV, sorted by (split in SPLITS order, image_id) whatever the input order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(stats, key=lambda row: (split_rank(row.split), row.split, row.image_id))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([field.name for field in fields(ImageStats)])
        writer.writerows(astuple(row) for row in rows)


def read_stats_csv(path: str | Path) -> dict[str, ImageStats]:
    """Read the stats CSV into `{image_id: ImageStats}`, in file order."""
    types = {field.name: _STATS_TYPES[field.type] for field in fields(ImageStats)}
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = [
            ImageStats(**{name: types[name](value) for name, value in record.items()})
            for record in csv.DictReader(f)
        ]
    return {row.image_id: row for row in rows}


def split_rank(split: str) -> int:
    """Sort key for splits: SPLITS order, anything else after them."""
    return SPLITS.index(split) if split in SPLITS else len(SPLITS)


def sequence_id(image_id: str) -> str:
    """The sequence (video) an image belongs to: the first field of its ID."""
    return image_id.split("_", 1)[0]


def is_still_collection(sequence: str) -> bool:
    """True for the 99999xx groups, which collect unrelated stills rather than one video."""
    return sequence.startswith(STILLS_PREFIX)


def read_labels(path: str | Path) -> dict[str, str]:
    """Read the hand-label CSV into `{key: label}`, in file order.

    A key is a sequence ID (labels every frame of it) or an image ID (overrides its sequence).
    Only the `key` and `label` columns are read; any others (notes, helper columns from the
    template) are ignored. Labels are stripped and lowercased; a blank label means unlabelled.
    Validation against the data happens in `prepare.resolve_labels`.

    Raises:
        ValueError: A missing `key` or `label` column, or a key listed twice.
    """
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"key", "label"} <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: needs a 'key' and a 'label' column")
        labels: dict[str, str] = {}
        for record in reader:
            key = (record["key"] or "").strip()
            if not key:
                continue
            if key in labels:
                raise ValueError(f"{path}: {key} is listed twice")
            labels[key] = (record["label"] or "").strip().lower()
    return labels


def set_labels(path: str | Path, updates: Mapping[str, str]) -> None:
    """Write labels into the hand-label CSV in place (a missing file is started).

    Keys already in the file get the new label, other keys are appended as new rows; every other
    row and column is kept as is. The file is rewritten through a temporary file and a rename, so
    an interrupted write never leaves half a file. An empty label clears one.

    Raises:
        ValueError: A label outside `LABELS` (nothing is written).
    """
    bad = {key: label for key, label in updates.items() if label and label not in LABELS}
    if bad:
        raise ValueError(f"unknown labels {bad}, expected one of {LABELS}")
    path = Path(path)
    fieldnames, rows = ["key", "label"], []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames, rows = list(reader.fieldnames or fieldnames), list(reader)
    pending = dict(updates)
    for row in rows:
        key = (row.get("key") or "").strip()
        if key in pending:
            row["label"] = pending.pop(key)
    rows += [{"key": key, "label": label} for key, label in pending.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    with partial.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames, restval="", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def read_manifest(path: str | Path) -> list[str]:
    """Image IDs, one per line, in file order (blank lines skipped)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def write_manifest(path: str | Path, image_ids: Iterable[str]) -> None:
    """Write image IDs one per line, sorted, so a manifest never depends on processing order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{image_id}\n" for image_id in sorted(image_ids)), encoding="utf-8")


def read_srgb8(path: str | Path) -> torch.Tensor:
    """Decode an image file to `(3, H, W)` uint8 sRGB codes (pixel grid as stored, no EXIF
    rotation, so annotations stay aligned)."""
    with Image.open(path) as image:
        pixels = np.array(image.convert("RGB"))
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def read_annotations(path: str | Path) -> torch.Tensor:
    """Parse a VisDrone-DET annotation file.

    Returns:
        `(N, 8)` int64, one row per line: left, top, width, height, score, category,
        truncation, occlusion (native pixels). `(0, 8)` for an empty file.
    """
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        values = [value for value in line.split(",") if value.strip()]  # some files end in ","
        if not values:
            continue
        if len(values) != 8:
            raise ValueError(f"{path}:{number}: expected 8 values, got {len(values)}")
        rows.append([int(value) for value in values])
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 8)


@dataclass(frozen=True)
class BoxFilter:
    """Which annotation boxes count as objects.

    Always dropped: boxes with score 0. Settings:
        drop_categories: 0 is "ignored regions", 11 is "others".
        min_size: boxes narrower or shorter than this many pixels, measured at the returned
            resolution after clipping to the image, are dropped.
    """

    drop_categories: tuple[int, ...] = (0, 11)
    min_size: float = 16.0

    @classmethod
    def from_config(cls, cfg: Mapping | None) -> BoxFilter:
        """From the `visdrone.box_filter` section of data.yaml (None gives the defaults)."""
        cfg = cfg or {}
        return cls(
            drop_categories=tuple(int(c) for c in cfg.get("drop_categories", cls.drop_categories)),
            min_size=float(cfg.get("min_size", cls.min_size)),
        )

    def keep_labels(self, annotations: torch.Tensor) -> torch.Tensor:
        """`(N,)` bool: the category and score rules only (resolution independent)."""
        dropped = torch.tensor(self.drop_categories, dtype=torch.int64)
        return ~torch.isin(annotations[:, 5], dropped) & (annotations[:, 4] != 0)

    def __call__(
        self,
        annotations: torch.Tensor,
        scale: tuple[float, float],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter and rescale boxes to a returned resolution.

        Args:
            annotations: `(N, 8)` from `read_annotations`, native pixels.
            scale: `(sx, sy)`, returned pixels per native pixel along x and y.
            image_size: Returned `(width, height)`; boxes are clipped to it.

        Returns:
            boxes: `(M, 4)` float32 `(x0, y0, x1, y1)` in returned pixels, inside the image.
            categories: `(M,)` int64.
        """
        kept = annotations[self.keep_labels(annotations)]
        (sx, sy), (width, height) = scale, image_size
        left, top, w, h = kept[:, :4].to(torch.float32).unbind(dim=1)
        boxes = torch.stack(
            [
                (left * sx).clamp(0, width),
                (top * sy).clamp(0, height),
                ((left + w) * sx).clamp(0, width),
                ((top + h) * sy).clamp(0, height),
            ],
            dim=1,
        )
        big = (boxes[:, 2] - boxes[:, 0] >= self.min_size) & (
            boxes[:, 3] - boxes[:, 1] >= self.min_size
        )
        return boxes[big], kept[big, 5]


# --- dataset -----------------------------------------------------------------------------------


class VisDroneDataset(Dataset):
    """VisDrone-DET images for one manifest, in one of three modes.

    | mode           | `day`                                         | used by               |
    |----------------|-----------------------------------------------|-----------------------|
    | `lowres`       | the 512-px copy, whole image, variable aspect | flow training         |
    | `native_crops` | a random `crop_size` crop of the raw image    | denoiser training     |
    | `eval`         | the whole raw image, plus filtered boxes      | evaluation (batch 1)  |

    Every item is a dict:
        day: `(3, h, w)` float32 linear RGB in [0, 1].
        image_id: str.
        native_size: `(2,)` int64 `(W, H)` of the raw image, from the stats CSV. Each returned
            pixel stands for `W * H / (w * h)` native pixels (for `lowres`; 1 otherwise), which is
            what the camera model needs to simulate binned pixels (docs/conventions.md 6).
        scale: float, returned pixels per native pixel along the long side (1.0 unless `lowres`).
        boxes, categories: `eval` only, from `BoxFilter` at the returned resolution.
    """

    def __init__(
        self,
        manifest: str | Path,
        layout: VisDroneLayout,
        mode: Mode = "lowres",
        ram_cache: bool = False,
        box_filter: BoxFilter | None = None,
        crop_size: int = 256,
        max_images: int | None = None,
    ) -> None:
        """
        Args:
            manifest: `<manifest_dir>/<split>/<subset>.txt`. Item order is the file order.
            layout: Where the raw images, stats CSV and 512-px copies live.
            mode: See the class docstring.
            ram_cache: Load every 512-px copy as uint8 at startup (a few GB for the day set) and
                linearize on the fly. `lowres` only.
            box_filter: Rules for `eval` boxes (defaults: drop categories 0 and 11, score 0, and
                boxes under 16 px).
            crop_size: Crop side for `native_crops`.
            max_images: Keep only the first N manifest entries (debug runs).

        Raises:
            ValueError: Unknown mode, `ram_cache` outside `lowres`, or a manifest ID missing
                from the stats CSV.
        """
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if ram_cache and mode != "lowres":
            raise ValueError("ram_cache holds the 512-px copies, so it needs mode='lowres'")
        self.layout = layout
        self.mode = mode
        self.box_filter = box_filter or BoxFilter()
        self.crop_size = crop_size
        self.image_ids = read_manifest(manifest)[:max_images]
        stats = read_stats_csv(layout.stats_csv)
        missing = [image_id for image_id in self.image_ids if image_id not in stats]
        if missing:
            raise ValueError(
                f"{len(missing)} IDs in {manifest} are not in {layout.stats_csv}: {missing[:5]}"
            )
        self._rows = [stats[image_id] for image_id in self.image_ids]
        self._cache: list[torch.Tensor] | None = None
        if ram_cache:
            with ThreadPoolExecutor(min(8, os.cpu_count() or 1)) as pool:  # PIL decodes off-GIL
                self._cache = list(pool.map(self._read_lowres, range(len(self))))

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        native_w, native_h = row.native_width, row.native_height
        item = {
            "image_id": row.image_id,
            "native_size": torch.tensor([native_w, native_h], dtype=torch.int64),
        }
        if self.mode == "lowres":
            srgb8 = self._cache[idx] if self._cache is not None else self._read_lowres(idx)
            item["scale"] = max(srgb8.shape[1:]) / max(native_w, native_h)
            return {"day": srgb8_to_linear(srgb8), **item}

        path = self.layout.raw_image(row.split, row.image_id)
        srgb8 = read_srgb8(path)
        if tuple(srgb8.shape[1:]) != (native_h, native_w):
            raise RuntimeError(
                f"{path} is {srgb8.shape[2]}x{srgb8.shape[1]}, the stats CSV says "
                f"{native_w}x{native_h}: rerun `prepare_visdrone.py stats`"
            )
        item["scale"] = 1.0
        if self.mode == "native_crops":
            return {"day": srgb8_to_linear(self._random_crop(srgb8, path)), **item}
        annotations = read_annotations(self.layout.raw_annotation(row.split, row.image_id))
        boxes, categories = self.box_filter(annotations, (1.0, 1.0), (native_w, native_h))
        return {"day": srgb8_to_linear(srgb8), **item, "boxes": boxes, "categories": categories}

    def _read_lowres(self, idx: int) -> torch.Tensor:
        row = self._rows[idx]
        return read_srgb8(self.layout.lowres_image(row.split, row.image_id))

    def _random_crop(self, img: torch.Tensor, path: Path) -> torch.Tensor:
        _, height, width = img.shape
        size = self.crop_size
        if height < size or width < size:
            raise ValueError(f"{path} ({width}x{height}) is smaller than crop_size {size}")
        top = int(torch.randint(0, height - size + 1, ()))
        left = int(torch.randint(0, width - size + 1, ()))
        return img[:, top : top + size, left : left + size]
