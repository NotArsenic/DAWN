"""DroneVehicle layout, file readers, border validation, and annotation parsing."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SPLITS = ("train", "val", "test")
BORDER = 100
FRAME_SIZE = (712, 840)
ASSET_KINDS = ("rgb", "thermal", "valid", "moving", "vehicles")
_WHITE = 250


class FrameError(ValueError):
    """Raised when an image or annotation frame does not match expected dimensions."""


class ThermalChannelError(ValueError):
    """Raised when a multi-channel thermal image contains non-identical channels."""


def image_id(split: str, stem: str) -> str:
    """Generate a global image identifier from split and stem."""
    return f"{split}_{stem}"


def parse_image_id(image_id_: str) -> tuple[str, str]:
    """Parse a global image identifier into split and stem."""
    split, sep, stem = image_id_.partition("_")
    if not sep or split not in SPLITS or not stem:
        raise ValueError(
            f"not an image ID (expected '<split>_<stem>', split in {SPLITS}): {image_id_!r}"
        )
    return split, stem


@dataclass(frozen=True)
class BorderCheck:
    """Thresholds for border validation."""

    min_value: int = 200
    max_ring_white: float = 0.5


@dataclass(frozen=True)
class DroneVehicleLayout:
    """Directory layout and configuration for the DroneVehicle dataset."""

    root: Path
    aligned_dir: Path
    stats_csv: Path
    manifest_dir: Path
    outputs_dir: Path
    frame_size: tuple[int, int] = FRAME_SIZE
    border: int = BORDER
    border_check: BorderCheck = field(default_factory=BorderCheck)
    class_map: Mapping[str, str | None] = field(default_factory=dict)
    max_pairs: int | None = None

    @classmethod
    def from_config(cls, cfg: Mapping) -> DroneVehicleLayout:
        """Create a layout instance from configuration."""
        check = cfg.get("border_check") or {}
        return cls(
            root=Path(cfg["root"]),
            aligned_dir=Path(cfg["aligned_dir"]),
            stats_csv=Path(cfg["stats_csv"]),
            manifest_dir=Path(cfg["manifest_dir"]),
            outputs_dir=Path(cfg["outputs_dir"]),
            frame_size=tuple(cfg.get("frame_size", FRAME_SIZE)),
            border=int(cfg.get("border", BORDER)),
            border_check=BorderCheck(
                min_value=int(check.get("min_value", BorderCheck.min_value)),
                max_ring_white=float(check.get("max_ring_white", BorderCheck.max_ring_white)),
            ),
            class_map={str(k).strip().lower(): v for k, v in (cfg.get("class_map") or {}).items()},
            max_pairs=cfg.get("max_pairs"),
        )

    @property
    def crop_size(self) -> tuple[int, int]:
        """Return cropped image dimensions (height, width)."""
        return (self.frame_size[0] - 2 * self.border, self.frame_size[1] - 2 * self.border)

    def rgb_image(self, split: str, stem: str) -> Path:
        return self.root / split / f"{split}img" / f"{stem}.jpg"

    def thermal_image(self, split: str, stem: str) -> Path:
        return self.root / split / f"{split}imgr" / f"{stem}.jpg"

    def rgb_label(self, split: str, stem: str) -> Path:
        return self.root / split / f"{split}label" / f"{stem}.xml"

    def thermal_label(self, split: str, stem: str) -> Path:
        return self.root / split / f"{split}labelr" / f"{stem}.xml"

    def aligned_path(self, image_id_: str, kind: str) -> Path:
        """Return the output path for an aligned asset."""
        if kind not in ASSET_KINDS:
            raise ValueError(f"kind must be one of {ASSET_KINDS}, got {kind!r}")
        split, _ = parse_image_id(image_id_)
        return self.aligned_dir / split / f"{image_id_}_{kind}.png"

    def manifest(self, name: str) -> Path:
        """Return the path to a split manifest file."""
        return self.manifest_dir / f"{name}.txt"

    def log(self, name: str) -> Path:
        """Return the path to a CSV log file."""
        return self.manifest_dir / "logs" / f"{name}.csv"


@dataclass(frozen=True)
class PairListing:
    """Listing of image stems partitioned by modality presence."""

    split: str
    paired: list[str]
    rgb_only: list[str]
    thermal_only: list[str]


def list_pairs(layout: DroneVehicleLayout, split: str) -> PairListing:
    """List and categorize image stems present in a split."""
    rgb = {p.stem for p in layout.rgb_image(split, "x").parent.glob("*.jpg")}
    thermal = {p.stem for p in layout.thermal_image(split, "x").parent.glob("*.jpg")}
    return PairListing(
        split=split,
        paired=sorted(rgb & thermal),
        rgb_only=sorted(rgb - thermal),
        thermal_only=sorted(thermal - rgb),
    )


def read_rgb8(path: str | Path) -> torch.Tensor:
    """Load an RGB image as a (3, H, W) uint8 tensor."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def read_thermal8(path: str | Path) -> torch.Tensor:
    """Load a thermal image as a (1, H, W) uint8 tensor."""
    with Image.open(path) as img:
        array = np.asarray(img)
    if array.ndim == 3:
        if array.shape[2] != 3 or not (
            (array[..., 0] == array[..., 1]).all() and (array[..., 1] == array[..., 2]).all()
        ):
            raise ThermalChannelError(f"{path}: thermal channels are not identical")
        array = array[..., 0]
    return torch.from_numpy(array.copy()).unsqueeze(0).contiguous()


def crop_border(
    img: torch.Tensor, border: int = BORDER, frame_size: tuple[int, int] = FRAME_SIZE
) -> torch.Tensor:
    """Crop white borders from an image tensor."""
    if tuple(img.shape[-2:]) != tuple(frame_size):
        raise FrameError(
            f"expected a {frame_size[1]}x{frame_size[0]} frame, got {img.shape[-1]}x{img.shape[-2]}"
        )
    return img[..., border : frame_size[0] - border, border : frame_size[1] - border]


def check_border(
    img: torch.Tensor, check: BorderCheck | None = None, border: int = BORDER
) -> str | None:
    """Validate white border boundaries, returning an error label if invalid."""
    check = check or BorderCheck()
    _, height, width = img.shape
    strips = (
        img[:, :border],
        img[:, height - border :],
        img[:, :, :border],
        img[:, :, width - border :],
    )
    if min(int(s.min()) for s in strips) < check.min_value:
        return "dark"
    inner = img[:, border : height - border, border : width - border]
    ring = torch.cat(
        [
            inner[:, 0].flatten(1),
            inner[:, -1].flatten(1),
            inner[:, :, 0].flatten(1),
            inner[:, :, -1].flatten(1),
        ],
        dim=1,
    )
    if (ring >= _WHITE).all(dim=0).float().mean().item() > check.max_ring_white:
        return "wide"
    return None


@dataclass(frozen=True)
class Box:
    """Bounding polygon and category for an annotated vehicle."""

    category: str
    polygon: tuple[tuple[float, float], ...]
    kind: str

    @property
    def vertices(self) -> np.ndarray:
        """Return polygon vertices as an (N, 2) float64 array."""
        return np.asarray(self.polygon, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        """Return the centroid coordinates (x, y) of the polygon."""
        return self.vertices.mean(axis=0)


@dataclass(frozen=True)
class Annotation:
    """Parsed XML annotation metadata and bounding boxes."""

    folder: str
    filename: str
    boxes: tuple[Box, ...]
    num_point_only: int
    num_dropped: int
    unknown_classes: tuple[str, ...]


def parse_annotation(
    path: str | Path, class_map: Mapping[str, str | None], border: int = BORDER
) -> Annotation:
    """Parse a VOC-style XML annotation file into bounding boxes."""
    root = ET.parse(path).getroot()
    boxes, num_point_only, num_dropped, unknown = [], 0, 0, []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip().lower()
        polygon, bndbox = obj.find("polygon"), obj.find("bndbox")
        if polygon is None and bndbox is None:
            if obj.find("point") is None:
                raise ValueError(
                    f"{path}: object {name!r} has no polygon, bndbox or point geometry"
                )
            num_point_only += 1
            continue
        if name not in class_map:
            if name not in unknown:
                unknown.append(name)
            continue
        category = class_map[name]
        if category is None:
            num_dropped += 1
            continue
        if polygon is not None:
            vertices = tuple(
                (
                    float(polygon.findtext(f"x{i}")) - border,
                    float(polygon.findtext(f"y{i}")) - border,
                )
                for i in range(1, 5)
            )
            kind = "polygon"
        else:
            x0, x1 = (float(bndbox.findtext(k)) - border for k in ("xmin", "xmax"))
            y0, y1 = (float(bndbox.findtext(k)) - border for k in ("ymin", "ymax"))
            vertices = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            kind = "bndbox"
        boxes.append(Box(category, vertices, kind))
    return Annotation(
        folder=(root.findtext("folder") or "").strip(),
        filename=(root.findtext("filename") or "").strip(),
        boxes=tuple(boxes),
        num_point_only=num_point_only,
        num_dropped=num_dropped,
        unknown_classes=tuple(unknown),
    )
