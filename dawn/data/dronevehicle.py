"""DroneVehicle: raw on-disk layout, image IDs, raw readers, border check and XML parsing.

Raw layout (paths from the `dronevehicle` section of configs/data.yaml, see `DroneVehicleLayout`):

    <root>/<split>/<split>img/<stem>.jpg      RGB
    <root>/<split>/<split>imgr/<stem>.jpg     thermal
    <root>/<split>/<split>label/<stem>.xml    RGB boxes
    <root>/<split>/<split>labelr/<stem>.xml   thermal boxes

The file stem pairs the two modalities. Stems repeat across splits (`train 00001` is not
`val 00001`), so the global image ID is `f"{split}_{stem}"` (`image_id`), used everywhere:
manifests, logs, file names and seeds.

Every frame is 840x712 with a clean 100-px white border on both modalities; the content is the
640x512 window `[100:612, 100:740]` (`crop_border`). Coordinates:

- Images are `(C, H, W)` tensors, `(3, ...)` for RGB and `(1, ...)` for thermal.
- Box coordinates are pixel-index coordinates, x = column and y = row, so the centre of the
  pixel in row r and column c is (c, r). The XML is in the 840x712 frame; `parse_annotation`
  subtracts the border, so boxes are in the cropped 640x512 frame like the cropped images.

Thermal is not sRGB: its 8-bit codes mean `code / 255` and never go through `srgb_to_linear`
(docs/conventions.md section 1). This module only reads raw 8-bit data; no color conversion
happens here.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SPLITS = ("train", "val", "test")
BORDER = 100  # px of white border on every side of a raw frame
FRAME_SIZE = (712, 840)  # raw (height, width)
ASSET_KINDS = ("rgb", "thermal", "valid", "moving", "vehicles")  # PNGs written per pair
_WHITE = 250  # an 8-bit value this high counts as white in the ring check


class FrameError(ValueError):
    """An image or label does not have the 840x712 frame this dataset is documented to have."""


class ThermalChannelError(ValueError):
    """A thermal JPEG stored as 3 channels whose channels are not identical."""


# --- ids -----------------------------------------------------------------------------------------


def image_id(split: str, stem: str) -> str:
    """The global ID of a pair: `f"{split}_{stem}"` (stems repeat across splits)."""
    return f"{split}_{stem}"


def parse_image_id(image_id_: str) -> tuple[str, str]:
    """`(split, stem)` of a global image ID.

    Raises:
        ValueError: The ID does not start with one of `SPLITS` and an underscore.
    """
    split, sep, stem = image_id_.partition("_")
    if not sep or split not in SPLITS or not stem:
        raise ValueError(
            f"not an image ID (expected '<split>_<stem>', split in {SPLITS}): {image_id_!r}"
        )
    return split, stem


# --- layout --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BorderCheck:
    """Bounds of `check_border`. Both come from `dronevehicle.border_check` in data.yaml."""

    min_value: int = 200  # every border pixel (8-bit) must be at least this
    max_ring_white: float = 0.5  # share of white pixels allowed in the content's outer 1-px ring


@dataclass(frozen=True)
class DroneVehicleLayout:
    """Where every DroneVehicle file lives (module docstring), plus the reader settings."""

    root: Path  # raw download
    aligned_dir: Path
    stats_csv: Path
    manifest_dir: Path
    outputs_dir: Path
    frame_size: tuple[int, int] = FRAME_SIZE  # raw (height, width), border included
    border: int = BORDER
    border_check: BorderCheck = field(default_factory=BorderCheck)
    class_map: Mapping[str, str | None] = field(default_factory=dict)  # raw name -> name / None
    max_pairs: int | None = None

    @classmethod
    def from_config(cls, cfg: Mapping) -> DroneVehicleLayout:
        """Build from the `dronevehicle` section of data.yaml."""
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
        """Cropped `(height, width)`: 512 x 640 for the default frame and border."""
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
        """`<aligned_dir>/<split>/<id>_<kind>.png` for `kind` in `ASSET_KINDS`."""
        if kind not in ASSET_KINDS:
            raise ValueError(f"kind must be one of {ASSET_KINDS}, got {kind!r}")
        split, _ = parse_image_id(image_id_)
        return self.aligned_dir / split / f"{image_id_}_{kind}.png"

    def manifest(self, name: str) -> Path:
        """`<manifest_dir>/<name>.txt`, e.g. `train_pairs`."""
        return self.manifest_dir / f"{name}.txt"

    def log(self, name: str) -> Path:
        """`<manifest_dir>/logs/<name>.csv`."""
        return self.manifest_dir / "logs" / f"{name}.csv"


@dataclass(frozen=True)
class PairListing:
    """The image stems of one split, by which modalities exist."""

    split: str
    paired: list[str]  # in both image folders
    rgb_only: list[str]  # RGB image but no thermal image
    thermal_only: list[str]  # thermal image but no RGB image


def list_pairs(layout: DroneVehicleLayout, split: str) -> PairListing:
    """Sorted stems of `split`, from the two image folders. Reports stems missing a modality."""
    rgb = {p.stem for p in layout.rgb_image(split, "x").parent.glob("*.jpg")}
    thermal = {p.stem for p in layout.thermal_image(split, "x").parent.glob("*.jpg")}
    return PairListing(
        split=split,
        paired=sorted(rgb & thermal),
        rgb_only=sorted(rgb - thermal),
        thermal_only=sorted(thermal - rgb),
    )


# --- raw readers ---------------------------------------------------------------------------------


def read_rgb8(path: str | Path) -> torch.Tensor:
    """Decode an RGB JPEG to a `(3, H, W)` uint8 tensor of sRGB codes (the whole raw frame)."""
    with Image.open(path) as img:
        array = np.asarray(img.convert("RGB"))
    return torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


def read_thermal8(path: str | Path) -> torch.Tensor:
    """Decode a thermal JPEG to a `(1, H, W)` uint8 tensor of raw codes (the whole raw frame).

    The files are stored either as one channel or as three identical channels. One channel is
    taken, never a luma mix, after checking that the three really are identical.

    Raises:
        ThermalChannelError: A 3-channel file whose channels differ.
    """
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
    """The content window of a raw frame: `(..., H, W)` -> `(..., H - 2b, W - 2b)`.

    Raises:
        FrameError: `img` is not `frame_size` = (height, width).
    """
    if tuple(img.shape[-2:]) != tuple(frame_size):
        raise FrameError(
            f"expected a {frame_size[1]}x{frame_size[0]} frame, got {img.shape[-1]}x{img.shape[-2]}"
        )
    return img[..., border : frame_size[0] - border, border : frame_size[1] - border]


def check_border(
    img: torch.Tensor, check: BorderCheck | None = None, border: int = BORDER
) -> str | None:
    """Check the white border of a raw `(C, H, W)` uint8 frame.

    Returns:
        None when the border is clean, `"dark"` when a border pixel is below `check.min_value`
        (content or a defect inside the border), `"wide"` when the content's outer 1-px ring is
        mostly white (a border wider than `border`, which the crop would leave in the image).
    """
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


# --- XML -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """One vehicle box: normalized class, 4 polygon vertices and where the geometry came from.

    `polygon` is `((x1, y1), ..., (x4, y4))` in pixel-index coordinates of the **cropped** frame.
    A `bndbox` becomes the rectangle (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax).
    """

    category: str
    polygon: tuple[tuple[float, float], ...]
    kind: str  # "polygon" or "bndbox": the form the XML used

    @property
    def vertices(self) -> np.ndarray:
        """`(4, 2)` float64 array of `(x, y)`."""
        return np.asarray(self.polygon, dtype=np.float64)

    @property
    def center(self) -> np.ndarray:
        """`(2,)` mean of the four vertices, `(x, y)`."""
        return self.vertices.mean(axis=0)


@dataclass(frozen=True)
class Annotation:
    """One XML file: the `<folder>` and `<filename>` fields and the usable boxes."""

    folder: str
    filename: str
    boxes: tuple[Box, ...]
    num_point_only: int  # objects that carry only a <point>: skipped
    num_dropped: int  # objects whose class maps to null (the "*" class): dropped
    unknown_classes: tuple[str, ...]  # class names missing from the map: dropped, listed once each


def parse_annotation(
    path: str | Path, class_map: Mapping[str, str | None], border: int = BORDER
) -> Annotation:
    """Parse one VOC-like XML into boxes in the cropped frame.

    Objects carry a `<polygon>` (x1..y4, an oriented quad) or a `<bndbox>`, or only a `<point>`
    (skipped and counted). Class names are matched case-insensitively through `class_map`
    (`dronevehicle.class_map`): a null target drops the box (counted), a name that is not in the
    map is dropped and listed in `unknown_classes` so a new typo shows up in the stats rather than
    silently shrinking the data. `<size><depth>` is wrong in this dataset and ignored.

    Raises:
        ValueError: An object with no polygon, bndbox or point.
    """
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
