"""Aligned thermal / RGB pairs on disk, and the dataset that serves them.

`write_aligned_pair` (the `write` stage of scripts/prepare_dronevehicle.py) stores each accepted
pair once, under `dronevehicle.aligned_dir/<split>/`, so training never warps on the fly:

    <id>_rgb.png      8-bit sRGB, the lossless crop of the decoded RGB JPEG (never resampled)
    <id>_thermal.png  8-bit L, the thermal warped onto the RGB frame (dawn.data.align.warp_thermal)
    <id>_valid.png    L, 0 / 255: 255 where the warped thermal is real data
    <id>_moving.png   L, 0 / 255: 255 on vehicles that moved between the two exposures

All four are 640x512 (`DroneVehicleLayout.crop_size`). The 8-bit thermal is deliberate: the
requantization noise (about 59 dB) is far below the JPEG noise already in the source.

`AlignedPairs` serves them as tensors. Conventions (docs/conventions.md sections 1 and 10):

- `thermal` is `code / 255`, float32 in [0, 1], `(1, H, W)`. Thermal values are codes, not sRGB:
  they never go through `srgb_to_linear`.
- `rgb` is decoded with `srgb8_to_linear`: linear float32 in [0, 1], `(3, H, W)`.
- `valid` is excluded from **every** loss, `moving` from the **pixel** losses.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dawn.data import align
from dawn.data import dronevehicle as dv
from dawn.data.color import srgb8_to_linear

logger = logging.getLogger(__name__)


# --- manifests -----------------------------------------------------------------------------------


def write_manifest(path: str | Path, image_ids: Iterable[str]) -> None:
    """One image ID per line, sorted and unique, LF line endings, UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(image_ids))
    path.write_bytes(("\n".join(ids) + "\n").encode("utf-8") if ids else b"")


def read_manifest(path: str | Path) -> list[str]:
    """The IDs in a manifest, in file order (blank lines ignored)."""
    return [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]


# --- writing the aligned PNGs --------------------------------------------------------------------


def _save_png(array: np.ndarray, path: Path) -> None:
    """Write through a temporary name so an interrupted run leaves no half-written PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.png")
    Image.fromarray(array).save(partial, format="PNG")
    os.replace(partial, path)


def write_aligned_pair(
    layout: dv.DroneVehicleLayout, cfg: align.AlignConfig, row: align.AlignRow, force: bool = False
) -> bool:
    """Write the four PNGs of one accepted pair, using the logged transform (no refit).

    The thermal is warped onto the RGB frame with `align.warp_thermal` (bilinear inverse mapping;
    not a downsample, so the area-averaging rule does not apply) and requantized to 8 bits. The
    moving-vehicle mask is rebuilt from the boxes (matching is deterministic) and the logged
    transform. An existing set is kept unless `force`.

    Returns:
        True when files were written, False when the pair was already complete.

    Raises:
        ValueError: `row` is not an accepted pair.
    """
    if not row.accepted:
        raise ValueError(f"only accepted pairs can be written: {row.image_id} is {row.reason}")
    paths = {kind: layout.aligned_path(row.image_id, kind) for kind in dv.ASSET_KINDS}
    if not force and all(path.exists() for path in paths.values()):
        return False
    split, stem = dv.parse_image_id(row.image_id)
    rgb8 = dv.crop_border(
        dv.read_rgb8(layout.rgb_image(split, stem)), layout.border, layout.frame_size
    )
    thermal8 = dv.crop_border(
        dv.read_thermal8(layout.thermal_image(split, stem)), layout.border, layout.frame_size
    )
    transform = align.transform_of(row)
    warped, valid = align.warp_thermal(thermal8.float() / 255, transform, layout.crop_size)
    thermal_codes = (warped[0].clamp(0, 1) * 255).round().to(torch.uint8)

    thermal_ann = dv.parse_annotation(
        layout.thermal_label(split, stem), layout.class_map, layout.border
    )
    rgb_ann = dv.parse_annotation(layout.rgb_label(split, stem), layout.class_map, layout.border)
    matches = align.match_boxes(thermal_ann.boxes, rgb_ann.boxes, cfg)
    moving, _ = align.moving_mask(
        thermal_ann.boxes, rgb_ann.boxes, matches, transform, layout.crop_size, cfg
    )

    _save_png(rgb8.permute(1, 2, 0).numpy(), paths["rgb"])
    _save_png(thermal_codes.numpy(), paths["thermal"])
    _save_png(valid[0].numpy().astype(np.uint8) * 255, paths["valid"])
    _save_png(moving[0].numpy().astype(np.uint8) * 255, paths["moving"])
    return True


# --- augmentation --------------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentConfig:
    """Training augmentation (the `augment` section of data.yaml).

    Flips are applied to all four tensors alike; the contrast curve to thermal only (the thermal
    camera rescales its gain frame by frame, the RGB does not). 90 degree rotations are not offered:
    they would turn 640x512 into 512x640, which breaks batching and the velocity net's learned
    40x32 position embedding.
    """

    hflip: float = 0.5  # probability
    vflip: float = 0.5  # probability; hflip and vflip together are a 180 degree rotation
    rot90: bool = False
    contrast_prob: float = 0.8  # probability that a thermal curve is applied at all
    black_point: tuple[float, float] = (0.0, 0.10)  # uniform range of the input mapped to 0
    white_point: tuple[float, float] = (0.90, 1.0)  # uniform range of the input mapped to 1
    gamma: tuple[float, float] = (0.7, 1.4)  # log-uniform range of the exponent

    def __post_init__(self) -> None:
        if self.rot90:
            raise ValueError(
                "rot90 is not supported: a 90 degree rotation turns 640x512 into 512x640"
            )

    @classmethod
    def from_config(cls, cfg: Mapping) -> AugmentConfig:
        """Build from the `augment` section of data.yaml."""
        contrast = cfg.get("thermal_contrast") or {}
        kwargs = {
            "hflip": cfg.get("hflip"),
            "vflip": cfg.get("vflip"),
            "rot90": cfg.get("rot90"),
            "contrast_prob": contrast.get("prob"),
            "black_point": tuple(contrast["black_point"]) if "black_point" in contrast else None,
            "white_point": tuple(contrast["white_point"]) if "white_point" in contrast else None,
            "gamma": tuple(contrast["gamma"]) if "gamma" in contrast else None,
        }
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


def _uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    return lo + (hi - lo) * torch.rand(1, generator=generator).item()


def sample_contrast_curve(
    aug: AugmentConfig, generator: torch.Generator | None
) -> tuple[float, float, float] | None:
    """Draw `(black, white, gamma)` for one image, or None when no curve is applied this time."""
    if aug.contrast_prob <= 0 or torch.rand(1, generator=generator).item() >= aug.contrast_prob:
        return None
    black = _uniform(*aug.black_point, generator)
    white = _uniform(*aug.white_point, generator)
    gamma = math.exp(_uniform(math.log(aug.gamma[0]), math.log(aug.gamma[1]), generator))
    return black, white, gamma


def apply_contrast_curve(x: torch.Tensor, black: float, white: float, gamma: float) -> torch.Tensor:
    """`((x - black) / (white - black)) ** gamma`, clipped to [0, 1]: monotone non-decreasing."""
    return ((x - black) / (white - black)).clamp(0.0, 1.0) ** gamma


# --- the dataset ---------------------------------------------------------------------------------


def _load_png(path: Path, image_id: str) -> np.ndarray:
    try:
        with Image.open(path) as img:
            return np.asarray(img).copy()
    except FileNotFoundError:
        raise FileNotFoundError(f"{image_id}: missing {path}") from None


class AlignedPairs(Dataset):
    """Aligned day pairs of one manifest (`train_pairs`, `val_pairs` or `test_pairs`).

    Each item is a dict:

        thermal   (1, 512, 640) float32  code / 255 of the thermal warped onto the RGB frame
        rgb       (3, 512, 640) float32  linear RGB (`srgb8_to_linear` of the stored 8-bit sRGB)
        valid     (1, 512, 640) bool     True where the warped thermal is real data
        moving    (1, 512, 640) bool     True on vehicles that moved between the two exposures
        image_id  str                    `<split>_<stem>`

    With `train=True`, a random horizontal and vertical flip is applied to all four tensors alike,
    then a random monotone contrast curve to thermal only. Validation and test (`train=False`) are
    deterministic and never augmented.

    Augmentation draws come from `generator` when one is passed, else from torch's default RNG,
    which a DataLoader seeds per worker. The global RNG is never reseeded here.

    Args:
        layout: Where the aligned PNGs and manifests live.
        manifest: Manifest name, read from `<manifest_dir>/<manifest>.txt`.
        train: Augment.
        augment: Augmentation settings (defaults to `AugmentConfig()` when `train`).
        max_pairs: Keep only the first N manifest entries; defaults to `dronevehicle.max_pairs`.
        generator: Explicit `torch.Generator` for the augmentation draws.
    """

    def __init__(
        self,
        layout: dv.DroneVehicleLayout,
        manifest: str,
        train: bool = False,
        augment: AugmentConfig | None = None,
        max_pairs: int | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        path = layout.manifest(manifest)
        if not path.exists():
            raise FileNotFoundError(f"manifest {manifest!r} not found: {path}")
        self.layout = layout
        self.train = train
        self.augment = augment or AugmentConfig()
        self.generator = generator
        limit = max_pairs if max_pairs is not None else layout.max_pairs
        self.image_ids = read_manifest(path)[:limit]

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        manifest: str,
        train: bool = False,
        generator: torch.Generator | None = None,
    ) -> AlignedPairs:
        """Build from the parsed data.yaml (`dronevehicle` and `augment` sections)."""
        return cls(
            dv.DroneVehicleLayout.from_config(cfg["dronevehicle"]),
            manifest,
            train=train,
            augment=AugmentConfig.from_config(cfg.get("augment") or {}),
            generator=generator,
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> dict:
        image_id = self.image_ids[index]
        path = lambda kind: self.layout.aligned_path(image_id, kind)  # noqa: E731
        rgb = torch.from_numpy(_load_png(path("rgb"), image_id)).permute(2, 0, 1)
        thermal = torch.from_numpy(_load_png(path("thermal"), image_id)).unsqueeze(0)
        valid = torch.from_numpy(_load_png(path("valid"), image_id) > 127).unsqueeze(0)
        moving = torch.from_numpy(_load_png(path("moving"), image_id) > 127).unsqueeze(0)
        item = {
            "thermal": thermal.float() / 255,
            "rgb": srgb8_to_linear(rgb.contiguous()),
            "valid": valid,
            "moving": moving,
            "image_id": image_id,
        }
        return self._augmented(item) if self.train else item

    def _augmented(self, item: dict) -> dict:
        aug, generator = self.augment, self.generator
        dims = []
        if aug.hflip > 0 and torch.rand(1, generator=generator).item() < aug.hflip:
            dims.append(-1)
        if aug.vflip > 0 and torch.rand(1, generator=generator).item() < aug.vflip:
            dims.append(-2)
        if dims:
            for key in ("thermal", "rgb", "valid", "moving"):
                item[key] = torch.flip(item[key], dims)
        curve = sample_contrast_curve(aug, generator)
        if curve is not None:
            item["thermal"] = apply_contrast_curve(item["thermal"], *curve)
        return item
