"""Dataset and export utilities for aligned thermal and RGB image pairs."""

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


def write_manifest(path: str | Path, image_ids: Iterable[str]) -> None:
    """Write sorted unique image identifiers to a manifest file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(image_ids))
    path.write_bytes(("\n".join(ids) + "\n").encode("utf-8") if ids else b"")


def read_manifest(path: str | Path) -> list[str]:
    """Read image identifiers from a manifest file."""
    return [
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _save_png(array: np.ndarray, path: Path) -> None:
    """Save an array as a PNG file via an atomic temporary write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.png")
    Image.fromarray(array).save(partial, format="PNG")
    os.replace(partial, path)


@dataclass(frozen=True)
class VehicleMaskConfig:
    """Configuration parameters for vehicle mask generation."""

    dilate_px: float = 3.0
    extend_px: float = 17.4
    window_margin_px: int = 12
    search_px: int = 6
    max_residual_px: int = 2
    min_prominence: float = 0.02
    min_window_px: int = 32
    max_frame_fraction: float = 0.30

    @classmethod
    def from_config(cls, cfg: Mapping) -> VehicleMaskConfig:
        """Create a VehicleMaskConfig instance from configuration."""
        static = cfg.get("static_test") or {}
        flat = {
            "dilate_px": cfg.get("dilate_px"),
            "extend_px": cfg.get("extend_px"),
            "max_frame_fraction": cfg.get("max_frame_fraction"),
            **{k: static.get(k) for k in ("window_margin_px", "search_px", "max_residual_px")},
            **{k: static.get(k) for k in ("min_prominence", "min_window_px")},
        }
        return cls(**{k: v for k, v in flat.items() if v is not None})

    @property
    def params(self) -> align.VehicleMaskParams:
        return align.VehicleMaskParams(
            dilate_px=self.dilate_px,
            extend_px=self.extend_px,
            window_margin_px=self.window_margin_px,
            search_px=self.search_px,
            max_residual_px=self.max_residual_px,
            min_prominence=self.min_prominence,
            min_window_px=self.min_window_px,
        )


@dataclass(frozen=True)
class VehicleResult:
    """Summary of generated vehicle mask metrics."""

    fraction: float
    branches: align.VehicleBranches


_BASE_KINDS = ("rgb", "thermal", "valid", "moving")


def write_aligned_pair(
    layout: dv.DroneVehicleLayout,
    cfg: align.AlignConfig,
    row: align.AlignRow,
    force: bool = False,
) -> bool:
    """Generate and write aligned image and mask files for an accepted pair."""
    if not row.accepted:
        raise ValueError(f"only accepted pairs can be written: {row.image_id} is {row.reason}")
    paths = {kind: layout.aligned_path(row.image_id, kind) for kind in _BASE_KINDS}
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
    if row.model == "edge_translation":
        moving = torch.zeros(1, *layout.crop_size, dtype=torch.bool)
    else:
        matches = align.match_boxes(thermal_ann.boxes, rgb_ann.boxes, cfg)
        moving, _ = align.moving_mask(
            thermal_ann.boxes, rgb_ann.boxes, matches, transform, layout.crop_size, cfg
        )
    _save_png(rgb8.permute(1, 2, 0).numpy(), paths["rgb"])
    _save_png(thermal_codes.numpy(), paths["thermal"])
    _save_png(valid[0].numpy().astype(np.uint8) * 255, paths["valid"])
    _save_png(moving[0].numpy().astype(np.uint8) * 255, paths["moving"])
    return True


def write_vehicles(
    layout: dv.DroneVehicleLayout,
    cfg: align.AlignConfig,
    masks: VehicleMaskConfig,
    row: align.AlignRow,
) -> VehicleResult:
    """Generate and save vehicle mask for an aligned pair."""
    split, stem = dv.parse_image_id(row.image_id)
    thermal8 = dv.crop_border(
        dv.read_thermal8(layout.thermal_image(split, stem)), layout.border, layout.frame_size
    )
    rgb8 = dv.crop_border(
        dv.read_rgb8(layout.rgb_image(split, stem)), layout.border, layout.frame_size
    )
    transform = align.transform_of(row)
    warped, valid = align.warp_thermal(thermal8.float() / 255, transform, layout.crop_size)
    sigma = cfg.grad_blur_sigma
    images = (
        align.gradient_magnitude(align.rgb_edge_luma(srgb8_to_linear(rgb8)), sigma),
        align.gradient_magnitude(warped[0].numpy(), sigma),
        valid[0].numpy(),
    )
    thermal_ann = dv.parse_annotation(
        layout.thermal_label(split, stem), layout.class_map, layout.border
    )
    rgb_ann = dv.parse_annotation(layout.rgb_label(split, stem), layout.class_map, layout.border)
    mask, branches = align.vehicle_mask(
        rgb_ann.boxes, thermal_ann.boxes, transform, layout.crop_size, masks.params, cfg, images
    )
    _save_png(mask[0].numpy().astype(np.uint8) * 255, layout.aligned_path(row.image_id, "vehicles"))
    return VehicleResult(fraction=align.vehicle_fraction(mask), branches=branches)


@dataclass(frozen=True)
class AugmentConfig:
    """Data augmentation configuration for training pairs."""

    hflip: float = 0.5
    vflip: float = 0.5
    rot90: bool = False
    contrast_prob: float = 0.8
    black_point: tuple[float, float] = (0.0, 0.10)
    white_point: tuple[float, float] = (0.90, 1.0)
    gamma: tuple[float, float] = (0.7, 1.4)

    def __post_init__(self) -> None:
        if self.rot90:
            raise ValueError(
                "rot90 is not supported: a 90 degree rotation turns 640x512 into 512x640"
            )

    @classmethod
    def from_config(cls, cfg: Mapping) -> AugmentConfig:
        """Create an AugmentConfig instance from configuration."""
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
    """Sample random thermal contrast adjustment parameters."""
    if aug.contrast_prob <= 0 or torch.rand(1, generator=generator).item() >= aug.contrast_prob:
        return None
    black = _uniform(*aug.black_point, generator)
    white = _uniform(*aug.white_point, generator)
    gamma = math.exp(_uniform(math.log(aug.gamma[0]), math.log(aug.gamma[1]), generator))
    return black, white, gamma


def apply_contrast_curve(x: torch.Tensor, black: float, white: float, gamma: float) -> torch.Tensor:
    """Apply contrast curve adjustments to a tensor."""
    return ((x - black) / (white - black)).clamp(0.0, 1.0) ** gamma


def _load_png(path: Path, image_id: str) -> np.ndarray:
    try:
        with Image.open(path) as img:
            return np.asarray(img).copy()
    except FileNotFoundError:
        raise FileNotFoundError(f"{image_id}: missing {path}") from None


class AlignedPairs(Dataset):
    """PyTorch Dataset serving aligned thermal and RGB image pairs."""

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
        fallback_path = layout.manifest("fallback_pairs")
        self.fallback_ids = set(read_manifest(fallback_path)) if fallback_path.exists() else set()

    @classmethod
    def from_config(
        cls,
        cfg: Mapping,
        manifest: str,
        train: bool = False,
        generator: torch.Generator | None = None,
    ) -> AlignedPairs:
        """Create an AlignedPairs dataset from configuration."""
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
        path = lambda kind: self.layout.aligned_path(image_id, kind)
        rgb = torch.from_numpy(_load_png(path("rgb"), image_id)).permute(2, 0, 1)
        thermal = torch.from_numpy(_load_png(path("thermal"), image_id)).unsqueeze(0)
        valid = torch.from_numpy(_load_png(path("valid"), image_id) > 127).unsqueeze(0)
        moving = torch.from_numpy(_load_png(path("moving"), image_id) > 127).unsqueeze(0)
        vehicles = torch.from_numpy(_load_png(path("vehicles"), image_id) > 127).unsqueeze(0)
        item = {
            "thermal": thermal.float() / 255,
            "rgb": srgb8_to_linear(rgb.contiguous()),
            "valid": valid,
            "moving": moving,
            "vehicles": vehicles,
            "image_id": image_id,
            "fallback": image_id in self.fallback_ids,
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
            for key in ("thermal", "rgb", "valid", "moving", "vehicles"):
                item[key] = torch.flip(item[key], dims)
        curve = sample_contrast_curve(aug, generator)
        if curve is not None:
            item["thermal"] = apply_contrast_curve(item["thermal"], *curve)
        return item
