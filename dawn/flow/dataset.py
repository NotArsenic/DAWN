"""The dataloader-worker side of stage 1: daytime image in, night / night_clean pair out.

`VisDroneDataset` deliberately only loads images; the degradation is "a separate transform in
the dataloader workers, which ends with fixed 256x256 `night`, `night_clean`, `day` and
`noise_params`" (docs/conventions.md section 7). `NightPairs` is that transform. It runs on the
CPU in the workers; the GPU side of a step (grid fitting, the encoder, the velocity net) is in
`dawn.flow.trainer`.

`SyntheticDay` is a stand-in day source for CPU debug runs and tests: procedurally generated
images with the same contract as `VisDroneDataset`, so the whole loop can run with no dataset
on disk. It is not a substitute for VisDrone in any measurement that matters.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

from dawn.data.degrade import FLOW_SIZE, degrade_image
from dawn.data.seeding import generator_for

VAL_LEVEL = "medium"  # the fixed val set renders at medium darkness


class NightPairs(Dataset):
    """Degrade a day-image dataset into matched night / night_clean / day triples.

    Every item is a dict:
        night        (3, S, S) linear RGB, what the model sees (noise, quantization, JPEG).
        night_clean  (3, S, S) linear RGB, the same render without the noise draws: the grid
                     fitting source (docs/conventions.md section 1).
        day          (3, S, S) linear RGB target, area-downsampled to the same size.
        noise_params dict with 'a', 'b' (3,) at native resolution, 'n' and 'quant_var'.
        image_id     str, and native_size (2,) int64 as the day source reported them.

    Args:
        day_source: Anything indexable yielding the `VisDroneDataset` dict ('day', 'image_id',
            'native_size').
        degradation: The `degradation` section of configs/data.yaml.
        calib: Parsed configs/camera_calib.yaml.
        level: None for a fresh training draw every epoch, or "easy"/"medium"/"hard" for a
            fixed, seeded set. Validation uses `VAL_LEVEL`.
        size: Square output side (the flow path's 256).
    """

    def __init__(
        self,
        day_source: Sequence[Mapping] | Dataset,
        degradation: Mapping,
        calib: Mapping,
        level: str | None = None,
        size: int = FLOW_SIZE,
    ) -> None:
        super().__init__()
        self.day_source = day_source
        self.degradation = degradation
        self.calib = calib
        self.level = level
        self.size = size

    def __len__(self) -> int:
        return len(self.day_source)

    def __getitem__(self, idx: int) -> dict:
        sample = self.day_source[idx]
        native = sample.get("native_size")
        result = degrade_image(
            sample["day"],
            self.degradation,
            self.calib,
            image_id=sample["image_id"],
            level=self.level,
            native_size=None if native is None else tuple(int(v) for v in native),
            size=self.size,
        )
        return {
            "night": result.night,
            "night_clean": result.night_clean,
            "day": result.day,
            "noise_params": {
                "a": result.noise_a,
                "b": result.noise_b,
                "n": float(result.n),
                "quant_var": float(result.quant_var),
            },
            "image_id": sample["image_id"],
            "native_size": sample.get("native_size", torch.zeros(2, dtype=torch.int64)),
        }


class SyntheticDay(Dataset):
    """Procedurally generated stand-in for `VisDroneDataset`, for CPU runs with no data.

    Each image is a smooth low-frequency colour field plus a handful of bright rectangles and
    a fine texture: enough spatial and brightness variation that a 16x16x8 grid has something
    to fit, and deterministic in the index so two runs see the same "dataset".

    Args:
        num_images: Dataset length.
        size: Square side of the generated day image.
        native_size: `(W, H)` reported as the original camera frame, which sets how many native
            pixels one output pixel averages (docs/conventions.md section 6).
        seed: Base seed; image i is drawn from `seed` and i alone, and `seed` is part of the
            image ID, so two sets built with different seeds never share an ID (and therefore
            never share a degradation draw either).
    """

    def __init__(
        self,
        num_images: int = 100,
        size: int = 512,
        native_size: tuple[int, int] = (2000, 1500),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.num_images = int(num_images)
        self.size = int(size)
        self.native_size = (int(native_size[0]), int(native_size[1]))
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> dict:
        if not 0 <= idx < self.num_images:
            raise IndexError(idx)
        image_id = f"synthetic_{self.seed:05d}_{idx:06d}"
        generator = generator_for(image_id)
        size = self.size
        coords = torch.linspace(0, 1, size)
        ys, xs = torch.meshgrid(coords, coords, indexing="ij")

        day = torch.zeros(3, size, size)
        for channel in range(3):
            for _ in range(3):
                freq = 1.0 + 3.0 * torch.rand((), generator=generator).item()
                phase = 2 * math.pi * torch.rand(2, generator=generator)
                day[channel] += torch.sin(2 * math.pi * freq * xs + phase[0]) * torch.sin(
                    2 * math.pi * freq * ys + phase[1]
                )
        day = 0.5 + 0.12 * day
        # A few bright patches: without them every pixel lands in the same brightness bins.
        for _ in range(6):
            box = (torch.rand(4, generator=generator) * size).to(torch.int64)
            y0, y1 = int(box[0].min(box[1])), int(box[0].max(box[1])) + 8
            x0, x1 = int(box[2].min(box[3])), int(box[2].max(box[3])) + 8
            colour = 0.55 + 0.45 * torch.rand(3, 1, 1, generator=generator)
            day[:, y0 : min(y1, size), x0 : min(x1, size)] = colour
        day = day + 0.02 * torch.randn(3, size, size, generator=generator)  # fine texture
        return {
            "day": day.clamp(0.0, 1.0),
            "image_id": image_id,
            "native_size": torch.tensor(self.native_size, dtype=torch.int64),
            "scale": size / max(self.native_size),
        }


def build_day_source(visdrone_cfg: Mapping, loader_cfg: Mapping, split: str) -> Dataset:
    """The day-image dataset for `split`: VisDrone's day manifest, or `SyntheticDay`.

    `loader.synthetic_images` is the debug escape hatch; with it set, nothing on disk is read.
    """
    synthetic = loader_cfg.get("synthetic_images")
    if synthetic:
        offset = 0 if split == visdrone_cfg.get("train_split", "train") else 10_000
        return SyntheticDay(num_images=int(synthetic), seed=offset)

    from dawn.data.visdrone import VisDroneDataset, VisDroneLayout

    layout = VisDroneLayout.from_config(visdrone_cfg)
    return VisDroneDataset(
        layout.manifest(split, "day"),
        layout,
        mode="lowres",
        max_images=visdrone_cfg.get("max_images"),
    )


def build_night_loaders(
    cfg: Mapping,
    calib: Mapping,
    batch_size: int | None = None,
    overfit: bool = False,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """Train and validation dataloaders for stage 1.

    Training draws a fresh degradation every epoch (`level=None`); validation is pinned to
    `loader.val_level` and seeded from the image ID, so the same val images come back
    identically on every run and on any machine.

    Args:
        cfg: configs/flow.yaml merged with configs/data.yaml (`visdrone`, `degradation`,
            `loader`, `optim` and `logging` sections).
        calib: Parsed configs/camera_calib.yaml.
        batch_size: Overrides `optim.batch_size`.
        overfit: Validate on the TRAINING images, still at the fixed val darkness. The
            sanity check: can the render PSNR reach the oracle on data the model has seen?
            The val render stays seeded even though training re-draws, so the curve is not
            measuring one lucky degradation against another.
    """
    from torch.utils.data import DataLoader

    visdrone, loader_cfg = cfg["visdrone"], cfg["loader"]
    batch_size = int(cfg["optim"]["batch_size"] if batch_size is None else batch_size)
    workers = int(loader_cfg.get("num_workers", 0))
    degradation = cfg["degradation"]

    train_source = build_day_source(visdrone, loader_cfg, visdrone.get("train_split", "train"))
    train = NightPairs(train_source, degradation, calib, level=None)
    val_source = (
        train_source
        if overfit
        else build_day_source(visdrone, loader_cfg, visdrone.get("val_split", "val"))
    )
    num_val = int(cfg["logging"]["num_val_samples"])
    if len(val_source) > num_val:
        val_source = Subset(val_source, range(num_val))
    val = NightPairs(val_source, degradation, calib, level=loader_cfg.get("val_level", VAL_LEVEL))

    common = dict(
        collate_fn=collate_night_pairs,
        num_workers=workers,
        pin_memory=bool(loader_cfg.get("pin_memory", False)),
        persistent_workers=workers > 0,
    )
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **common),
        DataLoader(val, batch_size=batch_size, shuffle=False, **common),
    )


def collate_night_pairs(batch: list[dict]) -> dict:
    """Stack `NightPairs` items, keeping `image_id` a list of strings.

    `torch.utils.data.default_collate` would do this too; spelling it out keeps the nested
    `noise_params` dict explicit, since the noise head's target is built from it.
    """
    out = {
        key: torch.stack([item[key] for item in batch])
        for key in ("night", "night_clean", "day", "native_size")
    }
    out["image_id"] = [item["image_id"] for item in batch]
    out["noise_params"] = {
        "a": torch.stack([item["noise_params"]["a"] for item in batch]),
        "b": torch.stack([item["noise_params"]["b"] for item in batch]),
        "n": torch.tensor([item["noise_params"]["n"] for item in batch]),
        "quant_var": torch.tensor([item["noise_params"]["quant_var"] for item in batch]),
    }
    return out
