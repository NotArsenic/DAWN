"""Data loaders for training, validation, and token statistics."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset

from dawn.data import dronevehicle as dv
from dawn.data.pairs import AlignedPairs, AugmentConfig
from dawn.flow.targets import prepare_batch
from dawn.progress import progress

SUBSETS = ("box_aligned", "fallback")


def load_config(config: str, data_config: str, overrides: Iterable[str] = ()) -> dict:
    """Merge configuration files and return resolved dictionary."""
    cfg = OmegaConf.merge(OmegaConf.load(config), OmegaConf.load(data_config))
    for path in overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return OmegaConf.to_container(cfg, resolve=True)


@dataclass
class FlowLoaders:
    """Container for training and validation data loaders."""

    train: DataLoader
    val: dict[str, DataLoader]
    train_size: int


def _loader(dataset: Dataset, cfg: Mapping, batch_size: int, shuffle: bool, drop_last: bool):
    workers = int(cfg["loader"]["num_workers"])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last and len(dataset) >= batch_size,
        num_workers=workers,
        pin_memory=bool(cfg["loader"]["pin_memory"]),
        persistent_workers=workers > 0,
    )


def split_by_pair_kind(dataset: AlignedPairs, max_per_subset: int | None) -> dict[str, Subset]:
    """Split dataset into box-aligned and fallback subsets."""
    kinds: dict[str, list[int]] = {name: [] for name in SUBSETS}
    for index, image_id in enumerate(dataset.image_ids):
        kinds["fallback" if image_id in dataset.fallback_ids else "box_aligned"].append(index)
    return {
        name: Subset(dataset, indices[:max_per_subset])
        for name, indices in kinds.items()
        if indices
    }


def build_loaders(cfg: Mapping, overfit: int | None = None) -> FlowLoaders:
    """Build training and validation data loaders from configuration.

    Args:
        cfg: Configuration mapping.
        overfit: Optional number of training samples to overfit on without augmentation.

    Returns:
        FlowLoaders container.
    """
    batch_size = int(cfg["optim"]["batch_size"])
    max_val = int(cfg["logging"]["num_val_samples"])
    if overfit:
        layout = dv.DroneVehicleLayout.from_config(cfg["dronevehicle"])
        train_set = AlignedPairs(
            layout, "train_pairs", train=False, augment=AugmentConfig(), max_pairs=overfit
        )
        val_sets = split_by_pair_kind(train_set, max_val)
    else:
        train_set = AlignedPairs.from_config(cfg, "train_pairs", train=True)
        val_sets = split_by_pair_kind(AlignedPairs.from_config(cfg, "val_pairs"), max_val)
    policy = cfg["pairs"]
    keep = {policy["eval_subset"], "box_aligned"}
    if policy["report_fallback_separately"]:
        keep.add("fallback")
    val_sets = {name: subset for name, subset in val_sets.items() if name in keep}
    val_batch = int(cfg["loader"].get("val_batch_size", 4))
    return FlowLoaders(
        train=_loader(train_set, cfg, batch_size, shuffle=True, drop_last=True),
        val={
            name: _loader(subset, cfg, val_batch, shuffle=False, drop_last=False)
            for name, subset in val_sets.items()
        },
        train_size=len(train_set),
    )


def stats_batches(
    cfg: Mapping, num_samples: int, device: torch.device | str = "cpu"
) -> Iterator[dict]:
    """Yield prepared training batches sampled across the dataset for normalization statistics."""
    dataset = AlignedPairs.from_config(cfg, "train_pairs", train=False)
    count = min(int(num_samples), len(dataset))
    indices = np.unique(np.linspace(0, len(dataset) - 1, count).round().astype(int)).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=8, num_workers=0)
    size = tuple(cfg["encoder"]["input_size"])
    for raw in progress(loader, f"token stats ({count} pairs)"):
        yield prepare_batch(raw, size, cfg["pairs"], device)


def cycle(loader: Iterable) -> Iterator:
    """Yield batches from an iterable loader infinitely."""
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("the training loader yielded no batches")

