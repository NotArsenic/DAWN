"""Grid coefficient standardization.

Flow matching targets should be roughly N(0, 1)-scaled. We compute a mean and std per
(coefficient, bin) from a subset of fitted RAW grids, then standardize / unstandardize.
`standardize` and `unstandardize` are the only places a grid changes space; where they are
called is pinned in docs/conventions.md section 5.

Grids are (B, 12, K, gy, gx); stats are (12, K), broadcast as (1, 12, K, 1, 1).
"""

from __future__ import annotations

from pathlib import Path

import torch

# std is floored by both standardize and unstandardize, so they stay exact inverses.
STD_FLOOR = 1e-6


@torch.no_grad()
def compute_grid_stats(
    dataset,
    num_samples: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Compute per-(coefficient, bin) mean and std from fitted RAW grids.

    Grids are fit on the fly: for each sample, `fit_grid(night_clean, day,
    compute_guidance(night_clean))`.

    Args:
        dataset: Dataset that yields dicts with 'night_clean' and 'day' (linear RGB).
        num_samples: Number of samples to use.
        device: Computation device.

    Grids are fit one sample at a time and only the running sums are kept, so the memory cost
    does not grow with `num_samples`. Statistics pool over samples and over both cell axes, but
    never over coefficients or bins: a bin-0 offset and a bin-7 offset live on different scales
    and sharing a mean would undo the point of standardizing.

    Returns:
        dict with 'mean' and 'std', each (12, K) float32, statistics over samples and cells.

    Raises:
        ValueError: `num_samples` is not positive, or the dataset yielded nothing usable.
    """
    from dawn.grid.fit import fit_grid
    from dawn.grid.guidance import compute_guidance

    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    count = 0.0
    sum_x: torch.Tensor | None = None
    sum_x2: torch.Tensor | None = None

    for index in range(min(num_samples, len(dataset))):
        sample = dataset[index]
        night_clean = sample["night_clean"].to(device=device, dtype=torch.float32)
        day = sample["day"].to(device=device, dtype=torch.float32)
        if night_clean.ndim == 3:
            night_clean, day = night_clean.unsqueeze(0), day.unsqueeze(0)
        grid, _ = fit_grid(night_clean, day, compute_guidance(night_clean))
        flat = grid.permute(1, 2, 0, 3, 4).reshape(grid.shape[1], grid.shape[2], -1)  # (12, K, M)
        batch_sum, batch_sq = flat.sum(dim=-1), flat.pow(2).sum(dim=-1)
        sum_x = batch_sum if sum_x is None else sum_x + batch_sum
        sum_x2 = batch_sq if sum_x2 is None else sum_x2 + batch_sq
        count += flat.shape[-1]

    if sum_x is None or count == 0:
        raise ValueError("dataset yielded no samples to compute grid stats from")

    mean = sum_x / count
    var = (sum_x2 / count - mean.pow(2)).clamp(min=0.0)
    if count > 1:  # unbiased, to match torch.std's default
        var = var * (count / (count - 1.0))
    return {
        "mean": mean.to(torch.float32).cpu(),
        "std": var.sqrt().clamp(min=STD_FLOOR).to(torch.float32).cpu(),
    }


def _broadcast(stat: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """(12, K) stats -> (1, 12, K, 1, 1), on the grid's device and dtype."""
    return stat.reshape(1, *stat.shape, 1, 1).to(device=grid.device, dtype=grid.dtype)


def standardize(
    grid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Raw -> standardized: (grid - mean) / std, per (coefficient, bin).

    Args:
        grid: (B, 12, K, gy, gx) raw coefficients.
        mean: (12, K) per-(coefficient, bin) mean.
        std: (12, K) per-(coefficient, bin) std.

    Returns:
        Standardized grid, same shape.
    """
    return (grid - _broadcast(mean, grid)) / _broadcast(std, grid).clamp(min=STD_FLOOR)


def unstandardize(
    grid: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Standardized -> raw: grid * std + mean, per (coefficient, bin).

    Args:
        grid: (B, 12, K, gy, gx) standardized coefficients.
        mean: (12, K) per-(coefficient, bin) mean.
        std: (12, K) per-(coefficient, bin) std.

    Returns:
        Raw grid, same shape.
    """
    return grid * _broadcast(std, grid).clamp(min=STD_FLOOR) + _broadcast(mean, grid)


def load_or_compute_stats(
    cache_path: str | Path,
    dataset=None,
    num_samples: int = 2000,
    device: torch.device = torch.device("cpu"),
) -> dict[str, torch.Tensor]:
    """Load cached stats or compute and save them.

    Training caches here; checkpoints embed the stats they were trained with and inference
    reads them from the checkpoint, not from this cache.

    Args:
        cache_path: Path to cached .pt file.
        dataset: Dataset to compute from (if cache missing).
        num_samples: Number of samples for computation.
        device: Computation device.

    Returns:
        dict with 'mean' and 'std' tensors, each (12, K).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return torch.load(cache_path, map_location=device, weights_only=True)

    if dataset is None:
        raise FileNotFoundError(f"Stats cache not found at {cache_path} and no dataset provided.")

    stats = compute_grid_stats(dataset, num_samples, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, cache_path)
    return stats
