"""Token standardization and running moment accumulators."""

from __future__ import annotations

import torch

STD_FLOOR = 1e-6


def _check(tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> None:
    if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
        raise ValueError(
            f"stats must be matching (D,) vectors, got {tuple(mean.shape)}, " f"{tuple(std.shape)}"
        )
    if tokens.shape[-1] != mean.shape[0]:
        raise ValueError(
            f"tokens have {tokens.shape[-1]} values per token but the stats have {mean.shape[0]}: "
            "the token layout and the stats disagree"
        )


def standardize(tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardize tokens elementwise using mean and standard deviation vectors.

    Args:
        tokens: (B, N, D) unnormalized token tensor.
        mean: (D,) per-feature mean vector.
        std: (D,) per-feature standard deviation vector.

    Returns:
        (B, N, D) standardized token tensor.
    """
    _check(tokens, mean, std)
    mean, std = mean.to(tokens), std.to(tokens)
    return (tokens - mean) / std.clamp(min=STD_FLOOR)


def unstandardize(tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Unstandardize tokens elementwise using mean and standard deviation vectors.

    Args:
        tokens: (B, N, D) standardized token tensor.
        mean: (D,) per-feature mean vector.
        std: (D,) per-feature standard deviation vector.

    Returns:
        (B, N, D) unstandardized token tensor.
    """
    _check(tokens, mean, std)
    mean, std = mean.to(tokens), std.to(tokens)
    return tokens * std.clamp(min=STD_FLOOR) + mean


class WeightedMoments:
    """Streaming accumulator for weighted mean and variance across variables."""

    def __init__(self, dim: int):
        self.total = torch.zeros((), dtype=torch.float64)
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64)

    @torch.no_grad()
    def add(self, values: torch.Tensor, weights: torch.Tensor | None) -> None:
        """Add a batch of values and optional weights to running sums."""
        values = values.detach().cpu().to(torch.float64)
        w = (
            torch.ones(values.shape[0], dtype=torch.float64)
            if weights is None
            else (weights.detach().cpu().to(torch.float64))
        )
        self.total += w.sum()
        self.sum += (values * w[:, None]).sum(dim=0)
        self.sum_sq += (values.pow(2) * w[:, None]).sum(dim=0)

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute final (mean, std) vectors as float32 tensors.

        Raises:
            ValueError: If no non-zero weights were added.
        """
        if float(self.total) <= 0:
            raise ValueError("no weight was added, so there are no statistics")
        mean = self.sum / self.total
        var = (self.sum_sq / self.total - mean.pow(2)).clamp(min=0.0)
        if float(self.total) > 1.0:
            var = var * (self.total / (self.total - 1.0))
        return mean.to(torch.float32), var.sqrt().clamp(min=STD_FLOOR).to(torch.float32)
