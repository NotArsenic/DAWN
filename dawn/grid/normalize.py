"""Token standardization.

Flow matching targets should be roughly N(0, 1)-scaled. `standardize` and `unstandardize` are the
only places tokens change space; where they are called is pinned in docs/conventions.md section 5.

Tokens are (B, N, D); stats are (D,) vectors. They are built from per-segment stats (brightness
per (coefficient, bin), colour per channel, correction) by `TokenLayout.stats_vector`, and
`WeightedMoments` is the accumulator that produces those per-segment stats.
"""

from __future__ import annotations

import torch

# std is floored by both standardize and unstandardize, so they stay exact inverses.
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
    """Raw -> standardized: (tokens - mean) / std, per value of the token.

    Args:
        tokens: (B, N, D) raw values.
        mean: (D,) per-value mean.
        std: (D,) per-value std.

    Returns:
        Standardized tokens, same shape.
    """
    _check(tokens, mean, std)
    mean, std = mean.to(tokens), std.to(tokens)
    return (tokens - mean) / std.clamp(min=STD_FLOOR)


def unstandardize(tokens: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Standardized -> raw: tokens * std + mean, per value of the token.

    Args:
        tokens: (B, N, D) standardized values.
        mean: (D,) per-value mean.
        std: (D,) per-value std.

    Returns:
        Raw tokens, same shape.
    """
    _check(tokens, mean, std)
    mean, std = mean.to(tokens), std.to(tokens)
    return tokens * std.clamp(min=STD_FLOOR) + mean


class WeightedMoments:
    """Streaming weighted mean and std of `dim` variables, in float64.

    Only running sums are kept, so the memory cost does not grow with the number of samples.
    Values with weight 0 do not count; the std is unbiased for unit weights (`n / (n - 1)`, to
    match `torch.std`) and uses the total weight as `n` for fractional ones.
    """

    def __init__(self, dim: int):
        self.total = torch.zeros((), dtype=torch.float64)
        self.sum = torch.zeros(dim, dtype=torch.float64)
        self.sum_sq = torch.zeros(dim, dtype=torch.float64)

    @torch.no_grad()
    def add(self, values: torch.Tensor, weights: torch.Tensor | None) -> None:
        """Add `values` (M, dim) with per-row `weights` (M,), or all ones when None."""
        values = values.detach().to("cpu", torch.float64)
        w = (
            torch.ones(values.shape[0], dtype=torch.float64)
            if weights is None
            else (weights.detach().to("cpu", torch.float64))
        )
        self.total += w.sum()
        self.sum += (values * w[:, None]).sum(dim=0)
        self.sum_sq += (values.pow(2) * w[:, None]).sum(dim=0)

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(mean, std)`, float32, `(dim,)` each.

        Raises:
            ValueError: nothing with positive weight was added.
        """
        if float(self.total) <= 0:
            raise ValueError("no weight was added, so there are no statistics")
        mean = self.sum / self.total
        var = (self.sum_sq / self.total - mean.pow(2)).clamp(min=0.0)
        if float(self.total) > 1.0:
            var = var * (self.total / (self.total - 1.0))
        return mean.to(torch.float32), var.sqrt().clamp(min=STD_FLOOR).to(torch.float32)
