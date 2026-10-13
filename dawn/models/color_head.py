"""Per-patch color classification head."""

from __future__ import annotations

import torch
import torch.nn as nn

from dawn.flow.color_bins import NUM_BINS


class ColorHead(nn.Module):
    """Linear classifier predicting color bin distributions from patch tokens."""

    def __init__(self, dino_dim: int = 384, num_bins: int = NUM_BINS):
        super().__init__()
        self.norm = nn.LayerNorm(dino_dim)
        self.proj = nn.Linear(dino_dim, num_bins)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized bin logits for patch tokens."""
        return self.proj(self.norm(patch_tokens))

