"""Per-patch colour classifier for L_color (training only).

A LayerNorm and one Linear map every ADAPTED DINO patch token to logits over the 261 a*b* bins
(`dawn.flow.color_bins`). Trained with cross-entropy against the soft histogram of the RGB patch's
colours (Zhang, Isola & Efros, ECCV 2016), it forces the LoRA-tuned encoder to know which colours
belong where, which is what the generated colour map then relies on. Nothing bigger is needed:
the head is a probe of what the tokens already hold, and a deeper head would learn the colour
itself and leave the encoder lazy.

It is part of the EMA and of the checkpoints and unused at inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dawn.flow.color_bins import NUM_BINS


class ColorHead(nn.Module):
    """LayerNorm + Linear from patch tokens `(B, N, dino_dim)` to logits `(B, N, num_bins)`."""

    def __init__(self, dino_dim: int = 384, num_bins: int = NUM_BINS):
        super().__init__()
        self.norm = nn.LayerNorm(dino_dim)
        self.proj = nn.Linear(dino_dim, num_bins)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Logits over the a*b* bins for every patch token (softmax is left to the loss)."""
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens must be (B, N, C), got {tuple(patch_tokens.shape)}")
        return self.proj(self.norm(patch_tokens))
