"""Velocity transformer for conditional flow matching."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dawn.flow.tokens import TokenLayout
from dawn.models.encoder import attention_to_bias


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization with zero initialization for time conditioning."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute modulated features and residual gate."""
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        x_normed = self.norm(x)
        return x_normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate


class AdaLNFinal(nn.Module):
    """Adaptive Layer Normalization for final layer modulation without residual gating."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaLN-Zero conditioning and optional attention bias."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        use_dino_attn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.adaln_attn = AdaLNZero(hidden_dim, hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.adaln_mlp = AdaLNZero(hidden_dim, hidden_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        self.use_dino_attn = use_dino_attn
        if use_dino_attn:
            self.dino_gate = nn.Parameter(torch.tensor(0.0))

    def _attention(self, x: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (B, heads, N, head_dim)
        mask = None
        if attn_bias is not None and self.use_dino_attn:
            mask = (self.dino_gate * attn_bias).to(query.dtype)
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        return self.out_proj(out.transpose(1, 2).reshape(batch, tokens, self.hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply self-attention and MLP blocks with adaptive modulation.

        Args:
            x: Input token features of shape (B, N, D).
            t_emb: Time embedding tensor of shape (B, D).
            attn_bias: Optional attention bias tensor of shape (B, 1, N, N).

        Returns:
            Updated token features of shape (B, N, D).
        """
        hidden, gate = self.adaln_attn(x, t_emb)
        x = x + gate.unsqueeze(1) * self._attention(hidden, attn_bias)
        hidden, gate = self.adaln_mlp(x, t_emb)
        return x + gate.unsqueeze(1) * self.mlp(hidden)


class VelocityTransformer(nn.Module):
    """Diffusion Transformer predicting velocity field for conditional flow matching."""


    def __init__(
        self,
        layout: TokenLayout,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 6,
        mlp_ratio: int = 4,
        dino_dim: int = 384,
        dropout: float = 0.0,
        detach_dino_attn: bool = True,
    ):
        super().__init__()
        self.layout = layout
        self.hidden_dim = hidden_dim
        self.num_tokens = layout.num_tokens
        self.token_dim = layout.dim
        self.detach_dino_attn = detach_dino_attn

        self.input_proj = nn.Linear(self.token_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.dino_proj = nn.Linear(dino_dim, hidden_dim)
        self.thumb_proj = nn.Linear(1, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)]
        )
        self.final_adaln = AdaLNFinal(hidden_dim, hidden_dim)

    @classmethod
    def from_config(cls, cfg, layout: TokenLayout) -> "VelocityTransformer":
        """Instantiate VelocityTransformer from configuration mapping and TokenLayout."""
        return cls(
            layout,
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"],
            mlp_ratio=cfg["mlp_ratio"],
            dino_dim=cfg.get("dino_dim", 384),
            dropout=cfg.get("dropout", 0.0),
            detach_dino_attn=cfg["detach_dino_attn"],
        )

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute sinusoidal embeddings for continuous time steps."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.to(torch.float32)[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def attn_gates(self) -> dict[str, float]:
        """Return learned attention gate values per transformer block."""
        return {
            f"gate_{i}": float(block.dino_gate.detach())
            for i, block in enumerate(self.blocks)
            if block.use_dino_attn
        }

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        dino_features: torch.Tensor,
        dino_attn: torch.Tensor | None = None,
        thumb: torch.Tensor | None = None,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field for noisy token inputs.

        Args:
            z_t: Standardized noisy tokens at time t of shape (B, N, D).
            t: Continuous time values of shape (B,) in [0, 1].
            dino_features: Conditioning patch tokens of shape (B, N, dino_dim).
            dino_attn: Optional patch attention weights of shape (B, N, N).
            thumb: Optional thermal thumbnail tensor of shape (B, 1, gy, gx).
            attn_bias: Optional precomputed log attention bias of shape (B, 1, N, N).

        Returns:
            Predicted velocity tensor of shape (B, N, D).
        """
        x = self.input_proj(z_t) + self.dino_proj(dino_features) + self.pos_embed
        if thumb is not None:
            x = x + self.thumb_proj(rearrange(thumb, "b c y x -> b (y x) c"))

        t_emb = self.time_mlp(self._sinusoidal_embedding(t, x.shape[-1]))

        bias = attn_bias
        if dino_attn is not None:
            bias = attention_to_bias(dino_attn.detach() if self.detach_dino_attn else dino_attn)
        if bias is not None and self.detach_dino_attn:
            bias = bias.detach()

        for block in self.blocks:
            x = block(x, t_emb, bias)
        return self.output_proj(self.final_adaln(x, t_emb))

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

