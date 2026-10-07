"""Velocity transformer for flow matching on the token layout.

6-block DiT that predicts the velocity field v(z_t, t, c) where:
    z_t: noisy STANDARDIZED tokens at time t, (B, N, D): one token per DINO patch, D values each
         (docs/conventions.md section 2.3: brightness grid, colour block, optional correction).
    t:   flow time in [0, 1] (0 = noise, 1 = data).
    c:   conditioning from the DINOv3 encoder.

Uses adaLN-Zero for time conditioning and an optional additive bias from DINO attention.

The net works on token tensors only; the token width D, the token count N and the (gy, gx) grid of
the thumbnail all come from the `TokenLayout` it is built with, so a different resolution or
`brightness_correction` setting changes nothing here. Token n of this transformer, DINO patch n,
brightness cell n and thumbnail pixel n all cover the same image region, so the conditioning
streams are added per token with no resampling anywhere:

    x = Linear(D -> 256)(z_t tokens)
      + Linear(384 -> 256)(DINO patch tokens)
      + Linear(1 -> 256)(thermal thumbnail)
      + learned positional embedding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dawn.flow.tokens import TokenLayout
from dawn.models.encoder import attention_to_bias


class AdaLNZero(nn.Module):
    """Adaptive Layer Normalization (Zero init) for time conditioning.

    Projects time embedding to (shift, scale, gate) per channel.
    Initialized so gate ≈ 0, making each block a no-op at init.
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (normalized_x * (1 + scale) + shift, gate)."""
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        x_normed = self.norm(x)
        return x_normed * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate


class AdaLNFinal(nn.Module):
    """adaLN without a residual gate, for the layer that feeds the zero-init output head.

    The final layer has nothing to gate (its output is the prediction, not a residual), so it
    projects (shift, scale) only. Zero init makes it the plain LayerNorm at step 0.
    """

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
    """Single Diffusion Transformer block.

    Multi-head self-attention + MLP, both gated via adaLN-Zero.
    Optional additive bias from DINO attention maps:
        logits = QK^T / sqrt(d) + dino_gate * log(dino_attn + 1e-6)
    with one learned scalar `dino_gate` per block, initialized at 0. The log is taken once, by
    `dawn.models.encoder.attention_to_bias`; this block receives the bias already in log space
    and passes `dino_gate * bias` to SDPA as a float `attn_mask`, which is exactly an additive
    term on the logits.
    """

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
        """Forward pass.

        Args:
            x: (B, N, D) hidden states (N = tokens, D = hidden_dim).
            t_emb: (B, D) time embedding.
            attn_bias: (B, 1, N, N) additive attention-logit bias from DINO, already in log
                space; broadcast over heads and scaled by this block's `dino_gate` (optional).

        Returns:
            x: (B, N, D) updated hidden states.
        """
        hidden, gate = self.adaln_attn(x, t_emb)
        x = x + gate.unsqueeze(1) * self._attention(hidden, attn_bias)
        hidden, gate = self.adaln_mlp(x, t_emb)
        return x + gate.unsqueeze(1) * self.mlp(hidden)


class VelocityTransformer(nn.Module):
    """Predicts velocity v(z_t, t, c) for conditional flow matching.

    Architecture:
        1. Linear(D -> hidden) of the tokens (D = layout.dim: 48, or 64 with the correction).
        2. Add Linear(384 -> hidden) of the DINO patch tokens, Linear(1 -> hidden) of the thermal
           thumbnail and a learned position embedding.
        3. A sinusoidal time embedding enters every block through adaLN-Zero; DINO attention is a
           per-block additive attention bias scaled by a learned gate initialized at 0.
        4. 6x DiTBlock, then a final adaLN and a zero-initialized Linear(hidden -> D).

    About 7M parameters, about 10 GMACs per step at 1,280 tokens (estimate).
    """

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
        """Build from the `velocity:` section of configs/flow.yaml and the token layout."""
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
        """Sinusoidal positional embedding for continuous time t in [0, 1].

        Args:
            t: (B,) time values.
            dim: Embedding dimension.

        Returns:
            (B, dim) time embedding.
        """
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t.to(torch.float32)[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)

    def attn_gates(self) -> dict[str, float]:
        """Each block's learned DINO-attention gate, for logging (`logging.log_attn_gates`)."""
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
        """Predict the velocity field.

        Args:
            z_t: (B, N, D) STANDARDIZED noisy tokens at time t.
            t: (B,) time in [0, 1].
            dino_features: (B, N, dino_dim) patch tokens from the encoder.
            dino_attn: (B, N, N) head-averaged patch-to-patch DINO attention for the attention
                bias. Detached first when `detach_dino_attn` is set, then turned into the
                additive bias `log(A + 1e-6)`.
            thumb: (B, 1, gy, gx) thermal thumbnail (`DINOEncoder.thumbnail`). Omitted means no
                thumbnail stream.
            attn_bias: (B, 1, N, N) the bias already in log space, as `encode` returns it.
                Mutually exclusive with `dino_attn`.

        Returns:
            v: (B, N, D) predicted velocity, standardized.

        Raises:
            ValueError: the shapes do not match the token layout this model was built for (a
                different resolution, bin count or brightness_correction setting).
        """
        n, d = self.num_tokens, self.token_dim
        if z_t.ndim != 3 or z_t.shape[1:] != (n, d):
            raise ValueError(
                f"z_t must be (B, {n}, {d}) for this token layout, got {tuple(z_t.shape)}: "
                "a different resolution, number of bins or brightness_correction setting?"
            )
        if t.ndim != 1 or t.shape[0] != z_t.shape[0]:
            raise ValueError(f"t must be (B,) with B = {z_t.shape[0]}, got {tuple(t.shape)}")
        if dino_features.shape[:2] != (z_t.shape[0], n):
            raise ValueError(f"dino_features must be (B, {n}, C), got {tuple(dino_features.shape)}")
        if thumb is not None and thumb.shape != (z_t.shape[0], 1, *self.layout.grid):
            raise ValueError(
                f"thumb must be (B, 1, {self.layout.grid[0]}, {self.layout.grid[1]}), "
                f"got {tuple(thumb.shape)}"
            )
        if dino_attn is not None and attn_bias is not None:
            raise ValueError("pass dino_attn or attn_bias, not both")

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
        """Total parameter count (about 7M at the plan's size)."""
        return sum(p.numel() for p in self.parameters())
