"""Velocity transformer for bilateral grid flow matching.

6-block DiT that predicts the velocity field v(z_t, t, c) where:
    z_t: noisy STANDARDIZED grid coefficients at time t, (B, 12, K, gy, gx).
    t:   flow time in [0, 1] (0 = noise, 1 = data).
    c:   conditioning from the DINOv3 encoder.

Uses adaLN-Zero for time conditioning and an optional additive bias from DINO attention.

The grid is tokenized here and only here, with `dawn.grid.layout.grid_to_tokens` /
`tokens_to_grid`: 256 tokens (one per spatial cell) of 96 values (K bins x 12 coefficients).
Layout and token order: docs/conventions.md section 2.

Token n of this transformer, DINO patch n and grid cell n all cover the same image region, so
the three conditioning streams are added per token with no resampling anywhere:

    x = Linear(96 -> D)(z_t tokens)
      + Linear(384 -> D)(DINO patch tokens)
      + Linear(3 -> D)(log thumbnail)
      + learned positional embedding
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dawn.grid.layout import NUM_COEFFS, grid_to_tokens, tokens_to_grid
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
            x: (B, N, D) hidden states (N = grid_h * grid_w = 256 tokens, D = hidden_dim).
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
        1. Tokenize the grid: (B, 12, K, gy, gx) → (B, 256, K*12) via `grid_to_tokens`,
           then Linear(K*12 → D).
        2. Add sinusoidal time embedding.
        3. Add DINO features (Linear 384 → D) as conditioning and, optionally, DINO attention
           as a per-block attention bias.
        4. 6x DiTBlock with adaLN-Zero.
        5. Linear proj back to K*12 values per token (zero-initialized).
        6. `tokens_to_grid` back to (B, 12, K, gy, gx).

    Total: ~7M params, ~1.2 GMACs/step.
    """

    def __init__(
        self,
        grid_h: int = 16,
        grid_w: int = 16,
        num_bins: int = 8,
        coeffs: int = 12,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_blocks: int = 6,
        mlp_ratio: int = 4,
        dino_dim: int = 384,
        dropout: float = 0.0,
        detach_dino_attn: bool = True,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_bins = num_bins
        self.num_tokens = grid_h * grid_w
        self.token_dim = num_bins * coeffs
        self.detach_dino_attn = detach_dino_attn

        self.input_proj = nn.Linear(self.token_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.token_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.dino_proj = nn.Linear(dino_dim, hidden_dim)
        self.thumb_proj = nn.Linear(3, hidden_dim)
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
    def from_config(cls, cfg, grid_cfg) -> "VelocityTransformer":
        """Build from the `velocity:` and `grid:` sections of configs/flow.yaml."""
        return cls(
            grid_h=grid_cfg["spatial_size"],
            grid_w=grid_cfg["spatial_size"],
            num_bins=grid_cfg["num_bins"],
            coeffs=grid_cfg["coeffs_per_bin"],
            hidden_dim=cfg["hidden_dim"],
            num_heads=cfg["num_heads"],
            num_blocks=cfg["num_blocks"],
            mlp_ratio=cfg["mlp_ratio"],
            dino_dim=cfg.get("dino_dim", 384),
            dropout=cfg.get("dropout", 0.0),
            detach_dino_attn=cfg["detach_dino_attn"],
        )

    def _sinusoidal_embedding(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        """Sinusoidal positional embedding for continuous time t ∈ [0, 1].

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
        """Predict velocity field.

        Args:
            z_t: (B, 12, K, gy, gx) STANDARDIZED noisy grid at time t.
            t: (B,) time in [0, 1].
            dino_features: (B, 256, dino_dim) patch tokens from the encoder.
            dino_attn: (B, 256, 256) head-averaged patch-to-patch DINO attention for the
                attention bias. Detached first when `detach_dino_attn` is set, then turned
                into the additive bias `log(A + 1e-6)`.
            thumb: (B, 3, gy, gx) log thumbnail of the night image (`DINOEncoder.thumbnail`).
                Omitted means no thumbnail stream.
            attn_bias: (B, 1, 256, 256) the bias already in log space, as `encode` returns it.
                Mutually exclusive with `dino_attn`.

        Returns:
            v: (B, 12, K, gy, gx) predicted velocity, standardized.
        """
        if z_t.ndim != 5 or z_t.shape[1] != NUM_COEFFS:
            raise ValueError(f"z_t must be (B, 12, K, gy, gx), got {tuple(z_t.shape)}")
        if z_t.shape[2:] != (self.num_bins, self.grid_h, self.grid_w):
            raise ValueError(
                f"z_t grid is {tuple(z_t.shape[2:])}, this model was built for "
                f"{(self.num_bins, self.grid_h, self.grid_w)}"
            )
        if t.ndim != 1 or t.shape[0] != z_t.shape[0]:
            raise ValueError(f"t must be (B,) with B = {z_t.shape[0]}, got {tuple(t.shape)}")
        if dino_features.shape[:2] != (z_t.shape[0], self.num_tokens):
            raise ValueError(
                f"dino_features must be (B, {self.num_tokens}, C), got "
                f"{tuple(dino_features.shape)}"
            )
        if dino_attn is not None and attn_bias is not None:
            raise ValueError("pass dino_attn or attn_bias, not both")

        x = self.input_proj(grid_to_tokens(z_t)) + self.dino_proj(dino_features) + self.pos_embed
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
        out = self.output_proj(self.final_adaln(x, t_emb))
        return tokens_to_grid(out, self.grid_h, self.grid_w, self.num_bins)

    def num_params(self) -> int:
        """Total parameter count (should be ~7M)."""
        return sum(p.numel() for p in self.parameters())
