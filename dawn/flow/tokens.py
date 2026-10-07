"""The one token layout of the flow stage (docs/conventions.md section 2.3).

A token is one DINO patch / brightness cell. Its values, in this order:

    brightness   2K     v = k*2 + c            (bin-major; c = gain, offset)
    color        2*b*b  v = 2K + ch*b*b + py*b + px   (its b x b block of the colour map; a*, b*)
    correction   b*b    v = 2K + 2*b*b + py*b + px    (only when enabled; the same block, in L)

with block `b = 4` (the plan's 16 brightness values, then 32 colour values). Token `n = y*gx + x`,
row-major, matches DINO patch `n`, the thermal thumbnail pixel `n` and brightness cell `n`.

`TokenLayout` is the only place that packs, unpacks, slices or counts those values: the velocity
net's width, the stats vector, the loss weights and the render path all derive from it, and no
other module hard-codes 16 / 32 / 48 / 64 (docs/conventions.md section 8, rule 3).

Raw "parts" are what the pixel side sees:

    grid  (B, 2, K, gy, gx)       gain and offset, mapping thermal to L = L*/100
    ab    (B, 2, b*gy, b*gx)      a*, b* in CIELAB units
    corr  (B, 1, b*gy, b*gx)      the L correction (only when enabled)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from einops import rearrange, repeat

from dawn.grid.layout import NUM_COEFFS

NUM_AB = 2  # a*, b*


@dataclass(frozen=True)
class TokenLayout:
    """Sizes and packing of the flow tokens.

    Args:
        num_bins: K, thermal-intensity bins of the brightness grid.
        grid: (gy, gx) token / cell / DINO patch grid.
        block: Colour pixels per token side; the colour map is `block` x the token grid.
        correction: Whether the generated L correction is part of every token.
    """

    num_bins: int = 8
    grid: tuple[int, int] = (32, 40)
    block: int = 4
    correction: bool = False

    @classmethod
    def from_config(cls, cfg: Mapping) -> TokenLayout:
        """Build from the parsed flow.yaml: `encoder.input_size`, `grid`, `color`, correction."""
        height, width = cfg["encoder"]["input_size"]
        patch = cfg["encoder"]["patch_size"]
        if height % patch or width % patch:
            raise ValueError(
                f"encoder.input_size {(height, width)} must be a multiple of the patch size {patch}"
            )
        return cls(
            num_bins=int(cfg["grid"]["num_bins"]),
            grid=(height // patch, width // patch),
            block=int(cfg["color"]["block"]),
            correction=bool(cfg["brightness_correction"]["enabled"]),
        )

    # --- sizes --------------------------------------------------------------------------------

    @property
    def num_tokens(self) -> int:
        return self.grid[0] * self.grid[1]

    @property
    def color_size(self) -> tuple[int, int]:
        """(H, W) of the colour map (and the correction map): `block` x the token grid."""
        return self.grid[0] * self.block, self.grid[1] * self.block

    @property
    def segments(self) -> dict[str, slice]:
        """Value slice of each segment, in token order: brightness, color, (correction)."""
        bright = NUM_COEFFS * self.num_bins
        color = NUM_AB * self.block**2
        out = {"brightness": slice(0, bright), "color": slice(bright, bright + color)}
        if self.correction:
            out["correction"] = slice(bright + color, bright + color + self.block**2)
        return out

    @property
    def dim(self) -> int:
        """D, values per token."""
        return max(seg.stop for seg in self.segments.values())

    # --- packing ------------------------------------------------------------------------------

    def _part_shapes(self, batch: int) -> dict[str, tuple[int, ...]]:
        height, width = self.color_size
        shapes = {
            "grid": (batch, NUM_COEFFS, self.num_bins, *self.grid),
            "ab": (batch, NUM_AB, height, width),
        }
        if self.correction:
            shapes["corr"] = (batch, 1, height, width)
        return shapes

    def pack(self, parts: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Raw parts -> tokens `(B, N, D)`. Works in whatever space the parts are in.

        Raises:
            ValueError: the parts are not exactly the ones this layout has, or are misshapen.
        """
        expected = self._part_shapes(parts["grid"].shape[0] if "grid" in parts else 0)
        if set(parts) != set(expected):
            raise ValueError(
                f"parts {sorted(parts)} do not match the layout's {sorted(expected)} "
                f"(brightness_correction {'on' if self.correction else 'off'})"
            )
        for name, shape in expected.items():
            if tuple(parts[name].shape) != shape:
                raise ValueError(f"part {name!r} must be {shape}, got {tuple(parts[name].shape)}")
        b = self.block
        pieces = [
            rearrange(parts["grid"], "b c k y x -> b (y x) (k c)"),
            rearrange(parts["ab"], "b ch (y py) (x px) -> b (y x) (ch py px)", py=b, px=b),
        ]
        if self.correction:
            pieces.append(
                rearrange(parts["corr"], "b ch (y py) (x px) -> b (y x) (ch py px)", py=b, px=b)
            )
        return torch.cat(pieces, dim=-1)

    def unpack(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tokens `(B, N, D)` -> raw parts. The inverse of `pack`.

        Raises:
            ValueError: `tokens` is not (B, num_tokens, dim).
        """
        if tokens.ndim != 3 or tokens.shape[1:] != (self.num_tokens, self.dim):
            raise ValueError(
                f"tokens must be (B, {self.num_tokens}, {self.dim}) for this layout, "
                f"got {tuple(tokens.shape)}"
            )
        gy, gx = self.grid
        b, seg = self.block, self.segments
        parts = {
            "grid": rearrange(
                tokens[..., seg["brightness"]],
                "b (y x) (k c) -> b c k y x",
                y=gy,
                x=gx,
                c=NUM_COEFFS,
            ),
            "ab": rearrange(
                tokens[..., seg["color"]],
                "b (y x) (ch py px) -> b ch (y py) (x px)",
                y=gy,
                x=gx,
                py=b,
                px=b,
            ),
        }
        if self.correction:
            parts["corr"] = rearrange(
                tokens[..., seg["correction"]],
                "b (y x) (ch py px) -> b ch (y py) (x px)",
                y=gy,
                x=gx,
                py=b,
                px=b,
            )
        return parts

    # --- stats and weights --------------------------------------------------------------------

    def _segment_stat_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {"brightness": (NUM_COEFFS, self.num_bins), "color": (NUM_AB,)}
        if self.correction:
            shapes["correction"] = (1,)
        return shapes

    def stats_vector(
        self, stats: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-segment stats -> the `(D,)` mean and std that `standardize` takes.

        Args:
            stats: `{segment: {"mean": ..., "std": ...}}` with brightness `(2, K)` indexed
                `[coefficient, bin]`, colour `(2,)` (a*, b*), correction `(1,)`.

        Raises:
            ValueError: a segment is missing or misshapen.
        """
        out = {"mean": [], "std": []}
        for name, shape in self._segment_stat_shapes().items():
            if name not in stats:
                raise ValueError(f"stats have no {name!r} segment (layout needs {sorted(shape)})")
            for kind in ("mean", "std"):
                value = stats[name][kind]
                if tuple(value.shape) != shape:
                    raise ValueError(f"{name} {kind} must be {shape}, got {tuple(value.shape)}")
                if name == "brightness":
                    out[kind].append(rearrange(value, "c k -> (k c)"))
                else:
                    out[kind].append(repeat(value, "ch -> (ch p)", p=self.block**2))
        return torch.cat(out["mean"]), torch.cat(out["std"])

    def weights(self, node_weights: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
        """Loss weights `(B, N, D)`, each segment normalized to mean 1 per sample.

        Brightness values take the node weights (pixel count per cell and bin) broadcast over
        gain and offset; colour and correction values take the mask fraction of their pixel.
        A segment with no weight at all (a fully masked sample) stays all zero.

        Args:
            node_weights: (B, K, gy, gx) from `fit_affine_grid`.
            fraction: (B, 1, b*gy, b*gx) kept share of every colour-map pixel, from the area
                average of the pixel mask.
        """
        b = self.block
        bright = repeat(
            rearrange(node_weights, "b k y x -> b (y x) k"), "b n k -> b n (k c)", c=NUM_COEFFS
        )
        pixel = rearrange(fraction, "b 1 (y py) (x px) -> b (y x) (py px)", py=b, px=b)
        pieces = [bright, repeat(pixel, "b n p -> b n (ch p)", ch=NUM_AB)]
        if self.correction:
            pieces.append(pixel)
        return torch.cat([_mean_one(p.to(torch.float32)) for p in pieces], dim=-1)

    # --- checkpoints --------------------------------------------------------------------------

    def descriptor(self) -> dict:
        """What a checkpoint stores to prove which layout it was trained with."""
        return {
            "num_bins": self.num_bins,
            "grid": list(self.grid),
            "block": self.block,
            "correction": self.correction,
            "dim": self.dim,
        }

    @classmethod
    def from_descriptor(cls, descriptor: Mapping) -> TokenLayout:
        return cls(
            num_bins=int(descriptor["num_bins"]),
            grid=(int(descriptor["grid"][0]), int(descriptor["grid"][1])),
            block=int(descriptor["block"]),
            correction=bool(descriptor["correction"]),
        )

    def check(self, descriptor: Mapping) -> None:
        """Raise a clear error if `descriptor` is not this layout (e.g. another resolution)."""
        if self.descriptor() != dict(descriptor):
            raise ValueError(
                f"token layout mismatch: this run has {self.descriptor()}, the checkpoint was "
                f"trained with {dict(descriptor)} (different resolution, bins, block or "
                "brightness_correction)"
            )


def _mean_one(weights: torch.Tensor) -> torch.Tensor:
    """Scale `(B, N, V)` weights to mean 1 per sample; all-zero samples stay zero."""
    mean = weights.mean(dim=(1, 2), keepdim=True)
    return torch.where(mean > 0, weights / mean.clamp(min=torch.finfo(weights.dtype).tiny), 0.0)
