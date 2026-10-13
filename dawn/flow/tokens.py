"""Token layout specification, packing, unpacking, and normalization weighting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from einops import rearrange, repeat

from dawn.grid.layout import NUM_COEFFS

NUM_AB = 2  # a*, b*


@dataclass(frozen=True)
class TokenLayout:
    """Structure and dimension specifications for flow tokens.

    Args:
        num_bins: Number of intensity bins in the brightness grid.
        grid: Spatial grid dimensions (gy, gx) for tokens and patches.
        block: Spatial scale factor from token grid to color map.
        correction: Whether lightness residual correction is included in tokens.
    """

    num_bins: int = 8
    grid: tuple[int, int] = (32, 40)
    block: int = 4
    correction: bool = False

    @classmethod
    def from_config(cls, cfg: Mapping) -> TokenLayout:
        """Instantiate TokenLayout from configuration mapping."""
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

    # sizes
    @property
    def num_tokens(self) -> int:
        """Total number of spatial tokens."""
        return self.grid[0] * self.grid[1]

    @property
    def color_size(self) -> tuple[int, int]:
        """Spatial dimensions (height, width) of the color map."""
        return self.grid[0] * self.block, self.grid[1] * self.block

    @property
    def segments(self) -> dict[str, slice]:
        """Slice ranges for token segments: brightness, color, and optional correction."""
        bright = NUM_COEFFS * self.num_bins
        color = NUM_AB * self.block**2
        out = {"brightness": slice(0, bright), "color": slice(bright, bright + color)}
        if self.correction:
            out["correction"] = slice(bright + color, bright + color + self.block**2)
        return out

    @property
    def dim(self) -> int:
        """Total number of channels per token."""
        return max(seg.stop for seg in self.segments.values())

    # packing
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
        """Pack raw part tensors into flattened tokens of shape (B, N, D).

        Args:
            parts: Mapping of part names ('grid', 'ab', optional 'corr') to tensors.

        Returns:
            Packed token tensor of shape (B, N, D).

        Raises:
            ValueError: If part shapes do not match expected layout.
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
        """Unpack token tensor of shape (B, N, D) into structured part tensors.

        Args:
            tokens: Packed token tensor of shape (B, N, D).

        Returns:
            Dictionary of unpacked part tensors.

        Raises:
            ValueError: If token shape does not match layout dimensions.
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

    # stats and weights
    def _segment_stat_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {"brightness": (NUM_COEFFS, self.num_bins), "color": (NUM_AB,)}
        if self.correction:
            shapes["correction"] = (1,)
        return shapes

    def stats_vector(
        self, stats: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten per-segment normalization moments into 1D mean and std vectors of shape (D,).

        Args:
            stats: Nested dictionary mapping segment names to 'mean' and 'std' tensors.

        Returns:
            Tuple of (mean, std) vectors of shape (D,).

        Raises:
            ValueError: If required segment statistics are missing or misshapen.
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
        """Construct loss weighting tensor of shape (B, N, D) normalized per sample.

        Args:
            node_weights: Grid fitting sample counts of shape (B, K, gy, gx).
            fraction: Mask coverage fraction tensor of shape (B, 1, hc, wc).

        Returns:
            Loss weights tensor of shape (B, N, D).
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

    # checkpoints
    def descriptor(self) -> dict:
        """Serialize layout metadata for checkpoint compatibility checks."""
        return {
            "num_bins": self.num_bins,
            "grid": list(self.grid),
            "block": self.block,
            "correction": self.correction,
            "dim": self.dim,
        }

    @classmethod
    def from_descriptor(cls, descriptor: Mapping) -> TokenLayout:
        """Instantiate TokenLayout from descriptor dictionary."""
        return cls(
            num_bins=int(descriptor["num_bins"]),
            grid=(int(descriptor["grid"][0]), int(descriptor["grid"][1])),
            block=int(descriptor["block"]),
            correction=bool(descriptor["correction"]),
        )

    def check(self, descriptor: Mapping) -> None:
        """Verify compatibility with a stored layout descriptor."""
        if self.descriptor() != dict(descriptor):
            raise ValueError(
                f"token layout mismatch: this run has {self.descriptor()}, the checkpoint was "
                f"trained with {dict(descriptor)} (different resolution, bins, block or "
                "brightness_correction)"
            )


def _mean_one(weights: torch.Tensor) -> torch.Tensor:
    """Normalize weights to mean 1 along token and feature dimensions per sample."""
    mean = weights.mean(dim=(1, 2), keepdim=True)
    return torch.where(mean > 0, weights / mean.clamp(min=torch.finfo(weights.dtype).tiny), 0.0)
