"""The Stage 2 local refiner: a small bounded sharpener on L (the plan's "local refiner").

Thermal is softer than RGB: its optics and its 8-bit codes round off edges the daytime camera
resolves, so the brightness Stage 1 fits from thermal is right in the large and mushy at object
boundaries. Two depthwise-separable 5x5 blocks of width 24 (about 2.5K parameters) read `L` and
the thermal that produced it and predict `residual_scale * tanh(.)`: a delta of at most +/- 0.1
in `L = L*/100` units, i.e. +/- 10 L* (docs/conventions.md sections 1 and 2.2).

Three properties are the design, not side effects:

- **Brightness only.** It takes `L` and returns `L`; `apply_refiner` passes a* and b* through
  bit-identical. Colour comes from the flow's generated a*b* map, conditioned on DINO features
  this net cannot see, so letting a 9-pixel CNN move colour could only undo that work.
- **Bounded.** `tanh` caps the delta at +/- 10 L* whatever the weights learn, so the refiner can
  steepen an edge the grid rounded off but cannot paint a new object out of nothing. The plan
  accepts invented detail and says the bound may be relaxed, which is why `residual_scale` is a
  constructor argument and not a literal.
- **Local.** The receptive field is `1 + num_blocks * (kernel_size - 1)` = 9 px at the defaults;
  the 1x1 projections at both ends mix channels and add nothing to it. A DroneVehicle car is tens
  of pixels across, so the refiner cannot see a whole vehicle, which is the point: it can only
  act on the handful of pixels either side of an edge, which is all that sharpening needs.

It runs as step 6 of the plan's inference list, strictly after `dawn.flow.render.render` (the one
render path, docs/conventions.md section 6.4), never instead of a step of it. Stage 2 is also the
first thing the plan cuts, so `apply_refiner(lab, thermal, None)` is an exact identity pass: if no
refiner was trained, inference drops it and no caller needs a branch.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from dawn.data.color import lab_to_linear
from dawn.data.resample import area_downsample
from dawn.models.denoiser import DepthwiseSeparableConv

IN_CHANNELS = 2  # L and thermal, concatenated on the channel axis


class LocalRefiner(nn.Module):
    """A bounded, local, brightness-only sharpener: `(L, thermal) -> L`.

    `1x1 Conv(2 -> hidden)`, then `num_blocks` x `[DepthwiseSeparableConv(kernel_size), GELU]`,
    then `1x1 Conv(hidden -> 1)`. The separable blocks are the only layers with spatial reach, so
    the field is exactly `1 + num_blocks * (kernel_size - 1)`: 9 px and about 2.5K parameters at
    the defaults. That cost matters because the refiner runs once per 640x512 frame, where the
    plan budgets it (with the grid slice and the colour upsampling) 1-2 GMACs against DINO's 45.

    The output 1x1 is zero-initialized, so an untrained refiner predicts `tanh(0) = 0` and is the
    exact identity: a Stage 2 that was never run, or stopped early, degrades to Stage 1 rather
    than to noise, and the `None` path of `apply_refiner` is the same image as a fresh model.

    Args:
        hidden_channels: Width of the blocks (the plan's 24).
        num_blocks: Depthwise-separable blocks (the plan's 2); with `kernel_size` this is the
            receptive field, so it is the knob that decides what the refiner is allowed to see.
        kernel_size: Depthwise kernel side; odd, so the `k // 2` padding keeps the image size.
        residual_scale: Bound on the delta, in `L = L*/100` units: 0.1 is +/- 10 L*.
    """

    def __init__(
        self,
        hidden_channels: int = 24,
        num_blocks: int = 2,
        kernel_size: int = 5,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        self.residual_scale = float(residual_scale)

        width, kernel = self.hidden_channels, self.kernel_size
        layers: list[nn.Module] = [nn.Conv2d(IN_CHANNELS, width, 1)]
        for _ in range(self.num_blocks):
            layers += [DepthwiseSeparableConv(width, width, kernel), nn.GELU()]
        head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.net = nn.Sequential(*layers, head)

    @classmethod
    def from_config(cls, cfg: Mapping) -> LocalRefiner:
        """Build from the `refiner` section of configs/refiner.yaml.

        Reads `hidden_channels`, `num_blocks`, `kernel_size` and `residual_scale`; each is
        optional and falls through to the `__init__` default, so a config may pin only what it
        changes. Either the whole merged config (the `refiner` section is taken from it) or that
        section on its own is accepted, because the trainer holds the merged config while a
        checkpoint stores the section alone.
        """
        section = cfg["refiner"] if "refiner" in cfg else cfg
        casts = {
            "hidden_channels": int,
            "num_blocks": int,
            "kernel_size": int,
            "residual_scale": float,
        }
        given = {name: cast(section[name]) for name, cast in casts.items() if name in section}
        return cls(**given)

    @property
    def receptive_field(self) -> int:
        """Side of the square receptive field in pixels: `1 + num_blocks * (kernel_size - 1)`.

        The 1x1 convolutions at both ends contribute nothing, so this is the whole story: an
        output pixel is a function of its own `receptive_field` x `receptive_field` neighbourhood
        and of nothing else. 9 px at the defaults, far smaller than any vehicle.
        """
        return 1 + self.num_blocks * (self.kernel_size - 1)

    def num_params(self) -> int:
        """Total parameter count (about 2.5K at the defaults)."""
        return sum(p.numel() for p in self.parameters())

    def residual(self, lightness: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        """The bounded delta `residual_scale * tanh(net(cat([lightness, thermal])))`.

        Returned on its own, and not only folded into `forward`, because Stage 2's loss is
        `L1 to the RGB brightness + 0.01 * |residual|` (masked): the penalty needs the delta
        itself, and running the net twice to get it would double the cost of a training step.

        Args:
            lightness: (B, 1, H, W) `L = L*/100` in [0, 1], the rendered brightness.
            thermal: (B, 1, H, W) `code / 255` in [0, 1], the same size and never sRGB-converted
                (docs/conventions.md section 1). `apply_refiner` downsamples a larger one first.

        Returns:
            (B, 1, H, W) float32 in `[-residual_scale, residual_scale]`, in `L = L*/100` units,
            so the default bound is +/- 10 L*.
        """
        pair = torch.cat([lightness.to(torch.float32), thermal.to(torch.float32)], dim=1)
        return self.residual_scale * torch.tanh(self.net(pair))

    def forward(self, lightness: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        """The refined brightness `clamp(lightness + residual(lightness, thermal), 0, 1)`.

        The clamp is the usual one on anything that can leave `[0, 1]` (docs/conventions.md
        section 1): a +10 L* delta on an already-white pixel must not render above white.

        Args:
            lightness: (B, 1, H, W) `L = L*/100` in [0, 1].
            thermal: (B, 1, H, W) `code / 255` in [0, 1], the same size.

        Returns:
            (B, 1, H, W) float32 `L = L*/100` in [0, 1], within `residual_scale` of the input.
        """
        return (lightness.to(torch.float32) + self.residual(lightness, thermal)).clamp(0.0, 1.0)


def apply_refiner(
    lab: torch.Tensor,
    thermal: torch.Tensor,
    refiner: LocalRefiner | None,
) -> dict[str, torch.Tensor]:
    """Step 6 of the plan's inference list, strictly AFTER `flow.render.render`.

    The refiner sharpens the rendered brightness and nothing else, so this re-assembles L*a*b*
    from the new L* and the untouched a*b* and converts it once more to linear RGB. It is not a
    second render path (docs/conventions.md section 8, rule 9): every step up to `lab` already
    happened in `render`.

    Args:
        lab: (B, 3, h, w) L*, a*, b* from `render`. a*b* pass through untouched -- the refiner is
            brightness only.
        thermal: (B, 1, H, W) `code / 255`, area-downsampled here if larger than `lab` (the one
            downsampling function, docs/conventions.md section 4a). Never sRGB-converted.
        refiner: None is an identity pass, so callers need no branch: Stage 2 is optional and the
            plan's first cut.

    Returns:
        `lab` (B, 3, h, w), `rgb` (B, 3, h, w) linear clamped to [0, 1], `residual` (B, 1, h, w)
        in `L = L*/100` units (zeros when `refiner` is None).
    """
    size = (int(lab.shape[-2]), int(lab.shape[-1]))
    lab = lab.to(torch.float32)
    if refiner is None:
        residual = torch.zeros_like(lab[:, :1])
    else:
        thermal = thermal.to(torch.float32)
        if tuple(thermal.shape[-2:]) != size:
            thermal = area_downsample(thermal, size)
        lightness = lab[:, :1] / 100.0
        residual = refiner.residual(lightness, thermal)  # one net pass: `forward` is this + clamp
        lab = torch.cat([100.0 * (lightness + residual).clamp(0.0, 1.0), lab[:, 1:]], dim=1)
    return {"lab": lab, "rgb": lab_to_linear(lab).clamp(0.0, 1.0), "residual": residual}
