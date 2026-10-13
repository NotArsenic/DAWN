"""Local residual refiner for lightness enhancement."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from dawn.data.color import lab_to_linear
from dawn.data.resample import area_downsample
from dawn.models.denoiser import DepthwiseSeparableConv

IN_CHANNELS = 2  # L and thermal, concatenated on the channel axis


class LocalRefiner(nn.Module):
    """Bounded, depthwise-separable convolutional network for lightness refinement.

    Args:
        hidden_channels: Number of hidden feature channels.
        num_blocks: Number of depthwise-separable convolution blocks.
        kernel_size: Kernel size for depthwise convolutions.
        residual_scale: Maximum residual magnitude applied to lightness.
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
        """Instantiate refiner from configuration mapping."""
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
        """Receptive field side length in pixels."""
        return 1 + self.num_blocks * (self.kernel_size - 1)

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def residual(self, lightness: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        """Compute bounded lightness residual delta.

        Args:
            lightness: Rendered lightness tensor of shape (B, 1, H, W) in [0, 1].
            thermal: Thermal guide tensor of shape (B, 1, H, W) in [0, 1].

        Returns:
            Residual tensor of shape (B, 1, H, W) in [-residual_scale, residual_scale].
        """
        pair = torch.cat([lightness.to(torch.float32), thermal.to(torch.float32)], dim=1)
        return self.residual_scale * torch.tanh(self.net(pair))

    def forward(self, lightness: torch.Tensor, thermal: torch.Tensor) -> torch.Tensor:
        """Refine lightness map with bounded residual.

        Args:
            lightness: Rendered lightness tensor of shape (B, 1, H, W) in [0, 1].
            thermal: Thermal guide tensor of shape (B, 1, H, W) in [0, 1].

        Returns:
            Refined lightness tensor of shape (B, 1, H, W) clamped to [0, 1].
        """
        return (lightness.to(torch.float32) + self.residual(lightness, thermal)).clamp(0.0, 1.0)


def apply_refiner(
    lab: torch.Tensor,
    thermal: torch.Tensor,
    refiner: LocalRefiner | None,
) -> dict[str, torch.Tensor]:
    """Apply lightness refiner to rendered Lab image.

    Args:
        lab: Lab image tensor of shape (B, 3, H, W).
        thermal: Thermal image tensor of shape (B, 1, H, W).
        refiner: LocalRefiner instance or None for identity pass.

    Returns:
        Dictionary with refined 'lab', linear 'rgb', and 'residual' tensors.
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
        residual = refiner.residual(lightness, thermal)
        lab = torch.cat([100.0 * (lightness + residual).clamp(0.0, 1.0), lab[:, 1:]], dim=1)
    return {"lab": lab, "rgb": lab_to_linear(lab).clamp(0.0, 1.0), "residual": residual}

