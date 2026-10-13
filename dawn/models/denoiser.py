"""Bounded residual denoiser."""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable 2D convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=kernel_size // 2, groups=in_ch
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class BoundedDenoiser(nn.Module):
    """Compact residual network with bounded tanh output for image refinement."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 24,
        out_channels: int = 3,
        num_blocks: int = 2,
        kernel_size: int = 5,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.residual_scale = residual_scale

        layers = [DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size), nn.GELU()]
        for _ in range(num_blocks - 1):
            layers.extend(
                [
                    DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size),
                    nn.GELU(),
                ]
            )
        layers.append(nn.Conv2d(hidden_channels, out_channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        enhanced: torch.Tensor,
        log_variance: torch.Tensor,
    ) -> torch.Tensor:
        """Refine enhanced image using variance information.

        Args:
            enhanced: Enhanced linear RGB image of shape (B, 3, H, W).
            log_variance: Log variance map of shape (B, 1, H, W).

        Returns:
            Refined linear RGB image of shape (B, 3, H, W).
        """
        x = torch.cat([enhanced, log_variance], dim=1)
        residual = self.net(x)
        return (enhanced + self.residual_scale * torch.tanh(residual)).clamp(0.0, 1.0)

