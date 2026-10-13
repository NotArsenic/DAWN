"""Image rendering pipeline from predicted token parts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from dawn.data.color import lab_to_linear
from dawn.data.resample import area_downsample
from dawn.flow.tokens import TokenLayout
from dawn.grid.guidance import compute_guidance
from dawn.grid.slice import apply_gain_offset, slice_grid
from dawn.grid.upsample import UPSAMPLE_MODES, upsample_map


@dataclass(frozen=True)
class RenderSettings:
    """Configuration settings for rendering and upsampling."""

    layout: TokenLayout
    bgu_grid: tuple[int, int, int] = (8, 64, 80)
    bgu_ridge: float = 1e-1
    bgu_refine: int = 2
    upsample: str = "bilinear"

    def __post_init__(self) -> None:
        if self.upsample not in UPSAMPLE_MODES:
            raise ValueError(f"upsample must be one of {UPSAMPLE_MODES}, got {self.upsample!r}")

    @classmethod
    def from_config(cls, cfg: Mapping) -> RenderSettings:
        """Instantiate RenderSettings from configuration mapping."""
        bgu = cfg["color"]["bgu"]
        return cls(
            layout=TokenLayout.from_config(cfg),
            bgu_grid=(int(bgu["bins"]), *map(int, bgu["grid"])),
            bgu_ridge=float(bgu["ridge_alpha"]),
            bgu_refine=int(bgu["refine_steps"]),
            upsample=str(cfg["color"]["upsample"]),
        )


def render(
    parts: Mapping[str, torch.Tensor],
    thermal: torch.Tensor,
    size: tuple[int, int],
    settings: RenderSettings,
) -> dict[str, torch.Tensor]:
    """Render raw model parts to full Lab and RGB images at target size.

    Args:
        parts: Mapping containing 'grid', 'ab', and optional 'corr' tensors.
        thermal: Input thermal guide tensor of shape (B, 1, H, W).
        size: Target spatial dimensions (h, w) for rendering.
        settings: RenderSettings instance specifying layout and upsampling modes.

    Returns:
        Dictionary containing rendered 'lab', clamped linear 'rgb', and 'L_grid' tensors.

    Raises:
        ValueError: If parts do not match layout or size constraints are violated.
    """
    layout = settings.layout
    size = (int(size[0]), int(size[1]))
    if ("corr" in parts) != layout.correction:
        raise ValueError(
            f"parts {sorted(parts)} do not match the layout: 'corr' must be present exactly when "
            f"brightness_correction is on (it is {'on' if layout.correction else 'off'})"
        )
    if size[0] > thermal.shape[-2] or size[1] > thermal.shape[-1]:
        raise ValueError(
            f"render size {size} is larger than the thermal {tuple(thermal.shape[-2:])}"
        )
    if size[0] < layout.color_size[0] or size[1] < layout.color_size[1]:
        raise ValueError(f"render size {size} is smaller than the color map {layout.color_size}")

    thermal = thermal.to(torch.float32)
    if tuple(thermal.shape[-2:]) != size:
        thermal = area_downsample(thermal, size)
    grid = parts["grid"].to(torch.float32)

    guidance = compute_guidance(thermal)
    l_grid = apply_gain_offset(slice_grid(grid, guidance), thermal)  # (B, 1, h, w), clamped
    lightness = l_grid

    def upsample(low: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        return upsample_map(
            low.to(torch.float32),
            area_downsample(guide, layout.color_size),
            guide,
            settings.upsample,
            grid_size=settings.bgu_grid,
            ridge_alpha=settings.bgu_ridge,
            refine_steps=settings.bgu_refine,
        )

    if layout.correction:
        lightness = (l_grid + upsample(parts["corr"], l_grid)).clamp(0.0, 1.0)
    ab = upsample(parts["ab"], lightness)
    lab = torch.cat([100.0 * lightness, ab], dim=1)
    rgb = lab_to_linear(lab).clamp(0.0, 1.0)
    return {"lab": lab, "rgb": rgb, "L_grid": l_grid}

