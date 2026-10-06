"""The one render path: raw parts -> L*a*b* -> linear RGB (docs/conventions.md section 6.4).

L_render (at `flow.render_size`), validation and the oracle (at the input size) and inference all
call `render`; none re-implements a step of it.

    1. thermal_s = area_downsample(thermal, size) if the thermal is larger.
    2. g = compute_guidance(thermal_s); L_grid = clamp(gain * thermal_s + offset, 0, 1), sliced
       from the brightness grid.
    3. With the correction: L = clamp(L_grid + BGU(corr, area(L_grid), L_grid), 0, 1);
       otherwise L = L_grid.
    4. ab = BGU(ab_lo, area(L), L): colour is guided by the OUTPUT L, so its edges follow
       brightness edges.
    5. lab = (100 * L, ab); rgb = clamp(lab_to_linear(lab), 0, 1) (clipped to the gamut).

The grid and colour tokens are in RAW units here (`unstandardize` has already run).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from dawn.data.color import lab_to_linear
from dawn.data.resample import area_downsample
from dawn.flow.tokens import TokenLayout
from dawn.grid.guidance import compute_guidance
from dawn.grid.slice import apply_gain_offset, slice_grid
from dawn.grid.upsample import bilateral_guided_upsample


@dataclass(frozen=True)
class RenderSettings:
    """What `render` needs besides the parts: the token layout and the BGU settings.

    Args:
        layout: The token layout (colour-map size, whether the correction is on).
        bgu_grid: `(bins, rows, cols)` of the bilateral grid inside the upsampling.
        bgu_ridge: Ridge strength of the BGU fit.
        bgu_refine: Extra BGU fit passes.
    """

    layout: TokenLayout
    bgu_grid: tuple[int, int, int] = (8, 16, 20)
    bgu_ridge: float = 1e-3
    bgu_refine: int = 0

    @classmethod
    def from_config(cls, cfg: Mapping) -> RenderSettings:
        """Build from the parsed flow.yaml (`color.bgu`, plus the layout's own sections)."""
        bgu = cfg["color"]["bgu"]
        return cls(
            layout=TokenLayout.from_config(cfg),
            bgu_grid=(int(bgu["bins"]), *map(int, bgu["grid"])),
            bgu_ridge=float(bgu["ridge_alpha"]),
            bgu_refine=int(bgu["refine_steps"]),
        )


def render(
    parts: Mapping[str, torch.Tensor],
    thermal: torch.Tensor,
    size: tuple[int, int],
    settings: RenderSettings,
) -> dict[str, torch.Tensor]:
    """Render raw parts to an image at `size`.

    Args:
        parts: raw `grid (B, 2, K, gy, gx)`, `ab (B, 2, hc, wc)` and, if the layout has the
            correction, `corr (B, 1, hc, wc)`.
        thermal: (B, 1, H, W) thermal, `code / 255`, at least `size`.
        size: `(h, w)` to render at; at least the colour-map size, at most the thermal size.
        settings: layout and BGU settings.

    Returns:
        dict of float32 tensors: `lab (B, 3, h, w)` L*, a*, b*; `rgb (B, 3, h, w)` linear,
        clamped to [0, 1]; `L_grid (B, 1, h, w)` the brightness grid's L = L*/100 (before any
        correction). Differentiable in `parts`.

    Raises:
        ValueError: the parts do not match the layout, or `size` is not between the colour-map
            size and the thermal size.
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
        return bilateral_guided_upsample(
            low.to(torch.float32),
            area_downsample(guide, layout.color_size),
            guide,
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
