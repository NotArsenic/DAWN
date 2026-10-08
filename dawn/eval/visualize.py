"""Comparison grids: thermal | generated | real RGB | uncertainty.

The columns are the argument the project has to make visually. Thermal is one channel and is
shown as grey; the generated and real RGB sit next to each other; the two spread maps say where
the model disagreed with itself. Side-by-side is also the only honest way to read CIEDE2000,
which punishes a plausible colour that is not the true one.

Everything arrives as **linear** RGB and leaves as sRGB, because that is the output boundary
(docs/conventions.md section 1). Uncertainty maps are coloured on a **fixed** `[0, vmax]` ramp,
never per-image: two frames' maps are only comparable on a shared scale, and normalizing each
one on its own makes the most confident frame look as uncertain as the worst.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from dawn.data.color import linear_to_srgb

# Defaults matched to scripts/infer.py, so a grid and a saved map mean the same thing.
L_SCALE = 10.0  # L* units at the top of the ramp
AB_SCALE = 10.0  # a*b* units


def colorize(value: torch.Tensor, vmax: float, cmap: str = "magma") -> torch.Tensor:
    """`(B, 1, H, W)` single-channel map -> `(B, 3, H, W)` sRGB in `[0, 1]`, on a FIXED ramp.

    Args:
        value: the map; negatives are clamped to 0.
        vmax: the value at the top of the ramp. Anything above it saturates rather than
            rescaling the image.
        cmap: any matplotlib colormap name.

    Returns:
        `(B, 3, H, W)` float32 in `[0, 1]`.

    Raises:
        ValueError: `value` is not `(B, 1, H, W)`, or `vmax` is not positive.
    """
    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError(f"colorize needs a (B, 1, H, W) map, got {tuple(value.shape)}")
    if vmax <= 0:
        raise ValueError(f"vmax must be positive, got {vmax}")
    from matplotlib import colormaps

    normalized = (value.detach().float().clamp(min=0.0) / vmax).clamp(0.0, 1.0)
    table = torch.as_tensor(
        colormaps[cmap](torch.linspace(0, 1, 256).numpy())[:, :3], dtype=torch.float32
    )
    idx = (normalized[:, 0] * 255).round().long().clamp(0, 255)
    return table.to(value.device)[idx].permute(0, 3, 1, 2).contiguous()


def grey_to_rgb(thermal: torch.Tensor) -> torch.Tensor:
    """`(B, 1, H, W)` thermal codes in `[0, 1]` -> `(B, 3, H, W)`, repeated, not colour-mapped.

    Thermal is not sRGB and takes no gamma (docs/conventions.md section 1), so the codes are
    shown as they are. A false-colour thermal would imply a temperature scale this project does
    not have.
    """
    if thermal.ndim != 4 or thermal.shape[1] != 1:
        raise ValueError(f"expected (B, 1, H, W) thermal, got {tuple(thermal.shape)}")
    return thermal.detach().float().clamp(0.0, 1.0).expand(-1, 3, -1, -1)


def make_comparison_grid(
    thermal: torch.Tensor,
    generated: torch.Tensor,
    real: torch.Tensor,
    l_std: torch.Tensor | None = None,
    ab_std: torch.Tensor | None = None,
    pad: int = 2,
    l_scale: float = L_SCALE,
    ab_scale: float = AB_SCALE,
) -> torch.Tensor:
    """One row per pair: thermal | generated | real RGB [| L\\* spread | a\\*b\\* spread].

    Args:
        thermal: `(N, 1, H, W)` thermal codes in `[0, 1]`.
        generated, real: `(N, 3, H, W)` **linear** RGB in `[0, 1]`.
        l_std: `(N, 1, H, W)` L\\* spread, in L\\* units. Omitted if None.
        ab_std: `(N, 1, H, W)` a\\*b\\* spread, in a\\*b\\* units. Omitted if None.
        pad: white pixels between panels.
        l_scale, ab_scale: the tops of the two fixed colour ramps.

    Returns:
        `(3, N*(H+pad)-pad, C*(W+pad)-pad)` **sRGB** in `[0, 1]`, ready to save.

    Raises:
        ValueError: the panels disagree on batch size or spatial size.
    """
    panels = [grey_to_rgb(thermal), _srgb(generated), _srgb(real)]
    if l_std is not None:
        panels.append(colorize(l_std, l_scale))
    if ab_std is not None:
        panels.append(colorize(ab_std, ab_scale))

    shapes = {(p.shape[0], *p.shape[-2:]) for p in panels}
    if len(shapes) != 1:
        raise ValueError(f"panels disagree on (N, H, W): {sorted(shapes)}")
    n, h, w = panels[0].shape[0], *panels[0].shape[-2:]
    cols = len(panels)
    out = torch.ones(3, n * (h + pad) - pad, cols * (w + pad) - pad)
    for r in range(n):
        for c, panel in enumerate(panels):
            y, x = r * (h + pad), c * (w + pad)
            out[:, y : y + h, x : x + w] = panel[r].cpu()
    return out


def _srgb(linear: torch.Tensor) -> torch.Tensor:
    if linear.ndim != 4 or linear.shape[1] != 3:
        raise ValueError(f"expected (N, 3, H, W) linear RGB, got {tuple(linear.shape)}")
    return linear_to_srgb(linear.detach().float().clamp(0.0, 1.0))


def save_grid(grid: torch.Tensor, path: str | Path) -> Path:
    """Write a `(3, H, W)` sRGB grid as PNG, through a temporary name."""
    import os

    import numpy as np
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (grid.clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    partial = path.with_name(f"{path.stem}.partial.png")
    Image.fromarray(array).save(partial, format="PNG")
    os.replace(partial, path)
    return path


def save_comparison_sheets(
    output_dir: str | Path,
    thermal: torch.Tensor,
    generated: torch.Tensor,
    real: torch.Tensor,
    l_std: torch.Tensor | None = None,
    ab_std: torch.Tensor | None = None,
    image_ids: Sequence[str] | None = None,
    rows_per_sheet: int = 4,
    prefix: str = "examples",
    **grid_kwargs,
) -> list[Path]:
    """Split a batch into sheets of `rows_per_sheet` rows and write each as a PNG.

    Args:
        output_dir: where the sheets go.
        thermal, generated, real, l_std, ab_std: as `make_comparison_grid`.
        image_ids: one per pair; written alongside as a `.txt` so a row can be traced back to
            the frame it came from. A grid with no labels is hard to act on.
        rows_per_sheet: pairs per sheet.
        prefix: file stem; sheets are `<prefix>_000.png`, `_001.png`, ...

    Returns:
        the paths written, in order.
    """
    output_dir = Path(output_dir)
    total = thermal.shape[0]
    paths: list[Path] = []
    for sheet, start in enumerate(range(0, total, rows_per_sheet)):
        sl = slice(start, min(start + rows_per_sheet, total))
        grid = make_comparison_grid(
            thermal[sl],
            generated[sl],
            real[sl],
            None if l_std is None else l_std[sl],
            None if ab_std is None else ab_std[sl],
            **grid_kwargs,
        )
        paths.append(save_grid(grid, output_dir / f"{prefix}_{sheet:03d}.png"))
        if image_ids is not None:
            (output_dir / f"{prefix}_{sheet:03d}.txt").write_text(
                "\n".join(image_ids[sl]) + "\n", encoding="utf-8"
            )
    return paths
