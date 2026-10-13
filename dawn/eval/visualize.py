"""Visual comparison grid generation for qualitative evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from dawn.data.color import linear_to_srgb

L_SCALE = 10.0
AB_SCALE = 10.0


def colorize(value: torch.Tensor, vmax: float, cmap: str = "magma") -> torch.Tensor:
    """Colorize a single-channel map into an sRGB image tensor using a colormap.

    Args:
        value: (B, 1, H, W) single-channel input map.
        vmax: Maximum value corresponding to the top of the color ramp.
        cmap: Colormap name supported by matplotlib.

    Returns:
        (B, 3, H, W) float32 sRGB tensor in [0, 1].

    Raises:
        ValueError: If value is not (B, 1, H, W) or vmax is non-positive.
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
    """Convert a single-channel image tensor to 3-channel grayscale.

    Args:
        thermal: (B, 1, H, W) single-channel tensor in [0, 1].

    Returns:
        (B, 3, H, W) grayscale tensor.

    Raises:
        ValueError: If thermal is not (B, 1, H, W).
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
    """Build a side-by-side comparison grid of thermal, generated, and real images.

    Args:
        thermal: (N, 1, H, W) thermal image tensor in [0, 1].
        generated: (N, 3, H, W) generated linear RGB tensor in [0, 1].
        real: (N, 3, H, W) target linear RGB tensor in [0, 1].
        l_std: Optional (N, 1, H, W) lightness uncertainty map.
        ab_std: Optional (N, 1, H, W) chroma uncertainty map.
        pad: Pixel padding between image panels.
        l_scale: Lightness colormap maximum scale.
        ab_scale: Chroma colormap maximum scale.

    Returns:
        (3, H_grid, W_grid) composite sRGB grid tensor in [0, 1].

    Raises:
        ValueError: If input panels differ in batch size or spatial dimensions.
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
    """Save an sRGB grid tensor to disk as a PNG image atomically."""
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
    """Split comparison batches into multi-row PNG sheets and save to disk.

    Args:
        output_dir: Directory where sheets are saved.
        thermal: (N, 1, H, W) thermal image tensor.
        generated: (N, 3, H, W) generated linear RGB tensor.
        real: (N, 3, H, W) target linear RGB tensor.
        l_std: Optional (N, 1, H, W) lightness uncertainty map.
        ab_std: Optional (N, 1, H, W) chroma uncertainty map.
        image_ids: Optional sequence of identifier strings for each row.
        rows_per_sheet: Number of image pairs per sheet.
        prefix: Filename prefix for saved sheets.
        **grid_kwargs: Additional keyword arguments for make_comparison_grid.

    Returns:
        List of saved PNG file paths in order.
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
