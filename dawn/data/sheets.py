"""Contact sheets and the process pool behind the data scripts.

Both were first written for the VisDrone preparation and are kept because every prepare / audit
script wants them: `run_parallel` maps a function over jobs on a spawn pool with one torch thread
per worker, and `tile_sheet` lays labelled thumbnails on a dark background.

Thumbnails shrink by area averaging (docs/conventions.md section 4a): RGB in linear light, then
back to sRGB codes for display; thermal as the plain code average, because thermal values are not
sRGB (section 1).
"""

from __future__ import annotations

import logging
import math
import multiprocessing
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from dawn.data.color import linear_to_srgb8
from dawn.data.resample import area_downsample

logger = logging.getLogger(__name__)

THUMB_BOX = (144, 192)  # thumbnails fit inside (height, width)
BACKGROUND = (26, 26, 25)
_TITLE_INK, _LABEL_INK = (235, 235, 230), (195, 194, 183)


def run_parallel(fn: Callable, jobs: Sequence, workers: int, desc: str = "") -> list:
    """`[fn(*job) for job in jobs]` on `workers` processes (in-process when `workers <= 1`).

    Results come back in job order. Each worker runs torch single-threaded, so `workers` equal
    to the number of cores uses every core without oversubscribing them. `fn` and the jobs must
    be picklable (a module-level function). Logs progress every 5% when `desc` is given.
    """
    jobs = list(jobs)
    total, results = len(jobs), []
    every = max(1, total // 20)

    def progress(done: int) -> None:
        if desc and (done % every == 0 or done == total):
            logger.info(f"{desc}: {done}/{total}")

    if workers <= 1 or total <= 1:
        for done, job in enumerate(jobs, 1):
            results.append(fn(*job))
            progress(done)
        return results
    # spawn, not fork: forking a process that has already started torch's thread pools can hang.
    context = multiprocessing.get_context("spawn")
    chunksize = max(1, min(32, total // (workers * 8)))
    with ProcessPoolExecutor(workers, mp_context=context, initializer=_one_thread) as pool:
        for done, result in enumerate(pool.map(fn, *zip(*jobs), chunksize=chunksize), 1):
            results.append(result)
            progress(done)
    return results


def _one_thread() -> None:
    torch.set_num_threads(1)


def load_font(size: int) -> ImageFont.ImageFont:
    """A scalable default font (Pillow >= 10.1), the bitmap default on older Pillow."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def save_srgb8(img: torch.Tensor, path: str | Path, **save_kwargs) -> None:
    """Save a `(3, H, W)` or `(1, H, W)` uint8 tensor as an image (mode RGB or L)."""
    array = img.permute(1, 2, 0).numpy()
    Image.fromarray(array[..., 0] if array.shape[-1] == 1 else array).save(path, **save_kwargs)


def _fit_size(height: int, width: int, box: tuple[int, int]) -> tuple[int, int]:
    scale = min(1.0, box[0] / height, box[1] / width)  # never upsample
    return max(1, round(height * scale)), max(1, round(width * scale))


def thumbnail(img_linear: torch.Tensor, box: tuple[int, int] = THUMB_BOX) -> torch.Tensor:
    """`(3, h, w)` uint8 sRGB thumbnail of a linear RGB image, area-averaged in linear light to
    fit `box = (height, width)`. Never upsamples.

    Args:
        img_linear: `(3, H, W)` float32 linear RGB in [0, 1].
    """
    size = _fit_size(*img_linear.shape[-2:], box)
    return linear_to_srgb8(area_downsample(img_linear, size))


def thumbnail_thermal(thermal: torch.Tensor, box: tuple[int, int] = THUMB_BOX) -> torch.Tensor:
    """`(3, h, w)` uint8 grey thumbnail of a thermal image: codes averaged as they are (thermal is
    not sRGB, docs/conventions.md section 1), then copied to three channels for display.

    Args:
        thermal: `(1, H, W)` float32 in [0, 1] (code / 255).
    """
    size = _fit_size(*thermal.shape[-2:], box)
    small = (area_downsample(thermal, size).clamp(0, 1) * 255).round().to(torch.uint8)
    return small.expand(3, -1, -1).contiguous()


def tile_sheet(
    thumbs: Sequence[torch.Tensor],
    labels: Sequence[str],
    title: str,
    columns: int = 10,
    box: tuple[int, int] = THUMB_BOX,
) -> Image.Image:
    """Thumbnails in a grid on a dark background, each with a caption under it.

    Args:
        thumbs: `(3, h, w)` uint8 tensors, each at most `box` = (height, width).
        labels: One caption per thumbnail.
        title: Printed across the top.
        columns: Tiles per row.

    Raises:
        ValueError: `labels` and `thumbs` differ in length.
    """
    if len(labels) != len(thumbs):
        raise ValueError(f"{len(thumbs)} thumbnails but {len(labels)} labels")
    (cell_h, cell_w), label_h, title_h, pad = box, 20, 34, 6
    rows = max(1, math.ceil(len(thumbs) / columns))
    size = (columns * (cell_w + pad) + pad, title_h + rows * (cell_h + label_h) + pad)
    sheet = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((pad, 8), title, fill=_TITLE_INK, font=load_font(18))
    label_font = load_font(13)
    for index, (thumb, label) in enumerate(zip(thumbs, labels)):
        row, column = divmod(index, columns)
        x0 = pad + column * (cell_w + pad)
        y0 = title_h + row * (cell_h + label_h)
        _, h, w = thumb.shape
        tile = Image.fromarray(thumb.permute(1, 2, 0).numpy())
        sheet.paste(tile, (x0 + (cell_w - w) // 2, y0))
        draw.text((x0 + 2, y0 + h + 3), label, fill=_LABEL_INK, font=label_font)
    return sheet
