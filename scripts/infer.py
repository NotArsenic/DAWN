#!/usr/bin/env python3
"""Run inference on thermal frames to generate daytime RGB and uncertainty maps."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.data.color import linear_to_srgb8  # noqa: E402
from dawn.data.dronevehicle import FRAME_SIZE, crop_border, read_thermal8  # noqa: E402
from dawn.data.resample import area_downsample  # noqa: E402
from dawn.flow.inference import InferenceBundle, load_checkpoint, load_refiner  # noqa: E402
from dawn.models.refiner import LocalRefiner, apply_refiner  # noqa: E402
from dawn.uncertainty.spread import SpreadResult, sample_with_spread  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_inputs(paths: list[str], limit: int | None) -> list[Path]:
    """Find and return image paths from files or directories up to an optional limit."""
    files = set()
    for p_str in paths:
        p = Path(p_str)
        if p.is_file():
            if p.suffix.lower() in SUFFIXES:
                files.add(p)
        elif p.is_dir():
            for f in p.rglob("*"):
                if f.suffix.lower() in SUFFIXES:
                    files.add(f)
        else:
            raise FileNotFoundError(f"no such file or directory: {p}")
    if not files:
        raise FileNotFoundError(f"no images found in {paths}")
    files = sorted(list(files))
    return files[:limit] if limit else files


def read_thermal(path: Path, input_size: tuple[int, int]) -> torch.Tensor:
    """Read a thermal frame, crop borders if needed, and scale to the input resolution."""
    thermal = read_thermal8(path).to(torch.float32) / 255.0
    if tuple(thermal.shape[-2:]) == FRAME_SIZE:
        thermal = crop_border(thermal)
    height, width = thermal.shape[-2:]
    if (height, width) != input_size:
        if height < input_size[0] or width < input_size[1]:
            raise ValueError(
                f"{path} is {width}x{height}, smaller than the checkpoint's input "
                f"{input_size[1]}x{input_size[0]}; Dawn does not upsample thermal"
            )
        thermal = area_downsample(thermal, input_size)
    return thermal.unsqueeze(0)


def colorize(value: torch.Tensor, scale: float) -> np.ndarray:
    """Map a single-channel uncertainty map to an 8-bit RGB heatmap using a fixed scale."""
    from matplotlib import colormaps

    normalized = (value[0, 0].clamp(min=0.0) / scale).clamp(0.0, 1.0).cpu().numpy()
    return (colormaps["magma"](normalized)[..., :3] * 255).round().astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    """Save an array as a PNG image."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def to_srgb8(linear: torch.Tensor) -> np.ndarray:
    """Convert a linear RGB tensor to an 8-bit sRGB array."""
    return linear_to_srgb8(linear)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def write_outputs(out: SpreadResult, stem: str, directory: Path, args) -> None:
    """Write output RGB images and uncertainty maps to disk."""
    save_png(to_srgb8(out.sample_rgb), directory / f"{stem}_rgb.png")
    if args.save_mean:
        save_png(to_srgb8(out.rgb), directory / f"{stem}_mean.png")
    if out.num_samples > 1:
        save_png(colorize(out.l_std, args.l_scale), directory / f"{stem}_l_std.png")
        save_png(colorize(out.ab_std, args.ab_scale), directory / f"{stem}_ab_std.png")
        if args.save_raw:
            np.savez_compressed(
                directory / f"{stem}_uncertainty.npz",
                l_std=out.l_std[0, 0].cpu().numpy(),
                ab_std=out.ab_std[0, 0].cpu().numpy(),
            )


@torch.no_grad()
def run_one(
    bundle: InferenceBundle,
    path: Path,
    device: torch.device,
    args,
    refiner: LocalRefiner | None = None,
) -> tuple[SpreadResult, SpreadResult, torch.Tensor]:
    """Run model inference on a single thermal frame, optionally applying refinement."""
    thermal = read_thermal(path, bundle.input_size).to(device)
    encoded = bundle.encoder.encode(thermal, return_attn=True)
    out = sample_with_spread(
        bundle.velocity,
        encoded,
        bundle.layout,
        bundle.mean,
        bundle.std,
        thermal,
        bundle.input_size,
        bundle.settings,
        num_samples=args.samples,
        method=args.method,
        nfe=args.nfe,
        image_ids=[path.stem] if args.seed_from_name else None,
    )
    member = apply_refiner(out.sample_lab, thermal, refiner)
    ensemble = apply_refiner(out.lab, thermal, refiner)
    refined = replace(
        out,
        sample_lab=member["lab"],
        sample_rgb=member["rgb"],
        lab=ensemble["lab"],
        rgb=ensemble["rgb"],
    )
    return out, refined, member["residual"]


def main() -> None:
    """Run model inference CLI."""
    parser = argparse.ArgumentParser(
        description="Dawn inference: thermal -> day RGB + uncertainty maps"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained flow checkpoint")
    parser.add_argument(
        "--refiner",
        type=str,
        default=None,
        help="Stage 2 refiner checkpoint; without it the Stage 1 render is saved as it is",
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="Thermal image(s), or a directory"
    )
    parser.add_argument("--output", type=str, default="outputs/infer", help="Where to write")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N of a directory")
    parser.add_argument(
        "--samples", type=int, default=8, help="K members for the uncertainty maps; 1 skips them"
    )
    parser.add_argument("--nfe", type=int, default=2, help="Velocity evaluations per member")
    parser.add_argument(
        "--method", choices=("euler", "midpoint"), default="euler", help="ODE sampler"
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use the live weights of both stages, not their EMA averages",
    )
    parser.add_argument(
        "--seed-from-name",
        action="store_true",
        help="Draw z_0 from the file stem, so a frame renders identically on any machine",
    )
    parser.add_argument("--save-mean", action="store_true", help="Also write the ensemble mean")
    parser.add_argument("--save-raw", action="store_true", help="Also write the maps as .npz")
    parser.add_argument(
        "--l-scale", type=float, default=10.0, help="L* units at the top of the uncertainty ramp"
    )
    parser.add_argument(
        "--ab-scale", type=float, default=10.0, help="a*b* units at the top of the ramp"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    bundle = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    logger.info(
        "checkpoint step %d, input %s, layout %s, %s upsampling, EMA %s",
        bundle.step,
        bundle.input_size,
        bundle.layout.descriptor(),
        bundle.settings.upsample,
        bundle.ema,
    )

    refiner = None
    if args.refiner:
        refiner = load_refiner(args.refiner, device, use_ema=not args.no_ema)
        logger.info(
            "refiner: %d parameters, %d px receptive field, delta bounded to +/- %.1f L*; the "
            "uncertainty maps stay those of the UNREFINED ensemble",
            refiner.num_params(),
            refiner.receptive_field,
            100.0 * refiner.residual_scale,
        )

    files = find_inputs(args.input, args.limit)
    directory = Path(args.output)
    logger.info(
        "%d frame(s) -> %s at K=%d, %s, NFE %d",
        len(files),
        directory,
        args.samples,
        args.method,
        args.nfe,
    )

    times = []
    for index, path in enumerate(files, 1):
        start = time.perf_counter()
        unrefined, refined, residual = run_one(bundle, path, device, args, refiner)
        times.append(time.perf_counter() - start)

        write_outputs(unrefined, path.stem, directory, args)
        if refiner is not None:
            write_outputs(refined, path.stem, directory / "refined", args)

        refined_str = (
            "" if refiner is None else f", refiner |dL*| {100 * float(residual.abs().mean()):.2f}"
        )
        if unrefined.num_samples > 1:
            logger.info(
                "[%d/%d] %s  mean L* spread %.2f, a*b* spread %.2f%s  (%.2f s)",
                index,
                len(files),
                path.name,
                float(unrefined.l_std.mean()),
                float(unrefined.ab_std.mean()),
                refined_str,
                times[-1],
            )
        else:
            logger.info(
                "[%d/%d] %s%s  (%.2f s)", index, len(files), path.name, refined_str, times[-1]
            )

    logger.info(
        "done: %d frame(s), %.2f s/frame (this is not a latency benchmark: see "
        "scripts/benchmark_latency.py)",
        len(files),
        sum(times) / max(1, len(times)),
    )


if __name__ == "__main__":
    main()
