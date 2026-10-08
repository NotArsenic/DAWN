#!/usr/bin/env python3
"""Run a trained Dawn model on thermal frames: day RGB plus two uncertainty maps.

The inference path of the plan, in order: read the thermal frame, run DINO once, draw K samples
with 2 velocity evaluations each, render every member, and reduce them to one image and the
spread of the ensemble (docs/conventions.md sections 6.4 and 11). The encoder runs once per
frame however large K is, so K samples cost `nfe * K` velocity passes and nothing else.

Everything about the model comes out of the checkpoint -- the token layout, the resolution, the
upsampling mode and especially the normalization stats, which must never be read from
`grid.stats_cache` (section 5.1). The only things this script decides are K, the sampler and
where the files go.

Usage:
    python scripts/infer.py --checkpoint checkpoints/run1/flow_last.pt --input frame.jpg
    python scripts/infer.py --checkpoint ... --input data/DroneVehicle/test/testimgr --limit 20
    python scripts/infer.py --checkpoint ... --input frames/ --samples 8 --method midpoint
    python scripts/infer.py --checkpoint ... --refiner checkpoints/refiner/refiner_last.pt ...

Inputs may be raw DroneVehicle frames (840x712 with the 100-px border, cropped here) or frames
already at the checkpoint's input size. The local refiner of the plan's step 6 is optional
(`--refiner`, a Stage 2 checkpoint) and runs between the render and the save: it sharpens the
rendered brightness and leaves a* and b* alone. Without it the saved image is exactly what
Stage 1 renders.

**The refiner does not touch the uncertainty maps, by choice.** It is applied to the two images
this script writes -- the displayed member and the ensemble mean -- and not to the K members, so
`l_std` and `ab_std` stay the spread of the *unrefined* ensemble: the maps that come out of a
`--refiner` run describe the unrefined ensemble, and should be read that way. The reasoning is
that the refiner is deterministic given `L`, so it adds no disagreement between members; refining
each member would move `l_std` only through the refiner's own nonlinearity, a change in the
reported spread that no extra uncertainty in the model stands behind. And docs/conventions.md
section 11 makes `dawn/uncertainty/spread.py` the single place the spread is computed, which this
path leaves untouched rather than recomputing a second spread here.
"""

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
    """One or more files/directories, sorted so runs are reproducible."""
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
    """`(1, 1, H, W)` thermal, `code / 255`, at `input_size` (docs/conventions.md section 1).

    A raw 840x712 frame loses its border first. Anything larger than the model's input is area
    averaged down (section 4a); anything smaller is an error, because upsampling a thermal frame
    would invent the texture the grid is guided by.
    """
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
    """A `(1, 1, h, w)` map to an 8-bit RGB image on a FIXED `[0, scale]` ramp.

    Fixed, not per-image: two frames' uncertainty maps are only comparable if they share a scale,
    and a per-image normalization would make the most confident frame look as uncertain as the
    worst. Values above `scale` saturate.
    """
    from matplotlib import colormaps

    normalized = (value[0, 0].clamp(min=0.0) / scale).clamp(0.0, 1.0).cpu().numpy()
    return (colormaps["magma"](normalized)[..., :3] * 255).round().astype(np.uint8)


def save_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG")


def to_srgb8(linear: torch.Tensor) -> np.ndarray:
    """`(1, 3, h, w)` linear RGB -> `(h, w, 3)` uint8 sRGB (section 1, output boundary)."""
    return linear_to_srgb8(linear)[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def write_outputs(out: SpreadResult, stem: str, directory: Path, args) -> None:
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
    """One frame, end to end: read, one DINO pass, K members, render, then step 6.

    Returns the unrefined result, the refined result, and the refiner's residual `(1, 1, h, w)` in `L = L*/100` units, so the
    caller can report whether Stage 2 moved anything; with no refiner the residual is zeros.
    """
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
    # Step 6, between the render and the save, on the two IMAGES only: the displayed member and
    # the ensemble mean. `l_std` and `ab_std` are carried through untouched on purpose -- the
    # refiner is deterministic given L, so it adds no member-to-member disagreement, and the
    # spread stays the one dawn/uncertainty/spread.py computed over the unrefined members
    # (docs/conventions.md section 11, and the module docstring above). `apply_refiner` with no
    # refiner is an exact identity, so this costs the no-refiner run nothing but the Lab -> RGB
    # it would have done anyway.
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
    parser.add_argument("--input", nargs="+", required=True, help="Thermal image(s), or a directory")
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

        # The residual is in L = L*/100 units; report L*, the unit the maps and the metrics use,
        # so "the refiner did nothing" is visible per frame rather than only in the images.
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
            logger.info("[%d/%d] %s%s  (%.2f s)", index, len(files), path.name, refined_str, times[-1])

    logger.info(
        "done: %d frame(s), %.2f s/frame (this is not a latency benchmark: see "
        "scripts/benchmark_latency.py)",
        len(files),
        sum(times) / max(1, len(times)),
    )


if __name__ == "__main__":
    main()
