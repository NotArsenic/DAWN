#!/usr/bin/env python3
"""Sanity check: render daytime images through the synthetic nighttime degradation.

    python scripts/check_degradation.py --images data/day_samples

Point it at a folder of daytime photographs (or at individual files) and it writes, for each
one and each darkness level, the four images that make up a render:

    <id>_day.png           the source, linearized and written straight back
    <id>_<level>_clean.png night_clean: the same camera pipeline with every noise draw
                           replaced by its mean and no JPEG, so this is E[night]. It already
                           carries the exposure gain, the white balance and the ISP, so it is
                           NOT the scene radiance. This is what the grid is fitted to.
    <id>_<level>_night.png night: what the model sees. clean + sensor noise, 12-bit quantized,
                           then a JPEG round trip.
    <id>_<level>_noise.png the per-pixel standard deviation predicted by the reported (a, b),
                           normalized to its own maximum; the log prints that maximum.
    <id>_<level>_panel.png the four side by side, captioned.

The night render is seeded from the file stem and the level (docs/conventions.md section 10.3),
so rerunning this gives pixel-identical output, and two different levels of the same image are
independent draws rather than the same noise scaled.

By default it renders the NATIVE path: the input resolution, one native pixel per output pixel,
in-camera denoise and sharpening on. That is what a real camera would have delivered, and it is
the honest thing to look at. `--flow` instead renders the 256x256 view the flow model trains
on, where one pixel averages many native pixels and is correspondingly less noisy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LEVELS = ("easy", "medium", "hard")
CAPTIONS = {
    "day": "day (source)",
    "clean": "night_clean = E[night]",
    "night": "night (noise + 12-bit + JPEG)",
    "noise": "predicted noise sigma",
}


def collect_images(paths: list[str], count: int | None) -> list[Path]:
    """Every readable image under `paths`, sorted. Directories are searched one level deep.

    Raises:
        SystemExit: Nothing matched.
    """
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found += [p for p in sorted(path.iterdir()) if p.suffix.lower() in SUFFIXES]
        elif path.exists():
            found.append(path)
        else:
            logger.warning(f"Skipping {path}: not found")
    if not found:
        raise SystemExit(f"No images found in {', '.join(paths)} (looked for {sorted(SUFFIXES)})")
    return found[:count] if count else found


def load_day(path: Path, max_side: int | None):
    """One file as `(3, H, W)` linear float32, optionally area-downsampled to fit `max_side`."""
    from dawn.data.color import srgb8_to_linear
    from dawn.data.resample import area_downsample, fit_long_side
    from dawn.data.visdrone import read_srgb8

    image = srgb8_to_linear(read_srgb8(path))
    if max_side:
        _, height, width = image.shape
        target = fit_long_side(height, width, max_side)
        if target != (height, width):
            image = area_downsample(image, target)
    return image


def noise_map(result):
    """Per-pixel noise standard deviation as a grey `(3, H, W)` linear image, and its peak.

    Shown through the sRGB transfer like a photograph rather than scaled linearly, because the
    interesting structure is in the dark half: sigma follows sqrt(a*y + b), so it is nearly
    flat wherever the frame is bright and all its detail sits where the frame is not.
    """
    sigma = result.variance().clamp(min=0.0).sqrt().mean(dim=0, keepdim=True)
    peak = float(sigma.max())
    return (sigma / max(peak, 1e-12)).expand(3, -1, -1), peak


def save_panel(images: list, captions: list[str], path: Path) -> None:
    """The renders side by side on one strip, each captioned underneath."""
    from PIL import Image, ImageDraw

    from dawn.data.color import linear_to_srgb8
    from dawn.data.prepare import _font

    tiles = [
        Image.fromarray(linear_to_srgb8(image).permute(1, 2, 0).numpy()) for image in images
    ]
    pad, label_h = 8, 22
    width = sum(tile.width for tile in tiles) + pad * (len(tiles) + 1)
    height = max(tile.height for tile in tiles) + label_h + pad * 2
    strip = Image.new("RGB", (width, height), (26, 26, 25))
    draw = ImageDraw.Draw(strip)
    font = _font(14)
    x = pad
    for tile, caption in zip(tiles, captions):
        strip.paste(tile, (x, pad))
        draw.text((x + 2, pad + tile.height + 4), caption, fill=(220, 219, 210), font=font)
        x += tile.width + pad
    strip.save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Render day images through the synthetic night degradation",
        epilog="Put your daytime photographs in one folder and pass it to --images.",
    )
    parser.add_argument("--images", nargs="+", required=True,
                        help="Folders or files of DAYTIME images")
    parser.add_argument("--out", type=str, default="outputs/check_degradation",
                        help="Output directory (created)")
    parser.add_argument("--count", type=int, default=None, help="Use only the first N images")
    parser.add_argument("--levels", nargs="+", default=["medium"], choices=LEVELS,
                        help="Darkness levels to render per image")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["moonlit", "streetlit", "starlight"],
                        help="Force a lighting mode (default: chosen from the source image)")
    parser.add_argument("--flow", action="store_true",
                        help="Render the 256x256 training view instead of the native path")
    parser.add_argument("--max-side", type=int, default=1600,
                        help="Area-downsample sources longer than this; 0 keeps full size")
    parser.add_argument("--no-panel", dest="panel", action="store_false",
                        help="Skip the captioned side-by-side strip")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--calib-config", type=str, default="configs/camera_calib.yaml")
    args = parser.parse_args()

    from omegaconf import OmegaConf

    from dawn.data.color import linear_to_srgb8
    from dawn.data.degrade import degrade_image
    from dawn.data.prepare import _save_srgb8

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)["degradation"]
    calib = OmegaConf.to_container(OmegaConf.load(args.calib_config), resolve=True)
    if calib.get("source") != "measured":
        logger.warning(f"{args.calib_config} is {calib.get('source')}: the noise is plausible, "
                       "not fitted. Run scripts/calibrate_noise.py once the night frames exist.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_images(args.images, args.count)
    logger.info(f"{len(paths)} day images -> {out_dir} "
                f"({'256x256 flow view' if args.flow else 'native path'})")

    for path in paths:
        image_id = path.stem
        day = load_day(path, args.max_side or None)
        _save_srgb8(linear_to_srgb8(day), out_dir / f"{image_id}_day.png")
        for level in args.levels:
            result = degrade_image(
                day,
                cfg,
                calib,
                image_id=image_id,
                level=level,
                mode=args.mode,
                native_path=not args.flow,
                size=256 if args.flow else None,
            )
            sigma, peak = noise_map(result)
            stem = f"{image_id}_{level}"
            _save_srgb8(linear_to_srgb8(result.night_clean), out_dir / f"{stem}_clean.png")
            _save_srgb8(linear_to_srgb8(result.night), out_dir / f"{stem}_night.png")
            _save_srgb8(linear_to_srgb8(sigma), out_dir / f"{stem}_noise.png")
            if args.panel:
                save_panel(
                    [result.day, result.night_clean, result.night, sigma],
                    [CAPTIONS["day"], CAPTIONS["clean"], CAPTIONS["night"],
                     f"{CAPTIONS['noise']} (peak {peak:.4f})"],
                    out_dir / f"{stem}_panel.png",
                )
            params = result.params
            logger.info(
                f"  {stem:<34} {params.mode:<10} "
                f"target {params.camera.target:.5f}  "
                f"N_white {params.camera.n_white:>7.1f} e- "
                f"{'(clamped)' if params.camera.n_white_clamped else '         '}  "
                f"gain {result.gain:>5.2f}  "
                f"a {result.noise_a.mean():.2e}  b {result.noise_b.mean():.2e}  "
                f"n {result.n:g}  sigma_max {peak:.4f}"
            )
    logger.info(f"Done. Open {out_dir}/*_panel.png to compare them side by side.")


if __name__ == "__main__":
    main()
