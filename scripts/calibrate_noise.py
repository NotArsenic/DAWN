#!/usr/bin/env python3
"""Fit the sensor noise model and the in-camera ISP to real night frames.

    python scripts/calibrate_noise.py              # fit, write configs/camera_calib.yaml
    python scripts/calibrate_noise.py --self-test  # no dataset: recover a known answer
    python scripts/calibrate_noise.py --dry-run    # fit, print, write nothing

Two things are measured in one pass over the frames, and `dawn/data/calibration.py` explains
how and how well:

    variance vs mean, flattest patches only -> effective (a, b) of Var[y|x] = a*x + b
    spatial autocorrelation of the residual -> the Gaussian denoise + unsharp pair that fits it

The ISP answer is what turns the first into the sensor constants `camera.py` samples from,
since a filter that sums to 1 multiplies a white-noise variance by the sum of its squared
weights and leaves the mean alone.

Which frames: the `hard_test` manifest (real night, hand-labelled) plus the darkest `--darkest`
images in the stats CSV. `b` is the intercept of a line, so it is only pinned down where some
part of some frame is nearly black, and the hand-labelled night sequences alone do not reliably
contain one. The brightness target is read off `hard_test` only, for the opposite reason: it
says where real night frames sit, and the darkest-N frames are not a sample of that.

Frames are read at NATIVE resolution. Downsampling first would divide the very variance this
measures (docs/conventions.md sections 4a and 6).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# `camera.isp_kernel` needs five numbers; the fit supplies two and these three are fixed by the
# shape of the filter rather than by the data, so they are carried over from the previous file.
FIXED_ISP = ("denoise_radius", "sharpen_sigma", "sharpen_radius")
SELF_TEST_A, SELF_TEST_B = 2.0e-4, 1.6e-7
SELF_TEST_SIGMA, SELF_TEST_AMOUNT = 0.80, 0.60
SELF_TEST_TOLERANCE = 0.20  # the loosest case may be this far out before the CLI calls it a fail


# --- choosing the frames -------------------------------------------------------------------


def select_frames(cfg: dict, args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """`(every frame to measure, the subset that sets the exposure target)`, as file paths.

    Raises:
        SystemExit: The stats CSV is missing, or nothing at all could be selected.
    """
    from dawn.data.splitting import HARD_TEST
    from dawn.data.visdrone import VisDroneLayout, read_manifest, read_stats_csv

    layout = VisDroneLayout.from_config(cfg["visdrone"])
    if not layout.stats_csv.exists():
        raise SystemExit(f"No stats CSV at {layout.stats_csv}: run `prepare_visdrone.py stats`")
    stats = read_stats_csv(layout.stats_csv)

    manifest = layout.manifest_dir / f"{HARD_TEST}.txt"
    if manifest.exists():
        hard_ids = [image_id for image_id in read_manifest(manifest) if image_id in stats]
    else:
        hard_ids = []
        logger.warning(f"No {manifest}: run `prepare_visdrone.py split`. Using dark frames only.")

    chosen = set(hard_ids)
    rest = sorted(
        (row for row in stats.values() if row.image_id not in chosen),
        key=lambda row: (row.lum_median, row.image_id),
    )
    dark_ids = [row.image_id for row in rest[: args.darkest]]
    image_ids = hard_ids + dark_ids
    if not image_ids:
        raise SystemExit(f"No images selected from {layout.stats_csv}")
    if args.limit:
        image_ids = image_ids[: args.limit]
        hard_ids = [image_id for image_id in hard_ids if image_id in set(image_ids)]

    darkest = min(stats[image_id].lum_median for image_id in image_ids)
    logger.info(
        f"{len(image_ids)} frames: {len(hard_ids)} hard_test + {len(dark_ids)} darkest "
        f"(median luminance down to {darkest:.2e})"
    )
    paths = [layout.raw_image(stats[image_id].split, image_id) for image_id in image_ids]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"{len(missing)} selected frames are not on disk, e.g. {missing[0]}")
    return paths, [layout.raw_image(stats[image_id].split, image_id) for image_id in hard_ids]


# --- the output file -----------------------------------------------------------------------


def _format_calibration(calib, template: dict, provenance: dict) -> str:
    """The YAML text of one calibration. Hand-written, so the comments survive a rerun."""
    exposure, isp = template["exposure"], template["isp"]
    autocorr = ", ".join(f"{value:.4f}" for value in calib.autocorr)
    per_a = ", ".join(f"{value:.4e}" for value in calib.per_channel_a)
    per_b = ", ".join(f"{value:.4e}" for value in calib.per_channel_b)
    lines = "\n".join(f"  {key}: {value}" for key, value in provenance.items())
    return f"""\
# Camera calibration. Written by scripts/calibrate_noise.py; read by dawn/data/camera.py.
#
#   python scripts/calibrate_noise.py --out configs/camera_calib.yaml
#
# Fitted, not assumed: every number under `measured` and `isp` comes from real night frames.
# What was measured and how far it can be trusted: dawn/data/calibration.py.
source: measured

provenance:
{lines}

# --- auto exposure ---------------------------------------------------------------------
# Where real night frames actually sit, so the simulated camera aims at the same place. The
# median over the hard_test frames of each frame's median linear Rec. 709 luminance.
exposure:
  target: {calib.exposure_target:.4g}
  percentile: {exposure["percentile"]}  # the statistic of the scene driven to `target`
  # Digital gain cap, a design choice rather than a measurement: 3 stops of push. Hitting it
  # is the only thing that leaves an image below `target`, so it is what separates the easy /
  # medium / hard darkness levels in data.yaml.
  gain_max: {exposure["gain_max"]}

# --- sensor --------------------------------------------------------------------------------
# Var[y|x] = a*x + b at NATIVE resolution (docs/conventions.md section 6). `effective_*` is
# what the camera delivers, after its ISP; `noise_*` is the sensor underneath it, which is what
# camera.py samples from and then re-smears with its own copy of that ISP.
measured:
  # a = 1 / full_well in normalized units.
  noise_a: {calib.noise_a:.4e}
  # b = sigma_r^2 * a^2. Recorded for reference; camera.py recomputes it from the sigma_r it
  # samples per image out of `degradation.camera.read_noise_range`.
  noise_b: {calib.noise_b:.4e}
  full_well: {calib.full_well:.0f}        # electrons, = 1 / noise_a
  read_noise_sigma: {calib.read_noise_sigma:.2f}   # electrons, from noise_b at that well
  # Digital gain assumed applied by the real camera. Not observable without EXIF: the two
  # sensor constants above scale as 1/g and 1/g^2 with it, everything else is unaffected.
  reference_gain: {calib.reference_gain:.4g}
  # As delivered, i.e. including the ISP's variance scaling of {calib.isp_var_scale:.4f}.
  effective_a: {calib.effective_a:.4e}
  effective_b: {calib.effective_b:.4e}
  per_channel_a: [{per_a}]
  per_channel_b: [{per_b}]
  # Fit quality. `darkest_bin` is the faintest signal the line was fit through and
  # `read_noise_share` is b / (a*darkest + b): below a few percent, `b` is an extrapolation.
  r_squared: {calib.r_squared:.4f}
  darkest_bin: {calib.darkest_bin:.4e}
  read_noise_share: {calib.read_noise_share:.4f}
  num_patches: {calib.num_patches}
  num_images: {calib.num_images}
  num_bins_used: {calib.num_bins_used}
  # Normalized spatial autocorrelation of the noise residual at lags 1..3 px, as measured
  # through the patch detrend. White sensor noise would be ~0; in-camera denoising correlates
  # neighbours and pushes lag 1 up. This is what the `isp` block below is fitted to.
  autocorr: [{autocorr}]

# --- in-camera ISP ---------------------------------------------------------------------
# Applied on the NATIVE-resolution path only (denoiser crops); both filters are linear, run on
# linear RGB, and scale the noise variance by the sum of the squared weights of their composite
# kernel (dawn/data/camera.py). The frames are JPEGs, so part of `denoise_sigma` is really the
# codec; that is the right thing for a simulator whose own last step is also a JPEG.
isp:
  denoise_sigma: {calib.denoise_sigma:.4g}   # px, Gaussian. Fitted to `measured.autocorr`.
  denoise_radius: {isp["denoise_radius"]}     # px, kernel half-width
  sharpen_amount: {calib.sharpen_amount:.4g}  # unsharp mask strength, fitted with it
  sharpen_sigma: {isp["sharpen_sigma"]}
  sharpen_radius: {isp["sharpen_radius"]}
"""


# --- the two modes -------------------------------------------------------------------------


def run_fit(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    from dawn.data.calibration import calibrate_paths

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    template = OmegaConf.to_container(OmegaConf.load(args.calib_config), resolve=True)
    paths, exposure_paths = select_frames(cfg, args)

    calib = calibrate_paths(
        paths,
        template["isp"],
        workers=args.workers,
        flat_quantile=args.flat_quantile,
        gate_quantile=args.gate_quantile,
        gate_scale=args.gate_scale,
        num_bins=args.num_bins,
        min_bin_patches=args.min_bin_patches,
        reference_gain=args.reference_gain,
        exposure_percentile=float(template["exposure"]["percentile"]),
        exposure_paths=exposure_paths,
    )
    report(calib)

    provenance = {
        "fitted": datetime.date.today().isoformat(),
        "frames": f"{len(paths)} ({len(exposure_paths)} hard_test + {args.darkest} darkest)",
        "stats_csv": cfg["visdrone"]["stats_csv"],
        "flat_quantile": args.flat_quantile,
        "gate_quantile": args.gate_quantile,
        "gate_scale": args.gate_scale,
        "num_bins": args.num_bins,
        "min_bin_patches": args.min_bin_patches,
    }
    text = _format_calibration(calib, template, provenance)
    if args.dry_run:
        logger.info(f"--dry-run: {args.out} not written. It would have been:\n\n{text}")
        return
    Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    logger.info(f"Wrote {args.out}")


def report(calib) -> None:
    """Log everything a person needs to decide whether to believe the numbers."""
    logger.info(f"  effective a      {calib.effective_a:.4e}   per channel "
                f"{['%.3e' % value for value in calib.per_channel_a]}")
    logger.info(f"  effective b      {calib.effective_b:.4e}   per channel "
                f"{['%.3e' % value for value in calib.per_channel_b]}")
    logger.info(f"  sensor a, b      {calib.noise_a:.4e}, {calib.noise_b:.4e} "
                f"at gain {calib.reference_gain:g}")
    logger.info(f"  full well        {calib.full_well:.0f} e-, "
                f"read noise {calib.read_noise_sigma:.2f} e-")
    logger.info(f"  autocorrelation  {['%.4f' % value for value in calib.autocorr]} "
                f"(residual rms {calib.autocorr_rms:.5f})")
    logger.info(f"  ISP              denoise sigma {calib.denoise_sigma:.3f} px, sharpen "
                f"{calib.sharpen_amount:.3f}, variance x{calib.isp_var_scale:.4f}")
    logger.info(f"  exposure target  {calib.exposure_target:.4g}")
    logger.info(f"  fit              r2 {calib.r_squared:.4f} over {calib.num_bins_used} bins, "
                f"{calib.num_patches} patches, {calib.num_images} images")
    logger.info(f"  identifiability  darkest bin {calib.darkest_bin:.3e}, "
                f"read noise is {100 * calib.read_noise_share:.1f}% of the fit there")
    if calib.read_noise_share < 0.05:
        logger.warning("  `b` carries almost none of the fit: it is an extrapolation, not a "
                       "measurement. Add darker frames (--darkest) before trusting it.")
    if calib.r_squared < 0.9:
        logger.warning("  Poor line fit: texture is probably leaking into the flat patches.")


def run_self_test(args: argparse.Namespace) -> None:
    """Measure frames whose answer is known, and print how close the estimator came.

    Raises:
        SystemExit: Any case missed the effective `a` or the denoise strength by too much.
    """
    import torch
    from omegaconf import OmegaConf

    from dawn.data.calibration import _isp_params, calibrate, synthetic_frame
    from dawn.data.camera import isp_kernel

    template = OmegaConf.to_container(OmegaConf.load(args.calib_config), resolve=True)
    isp_cfg = {key: template["isp"][key] for key in FIXED_ISP}
    kernel = isp_kernel(_isp_params(SELF_TEST_SIGMA, SELF_TEST_AMOUNT, isp_cfg))
    scale = float(kernel.pow(2).sum())

    cases = [
        ("mostly flat", 0.2, [0.06]),
        ("half texture", 0.5, [0.06]),
        ("mostly texture", 0.8, [0.06]),
        ("busy + bright", 0.8, [0.60]),
        ("mixed levels", 0.6, [0.02, 0.05, 0.12, 0.40]),
        ("six frames", 0.5, [0.04, 0.15]),
    ]
    columns = ("eff a", "a_raw", "b_raw", "sigma", "amount")
    logger.info("Recovering a known sensor and a known ISP. Each number is fitted / true.")
    logger.info(f"{'case':<16}" + "".join(f"{name:>8}" for name in columns)
                + f"{'bins':>6}{'r2':>8}")
    failures = []
    for name, texture, levels in cases:
        generator = torch.Generator().manual_seed(args.seed)
        count = 6 if name == "six frames" else args.self_test_images
        frames = [
            synthetic_frame(
                args.self_test_size,
                args.self_test_size,
                generator,
                kernel,
                level=levels[index % len(levels)],
                a_raw=SELF_TEST_A,
                b_raw=SELF_TEST_B,
                texture_fraction=texture,
            )
            for index in range(count)
        ]
        got = calibrate(frames, isp_cfg, reference_gain=1.0)
        ratios = (
            got.effective_a / (SELF_TEST_A * scale),
            got.noise_a / SELF_TEST_A,
            got.noise_b / SELF_TEST_B,
            got.denoise_sigma / SELF_TEST_SIGMA,
            got.sharpen_amount / SELF_TEST_AMOUNT,
        )
        logger.info(f"{name:<16}" + "".join(f"{value:>8.3f}" for value in ratios)
                    + f"{got.num_bins_used:>6}{got.r_squared:>8.4f}")
        if abs(ratios[0] - 1) > SELF_TEST_TOLERANCE or abs(ratios[3] - 1) > SELF_TEST_TOLERANCE:
            failures.append(name)

    logger.info(f"ISP variance scale {scale:.4f}; `b_raw` is only identifiable in the dark cases.")
    if failures:
        raise SystemExit(f"Self-test FAILED on: {', '.join(failures)}")
    logger.info(f"Self-test passed: effective a and denoise sigma within "
                f"{100 * SELF_TEST_TOLERANCE:.0f}% in every case.")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Var[y|x] = a*x + b and the in-camera ISP from real night frames",
        epilog="With no dataset on disk, --self-test still exercises the whole estimator.",
    )
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--calib-config", type=str, default="configs/camera_calib.yaml",
                        help="Calibration file to take the fixed ISP radii and gain cap from")
    parser.add_argument("--out", type=str, default="configs/camera_calib.yaml",
                        help="Where to write the result (default: overwrite --calib-config)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="Processes (default: every core)")
    parser.add_argument("--darkest", type=int, default=200,
                        help="Darkest images from the stats CSV to add to hard_test")
    parser.add_argument("--limit", type=int, default=None, help="Cap the total frame count")
    parser.add_argument("--reference-gain", type=float, default=1.0,
                        help="Digital gain assumed applied by the real camera (see the YAML)")
    parser.add_argument("--dry-run", action="store_true", help="Fit and print, write nothing")

    tuning = parser.add_argument_group("estimator")
    tuning.add_argument("--flat-quantile", type=float, default=0.15,
                        help="Share of each brightness bin kept as flat")
    tuning.add_argument("--gate-quantile", type=float, default=0.25,
                        help="Share of each frame assumed featureless, setting the flatness gate")
    tuning.add_argument("--gate-scale", type=float, default=1.3,
                        help="How far above that gate a patch may still count as flat")
    tuning.add_argument("--num-bins", type=int, default=40, help="Brightness bins")
    tuning.add_argument("--min-bin-patches", type=int, default=64,
                        help="Bins with fewer flat patches than this are dropped from the fit")

    check = parser.add_argument_group("self-test")
    check.add_argument("--self-test", action="store_true",
                       help="Recover a known sensor from synthetic frames; touch no dataset")
    check.add_argument("--self-test-images", type=int, default=10, help="Frames per case")
    check.add_argument("--self-test-size", type=int, default=1024, help="Frame side, px")
    check.add_argument("--seed", type=int, default=23, help="Self-test RNG seed")

    args = parser.parse_args()
    run_self_test(args) if args.self_test else run_fit(args)


if __name__ == "__main__":
    main()
