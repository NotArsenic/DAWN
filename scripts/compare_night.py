#!/usr/bin/env python3
"""Compare synthetic nighttime degradation against real night frames side-by-side.

Evaluates real night images from VisDrone (e.g. hard_test) and synthetic night images
rendered from daytime sources at native resolution. Produces:
1. Contact sheets (sheet_<level>.jpg) comparing real vs synthetic at identical display size.
2. 1:1 native-resolution crops (crops_<level>.jpg) comparing fine texture and noise.
3. Per-image CSV (stats.csv) with darkness, color, highlight, noise, and clipping metrics.
4. Summary table (stats.md, and logged) with medians, IQRs, and synthetic/real ratios.

Usage:
    python scripts/compare_night.py
    python scripts/compare_night.py --real-manifest hard_test --count 20 --levels hard
    python scripts/compare_night.py --also night_normal --levels easy medium hard
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from dawn.data.color import linear_to_srgb8, luminance, srgb8_to_linear
from dawn.data.degrade import DARKNESS_LEVELS, degrade_image
from dawn.data.prepare import _font
from dawn.data.resample import area_downsample, fit_long_side
from dawn.data.visdrone import SPLITS, ImageStats, VisDroneLayout, read_srgb8, read_stats_csv
from dawn.eval.night_stats import NightStats, compute_night_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

STAT_FIELDS = [
    "lum_median",
    "lum_p90",
    "sat_mean",
    "sat_pixel_fraction",
    "highlight_ratio",
    "flat_noise_srgb",
    "flat_noise_linear",
    "flat_patch_mean",
    "noise_level",
    "code0_fraction",
]

STAT_LABELS = {
    "lum_median": "Median luminance (linear Rec. 709)",
    "lum_p90": "p90 luminance (linear Rec. 709)",
    "sat_mean": "Color saturation (HSV, luma >= 10)",
    "sat_pixel_fraction": "Saturated pixel fraction",
    "highlight_ratio": "Highlight ratio (p99.5 / median)",
    "flat_noise_srgb": "Flat-patch noise std (sRGB codes)",
    "flat_noise_linear": "Flat-patch noise std (linear)",
    "flat_patch_mean": "Flat-patch mean luminance (linear)",
    "noise_level": "Noise level (high-pass proxy)",
    "code0_fraction": "Code-0 pixel fraction (crushed black)",
}


def resolve_manifest_ids(name_or_path: str, manifest_dir: Path) -> list[str]:
    """Load image IDs from a manifest file name or path.

    Args:
        name_or_path: Manifest name (e.g. 'hard_test') or path to .txt file.
        manifest_dir: Directory where manifests are stored.

    Returns:
        List of cleaned, non-empty image IDs.

    Raises:
        FileNotFoundError: If the manifest file cannot be located.
        ValueError: If the manifest file is empty.
    """
    path = Path(name_or_path)
    if not path.is_file():
        candidate_with_ext = manifest_dir / f"{name_or_path}.txt"
        candidate_raw = manifest_dir / name_or_path
        if candidate_with_ext.is_file():
            path = candidate_with_ext
        elif candidate_raw.is_file():
            path = candidate_raw
        else:
            raise FileNotFoundError(
                f"Manifest '{name_or_path}' not found at '{path}' or '{candidate_with_ext}'. "
                f"Verify that manifests exist or pass an explicit file path via --real-manifest."
            )

    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"Manifest at '{path}' is empty.")
    return lines


def find_image_path(layout: VisDroneLayout, stats: dict[str, ImageStats], image_id: str) -> Path:
    """Locate raw image JPEG on disk using stats CSV split or searching splits."""
    if image_id in stats:
        split = stats[image_id].split
        path = layout.raw_image(split, image_id)
        if path.exists():
            return path
    for split in SPLITS:
        path = layout.raw_image(split, image_id)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Raw image '{image_id}.jpg' not found in any VisDrone split under {layout.root}."
    )


def extract_deterministic_crop(img_linear: torch.Tensor, crop_size: int = 384) -> torch.Tensor:
    """Extract a 1:1 crop from the window with the largest luminance contrast.

    Args:
        img_linear: (3, H, W) linear RGB tensor.
        crop_size: Size of the square crop (default 384).

    Returns:
        (3, crop_size, crop_size) linear RGB crop.
    """
    _, height, width = img_linear.shape
    if height < crop_size or width < crop_size:
        pad_h = max(0, crop_size - height)
        pad_w = max(0, crop_size - width)
        img_linear = torch.nn.functional.pad(
            img_linear.unsqueeze(0), (0, pad_w, 0, pad_h), mode="replicate"
        ).squeeze(0)
        _, height, width = img_linear.shape

    lum = luminance(img_linear).squeeze(0)  # (H, W)

    # Search candidates on a 32-pixel grid
    stride = 32
    row_starts = list(range(0, height - crop_size + 1, stride))
    if (height - crop_size) not in row_starts:
        row_starts.append(height - crop_size)
    col_starts = list(range(0, width - crop_size + 1, stride))
    if (width - crop_size) not in col_starts:
        col_starts.append(width - crop_size)

    best_spread = -1.0
    best_top, best_left = 0, 0
    for r in row_starts:
        for c in col_starts:
            window = lum[r : r + crop_size, c : c + crop_size]
            spread = (window.max() - window.min()).item()
            if spread > best_spread:
                best_spread = spread
                best_top, best_left = r, c

    return img_linear[:, best_top : best_top + crop_size, best_left : best_left + crop_size]


def create_tile(img_linear: torch.Tensor, display_size: int = 480) -> Image.Image:
    """Area-downsample image to display_size on long side and convert to PIL Image."""
    _, height, width = img_linear.shape
    target_size = fit_long_side(height, width, display_size)
    if target_size != (height, width):
        img_linear = area_downsample(img_linear, target_size)
    srgb8 = linear_to_srgb8(img_linear)
    return Image.fromarray(srgb8.permute(1, 2, 0).numpy())


def draw_block_grid(
    blocks: list[tuple[str, str, list[dict[str, Any]]]],
    columns: int = 4,
    pad: int = 10,
    banner_h: int = 40,
    label_h: int = 36,
    crop_mode: bool = False,
) -> Image.Image:
    """Build a multi-block contact sheet image with section banners and captioned tiles.

    Args:
        blocks: List of (block_title, bg_color_hex, items) tuples.
        columns: Number of grid columns.
        pad: Padding in pixels.
        banner_h: Height of banner header.
        label_h: Height of tile caption.
        crop_mode: Whether tiles are crops (affects caption details).

    Returns:
        PIL Image of the composite sheet.
    """
    all_tiles = [item["tile"] for _, _, items in blocks for item in items]
    if not all_tiles:
        return Image.new("RGB", (100, 100), (26, 26, 25))

    cell_w = max(tile.width for tile in all_tiles)
    cell_h = max(tile.height for tile in all_tiles)

    total_w = columns * (cell_w + pad) + pad
    total_h = pad

    for _, _, items in blocks:
        rows = math.ceil(len(items) / columns)
        block_h = banner_h + rows * (cell_h + label_h + pad) + pad
        total_h += block_h

    sheet = Image.new("RGB", (total_w, total_h), (26, 26, 25))
    draw = ImageDraw.Draw(sheet)

    title_font = _font(16)
    label_font = _font(11)

    curr_y = pad
    for title, banner_color, items in blocks:
        # Draw section banner
        draw.rectangle([pad, curr_y, total_w - pad, curr_y + banner_h - 4], fill=banner_color)
        draw.text((pad + 12, curr_y + 8), title, fill=(245, 245, 240), font=title_font)
        curr_y += banner_h

        for idx, item in enumerate(items):
            r, c = divmod(idx, columns)
            tile_x = pad + c * (cell_w + pad) + (cell_w - item["tile"].width) // 2
            tile_y = curr_y + r * (cell_h + label_h + pad)

            sheet.paste(item["tile"], (tile_x, tile_y))

            cap_x = pad + c * (cell_w + pad)
            cap_y = tile_y + item["tile"].height + 4
            stats: NightStats = item["stats"]

            line1 = f"[{item['group']}] {item['image_id']}"
            if crop_mode:
                line2 = (
                    f"flat_srgb={stats.flat_noise_srgb:.2f}  "
                    f"code0={stats.code0_fraction:.1%}  "
                    f"sat={stats.sat_mean:.2f}"
                )
            else:
                line2 = (
                    f"med={stats.lum_median:.4f}  "
                    f"p90={stats.lum_p90:.4f}  "
                    f"hl={stats.highlight_ratio:.1f}  "
                    f"c0={stats.code0_fraction:.1%}"
                )

            draw.text((cap_x + 2, cap_y), line1, fill=(220, 219, 210), font=label_font)
            draw.text((cap_x + 2, cap_y + 14), line2, fill=(160, 160, 150), font=label_font)

        rows = math.ceil(len(items) / columns)
        curr_y += rows * (cell_h + label_h + pad) + pad

    return sheet


def compute_group_summary(stats_list: list[NightStats]) -> dict[str, tuple[float, float]]:
    """Compute median and IQR for each statistic in NightStats.

    Returns:
        dict: {field_name: (median, iqr)}
    """
    summary = {}
    for field in STAT_FIELDS:
        vals = [getattr(s, field) for s in stats_list if not math.isnan(getattr(s, field))]
        if not vals:
            summary[field] = (float("nan"), float("nan"))
        else:
            arr = np.array(vals)
            med = float(np.median(arr))
            q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
            iqr = q75 - q25
            summary[field] = (med, iqr)
    return summary


def format_summary_markdown(
    group_summaries: dict[str, dict[str, tuple[float, float]]],
    primary_real_group: str,
) -> str:
    """Format medians, IQRs, and synthetic/real ratios into a GitHub Markdown table."""
    headers = ["Metric"]
    group_keys = list(group_summaries.keys())
    for gk in group_keys:
        headers.append(f"{gk} Median (IQR)")
        if gk.startswith("synthetic"):
            headers.append(f"Ratio ({gk.split(':')[-1].strip()} / Real)")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    real_summary = group_summaries.get(primary_real_group, {})

    for field in STAT_FIELDS:
        label = STAT_LABELS.get(field, field)
        row = [f"**{label}**"]
        real_med = real_summary.get(field, (float("nan"), float("nan")))[0]

        for gk in group_keys:
            med, iqr = group_summaries[gk].get(field, (float("nan"), float("nan")))
            if math.isnan(med):
                row.append("NaN")
            else:
                row.append(f"{med:.4g} (IQR {iqr:.4g})")

            if gk.startswith("synthetic"):
                if math.isnan(med) or math.isnan(real_med):
                    row.append("NaN")
                elif abs(real_med) < 1e-8:
                    row.append(f"{med:.4g} (real≈0)")
                else:
                    ratio = med / real_med
                    row.append(f"**{ratio:.2f}x**")

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare synthetic nighttime degradation against real night frames",
        epilog="Reads configs/data.yaml and configs/camera_calib.yaml.",
    )
    parser.add_argument(
        "--real-manifest",
        type=str,
        default="hard_test",
        help="Primary real night manifest name or path (default: hard_test)",
    )
    parser.add_argument(
        "--day-manifest",
        type=str,
        default="val_day",
        help="Day source manifest name or path (default: val_day)",
    )
    parser.add_argument(
        "--also",
        nargs="*",
        default=[],
        help="Additional real reference manifest groups (e.g. night_normal)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of day sources to synthesize (default: 20)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["hard"],
        choices=DARKNESS_LEVELS,
        help="Darkness levels to render per synthetic sample (default: hard)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/compare_night",
        help="Output directory (default: outputs/compare_night)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20261004,
        help="Random seed for day sampling and determinism (default: 20261004)",
    )
    parser.add_argument(
        "--display-size",
        type=int,
        default=480,
        help="Long side for contact sheet tiles (default: 480)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=384,
        help="Square crop size for 1:1 crops (default: 384)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data.yaml",
        help="Path to data configuration YAML",
    )
    parser.add_argument(
        "--calib-config",
        type=str,
        default="configs/camera_calib.yaml",
        help="Path to camera calibration YAML",
    )
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)
    calib = OmegaConf.to_container(OmegaConf.load(args.calib_config), resolve=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout = VisDroneLayout.from_config(cfg["visdrone"])
    stats_cache = read_stats_csv(layout.stats_csv)

    # 1. Resolve manifests
    logger.info("Resolving manifests...")
    real_manifest_name = Path(args.real_manifest).stem
    real_ids = resolve_manifest_ids(args.real_manifest, layout.manifest_dir)
    logger.info(f"Loaded {len(real_ids)} primary real images from {args.real_manifest}")

    also_groups: dict[str, list[str]] = {}
    for also_name in args.also:
        also_ids = resolve_manifest_ids(also_name, layout.manifest_dir)
        also_groups[Path(also_name).stem] = also_ids
        logger.info(f"Loaded {len(also_ids)} reference real images from {also_name}")

    all_day_ids = resolve_manifest_ids(args.day_manifest, layout.manifest_dir)
    rng = random.Random(f"{args.seed}/day_select")
    sample_count = min(args.count, len(all_day_ids))
    day_ids = sorted(rng.sample(all_day_ids, sample_count))
    logger.info(f"Sampled {len(day_ids)} daytime source images from {args.day_manifest}")

    # Storage for all evaluated samples
    all_evaluated_rows: list[dict[str, Any]] = []

    # Cache for contact sheet tiles and crops: group -> list of item dicts
    real_items: dict[str, list[dict[str, Any]]] = {real_manifest_name: []}
    for gname in also_groups:
        real_items[gname] = []

    # 2. Process real images
    logger.info(f"Processing primary real group '{real_manifest_name}' ({len(real_ids)} images)...")
    for img_id in real_ids:
        path = find_image_path(layout, stats_cache, img_id)
        img_lin = srgb8_to_linear(read_srgb8(path))
        st = compute_night_stats(img_lin)
        tile = create_tile(img_lin, display_size=args.display_size)
        crop_lin = extract_deterministic_crop(img_lin, crop_size=args.crop_size)
        crop_tile = Image.fromarray(linear_to_srgb8(crop_lin).permute(1, 2, 0).numpy())

        row = {"group": real_manifest_name, "level": "real", "image_id": img_id, **st.as_dict()}
        all_evaluated_rows.append(row)
        real_items[real_manifest_name].append({
            "image_id": img_id,
            "group": real_manifest_name,
            "tile": tile,
            "crop_tile": crop_tile,
            "stats": st,
        })

    # Process additional real groups (e.g. night_normal)
    for gname, ids in also_groups.items():
        logger.info(f"Processing reference real group '{gname}' ({len(ids)} images)...")
        for img_id in ids:
            path = find_image_path(layout, stats_cache, img_id)
            img_lin = srgb8_to_linear(read_srgb8(path))
            st = compute_night_stats(img_lin)
            tile = create_tile(img_lin, display_size=args.display_size)
            crop_lin = extract_deterministic_crop(img_lin, crop_size=args.crop_size)
            crop_tile = Image.fromarray(linear_to_srgb8(crop_lin).permute(1, 2, 0).numpy())

            row = {"group": gname, "level": "real", "image_id": img_id, **st.as_dict()}
            all_evaluated_rows.append(row)
            real_items[gname].append({
                "image_id": img_id,
                "group": gname,
                "tile": tile,
                "crop_tile": crop_tile,
                "stats": st,
            })

    # 3. Process synthetic images for each level
    synthetic_items: dict[str, list[dict[str, Any]]] = {}
    for level in args.levels:
        synthetic_items[level] = []
        logger.info(f"Synthesizing {len(day_ids)} native-resolution images for level '{level}'...")
        for day_id in day_ids:
            day_path = find_image_path(layout, stats_cache, day_id)
            day_lin = srgb8_to_linear(read_srgb8(day_path))

            res = degrade_image(
                day_lin,
                cfg["degradation"],
                calib,
                image_id=day_id,
                level=level,
                native_path=True,
            )
            night_lin = res.night
            st = compute_night_stats(night_lin)
            tile = create_tile(night_lin, display_size=args.display_size)
            crop_lin = extract_deterministic_crop(night_lin, crop_size=args.crop_size)
            crop_tile = Image.fromarray(linear_to_srgb8(crop_lin).permute(1, 2, 0).numpy())

            row = {
                "group": f"synthetic_{level}",
                "level": level,
                "image_id": day_id,
                **st.as_dict(),
            }
            all_evaluated_rows.append(row)
            synthetic_items[level].append({
                "image_id": day_id,
                "group": f"synth_{level}",
                "tile": tile,
                "crop_tile": crop_tile,
                "stats": st,
            })

    # 4. Generate contact sheets and 1:1 crop sheets per level
    for level in args.levels:
        logger.info(f"Generating contact sheets for level '{level}'...")
        real_block_title = f"REAL: {real_manifest_name} (N={len(real_ids)})"
        blocks = [
            (real_block_title, "#1e293b", real_items[real_manifest_name])
        ]
        crop_blocks = [
            (
                f"{real_block_title} 1:1 crops",
                "#1e293b",
                [{**it, "tile": it["crop_tile"]} for it in real_items[real_manifest_name]],
            )
        ]

        for gname in also_groups:
            title_also = f"REAL: {gname} (N={len(also_groups[gname])})"
            blocks.append((title_also, "#334155", real_items[gname]))
            crop_blocks.append((
                f"{title_also} 1:1 crops",
                "#334155",
                [{**it, "tile": it["crop_tile"]} for it in real_items[gname]],
            ))

        synth_list = synthetic_items[level]
        blocks.append((f"SYNTHETIC: {level} (N={len(synth_list)})", "#451a03", synth_list))
        crop_blocks.append((
            f"SYNTHETIC: {level} (N={len(synth_list)}, 1:1 crops)",
            "#451a03",
            [{**it, "tile": it["crop_tile"]} for it in synth_list],
        ))

        sheet_img = draw_block_grid(blocks, columns=4, crop_mode=False)
        sheet_path = out_dir / f"sheet_{level}.jpg"
        sheet_img.save(sheet_path, quality=92)
        logger.info(f"Saved contact sheet: {sheet_path}")

        crops_img = draw_block_grid(crop_blocks, columns=4, crop_mode=True)
        crops_path = out_dir / f"crops_{level}.jpg"
        crops_img.save(crops_path, quality=92)
        logger.info(f"Saved 1:1 crops sheet: {crops_path}")

    # 5. Write stats.csv
    csv_path = out_dir / "stats.csv"
    csv_fields = ["group", "level", "image_id"] + STAT_FIELDS
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for r in all_evaluated_rows:
            formatted_row = {
                k: f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in csv_fields
            }
            writer.writerow(formatted_row)
    logger.info(f"Saved detailed stats to {csv_path}")

    # 6. Aggregate summary table
    group_summaries: dict[str, dict[str, tuple[float, float]]] = {}
    primary_key = f"real: {real_manifest_name}"
    group_summaries[primary_key] = compute_group_summary([
        it["stats"] for it in real_items[real_manifest_name]
    ])

    for gname in also_groups:
        group_summaries[f"real: {gname}"] = compute_group_summary([
            it["stats"] for it in real_items[gname]
        ])

    for level in args.levels:
        group_summaries[f"synthetic: {level}"] = compute_group_summary([
            it["stats"] for it in synthetic_items[level]
        ])

    summary_md = format_summary_markdown(group_summaries, primary_real_group=primary_key)
    md_path = out_dir / "stats.md"
    md_path.write_text(summary_md + "\n", encoding="utf-8")
    logger.info(f"Saved summary table to {md_path}")

    banner_sep = "=" * 80
    logger.info(f"\n{banner_sep}\nSUMMARY TABLE (Median and IQR):\n{summary_md}\n{banner_sep}")


if __name__ == "__main__":
    main()
