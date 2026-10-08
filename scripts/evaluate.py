#!/usr/bin/env python3
"""Evaluate a trained Dawn model on the real paired RGB: the plan's whole metric table.

Scores `test_pairs` (or any manifest) against the daytime RGB those thermal frames are paired
with, on whole frames and over the vehicle region, with the masked pixels excluded everywhere.

    structure   L* PSNR (peak 100) and L* SSIM
    colour      mean CIEDE2000
    vividness   colorfulness (Hasler & Suesstrunk), beside the real image's own
    perceptual  LPIPS
    vehicles    the pixel metrics again over the vehicle region
    spread      mean L* and a*b* sample spread, for dawn/eval/calibration.py to rank

Two things this script does NOT decide. The **model** side -- token layout, resolution,
upsampling mode, normalization stats -- all comes out of the checkpoint, never out of a config
(docs/conventions.md section 5.1). And the headline is **box-aligned pairs only**: fallback pairs
were aligned by edge correlation alone, so they are reported as a separate robustness number
rather than averaged in.

Colour difference punishes a plausible colour that is not the true one -- a blue car painted red
scores badly -- so CIEDE2000 is read next to colorfulness and the example grids, never alone.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/run1/flow_best.pt --device cuda
    python scripts/evaluate.py --checkpoint ... --split val_pairs --max-pairs 40 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.data.pairs import AlignedPairs  # noqa: E402
from dawn.data.resample import area_downsample  # noqa: E402
from dawn.eval.metrics import evaluate_batch  # noqa: E402
from dawn.flow.inference import load_checkpoint  # noqa: E402
from dawn.flow.loaders import load_config, split_by_pair_kind  # noqa: E402
from dawn.flow.targets import prepare_batch  # noqa: E402
from dawn.progress import progress  # noqa: E402
from dawn.uncertainty.spread import sample_with_spread  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def vehicle_region(raw_batch, size, mode: str, min_pixels: int, device) -> torch.Tensor | None:
    """The region the "do vehicles survive?" row is scored over, as a `(B, 1, H, W)` weight.

    The plan asks for crops from the RGB boxes, but boxes are not persisted with the aligned
    pairs (`ASSET_KINDS` is rgb/thermal/valid/moving/vehicles) and re-deriving them needs the raw
    dataset plus the transform in `logs/align.csv`. `metrics.vehicle_region` picks the stand-in:

        mask    the `vehicles` mask itself, area-averaged to the working size. Dilated 3 px and
                stretched 17.4 px for vehicles that may have moved, so it is larger than the
                true boxes.
        boxes   the bounding box of each connected component of that mask, which is closer to a
                crop. Vehicles that touch merge into one component.
        none    no vehicle rows.
    """
    if mode == "none" or "vehicles" not in raw_batch:
        return None
    veh = raw_batch["vehicles"].to(device).float()
    if tuple(veh.shape[-2:]) != tuple(size):
        veh = area_downsample(veh, size)
    if mode == "mask":
        return veh
    if mode != "boxes":
        raise ValueError(f"metrics.vehicle_region must be mask | boxes | none, got {mode!r}")

    from scipy import ndimage

    out = torch.zeros_like(veh)
    for i in range(veh.shape[0]):
        labelled, n = ndimage.label((veh[i, 0] > 0.5).cpu().numpy())
        for sl in ndimage.find_objects(labelled):
            if sl is None:
                continue
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            if h * w >= min_pixels:
                out[i, 0, sl[0], sl[1]] = 1.0
    return out


@torch.no_grad()
def score_subset(bundle, dataset, cfg, ecfg, mcfg, method, device, lpips_fn) -> list[dict]:
    """One row per pair: every metric, plus the sample spread and the pair's kind."""
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(ecfg["batch_size"]),
        num_workers=int(ecfg["workers"]),
        shuffle=False,
    )
    size = bundle.input_size
    rows: list[dict] = []
    for raw in progress(loader, f"evaluate:{method}"):
        batch = prepare_batch(raw, size, cfg["pairs"], device)
        encoded = bundle.encoder.encode(batch["thermal"], return_attn=True)
        out = sample_with_spread(
            bundle.velocity,
            encoded,
            bundle.layout,
            bundle.mean,
            bundle.std,
            batch["thermal"],
            size,
            bundle.settings,
            num_samples=int(ecfg["num_samples"]),
            method=method,
            nfe=int(ecfg["nfe"]),
            image_ids=batch["image_id"] if ecfg["seed_from_image_id"] else None,
        )
        from dawn.data.color import linear_to_lab

        lab_target = linear_to_lab(batch["rgb"].clamp(0.0, 1.0))
        veh = vehicle_region(
            raw, size, str(mcfg["vehicle_region"]), int(mcfg["min_vehicle_pixels"]), device
        )
        # The single sample is the headline: it is what inference actually emits. The ensemble
        # mean is reported beside it because averaging helps CIEDE2000 and hurts colorfulness,
        # which is exactly the trade the plan warns about.
        scored = evaluate_batch(
            out.sample_lab, lab_target, out.sample_rgb, batch["rgb"],
            batch["mask"], vehicle_mask=veh, lpips_fn=lpips_fn,
        )
        mean_scored = evaluate_batch(out.lab, lab_target, out.rgb, batch["rgb"], batch["mask"])
        for n, image_id in enumerate(batch["image_id"]):
            row = {"id": image_id, "fallback": bool(batch["fallback"][n]), "method": method}
            row |= {k: float(v[n]) for k, v in scored.items()}
            row |= {f"mean_{k}": float(v[n]) for k, v in mean_scored.items()
                    if k in ("psnr", "ciede2000", "colorfulness")}
            if out.num_samples > 1:
                row["l_std"] = float(out.l_std[n].mean())
                row["ab_std"] = float(out.ab_std[n].mean())
            rows.append(row)
    return rows


def summarize(rows: list[dict], percentiles) -> dict[str, dict[str, float]]:
    """`{metric: {mean, p10, p50, p90}}` over the rows given."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], float)] if rows else []
    out = {}
    for k in keys:
        vals = sorted(r[k] for r in rows if k in r)
        if not vals:
            continue
        entry = {"mean": statistics.fmean(vals)}
        for p in percentiles:
            entry[f"p{p}"] = vals[min(len(vals) - 1, int(round((p / 100) * (len(vals) - 1))))]
        out[k] = entry
    return out


def table(summary: dict[str, dict[str, float]], percentiles) -> str:
    head = "| metric | mean | " + " | ".join(f"p{p}" for p in percentiles) + " |"
    rule = "|---|---|" + "---|" * len(percentiles)
    lines = [head, rule]
    for k, v in summary.items():
        cells = " | ".join(f"{v[f'p{p}']:.3f}" for p in percentiles)
        lines.append(f"| {k} | {v['mean']:.3f} | {cells} |")
    return "\n".join(lines)


def build_report(args, bundle, ecfg, mcfg, by_key, percentiles) -> str:
    lines = [
        "# Evaluation",
        "",
        f"- checkpoint: `{args.checkpoint}` (step {bundle.step}, EMA {bundle.ema})",
        f"- resolution (H, W): {list(bundle.input_size)}; "
        f"colour upsampling `{bundle.settings.upsample}`; "
        f"correction {'on' if bundle.layout.correction else 'off'}",
        f"- split: `{ecfg['split']}`, NFE {ecfg['nfe']}, K={ecfg['num_samples']}",
        f"- vehicle region: `{mcfg['vehicle_region']}` "
        "(the `vehicles` mask stands in for the plan's RGB boxes; see configs/eval.yaml)",
        "- masks: `valid & ~moving` (box-aligned), `valid & ~vehicles` (fallback)",
        "- structure on L\\* with the peak at 100; LPIPS is whole-frame and unmasked",
        "",
        "**The headline is box-aligned pairs.** Fallback pairs were aligned by edge correlation "
        "alone, so they are a robustness number, not the result. CIEDE2000 punishes a plausible "
        "colour that is not the true one, so read it beside colorfulness and the examples.",
        "",
    ]
    for (subset, method), rows in sorted(by_key.items()):
        lines += [
            f"## {subset} / {method} ({len(rows)} pairs)",
            "",
            table(summarize(rows, percentiles), percentiles),
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dawn against the real paired RGB")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained flow checkpoint")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Flow config")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--eval-config", type=str, default="configs/eval.yaml", help="Eval config")
    parser.add_argument("--override", nargs="*", default=[], help="Extra YAML files, in order")
    parser.add_argument("--split", type=str, default=None, help="Manifest (default: eval.split)")
    parser.add_argument("--max-pairs", type=int, default=None, help="Only the first N per subset")
    parser.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    parser.add_argument("--no-ema", action="store_true", help="Live weights, not the EMA average")
    parser.add_argument("--output", type=str, default=None, help="Output dir (default: config)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.data_config, [args.eval_config, *args.override])
    ecfg, mcfg, rcfg = cfg["eval"], cfg["metrics"], cfg["report"]
    if args.split:
        ecfg["split"] = args.split
    device = torch.device(args.device)

    bundle = load_checkpoint(args.checkpoint, device, use_ema=not args.no_ema)
    logger.info(
        "checkpoint step %d, %s, correction %s, upsample %s",
        bundle.step, bundle.input_size, bundle.layout.correction, bundle.settings.upsample,
    )

    lpips_fn = None
    if mcfg["lpips"]:
        try:
            import lpips as lpips_mod

            lpips_fn = lpips_mod.LPIPS(net=str(mcfg["lpips_net"])).to(device).eval()
        except ImportError:
            logger.warning("lpips is not installed; skipping the LPIPS column")

    dataset = AlignedPairs.from_config(cfg, ecfg["split"], train=False)
    subsets = split_by_pair_kind(dataset, args.max_pairs)
    logger.info("%s: %s", ecfg["split"], {k: len(v) for k, v in subsets.items()})

    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    all_rows: list[dict] = []
    for subset, data in subsets.items():
        for method in ecfg["methods"]:
            rows = score_subset(bundle, data, cfg, ecfg, mcfg, method, device, lpips_fn)
            for r in rows:
                r["subset"] = subset
            by_key[(subset, method)] = rows
            all_rows += rows

    out_dir = Path(args.output or ecfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args, bundle, ecfg, mcfg, by_key, rcfg["percentiles"])
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    if rcfg["save_csv"] and all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with (out_dir / "per_pair.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)
    summary = {f"{s}/{m}": summarize(r, rcfg["percentiles"]) for (s, m), r in by_key.items()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    headline = ecfg["headline_subset"]
    for (subset, method), rows in sorted(by_key.items()):
        if subset == headline and rows:
            s = summarize(rows, rcfg["percentiles"])
            logger.info(
                "%s/%s: L* PSNR %.2f dB, SSIM %.4f, CIEDE2000 %.3f, colorfulness %.1f (real %.1f)",
                subset, method, s["psnr"]["mean"], s["ssim"]["mean"], s["ciede2000"]["mean"],
                s["colorfulness"]["mean"], s["colorfulness_target"]["mean"],
            )
    logger.info("wrote %s", out_dir / "report.md")


if __name__ == "__main__":
    main()
