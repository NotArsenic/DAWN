"""The brightness and colour oracle: the best any model could do with this parametrization.

`scripts/check_oracle.py` is a thin CLI over this module. Per pair, with the pixel mask of
`configs/flow.yaml: pairs`, at the configured resolution, it measures (all against the RGB's
CIELAB):

    grid-only    the fitted brightness grid, rendered, at refine 0 and 1: L* PSNR (peak 100)
    + correction the same grid plus the generated-L correction at the colour-map size, upsampled
                 by the BGU (the render path) and, as the reference the plan's numbers used, by
                 plain bilinear upsampling
    colour       the true a*b* area-averaged to the colour-map size and brought back up by the BGU
                 guided by the true L, against bilinear: mean |delta a*b*|
    assembled    the whole oracle image through `render` (grid-only and with the correction):
                 CIEDE2000 next to the L* PSNR

Everything goes through the project's own `build_targets` and `render`, so the oracle is exactly
the ceiling the trainer's validation compares against (docs/conventions.md rules 9 and 10).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from dawn.data.resample import area_downsample
from dawn.data.sheets import thumbnail, thumbnail_thermal, tile_sheet
from dawn.eval.metrics import ciede2000, masked_mean, masked_psnr
from dawn.flow.render import RenderSettings, render
from dawn.flow.targets import FitSettings, build_targets, prepare_batch
from dawn.flow.tokens import TokenLayout
from dawn.grid.upsample import bilateral_guided_upsample

logger = logging.getLogger(__name__)

METRICS = (
    "psnr_grid_r0",
    "psnr_grid_r1",
    "psnr_corr_bgu_r0",
    "psnr_corr_bgu_r1",
    "psnr_corr_bilinear_r1",
    "ab_err_bgu",
    "ab_err_bilinear",
    "de_grid_only",
    "de_with_correction",
)
LABELS = {
    "psnr_grid_r0": "grid only, refine 0 (L* PSNR dB)",
    "psnr_grid_r1": "grid only, refine 1 (L* PSNR dB)",
    "psnr_corr_bgu_r0": "grid + correction (BGU), refine 0 (dB)",
    "psnr_corr_bgu_r1": "grid + correction (BGU), refine 1 (dB)",
    "psnr_corr_bilinear_r1": "grid + correction (bilinear), refine 1 (dB)",
    "ab_err_bgu": "colour ceiling, BGU (mean abs delta a*b*)",
    "ab_err_bilinear": "colour ceiling, bilinear (mean abs delta a*b*)",
    "de_grid_only": "assembled oracle, grid only (CIEDE2000)",
    "de_with_correction": "assembled oracle, with correction (CIEDE2000)",
}


def layouts(cfg: Mapping) -> tuple[TokenLayout, TokenLayout]:
    """`(correction off, correction on)` layouts at the configured resolution."""
    off = dataclasses.replace(TokenLayout.from_config(cfg), correction=False)
    return off, dataclasses.replace(off, correction=True)


def measure_batch(batch: Mapping, cfg: Mapping) -> dict[str, torch.Tensor]:
    """Every oracle metric for one prepared batch (`prepare_batch`), each `(B,)`."""
    off, on = layouts(cfg)
    ridge = float(cfg["grid"]["ridge_alpha"])
    size = tuple(int(v) for v in cfg["encoder"]["input_size"])
    base = RenderSettings.from_config(cfg)
    settings = {
        False: RenderSettings(off, base.bgu_grid, base.bgu_ridge, base.bgu_refine),
        True: RenderSettings(on, base.bgu_grid, base.bgu_ridge, base.bgu_refine),
    }
    thermal, rgb, mask = batch["thermal"], batch["rgb"], batch["mask"]
    out: dict[str, torch.Tensor] = {}
    for refine in (0, 1):
        fit = FitSettings(ridge, refine)
        plain = build_targets(thermal, rgb, mask, off, fit)
        withc = build_targets(thermal, rgb, mask, on, fit)
        true_l = plain.lab[:, :1]
        out[f"psnr_grid_r{refine}"] = masked_psnr(100.0 * plain.L_grid, true_l, mask, 100.0)
        bgu = render(withc.parts, thermal, size, settings[True])
        out[f"psnr_corr_bgu_r{refine}"] = masked_psnr(bgu["lab"][:, :1], true_l, mask, 100.0)
        if refine == 1:
            up = F.interpolate(withc.parts["corr"], size=size, mode="bilinear", align_corners=False)
            bilinear_l = 100.0 * (plain.L_grid + up).clamp(0.0, 1.0)
            out["psnr_corr_bilinear_r1"] = masked_psnr(bilinear_l, true_l, mask, 100.0)
            grid_only = render(plain.parts, thermal, size, settings[False])
            out["de_grid_only"] = masked_mean(ciede2000(grid_only["lab"], plain.lab), mask)
            out["de_with_correction"] = masked_mean(ciede2000(bgu["lab"], plain.lab), mask)
            ab_up, ab_bil = colour_ceiling(plain, base, size, off)
            true_ab = plain.lab[:, 1:]
            out["ab_err_bgu"] = masked_mean(torch.linalg.vector_norm(ab_up - true_ab, dim=1), mask)
            out["ab_err_bilinear"] = masked_mean(
                torch.linalg.vector_norm(ab_bil - true_ab, dim=1), mask
            )
    return out


def colour_ceiling(
    targets, render_settings: RenderSettings, size: tuple[int, int], layout: TokenLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """True a*b* at the colour-map size, brought back to `size` by the BGU (guided by the true L)
    and by plain bilinear upsampling. The `F.interpolate` baseline is the one sanctioned upsample
    outside the BGU (docs/conventions.md rule 5): it is what the BGU is measured against."""
    low = targets.parts["ab"]
    guide_hi = targets.lab[:, :1] / 100.0
    guide_lo = area_downsample(guide_hi, layout.color_size)
    up = bilateral_guided_upsample(
        low,
        guide_lo,
        guide_hi,
        targets.fraction,
        grid_size=render_settings.bgu_grid,
        ridge_alpha=render_settings.bgu_ridge,
        refine_steps=render_settings.bgu_refine,
    )
    return up, F.interpolate(low, size=size, mode="bilinear", align_corners=False)


def bgu_sweep_batch(
    batch: Mapping,
    cfg: Mapping,
    settings: Sequence[tuple[int, int, int, float]],
) -> dict[tuple, torch.Tensor]:
    """Colour-ceiling error `(B,)` of each BGU setting `(bins, rows, cols, ridge)` on a batch."""
    off, _ = layouts(cfg)
    targets = build_targets(
        batch["thermal"], batch["rgb"], batch["mask"], off, FitSettings(1e-3, 0)
    )
    true_ab = targets.lab[:, 1:]
    guide_hi = targets.lab[:, :1] / 100.0
    guide_lo = area_downsample(guide_hi, off.color_size)
    low = targets.parts["ab"]
    bilinear = F.interpolate(low, size=true_ab.shape[-2:], mode="bilinear", align_corners=False)
    out = {
        ("bilinear",): masked_mean(
            torch.linalg.vector_norm(bilinear - true_ab, dim=1), batch["mask"]
        )
    }
    for bins, rows, cols, ridge in settings:
        up = bilateral_guided_upsample(
            targets.parts["ab"],
            guide_lo,
            guide_hi,
            targets.fraction,
            grid_size=(bins, rows, cols),
            ridge_alpha=ridge,
        )
        out[(bins, rows, cols, ridge)] = masked_mean(
            torch.linalg.vector_norm(up - true_ab, dim=1), batch["mask"]
        )
    return out


def _loader(dataset, indices, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        Subset(dataset, list(indices)), batch_size=batch_size, num_workers=workers, shuffle=False
    )


@torch.no_grad()
def run_oracle(
    dataset,
    indices: Iterable[int],
    cfg: Mapping,
    device: torch.device | str = "cpu",
    batch_size: int = 4,
    workers: int = 0,
    sweep: Sequence[tuple[int, int, int, float]] = (),
    sweep_pairs: int = 40,
) -> tuple[list[dict], dict[tuple, list[float]]]:
    """Per-pair oracle rows (`id`, `fallback`, every metric) and the BGU sweep sums.

    `sweep` settings are measured on the first `sweep_pairs` pairs only (box and fallback alike).
    """
    size = tuple(int(v) for v in cfg["encoder"]["input_size"])
    rows: list[dict] = []
    sweeps: dict[tuple, list[float]] = {}
    swept = 0
    for batch_index, raw in enumerate(_loader(dataset, indices, batch_size, workers)):
        batch = prepare_batch(raw, size, cfg["pairs"], device)
        measured = measure_batch(batch, cfg)
        for n, image_id in enumerate(batch["image_id"]):
            rows.append(
                {"id": image_id, "fallback": bool(batch["fallback"][n])}
                | {k: float(v[n]) for k, v in measured.items()}
            )
        remaining = sweep_pairs - swept
        if sweep and remaining > 0:
            head = {
                k: v[:remaining] if torch.is_tensor(v) else v[:remaining] for k, v in batch.items()
            }
            for setting, values in bgu_sweep_batch(head, cfg, sweep).items():
                sweeps.setdefault(setting, []).extend(float(v) for v in values)
            swept += len(head["image_id"])
        if batch_index % 10 == 0:
            logger.info("oracle: %d pairs done", len(rows))
    return rows, sweeps


# --- summaries --------------------------------------------------------------------------------


def summarize(rows: Sequence[Mapping], key: str) -> dict[str, float]:
    """mean / p10 / p50 / p90 / n of `key` over `rows` (NaN when there are none)."""
    values = np.array([r[key] for r in rows if key in r], dtype=np.float64)
    if values.size == 0:
        return {
            "mean": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "n": 0,
        }
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    return {
        "mean": float(values.mean()),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "n": int(values.size),
    }


def metrics_table(rows: Sequence[Mapping], keys: Sequence[str] = METRICS) -> str:
    """A markdown table: one row per metric, mean / p10 / p50 / p90 for all, box and fallback."""
    groups = {
        "all": list(rows),
        "box-aligned": [r for r in rows if not r["fallback"]],
        "fallback": [r for r in rows if r["fallback"]],
    }
    header = (
        "| metric | "
        + " | ".join(
            f"{name} mean (n={len(g)}) | {name} p10 / p50 / p90" for name, g in groups.items()
        )
        + " |"
    )
    lines = [header, "|---|" + "---|---|" * len(groups)]
    for key in keys:
        cells = []
        for group in groups.values():
            s = summarize(group, key)
            cells += [f"{s['mean']:.2f}", f"{s['p10']:.2f} / {s['p50']:.2f} / {s['p90']:.2f}"]
        lines.append(f"| {LABELS[key]} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sweep_table(sweeps: Mapping[tuple, Sequence[float]]) -> tuple[str, tuple | None, float | None]:
    """A markdown table of the BGU sweep (mean |delta a*b*|, lower is better).

    Returns `(table, best BGU setting, bilinear baseline on the same pairs)`; the bilinear row
    (key `("bilinear",)`) is the baseline, not a candidate.
    """
    baseline = sweeps.get(("bilinear",))
    base_mean = float(np.mean(baseline)) if baseline else None
    ranked = sorted(
        ((float(np.mean(v)), k, len(v)) for k, v in sweeps.items() if len(v) and len(k) == 4),
        key=lambda t: t[0],
    )
    lines = [
        "| bins | grid (rows x cols) | ridge | mean abs delta a*b* | pairs |",
        "|---|---|---|---|---|",
    ]
    if base_mean is not None:
        lines.append(f"| bilinear (baseline) | - | - | {base_mean:.3f} | {len(baseline)} |")
    for mean, (bins, rows, cols, ridge), n in ranked:
        lines.append(f"| {bins} | {rows} x {cols} | {ridge:g} | {mean:.3f} | {n} |")
    return "\n".join(lines), (ranked[0][1] if ranked else None), base_mean


# --- the worst-10% sheet ----------------------------------------------------------------------


def worst_ids(
    rows: Sequence[Mapping], key: str = "psnr_grid_r1", fraction: float = 0.10
) -> list[str]:
    """IDs of the worst `fraction` of `rows` by `key` (at least one), worst first."""
    count = max(1, int(round(len(rows) * fraction)))
    return [r["id"] for r in sorted(rows, key=lambda r: r[key])[:count]]


@torch.no_grad()
def write_worst_sheets(
    dataset,
    ids: Sequence[str],
    rows: Sequence[Mapping],
    cfg: Mapping,
    out_dir: str | Path,
    device: torch.device | str = "cpu",
    scenes_per_sheet: int = 10,
    prefix: str = "worst10_grid_only",
) -> list[Path]:
    """Contact sheets of the scenes the grid cannot express: one row per scene,
    thermal | RGB | true L* | oracle L* | |delta L*|, labelled with the ID and the PSNR.

    The oracle is the grid-only fit at the configured refine steps, rendered at the input size.
    """
    off, _ = layouts(cfg)
    size = tuple(int(v) for v in cfg["encoder"]["input_size"])
    base = RenderSettings.from_config(cfg)
    settings = RenderSettings(off, base.bgu_grid, base.bgu_ridge, base.bgu_refine)
    fit = FitSettings.from_config(cfg)
    by_id = {image_id: index for index, image_id in enumerate(dataset.image_ids)}
    psnr = {r["id"]: r["psnr_grid_r1"] for r in rows}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, scenes = [], []
    for image_id in ids:
        raw = default_collate([dataset[by_id[image_id]]])
        batch = prepare_batch(raw, size, cfg["pairs"], device)
        targets = build_targets(batch["thermal"], batch["rgb"], batch["mask"], off, fit)
        oracle = render(targets.parts, batch["thermal"], size, settings)["lab"][:, :1]
        true_l, mask = targets.lab[:, :1], batch["mask"]
        error = ((oracle - true_l).abs() * mask / 30.0).clamp(0.0, 1.0)
        scenes.append(
            (
                [
                    thumbnail_thermal(batch["thermal"][0].cpu()),
                    thumbnail(batch["rgb"][0].cpu()),
                    thumbnail_thermal((true_l[0] / 100.0).cpu()),
                    thumbnail_thermal((oracle[0] / 100.0).cpu()),
                    thumbnail_thermal(error[0].cpu()),
                ],
                [
                    f"{image_id} thermal",
                    "RGB",
                    "true L*",
                    f"oracle L* {psnr[image_id]:.1f} dB",
                    "|dL*| 0-30",
                ],
            )
        )
    for start in range(0, len(scenes), scenes_per_sheet):
        chunk = scenes[start : start + scenes_per_sheet]
        thumbs = [t for s in chunk for t in s[0]]
        labels = [label for s in chunk for label in s[1]]
        number = start // scenes_per_sheet + 1
        sheet = tile_sheet(
            thumbs,
            labels,
            f"Worst 10% grid-only oracle scenes, {start + 1}-{start + len(chunk)} of {len(scenes)}"
            " (worst first): thermal | RGB | true L* | oracle L* | |dL*|",
            columns=5,
        )
        path = out_dir / f"{prefix}_{number:02d}.jpg"
        sheet.save(path, quality=90)
        paths.append(path)
    return paths


def save_rows(rows: Sequence[Mapping], path: str | Path) -> None:
    """Per-pair numbers as JSON lines, for later comparison and plots."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
