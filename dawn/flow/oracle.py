"""The brightness and colour oracle: the best any model could do with this parametrization.

`scripts/check_oracle.py` is a thin CLI over this module. Per pair, with the pixel mask of
`configs/flow.yaml: pairs`, at the configured resolution, it measures (all against the RGB's
CIELAB):

    grid-only    the fitted brightness grid, rendered, at refine 0 and 1: L* PSNR (peak 100)
    + correction the same grid plus the generated-L correction at the colour-map size, upsampled
                 bilinearly and by the BGU (both through `render`)
    colour       the true a*b* area-averaged to the colour-map size and brought back up by the BGU
                 guided by the true L, against bilinear: mean |delta a*b*|
    assembled    the whole oracle image through `render` (grid-only and with the correction, each
                 upsample mode): CIEDE2000 next to the L* PSNR

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
from torch.utils.data import DataLoader, Subset, default_collate

from dawn.data.resample import area_downsample
from dawn.data.sheets import thumbnail, thumbnail_thermal, tile_sheet
from dawn.eval.metrics import ciede2000, masked_mean, masked_psnr
from dawn.flow.render import RenderSettings, render
from dawn.flow.targets import FitSettings, build_targets, prepare_batch
from dawn.flow.tokens import TokenLayout
from dawn.grid.upsample import upsample_map

logger = logging.getLogger(__name__)

MODES = ("bilinear", "bgu")

METRICS = (
    "psnr_grid_r0",
    "psnr_grid_r1",
    "psnr_corr_bilinear_r0",
    "psnr_corr_bilinear_r1",
    "psnr_corr_bgu_r0",
    "psnr_corr_bgu_r1",
    "ab_err_bilinear",
    "ab_err_bgu",
    "de_grid_only_bilinear",
    "de_grid_only_bgu",
    "de_with_correction_bilinear",
    "de_with_correction_bgu",
)
LABELS = {
    "psnr_grid_r0": "grid only, refine 0 (L* PSNR dB)",
    "psnr_grid_r1": "grid only, refine 1 (L* PSNR dB)",
    "psnr_corr_bilinear_r0": "grid + correction (bilinear), refine 0 (dB)",
    "psnr_corr_bilinear_r1": "grid + correction (bilinear), refine 1 (dB)",
    "psnr_corr_bgu_r0": "grid + correction (BGU), refine 0 (dB)",
    "psnr_corr_bgu_r1": "grid + correction (BGU), refine 1 (dB)",
    "ab_err_bilinear": "colour ceiling, bilinear (mean abs delta a*b*)",
    "ab_err_bgu": "colour ceiling, BGU (mean abs delta a*b*)",
    "de_grid_only_bilinear": "assembled oracle, grid only, bilinear colour (CIEDE2000)",
    "de_grid_only_bgu": "assembled oracle, grid only, BGU colour (CIEDE2000)",
    "de_with_correction_bilinear": "assembled oracle, with correction, bilinear (CIEDE2000)",
    "de_with_correction_bgu": "assembled oracle, with correction, BGU (CIEDE2000)",
}


def layouts(cfg: Mapping) -> tuple[TokenLayout, TokenLayout]:
    """`(correction off, correction on)` layouts at the configured resolution."""
    off = dataclasses.replace(TokenLayout.from_config(cfg), correction=False)
    return off, dataclasses.replace(off, correction=True)


def measure_batch(batch: Mapping, cfg: Mapping) -> dict[str, torch.Tensor]:
    """Every oracle metric for one prepared batch (`prepare_batch`), each `(B,)`.

    Both upsample modes are always measured (whatever `color.upsample` is), so the report keeps
    the BGU's cost against bilinear in view.
    """
    off, on = layouts(cfg)
    ridge = float(cfg["grid"]["ridge_alpha"])
    size = tuple(int(v) for v in cfg["encoder"]["input_size"])
    base = RenderSettings.from_config(cfg)

    def settings(layout: TokenLayout, mode: str) -> RenderSettings:
        return dataclasses.replace(base, layout=layout, upsample=mode)

    thermal, rgb, mask = batch["thermal"], batch["rgb"], batch["mask"]
    out: dict[str, torch.Tensor] = {}
    for refine in (0, 1):
        fit = FitSettings(ridge, refine)
        plain = build_targets(thermal, rgb, mask, off, fit)
        withc = build_targets(thermal, rgb, mask, on, fit)
        true_l = plain.lab[:, :1]
        out[f"psnr_grid_r{refine}"] = masked_psnr(100.0 * plain.L_grid, true_l, mask, 100.0)
        corrected = {mode: render(withc.parts, thermal, size, settings(on, mode)) for mode in MODES}
        for mode, rendered in corrected.items():
            out[f"psnr_corr_{mode}_r{refine}"] = masked_psnr(
                rendered["lab"][:, :1], true_l, mask, 100.0
            )
        if refine == 1:
            for mode in MODES:
                grid_only = render(plain.parts, thermal, size, settings(off, mode))
                out[f"de_grid_only_{mode}"] = masked_mean(
                    ciede2000(grid_only["lab"], plain.lab), mask
                )
                out[f"de_with_correction_{mode}"] = masked_mean(
                    ciede2000(corrected[mode]["lab"], plain.lab), mask
                )
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
    """True a*b* at the colour-map size, brought back to `size` by the BGU (guided by the true L,
    at the configured BGU settings) and by plain bilinear upsampling: `(bgu, bilinear)`."""
    low = targets.parts["ab"]
    guide_hi = targets.lab[:, :1] / 100.0
    guide_lo = area_downsample(guide_hi, layout.color_size)
    kwargs = {
        "grid_size": render_settings.bgu_grid,
        "ridge_alpha": render_settings.bgu_ridge,
        "refine_steps": render_settings.bgu_refine,
    }
    up = upsample_map(low, guide_lo, guide_hi, "bgu", targets.fraction, **kwargs)
    return up, upsample_map(low, guide_lo, guide_hi, "bilinear")


def bgu_sweep_batch(
    batch: Mapping,
    cfg: Mapping,
    settings: Sequence[tuple[int, int, int, float, int]],
) -> dict[tuple, torch.Tensor]:
    """Colour-ceiling error `(B,)` of each BGU setting `(bins, rows, cols, ridge, refine)` on a
    batch, plus the bilinear baseline under the key `("bilinear",)`."""
    off, _ = layouts(cfg)
    targets = build_targets(
        batch["thermal"], batch["rgb"], batch["mask"], off, FitSettings(1e-3, 0)
    )
    true_ab, mask = targets.lab[:, 1:], batch["mask"]
    guide_hi = targets.lab[:, :1] / 100.0
    guide_lo = area_downsample(guide_hi, off.color_size)
    low = targets.parts["ab"]

    def error(up: torch.Tensor) -> torch.Tensor:
        return masked_mean(torch.linalg.vector_norm(up - true_ab, dim=1), mask)

    out = {("bilinear",): error(upsample_map(low, guide_lo, guide_hi, "bilinear"))}
    for bins, rows, cols, ridge, refine in settings:
        up = upsample_map(
            low,
            guide_lo,
            guide_hi,
            "bgu",
            targets.fraction,
            grid_size=(bins, rows, cols),
            ridge_alpha=ridge,
            refine_steps=refine,
        )
        out[(bins, rows, cols, ridge, refine)] = error(up)
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
    sweep: Sequence[tuple[int, int, int, float, int]] = (),
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
        ((float(np.mean(v)), k, len(v)) for k, v in sweeps.items() if len(v) and len(k) == 5),
        key=lambda t: t[0],
    )
    lines = [
        "| bins | grid (rows x cols) | ridge | refine | mean abs delta a*b* | pairs |",
        "|---|---|---|---|---|---|",
    ]
    if base_mean is not None:
        lines.append(f"| bilinear (baseline) | - | - | - | {base_mean:.3f} | {len(baseline)} |")
    for mean, (bins, rows, cols, ridge, refine), n in ranked:
        lines.append(f"| {bins} | {rows} x {cols} | {ridge:g} | {refine} | {mean:.3f} | {n} |")
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
