#!/usr/bin/env python3
"""Plot a campaign's training telemetry and test results as the README's figures.

Reads only the local mirrors that `dawn.runlog.RunLog` writes beside every run
(`<run>/metrics/<kind>.csv`, `<run>/run_config.json`) and the evaluation's `per_pair.csv`, so no
W&B access is needed. `--runs` points at a directory laid out like a campaign's `runs/`:

    baseline/   overfit/   refiner/   evaluation/per_pair.csv

`results/` in the repo is that layout with the checkpoints left out (committed); `--export`
refreshes it from a real `runs/`. Every figure is written twice, `<name>.png` for a light page and
`<name>_dark.png` for a dark one, and every number the README quotes from these curves is written
to `numbers.json` next to them.

    python scripts/plot_runs.py                          # results/ -> docs/images/plots/
    python scripts/plot_runs.py --runs runs              # straight from a campaign's out-dirs
    python scripts/plot_runs.py --runs runs --export results   # copy the small files, then plot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FLOW_RUNS = ("baseline", "overfit", "gpu_probe")
EXPORT_RUNS = (*FLOW_RUNS, "refiner")
EVAL_FILES = ("report.md", "summary.json", "per_pair.csv")
ORACLE_FILES = ("report.md", "rows_val.jsonl")
NUM_GATES = 6

# The reference palette (categorical slots 1-6 in their validated order) and its chart chrome.
THEMES: Mapping[str, Mapping] = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "band": "#eeede8",
        "band_ramp": "#f5f5f1",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "band": "#2a2a28",
        "band_ramp": "#222221",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"),
    },
}


# --- reading -------------------------------------------------------------------------------------


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a RunLog CSV mirror into float columns; empty cells become NaN."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no rows")
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [row.get(key, "") for row in rows]
        columns[key] = np.array([float(v) if v not in ("", None) else math.nan for v in values])
    return columns


def read_per_pair(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def schedule_of(run_dir: Path) -> tuple[int, int, int]:
    """Return (warmup_steps, warmup_ramp_steps, num_train_steps) from a run's saved config."""
    flow = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))["flow"]
    warmup = int(flow["warmup_steps"])
    ramp = flow.get("warmup_ramp_steps")
    ramp = max(1, warmup // 6) if ramp is None else int(ramp)
    return warmup, ramp, int(flow["num_train_steps"])


def column(table: Mapping[str, np.ndarray], suffix: str) -> np.ndarray:
    """The one column whose name ends in `suffix` (CSV columns carry their kind as a prefix)."""
    matches = [key for key in table if key == suffix or key.endswith("/" + suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one column ending in {suffix!r}, found {matches}")
    return table[matches[0]]


# --- numbers -------------------------------------------------------------------------------------


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (ties are vanishingly rare in float metrics; no tie correction)."""
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    return float(np.corrcoef(ra, rb)[0, 1])


def lstar_rmse(psnr: np.ndarray) -> np.ndarray:
    """L* RMSE implied by an L* PSNR with the peak at 100."""
    return 100.0 * 10.0 ** (-psnr / 20.0)


def subset(rows: list[dict[str, str]], name: str, method: str = "euler") -> dict[str, np.ndarray]:
    picked = [r for r in rows if r["subset"] == name and r["method"] == method]
    keys = ("psnr", "ciede2000", "l_std", "ab_std", "colorfulness", "colorfulness_target")
    return {key: np.array([float(r[key]) for r in picked]) for key in keys}


def gate_numbers(train: Mapping[str, np.ndarray], warmup: int) -> dict[str, dict[str, float]]:
    steps = train["step"]
    out = {}
    for i in range(NUM_GATES):
        gate = column(train, f"gate_{i}")
        peak = int(np.nanargmax(gate))
        at_warmup = int(np.argmin(np.abs(steps - warmup)))
        out[f"block_{i}"] = {
            "at_warmup": round(float(gate[at_warmup]), 3),
            "peak": round(float(gate[peak]), 3),
            "peak_step": int(steps[peak]),
            "final": round(float(gate[-1]), 3),
        }
    return out


def derive_numbers(data: Mapping) -> dict:
    """Every number the README quotes from the curves, in one place."""
    base, val = data["baseline"]["train"], data["baseline"]["val"]
    warmup, ramp, total = data["baseline"]["schedule"]
    ciede = column(val, "ciede2000/euler")
    best = int(np.nanargmin(ciede))
    tail = val["step"] >= total - 4000
    numbers: dict = {
        "schedule": {"warmup_steps": warmup, "ramp_steps": ramp, "num_train_steps": total},
        "gates": gate_numbers(base, warmup),
        "validation": {
            "best_step": int(val["step"][best]),
            "best_ciede2000": round(float(ciede[best]), 3),
            "last_4k_ciede2000_range": round(float(np.ptp(ciede[tail])), 4),
            "retrieval_top1_final": round(float(column(val, "retrieval/top1")[-1]), 3),
            "retrieval_within1_final": round(float(column(val, "retrieval/within_1")[-1]), 3),
            "retrieval_within1_at_warmup": round(
                float(column(val, "retrieval/within_1")[val["step"] == warmup][0]), 3
            ),
        },
    }
    encoder_grad = column(base, "grad/encoder")
    before = (base["step"] >= warmup - 1000) & (base["step"] < warmup)
    after = (base["step"] >= warmup + ramp) & (base["step"] < warmup + ramp + 1000)
    numbers["encoder_grad_norm"] = {
        "1k_before_detach_lifts": round(float(np.nanmedian(encoder_grad[before])), 3),
        "1k_after_ramp": round(float(np.nanmedian(encoder_grad[after])), 3),
    }
    if "overfit" in data:
        overfit = data["overfit"]["train"]
        numbers["overfit_gates_final"] = {
            f"block_{i}": round(float(column(overfit, f"gate_{i}")[-1]), 3)
            for i in range(NUM_GATES)
        }
    if "per_pair" in data:
        per_frame = {}
        for name in ("box_aligned", "fallback"):
            for method in ("euler", "midpoint"):
                s = subset(data["per_pair"], name, method)
                if not len(s["psnr"]):
                    continue
                rmse = lstar_rmse(s["psnr"])
                per_frame[f"{name}/{method}"] = {
                    "frames": int(len(rmse)),
                    "spearman_l_std_vs_lstar_rmse": round(spearman(s["l_std"], rmse), 3),
                    "spearman_ab_std_vs_ciede2000": round(spearman(s["ab_std"], s["ciede2000"]), 3),
                    "mean_lstar_rmse": round(float(rmse.mean()), 2),
                    "mean_l_std": round(float(s["l_std"].mean()), 2),
                    "error_over_spread": round(float(rmse.mean() / s["l_std"].mean()), 2),
                    "colourfulness_ratio_of_means": round(
                        float(s["colorfulness"].mean() / s["colorfulness_target"].mean()), 3
                    ),
                }
        numbers["per_frame"] = per_frame
    if "refiner" in data:
        refiner = data["refiner"]
        numbers["refiner"] = {
            name: {
                "improvement_db_final": round(float(column(table, "psnr_L/improvement")[-1]), 4),
                "abs_residual_lstar_final": round(
                    float(column(table, "residual/abs_mean_Lstar")[-1]), 3
                ),
            }
            for name, table in refiner.items()
        }
    return numbers


# --- drawing -------------------------------------------------------------------------------------


def k_steps(x: float, _pos=None) -> str:
    return "0" if x == 0 else f"{x / 1000:g}K"


def apply_theme(theme: Mapping) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": theme["ink"],
            "axes.labelcolor": theme["ink2"],
            "axes.labelsize": 8.5,
            "axes.edgecolor": theme["axis"],
            "axes.linewidth": 0.8,
            "axes.facecolor": theme["surface"],
            "figure.facecolor": theme["surface"],
            "savefig.facecolor": theme["surface"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
            "xtick.labelcolor": theme["ink2"],
            "ytick.labelcolor": theme["ink2"],
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "text.color": theme["ink"],
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def style(ax, theme: Mapping, steps_axis: bool = True) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)
    if steps_axis:
        ax.xaxis.set_major_formatter(FuncFormatter(k_steps))


def shade_detach(
    ax, warmup: int, ramp: int, theme: Mapping, label: bool = False, label_y: float = 0.985
) -> None:
    """Grey band over the steps where the encoder's tokens reach the velocity net detached.

    The lighter band after it is the ramp, where that gradient is eased in. `label_y` (axes
    fraction, top of the text) moves the label off whatever curve starts in that corner.
    """
    ax.axvspan(0, warmup, color=theme["band"], lw=0, zorder=0)
    ax.axvspan(warmup, warmup + ramp, color=theme["band_ramp"], lw=0, zorder=0)
    if label:
        ax.text(
            0.012,
            label_y,
            "LoRA\ndetached",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=theme["muted"],
            linespacing=1.1,
        )


def smooth(y: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    out, acc = np.empty_like(y), math.nan
    for i, value in enumerate(y):
        acc = value if math.isnan(acc) else (1 - alpha) * acc + alpha * value
        out[i] = acc
    return out


def spread_positions(values: np.ndarray, gap: float, lo: float, hi: float) -> np.ndarray:
    """Move label anchors apart by at least `gap`, keeping them inside [lo, hi]."""
    order = np.argsort(values)
    pos = values[order].astype(float).copy()
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    if pos[-1] > hi:
        pos[-1] = hi
        for i in range(len(pos) - 2, -1, -1):
            pos[i] = min(pos[i], pos[i + 1] - gap)
    pos = np.maximum(pos, lo)
    out = np.empty_like(pos)
    out[order] = pos
    return out


def end_labels(ax, x_end, ys, texts, colors, theme, gap, dx) -> None:
    """End dots in the series colour, labels in ink, leader lines where a label had to move."""
    ys = np.asarray(ys, dtype=float)
    lo, hi = ax.get_ylim()
    anchors = spread_positions(ys, gap, lo, hi)
    for y, anchor, text, color in zip(ys, anchors, texts, colors):
        ax.plot(
            [x_end],
            [y],
            "o",
            ms=6,
            color=color,
            mec=theme["surface"],
            mew=1.5,
            zorder=5,
            clip_on=False,
        )
        ax.plot(
            [x_end, x_end + dx * 0.55],
            [y, anchor],
            color=theme["muted"],
            lw=0.6,
            zorder=4,
            clip_on=False,
        )
        ax.text(
            x_end + dx * 0.65,
            anchor,
            text,
            va="center",
            ha="left",
            fontsize=8,
            color=theme["ink2"],
            clip_on=False,
        )


def series_legend(fig, labels, colors, theme, y=1.0, ncol=None) -> None:
    handles = [Line2D([], [], color=c, lw=2) for c in colors]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol or len(labels),
        handlelength=1.6,
        columnspacing=1.4,
        labelcolor=theme["ink2"],
    )


# --- figures -------------------------------------------------------------------------------------


def fig_gates(data: Mapping, numbers: Mapping, theme: Mapping):
    """The six DINO-attention gates over the run, beside the same gates on 100 memorised frames."""
    colors = theme["series"]
    base = data["baseline"]["train"]
    warmup, ramp, total = data["baseline"]["schedule"]
    has_overfit = "overfit" in data
    fig = plt.figure(figsize=(9.0, 3.6))
    grid = fig.add_gridspec(
        1, 2 if has_overfit else 1, width_ratios=[3.0, 1.2] if has_overfit else [1]
    )
    ax = fig.add_subplot(grid[0, 0])
    style(ax, theme)
    shade_detach(ax, warmup, ramp, theme, label=True)
    steps = base["step"]
    finals = []
    for i in range(NUM_GATES):
        gate = column(base, f"gate_{i}")
        ax.plot(steps, gate, color=colors[i], lw=2, zorder=3)
        finals.append(gate[-1])
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("gate  g")
    ax.set_xlabel("training step")
    ax.set_title("Full run: 3,662 pairs, 20K steps")
    end_labels(
        ax,
        steps[-1],
        finals,
        [f"block {i}  {v:.2f}" for i, v in enumerate(finals)],
        colors,
        theme,
        gap=0.065,
        dx=total * 0.03,
    )
    peak = numbers["gates"]["block_5"]
    ax.plot(
        [peak["peak_step"]],
        [peak["peak"]],
        "o",
        ms=6,
        color=colors[5],
        mec=theme["surface"],
        mew=1.5,
        zorder=6,
    )
    ax.annotate(
        f"block 5 peaks at {peak['peak']:.2f} (step {peak['peak_step']:,}),\n"
        "then closes for the rest of the run",
        xy=(peak["peak_step"], peak["peak"]),
        xytext=(peak["peak_step"] + 3200, 0.17),
        fontsize=7.5,
        color=theme["ink2"],
        arrowprops={"arrowstyle": "-", "color": theme["muted"], "lw": 0.6},
    )
    if has_overfit:
        over = data["overfit"]["train"]
        o_warmup, o_ramp, o_total = data["overfit"]["schedule"]
        ax2 = fig.add_subplot(grid[0, 1], sharey=ax)
        style(ax2, theme)
        shade_detach(ax2, o_warmup, o_ramp, theme)
        for i in range(NUM_GATES):
            ax2.plot(over["step"], column(over, f"gate_{i}"), color=colors[i], lw=2, zorder=3)
        ax2.set_xlim(0, o_total)
        ax2.set_xlabel("training step")
        ax2.set_title("Overfit run: 100 frames")
        ax2.tick_params(labelleft=False)
    series_legend(fig, [f"block {i}" for i in range(NUM_GATES)], colors, theme, y=1.0)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.83, wspace=0.42)
    return fig


def fig_schedule(data: Mapping, theme: Mapping):
    """Learning rates, the conditioning-gradient scale and the EMA horizon over the run."""
    colors = theme["series"]
    base = data["baseline"]["train"]
    warmup, ramp, total = data["baseline"]["schedule"]
    steps = base["step"]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    ax = axes[0]
    style(ax, theme)
    shade_detach(ax, warmup, ramp, theme)
    for key, label, color in (
        ("lr_main", "velocity + head", colors[0]),
        ("lr_lora", "LoRA", colors[1]),
    ):
        lr = column(base, f"loss/{key}")
        ax.plot(steps, lr * 1e4, color=color, lw=2, label=label)
    ax.set_title("Learning rate (×1e-4)")
    ax.legend(loc="upper right", labelcolor=theme["ink2"])
    ax = axes[1]
    style(ax, theme)
    shade_detach(ax, warmup, ramp, theme, label=True)
    ax.plot(steps, column(base, "loss/cond_scale"), color=colors[0], lw=2)
    ax.set_ylim(-0.03, 1.12)
    ax.set_title("Velocity gradient into LoRA,  s(k)")
    ax = axes[2]
    style(ax, theme)
    shade_detach(ax, warmup, ramp, theme)
    decay = column(base, "loss/ema_decay")
    ax.plot(steps, 1.0 / (1.0 - decay), color=colors[0], lw=2)
    ax.set_yscale("log")
    ax.set_title("EMA horizon  1/(1-β)  (steps)")
    for ax in axes:
        ax.set_xlim(0, total)
        ax.set_xlabel("training step")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.2, top=0.86, wspace=0.28)
    return fig


def fig_losses(data: Mapping, theme: Mapping):
    """Each training term, train (smoothed) against the same objective on held-out pairs."""
    colors = theme["series"]
    base, held = data["baseline"]["train"], data["baseline"]["valloss"]
    warmup, ramp, total = data["baseline"]["schedule"]
    panels = (
        ("total", "Total objective"),
        ("flow_brightness", "L_flow · brightness segment"),
        ("flow_color", "L_flow · colour segment"),
        ("render", "L_render  (Lab L1 / 100)"),
        ("align", "L_align  (1 − cos)"),
        ("color", "L_color  (CE, 261 bins)"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.9))
    for ax, (key, title) in zip(axes.flat, panels):
        style(ax, theme)
        shade_detach(ax, warmup, ramp, theme, label=key == "total", label_y=0.2)
        train = column(base, f"loss/{key}")
        ax.plot(base["step"], train, color=colors[0], lw=0.8, alpha=0.3)
        ax.plot(base["step"], smooth(train), color=colors[0], lw=2)
        ax.plot(held["step"], column(held, f"loss/{key}"), color=colors[1], lw=2)
        ax.set_title(title)
        ax.set_xlim(0, total)
        finite = np.concatenate([train[np.isfinite(train)], column(held, f"loss/{key}")])
        top = np.nanpercentile(finite, 99)
        ax.set_ylim(min(0.0, np.nanmin(finite)), top * 1.08)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    series_legend(
        fig,
        ["train (one batch per point, smoothed)", "held-out box-aligned val (8 batches)"],
        colors[:2],
        theme,
        y=1.0,
    )
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.1, top=0.88, wspace=0.28, hspace=0.42)
    return fig


def fig_validation(data: Mapping, numbers: Mapping, theme: Mapping):
    """Sampled image quality on 64 box-aligned val pairs (EMA weights), against their ceiling."""
    colors = theme["series"]
    val = data["baseline"]["val"]
    warmup, ramp, total = data["baseline"]["schedule"]
    steps = val["step"]
    best = numbers["validation"]["best_step"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.0))
    panels = (
        ("ciede2000", "CIEDE2000  (lower is better)", "ceiling"),
        ("psnr_L", "L* PSNR, dB  (higher is better)", "ceiling"),
        ("colorfulness", "Colourfulness / real photo", "ceiling"),
    )
    for ax, (key, title, ceiling_label) in zip(axes.flat, panels):
        style(ax, theme)
        shade_detach(ax, warmup, ramp, theme, label=key == "colorfulness", label_y=0.8)
        if key == "colorfulness":
            euler = column(val, "colorfulness/ratio_euler")
            midpoint = column(val, "colorfulness/ratio_midpoint")
            ceiling = column(val, "colorfulness/ratio_oracle")
        else:
            euler = column(val, f"{key}/euler")
            midpoint = column(val, f"{key}/midpoint")
            ceiling = column(val, f"{key}/oracle")
        ax.plot(steps, euler, color=colors[0], lw=2, zorder=3)
        ax.plot(steps, midpoint, color=colors[1], lw=2, zorder=3)
        ax.axhline(ceiling[-1], color=theme["muted"], lw=1, ls=(0, (4, 3)), zorder=2)
        ax.text(
            total * 0.985,
            ceiling[-1],
            f"{ceiling_label} {ceiling[-1]:.2f}",
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=theme["ink2"],
        )
        ax.axvline(best, color=theme["axis"], lw=0.8, zorder=1)
        ax.set_title(title)
        ax.set_xlim(0, total)
    axes[0, 0].text(
        best,
        0.985,
        f" best (EMA)\n {best / 1000:g}K",
        transform=axes[0, 0].get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=7.5,
        color=theme["muted"],
    )
    ax = axes[1, 1]
    style(ax, theme)
    shade_detach(ax, warmup, ramp, theme)
    within = column(val, "retrieval/within_1")
    top1 = column(val, "retrieval/top1")
    ax.plot(steps, within, color=colors[2], lw=2)
    ax.plot(steps, top1, color=colors[3], lw=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, total)
    ax.set_title("Patch retrieval: thermal token → own RGB token")
    for y, text in (
        (within[-1], f"within one patch  {within[-1]:.2f}"),
        (top1[-1], f"exact patch  {top1[-1]:.2f}"),
    ):
        ax.text(
            total * 0.985, y + 0.03, text, ha="right", va="bottom", fontsize=8, color=theme["ink2"]
        )
    ax.text(
        total * 0.985,
        0.04,
        "chance 1/1,280 = 0.0008",
        ha="right",
        fontsize=7.5,
        color=theme["muted"],
    )
    for ax in axes[1]:
        ax.set_xlabel("training step")
    series_legend(fig, ["Euler, NFE 2", "midpoint, NFE 2"], colors[:2], theme, y=1.0)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.09, top=0.89, wspace=0.22, hspace=0.42)
    return fig


def _scatter(ax, x, y, theme, color) -> None:
    ax.scatter(x, y, s=11, color=color, alpha=0.55, linewidths=0.3, edgecolors=theme["surface"])


def fig_per_frame(data: Mapping, numbers: Mapping, theme: Mapping):
    """Per test frame: sample spread against realized error, and colourfulness against the photo."""
    colors = theme["series"]
    s = subset(data["per_pair"], "box_aligned")
    stats = numbers["per_frame"]["box_aligned/euler"]
    rmse = lstar_rmse(s["psnr"])
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    panels = (
        (
            s["l_std"],
            rmse,
            "spread  l_std  (L*)",
            "realized L* RMSE",
            "Brightness: spread vs error",
            f"Spearman ρ = {stats['spearman_l_std_vs_lstar_rmse']:.2f}",
        ),
        (
            s["ab_std"],
            s["ciede2000"],
            "spread  ab_std  (a*b*)",
            "realized CIEDE2000",
            "Colour: spread vs error",
            f"Spearman ρ = {stats['spearman_ab_std_vs_ciede2000']:.2f}",
        ),
        (
            s["colorfulness_target"],
            s["colorfulness"],
            "colourfulness of the real photo",
            "colourfulness of Dawn's output",
            "Colourfulness",
            f"ratio of means {stats['colourfulness_ratio_of_means']:.2f}",
        ),
    )
    for ax, (x, y, xlabel, ylabel, title, note) in zip(axes, panels):
        style(ax, theme, steps_axis=False)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
        _scatter(ax, x, y, theme, colors[0])
        hi = float(max(np.percentile(x, 99.5), np.percentile(y, 99.5))) * 1.05
        ax.plot([0, hi], [0, hi], color=theme["muted"], lw=1, ls=(0, (4, 3)))
        ax.set_xlim(0, hi if title == "Colourfulness" else float(np.percentile(x, 99.5)) * 1.15)
        ax.set_ylim(0, hi if title == "Colourfulness" else float(np.percentile(y, 99.5)) * 1.08)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.text(0.04, 0.95, note, transform=ax.transAxes, va="top", fontsize=8, color=theme["ink"])
    ratio = stats["colourfulness_ratio_of_means"]
    hi = axes[2].get_xlim()[1]
    axes[2].plot([0, hi], [0, ratio * hi], color=colors[1], lw=1.5)
    axes[2].text(
        hi * 0.98,
        ratio * hi * 0.93,
        f"{ratio:.2f} × the photo",
        ha="right",
        va="top",
        fontsize=7.5,
        color=theme["ink2"],
    )
    axes[2].text(
        hi * 0.97,
        hi * 0.97,
        "as colourful\nas the photo",
        ha="right",
        va="top",
        fontsize=7.5,
        color=theme["ink2"],
    )
    axes[0].text(
        0.96,
        0.05,
        "dashed: spread = error",
        transform=axes[0].transAxes,
        ha="right",
        fontsize=7.5,
        color=theme["ink2"],
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.17, top=0.9, wspace=0.3)
    return fig


def oracle_columns(rows: list[dict]) -> dict[str, np.ndarray]:
    """The configured oracle (grid only, refine 1, bilinear colour) on box-aligned val pairs."""
    box = [r for r in rows if not r["fallback"]]
    return {
        "psnr": np.array([r["psnr_grid_r1"] for r in box], dtype=float),
        "ciede2000": np.array([r["de_grid_only_bilinear"] for r in box], dtype=float),
    }


def fig_test_distributions(data: Mapping, theme: Mapping):
    """Per-frame L* PSNR and CIEDE2000 on the test split, beside the oracle's per-frame ceiling."""
    colors = theme["series"]
    box, fallback = subset(data["per_pair"], "box_aligned"), subset(data["per_pair"], "fallback")
    oracle = oracle_columns(data["oracle"]) if "oracle" in data else None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.9))
    for ax, key, label in ((axes[0], "psnr", "L* PSNR, dB"), (axes[1], "ciede2000", "CIEDE2000")):
        style(ax, theme, steps_axis=False)
        groups = [(box, colors[0], "-"), (fallback, colors[1], "-")]
        if oracle is not None:
            groups.append((oracle, theme["muted"], (0, (4, 3))))
        values = np.concatenate([g[0][key] for g in groups])
        bins = np.linspace(np.percentile(values, 0.5), np.percentile(values, 99.5), 30)
        for index, (s, color, ls) in enumerate(groups):
            ax.hist(s[key], bins=bins, density=True, histtype="step", lw=2, color=color, ls=ls)
            ax.axvline(s[key].mean(), color=color, lw=1, ls=ls)
            if index != 1:  # the fallback mean sits on the box-aligned one; the legend has it
                ax.text(
                    s[key].mean(),
                    1.0,
                    f" {s[key].mean():.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    color=theme["ink2"],
                )
        ax.set_xlabel(f"{label}, one value per frame (vertical lines: means)")
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
    labels = [
        f"Dawn, box-aligned test ({len(box['psnr'])})",
        f"Dawn, fallback test ({len(fallback['psnr'])})",
    ]
    legend_colors = list(colors[:2])
    if oracle is not None:
        labels.append(f"oracle ceiling, box-aligned val ({len(oracle['psnr'])})")
        legend_colors.append(theme["muted"])
    series_legend(fig, labels, legend_colors, theme, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.19, top=0.84, wspace=0.12)
    return fig


def fig_refiner(data: Mapping, theme: Mapping):
    """Stage 2: what the refiner adds over the unrefined grid, and how far it moves L*."""
    colors = theme["series"]
    tables = data["refiner"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.6))
    names = [n for n in ("val_box_aligned", "val_fallback") if n in tables]
    for i, name in enumerate(names):
        table = tables[name]
        axes[0].plot(table["step"], column(table, "psnr_L/improvement"), color=colors[i], lw=2)
        axes[1].plot(table["step"], column(table, "residual/abs_mean_Lstar"), color=colors[i], lw=2)
    axes[0].axhline(0, color=theme["axis"], lw=0.8)
    axes[0].set_title("Refined − unrefined L* PSNR, dB  (gate: ≥ 0)")
    axes[1].set_title("Mean |residual| the refiner applies, L*")
    for ax in axes:
        style(ax, theme)
        ax.set_xlabel("refiner step")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    labels = {"val_box_aligned": "box-aligned val", "val_fallback": "fallback val"}
    series_legend(fig, [labels[n] for n in names], colors[: len(names)], theme, y=1.0)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.8, wspace=0.22)
    return fig


# --- main ----------------------------------------------------------------------------------------


def load(runs: Path) -> dict:
    data: dict = {}
    for name in ("baseline", "overfit"):
        metrics = runs / name / "metrics"
        if not (metrics / "train.csv").exists():
            continue
        data[name] = {
            "train": read_csv(metrics / "train.csv"),
            "val": read_csv(metrics / "val_box_aligned.csv"),
            "valloss": read_csv(metrics / "valloss_box_aligned.csv"),
            "schedule": schedule_of(runs / name),
        }
    if "baseline" not in data:
        raise FileNotFoundError(f"{runs / 'baseline' / 'metrics' / 'train.csv'} is required")
    refiner = runs / "refiner" / "metrics"
    if refiner.exists():
        data["refiner"] = {p.stem: read_csv(p) for p in sorted(refiner.glob("val_*.csv"))}
    per_pair = runs / "evaluation" / "per_pair.csv"
    if per_pair.exists():
        data["per_pair"] = read_per_pair(per_pair)
    oracle = runs / "oracle" / "rows_val.jsonl"
    if oracle.exists():
        lines = oracle.read_text(encoding="utf-8").splitlines()
        data["oracle"] = [json.loads(line) for line in lines if line.strip()]
    return data


def export(runs: Path, dest: Path, oracle: Path | None) -> None:
    """Copy the small, checkpoint-free part of a campaign into `dest` (the committed results/)."""
    for name in EXPORT_RUNS:
        src = runs / name
        if not src.exists():
            continue
        for path in [src / "run_config.json", *sorted((src / "metrics").glob("*.csv"))]:
            if path.exists():
                target = dest / path.relative_to(runs)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for name in EVAL_FILES:
        if (runs / "evaluation" / name).exists():
            (dest / "evaluation").mkdir(parents=True, exist_ok=True)
            shutil.copy2(runs / "evaluation" / name, dest / "evaluation" / name)
    if (runs / "pipeline_state.json").exists():
        shutil.copy2(runs / "pipeline_state.json", dest / "pipeline_state.json")
    if oracle is not None:
        for name in ORACLE_FILES:
            if (oracle / name).exists():
                (dest / "oracle").mkdir(parents=True, exist_ok=True)
                shutil.copy2(oracle / name, dest / "oracle" / name)
    logger.info("exported telemetry from %s to %s", runs, dest)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "results",
        help="campaign directory: baseline/, overfit/, refiner/, evaluation/ "
        "(default: the committed results/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "images" / "plots",
        help="where the PNGs and numbers.json go (default: docs/images/plots)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        metavar="DIR",
        help="first copy the CSV mirrors, run configs and evaluation files from "
        "--runs into DIR (e.g. results), then plot from DIR",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=ROOT / "outputs" / "oracle",
        help="with --export: the check_oracle.py output to copy as well (default: outputs/oracle)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(THEMES),
        default=["light", "dark"],
        help="which page themes to render (default: both)",
    )
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args(argv)

    runs = args.runs
    if args.export is not None:
        export(runs, args.export, args.oracle)
        runs = args.export
    data = load(runs)
    numbers = derive_numbers(data)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n", encoding="utf-8")

    figures = {
        "gates": lambda t: fig_gates(data, numbers, t),
        "schedule": lambda t: fig_schedule(data, t),
        "losses": lambda t: fig_losses(data, t),
        "validation": lambda t: fig_validation(data, numbers, t),
    }
    if "per_pair" in data:
        figures["per_frame"] = lambda t: fig_per_frame(data, numbers, t)
        figures["test_distributions"] = lambda t: fig_test_distributions(data, t)
    if "refiner" in data:
        figures["refiner"] = lambda t: fig_refiner(data, t)

    for mode in args.modes:
        theme = THEMES[mode]
        apply_theme(theme)
        suffix = "" if mode == "light" else f"_{mode}"
        for name, build in figures.items():
            fig = build(theme)
            path = args.out / f"{name}{suffix}.png"
            fig.savefig(path, dpi=args.dpi)
            plt.close(fig)
            logger.info("wrote %s", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    logger.info("wrote %s", args.out / "numbers.json")


if __name__ == "__main__":
    main()
