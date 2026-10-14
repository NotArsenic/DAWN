#!/usr/bin/env python3
"""Plot a campaign's training telemetry and test results as the README's figures.

Reads only local files: the mirrors that `dawn.runlog.RunLog` writes beside every run
(`<run>/metrics/<kind>.csv`, `<run>/run_config.json`), the evaluation's `per_pair.csv`, the
oracle's `rows_val.jsonl` and, when present, `baseline/histograms.npz`: the weight and gradient
histograms that `wandb.watch` recorded, extracted from the run's local `.wandb` log by
`--wandb-log`. `--runs` points at a directory laid out like a campaign's `runs/`:

    baseline/   overfit/   refiner/   evaluation/   oracle/

`results/` in the repo is that layout with the checkpoints left out (committed); `--export`
refreshes it from a real `runs/`. Every figure is written twice, `<name>.svg` for a light page and
`<name>_dark.svg` for a dark one (white and black backgrounds), and every
number the README quotes from these curves is written to `numbers.json` next to them.

    python scripts/plot_runs.py                          # results/ -> docs/images/plots/
    python scripts/plot_runs.py --runs runs              # straight from a campaign's out-dirs
    python scripts/plot_runs.py --runs runs --export results --oracle outputs/oracle \\
        --wandb-log wandb/run-<id>/run-<id>.wandb        # refresh results/, then plot
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import struct
import zlib
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import PowerNorm, SymLogNorm  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FuncFormatter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
FLOW_RUNS = ("baseline", "overfit", "gpu_probe")
EXPORT_RUNS = (*FLOW_RUNS, "refiner")
EVAL_FILES = ("report.md", "summary.json", "per_pair.csv")
ORACLE_FILES = ("report.md", "rows_val.jsonl")
NUM_GATES = 6
WIDTH = 7.0  # inches: a double-column figure

# The wandb.watch histograms kept in results/: the zero-initialised paths, a randomly
# initialised one for contrast, and the scalar attention gates (whose "histogram" is one value).
HIST_TENSORS = tuple(
    [f"blocks.{i}.adaln_{s}.proj.weight" for i in range(NUM_GATES) for s in ("attn", "mlp")]
    + ["output_proj.weight", "final_adaln.proj.weight"]
    + [f"blocks.{i}.qkv.weight" for i in range(NUM_GATES)]
    + [f"blocks.{i}.dino_gate" for i in range(NUM_GATES)]
)
ADALN = tuple(n for n in HIST_TENSORS if ".adaln_" in n)
QKV = tuple(n for n in HIST_TENSORS if n.endswith(".qkv.weight"))

# Paul Tol's colour-blind-safe schemes: "bright" on a light page, "light" on a dark one.
THEMES: Mapping[str, Mapping] = {
    "light": {
        "background": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "band": "#1f2328",
        "band_alpha": 0.07,
        "blue": "#4477AA",
        "red": "#EE6677",
        "green": "#228833",
        "yellow": "#CCBB44",
        "grey": "#BBBBBB",
        "gates": (0.0, 0.85),
        "density": "Blues",
        "diverging": "RdBu",
    },
    "dark": {
        "background": "#000000",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "band": "#e6edf3",
        "band_alpha": 0.14,
        "blue": "#77AADD",
        "red": "#EE8866",
        "green": "#44BB99",
        "yellow": "#EEDD88",
        "grey": "#6e7681",
        "gates": (0.3, 1.0),
        "density": "magma",
        "diverging": "berlin_r",
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


def wandb_records(path: Path) -> Iterator[bytes]:
    """Yield the raw records of a local `.wandb` log.

    The file is a LevelDB-style log: a 7-byte file header (":W&B", magic 0xBEE1, version), then
    32 KiB blocks of records, each `<crc32 uint32><length uint16><type uint8>` plus data, a record
    split across blocks as FIRST, MIDDLE..., LAST pieces, and a block tail shorter than a header
    zero-padded. Read here directly because wandb >= 0.20 no longer ships a Python reader.
    """
    block, header = 32768, 7
    full, first, middle, last = 1, 2, 3, 4
    with open(path, "rb") as handle:
        ident, magic, _version = struct.unpack("<4sHB", handle.read(header))
        if ident != b":W&B" or magic != 0xBEE1:
            raise ValueError(f"{path} is not a W&B run log")
        index, pending = header, b""
        while True:
            left = block - index % block
            if left < header:
                handle.read(left)
                index += left
            head = handle.read(header)
            if len(head) < header:
                return
            checksum, length, kind = struct.unpack("<IHB", head)
            data = handle.read(length)
            index += header + length
            if zlib.crc32(data, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF != checksum:
                raise ValueError(f"{path}: record checksum mismatch at byte {index}")
            if kind == full:
                yield data
            elif kind == first:
                pending = data
            elif kind == middle:
                pending += data
            elif kind == last:
                yield pending + data


def read_wandb_histograms(path: Path, tensors: Sequence[str] = HIST_TENSORS) -> dict:
    """The `wandb.watch` histograms of `tensors` (weights and gradients) from a `.wandb` log."""
    from wandb.proto import wandb_internal_pb2  # only needed for this one export step

    wanted = {f"{kind}/{name}" for kind in ("parameters", "gradients") for name in tensors}
    found: dict[str, dict[str, list]] = {}
    for data in wandb_records(path):
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "history":
            continue
        items = {
            "/".join(item.nested_key) if item.nested_key else item.key: item.value_json
            for item in record.history.item
        }
        step = int(json.loads(items["_step"]))
        for key in wanted:
            if key + "/values" in items:
                entry = found.setdefault(key, {"steps": [], "counts": [], "bins": []})
                entry["steps"].append(step)
                entry["counts"].append(json.loads(items[key + "/values"]))
                entry["bins"].append(json.loads(items[key + "/bins"]))
    return {
        key: {
            "steps": np.asarray(e["steps"], dtype=np.int64),
            "counts": np.asarray(e["counts"], dtype=np.int64),
            "bins": np.asarray(e["bins"], dtype=np.float32),
        }
        for key, e in found.items()
    }


def save_histograms(histograms: Mapping, path: Path) -> None:
    flat = {f"{key}::{part}": value for key, e in histograms.items() for part, value in e.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_histograms(path: Path) -> dict:
    with np.load(path) as archive:
        out: dict[str, dict[str, np.ndarray]] = {}
        for flat in archive.files:
            key, part = flat.split("::")
            out.setdefault(key, {})[part] = archive[flat]
    return out


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


def scalar_series(entry: Mapping[str, np.ndarray]) -> np.ndarray:
    """The value a one-element tensor's histogram records (its bins collapse onto it)."""
    return 0.5 * (entry["bins"][:, 0] + entry["bins"][:, -1]).astype(float)


def gate_gradients(histograms: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """(steps, -dL/dg) per block: positive means a descent step opens that gate."""
    entries = [histograms[f"gradients/blocks.{i}.dino_gate"] for i in range(NUM_GATES)]
    return entries[0]["steps"], -np.stack([scalar_series(e) for e in entries])


def hist_std(entry: Mapping[str, np.ndarray]) -> np.ndarray:
    """Per-snapshot standard deviation of a histogram, from its bin centres."""
    bins, counts = entry["bins"].astype(float), entry["counts"].astype(float)
    mids = 0.5 * (bins[:, 1:] + bins[:, :-1])
    weight = counts / counts.sum(axis=1, keepdims=True)
    mean = (weight * mids).sum(axis=1, keepdims=True)
    return np.sqrt((weight * (mids - mean) ** 2).sum(axis=1))


def pooled_std(histograms: Mapping, kind: str, names: Sequence[str]) -> np.ndarray:
    """Pooled standard deviation over several tensors (equal-size tensors, so a plain RMS)."""
    return np.sqrt(np.mean([hist_std(histograms[f"{kind}/{n}"]) ** 2 for n in names], axis=0))


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
    if "histograms" in data:
        hist = data["histograms"]
        steps, push = gate_gradients(hist)
        turn = numbers["gates"]["block_5"]["peak_step"]
        early, late = steps <= turn, steps > turn
        numbers["gate_gradient_snapshots"] = {
            "turn_step": turn,
            **{
                f"block_{i}": {
                    "opening_before_turn": f"{int((push[i, early] > 0).sum())}/{int(early.sum())}",
                    "closing_after_turn": f"{int((push[i, late] < 0).sum())}/{int(late.sum())}",
                }
                for i in range(NUM_GATES)
            },
        }
        first, last = 0, -1
        adaln_w, adaln_g = (
            pooled_std(hist, "parameters", ADALN),
            pooled_std(hist, "gradients", ADALN),
        )
        qkv_w = pooled_std(hist, "parameters", QKV)
        out_w = hist_std(hist["parameters/output_proj.weight"])
        snap = hist[f"parameters/{ADALN[0]}"]["steps"]
        grad_snap = hist[f"gradients/{ADALN[0]}"]["steps"]
        numbers["weight_distributions"] = {
            "first_snapshot_step": int(snap[first]),
            "last_snapshot_step": int(snap[last]),
            "adaln_weight_std": [float(f"{adaln_w[first]:.3g}"), float(f"{adaln_w[last]:.3g}")],
            "output_proj_weight_std": [float(f"{out_w[first]:.3g}"), float(f"{out_w[last]:.3g}")],
            "qkv_weight_std": [float(f"{qkv_w[first]:.3g}"), float(f"{qkv_w[last]:.3g}")],
            "adaln_grad_std": [float(f"{adaln_g[first]:.3g}"), float(f"{adaln_g[last]:.3g}")],
            "adaln_grad_snapshot_steps": [int(grad_snap[first]), int(grad_snap[last])],
        }
        early = int(np.argmin(np.abs(grad_snap - 2400)))
        groups = {"adaln": ADALN, "output_proj": ("output_proj.weight",), "qkv": QKV}
        shrink = {name: pooled_std(hist, "gradients", keys) for name, keys in groups.items()}
        numbers["gradient_shrink"] = {
            "early_step": int(grad_snap[early]),
            **{
                name: {
                    "by_early_step": round(float(g[first] / g[early]), 1),
                    "overall": round(float(g[first] / g[last]), 1),
                }
                for name, g in shrink.items()
            },
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
        numbers["refiner"] = {
            name: {
                "improvement_db_final": round(float(column(table, "psnr_L/improvement")[-1]), 4),
                "abs_residual_lstar_final": round(
                    float(column(table, "residual/abs_mean_Lstar")[-1]), 3
                ),
            }
            for name, table in data["refiner"].items()
        }
    return numbers


# --- style ---------------------------------------------------------------------------------------


def apply_theme(theme: Mapping) -> None:
    """A journal style: STIX type with matching math, a closed frame, inward ticks, no grid."""
    ink = theme["ink"]
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": ink,
            "axes.edgecolor": ink,
            "axes.labelcolor": ink,
            "axes.titlecolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
            "axes.linewidth": 0.6,
            "axes.titlelocation": "left",
            "axes.titlepad": 4,
            "axes.labelpad": 2,
            "axes.facecolor": theme["background"],
            "figure.facecolor": theme["background"],
            "savefig.facecolor": theme["background"],
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "lines.linewidth": 1.1,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.borderaxespad": 0.3,
            "svg.fonttype": "path",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def panel(ax, letter: str, title: str = "") -> None:
    ax.set_title(rf"$\mathbf{{({letter})}}$  {title}", loc="left")


def steps_axis(ax, label: bool = True) -> None:
    """Training steps in thousands."""
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x / 1000:g}"))
    if label:
        ax.set_xlabel(r"Training step ($\times 10^3$)")


def shade_detach(
    ax,
    warmup: int,
    ramp: int,
    theme: Mapping,
    label: bool = False,
    label_y: float = 0.5,
    vertical: bool = True,
) -> None:
    """Grey band over the steps where the encoder's tokens reach the velocity net detached.

    The lighter band after it is the ramp, where that gradient is eased in. The label is
    centred at `label_y` (axes fraction), set vertically inside the band unless the band is wide
    enough for it to read across.
    """
    ax.axvspan(0, warmup, color=theme["band"], alpha=theme["band_alpha"], lw=0, zorder=0)
    ax.axvspan(
        warmup, warmup + ramp, color=theme["band"], alpha=theme["band_alpha"] / 2, lw=0, zorder=0
    )
    if label:
        ax.text(
            warmup * 0.5,
            label_y,
            "LoRA detached" if vertical else "LoRA\ndetached",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=6.5,
            style="italic",
            color=theme["muted"],
            linespacing=1.0,
            rotation=90 if vertical else 0,
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


def sci(value: float) -> str:
    """A tick label in math: 10^-2, -10^-2, 3 x 10^-3."""
    if value == 0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = abs(value) / 10.0**exponent
    sign = "-" if value < 0 else ""
    if abs(mantissa - 1.0) < 0.05:
        return rf"${sign}10^{{{exponent}}}$"
    return rf"${sign}{mantissa:.0f}\times 10^{{{exponent}}}$"


def symlog_ticks(limit: float, linthresh: float) -> list[float]:
    """Zero and every other decade between the linear threshold and the limit, both signs."""
    top = math.floor(math.log10(limit))
    exps = [e for e in range(top, math.floor(math.log10(linthresh)), -2)]
    return sorted([-(10.0**e) for e in exps] + [0.0] + [10.0**e for e in exps])


def gate_colors(theme: Mapping) -> list:
    lo, hi = theme["gates"]
    return [matplotlib.colormaps["viridis"](x) for x in np.linspace(lo, hi, NUM_GATES)]


def rebin(counts: np.ndarray, bins: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """One histogram's counts on new bin edges, each source bin spread uniformly over its width."""
    counts = counts.astype(float)
    if bins[-1] <= bins[0]:  # a one-value tensor: every count sits at that value
        out = np.zeros(len(edges) - 1)
        out[np.clip(np.searchsorted(edges, bins[0]) - 1, 0, len(out) - 1)] = counts.sum()
        return out
    cdf = np.concatenate([[0.0], np.cumsum(counts)])
    return np.diff(np.interp(edges, bins.astype(float), cdf))


def pooled_density(histograms: Mapping, keys: Sequence[str], edges: np.ndarray) -> np.ndarray:
    """(snapshots, bins) counts of several tensors on common edges, each column scaled to max 1."""
    total = None
    for key in keys:
        entry = histograms[key]
        grid = np.stack([rebin(c, b, edges) for c, b in zip(entry["counts"], entry["bins"])])
        total = grid if total is None else total + grid
    return total / np.maximum(total.max(axis=1, keepdims=True), 1e-12)


def step_edges(steps: np.ndarray) -> np.ndarray:
    steps = steps.astype(float)
    mids = 0.5 * (steps[1:] + steps[:-1])
    return np.concatenate(
        [[steps[0] - (mids[0] - steps[0])], mids, [steps[-1] + (steps[-1] - mids[-1])]]
    )


def symlog_edges(limit: float, linthresh: float, count: int) -> np.ndarray:
    """Bin edges uniform on a symmetric-log axis over [-limit, limit]."""
    top = math.log10(1.0 + limit / linthresh)
    t = np.linspace(-top, top, count + 1)
    return np.sign(t) * linthresh * (10.0 ** np.abs(t) - 1.0)


# --- figures -------------------------------------------------------------------------------------


def fig_gates(data: Mapping, numbers: Mapping, theme: Mapping):
    """The attention gates: their trajectories, the same gates memorising, and their gradients."""
    colors = gate_colors(theme)
    base = data["baseline"]["train"]
    warmup, ramp, total = data["baseline"]["schedule"]
    has_overfit, has_grad = "overfit" in data, "histograms" in data
    rows = 2 if has_grad else 1
    fig = plt.figure(figsize=(WIDTH, 4.3 if has_grad else 2.8))
    grid = fig.add_gridspec(
        rows,
        2,
        width_ratios=[3.0, 1.15],
        height_ratios=[2.3, 1.0][:rows],
        hspace=0.42,
        wspace=0.28,
    )
    ax = fig.add_subplot(grid[0, 0])
    shade_detach(ax, warmup, ramp, theme, label=True, label_y=0.88, vertical=False)
    steps = base["step"]
    finals = []
    for i in range(NUM_GATES):
        gate = column(base, f"gate_{i}")
        ax.plot(steps, gate, color=colors[i], zorder=3)
        finals.append(gate[-1])
    ax.set_xlim(0, total)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(r"Gate $g_\ell$")
    steps_axis(ax, label=not has_grad)
    panel(ax, "a", "Full run: 3,662 pairs")
    anchors = spread_positions(np.asarray(finals), 0.062, 0.0, 1.0)
    for i, (y, anchor) in enumerate(zip(finals, anchors)):
        ax.annotate(
            rf"$\ell={i}$  {y:.2f}",
            xy=(total, y),
            xytext=(total * 1.025, anchor),
            va="center",
            fontsize=7,
            annotation_clip=False,
            arrowprops={
                "arrowstyle": "-",
                "lw": 0.4,
                "color": theme["muted"],
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
    peak = numbers["gates"]["block_5"]
    ax.plot([peak["peak_step"]], [peak["peak"]], "o", ms=3.2, color=colors[5], zorder=5)
    ax.annotate(
        rf"$g_5$ peaks at {peak['peak']:.2f} (step {peak['peak_step']:,}), then closes",
        xy=(peak["peak_step"], peak["peak"]),
        xytext=(peak["peak_step"] + 3500, 0.13),
        fontsize=7,
        color=theme["ink"],
        arrowprops={
            "arrowstyle": "-|>",
            "lw": 0.5,
            "color": theme["muted"],
            "mutation_scale": 6,
            "shrinkB": 2,
        },
    )
    if has_overfit:
        over = data["overfit"]["train"]
        o_warmup, o_ramp, o_total = data["overfit"]["schedule"]
        ax2 = fig.add_subplot(grid[0, 1], sharey=ax)
        shade_detach(ax2, o_warmup, o_ramp, theme)
        for i in range(NUM_GATES):
            ax2.plot(over["step"], column(over, f"gate_{i}"), color=colors[i], zorder=3)
        ax2.set_xlim(0, o_total)
        steps_axis(ax2)
        ax2.tick_params(labelleft=False)
        panel(ax2, "b", "Overfit run: 100 frames")
    if has_grad:
        g_steps, push = gate_gradients(data["histograms"])
        ax3 = fig.add_subplot(grid[1, 0], sharex=ax)
        limit = float(np.nanpercentile(np.abs(push), 98))
        norm = SymLogNorm(linthresh=limit / 30, vmin=-limit, vmax=limit, base=10)
        mesh = ax3.pcolormesh(
            step_edges(g_steps),
            np.arange(NUM_GATES + 1) - 0.5,
            push,
            cmap=theme["diverging"],
            norm=norm,
            rasterized=True,
        )
        ax3.set_ylim(NUM_GATES - 0.5, -0.5)
        ax3.set_yticks(range(NUM_GATES))
        ax3.tick_params(axis="y", which="minor", left=False, right=False)
        ax3.set_ylabel(r"Block $\ell$")
        ax3.axvline(warmup, color=theme["ink"], lw=0.5, ls=(0, (2, 2)))
        ax3.axvline(
            numbers["gates"]["block_5"]["peak_step"], color=theme["ink"], lw=0.5, ls=(0, (1, 1.5))
        )
        steps_axis(ax3)
        panel(
            ax3,
            "c",
            r"Gradient on each gate, $-\partial\mathcal{L}/\partial g_\ell$ (one batch per column)",
        )
        cax = fig.add_subplot(grid[1, 1])
        cax.set_axis_off()
        inset = cax.inset_axes([0.05, 0.45, 0.9, 0.16])
        bar = fig.colorbar(mesh, cax=inset, orientation="horizontal")
        bar.outline.set_linewidth(0.5)
        bar.ax.tick_params(labelsize=6, direction="in", length=2, width=0.4, which="both")
        bar.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        bar.set_ticks([-limit, 0, limit])
        bar.set_ticklabels([sci(-limit), "$0$", sci(limit)])
        inset.text(
            0.0,
            1.6,
            "closes",
            transform=inset.transAxes,
            fontsize=6.5,
            ha="left",
            style="italic",
            color=theme["muted"],
        )
        inset.text(
            1.0,
            1.6,
            "opens",
            transform=inset.transAxes,
            fontsize=6.5,
            ha="right",
            style="italic",
            color=theme["muted"],
        )
        inset.text(
            0.5,
            -2.6,
            "descent step on $g_\\ell$",
            transform=inset.transAxes,
            fontsize=6.5,
            ha="center",
            color=theme["muted"],
        )
    return fig


def fig_distributions(data: Mapping, theme: Mapping):
    """Weight and gradient distributions of the velocity net over training (wandb.watch)."""
    hist = data["histograms"]
    warmup, ramp, total = data["baseline"]["schedule"]
    groups = (
        ("adaLN-Zero modulation", ADALN, "zero init, 12 tensors"),
        ("Output projection", ("output_proj.weight",), "zero init"),
        (r"Attention $W_{qkv}$", QKV, "random init, 6 tensors"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.9), sharex=True)
    cmap = matplotlib.colormaps[theme["density"]].copy()
    cmap.set_bad(alpha=0.0)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=1.0)
    letters = iter("abcdef")
    mesh = None
    for row, kind in enumerate(("parameters", "gradients")):
        for col, (title, names, init) in enumerate(groups):
            ax = axes[row, col]
            keys = [f"{kind}/{n}" for n in names]
            entries = [hist[k] for k in keys]
            last = np.concatenate([np.abs(e["bins"][-1]) for e in entries])
            if kind == "parameters":
                limit = float(
                    np.percentile(np.abs(np.concatenate([e["bins"][-1] for e in entries])), 99)
                )
                limit = max(limit, float(last.max()) * 0.6)
                edges = np.linspace(-limit, limit, 161)
            else:
                limit = float(max(np.abs(e["bins"]).max() for e in entries))
                linthresh = limit / 1e3
                edges = symlog_edges(limit, linthresh, 160)
                ax.set_yscale("symlog", linthresh=linthresh, linscale=0.3)
                ticks = symlog_ticks(limit, linthresh)
                ax.set_yticks(ticks, [sci(t) for t in ticks])
                ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            density = pooled_density(hist, keys, edges)
            density = np.ma.masked_less(density, 2e-3)
            mesh = ax.pcolormesh(
                step_edges(entries[0]["steps"]),
                edges,
                density.T,
                cmap=cmap,
                norm=norm,
                rasterized=True,
            )
            ax.axvline(warmup, color=theme["ink"], lw=0.5, ls=(0, (2, 2)))
            ax.set_xlim(0, total)
            ax.set_ylim(edges[0], edges[-1])
            steps_axis(ax, label=row == 1)
            letter = next(letters)
            if kind == "parameters":
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                panel(ax, letter, title)
                ax.text(
                    0.04,
                    0.95,
                    init,
                    transform=ax.transAxes,
                    va="top",
                    fontsize=6.5,
                    style="italic",
                    color=theme["muted"],
                )
            else:
                panel(ax, letter, f"Gradients of ({'abc'[col]})")
        axes[row, 0].set_ylabel("Weight value" if kind == "parameters" else "Gradient value")
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.11, top=0.93, wspace=0.32, hspace=0.32)
    cax = fig.add_axes([0.915, 0.25, 0.012, 0.5])
    bar = fig.colorbar(mesh, cax=cax)
    bar.outline.set_linewidth(0.5)
    bar.ax.tick_params(labelsize=6, direction="in", width=0.4, which="both")
    bar.set_label("Density, each step scaled to 1", fontsize=7)
    return fig


def fig_schedule(data: Mapping, theme: Mapping):
    """Learning rates, the conditioning-gradient scale and the EMA horizon over the run."""
    base = data["baseline"]["train"]
    warmup, ramp, total = data["baseline"]["schedule"]
    steps = base["step"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 1.95))
    ax = axes[0]
    shade_detach(ax, warmup, ramp, theme)
    ax.plot(
        steps, column(base, "loss/lr_main") * 1e4, color=theme["blue"], label="velocity net, head"
    )
    ax.plot(steps, column(base, "loss/lr_lora") * 1e4, color=theme["red"], label="LoRA")
    ax.set_ylabel(r"Learning rate ($\times 10^{-4}$)")
    ax.set_ylim(0, 2.25)
    ax.legend(loc="upper right")
    panel(ax, "a", "Learning rate")
    ax = axes[1]
    shade_detach(ax, warmup, ramp, theme, label=True, label_y=0.5)
    ax.plot(steps, column(base, "loss/cond_scale"), color=theme["blue"])
    ax.set_ylim(-0.04, 1.1)
    ax.set_ylabel(r"$s(k)$")
    panel(ax, "b", "Velocity gradient into LoRA")
    ax = axes[2]
    shade_detach(ax, warmup, ramp, theme)
    decay = column(base, "loss/ema_decay")
    ax.plot(steps, 1.0 / (1.0 - decay), color=theme["blue"])
    ax.set_yscale("log")
    ax.set_ylabel(r"$1/(1-\beta)$ (steps)")
    panel(ax, "c", "EMA horizon")
    for ax in axes:
        ax.set_xlim(0, total)
        steps_axis(ax)
    fig.subplots_adjust(wspace=0.42)
    return fig


def fig_losses(data: Mapping, theme: Mapping):
    """Each training term, train (smoothed) against the same objective on held-out pairs."""
    base, held = data["baseline"]["train"], data["baseline"]["valloss"]
    warmup, ramp, total = data["baseline"]["schedule"]
    panels = (
        ("total", r"Total $\mathcal{L}$"),
        ("flow_brightness", r"$\mathcal{L}_\mathrm{flow}$, brightness segment"),
        ("flow_color", r"$\mathcal{L}_\mathrm{flow}$, colour segment"),
        ("render", r"$\mathcal{L}_\mathrm{render}$"),
        ("align", r"$\mathcal{L}_\mathrm{align}$"),
        ("color", r"$\mathcal{L}_\mathrm{color}$"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 3.6), sharex=True)
    for (key, title), ax, letter in zip(panels, axes.flat, "abcdef"):
        shade_detach(ax, warmup, ramp, theme, label=key == "total", label_y=0.2)
        train = column(base, f"loss/{key}")
        heldout = column(held, f"loss/{key}")
        ax.plot(base["step"], train, color=theme["grey"], lw=0.5, zorder=2)
        ax.plot(base["step"], smooth(train), color=theme["ink"], lw=1.0, zorder=3, label="train")
        ax.plot(held["step"], heldout, color=theme["blue"], lw=1.2, zorder=4, label="held-out val")
        finite = np.concatenate([train[np.isfinite(train)], heldout])
        ax.set_ylim(0.0, float(np.nanpercentile(finite, 99)) * 1.08)
        ax.set_xlim(0, total)
        steps_axis(ax, label=letter in "def")
        panel(ax, letter, title)
    axes[0, 0].legend(loc="upper right")
    fig.subplots_adjust(wspace=0.3, hspace=0.38)
    return fig


def fig_validation(data: Mapping, numbers: Mapping, theme: Mapping):
    """Sampled image quality on 64 box-aligned val pairs (EMA weights), against their ceiling."""
    val = data["baseline"]["val"]
    warmup, ramp, total = data["baseline"]["schedule"]
    steps = val["step"]
    best = numbers["validation"]["best_step"]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.9), sharex=True)
    panels = (
        ("ciede2000", r"CIEDE2000 $\Delta E_{00}$ ($\downarrow$)"),
        ("psnr_L", r"$L^*$ PSNR, dB ($\uparrow$)"),
        ("colorfulness", r"Colourfulness / real photograph ($\uparrow$)"),
    )
    for (key, title), ax, letter in zip(panels, axes.flat, "abc"):
        shade_detach(ax, warmup, ramp, theme, label=key == "colorfulness", label_y=0.68)
        if key == "colorfulness":
            euler, midpoint, ceiling = (
                column(val, f"colorfulness/ratio_{m}") for m in ("euler", "midpoint", "oracle")
            )
        else:
            euler, midpoint, ceiling = (
                column(val, f"{key}/{m}") for m in ("euler", "midpoint", "oracle")
            )
        ax.plot(steps, euler, color=theme["blue"], marker="o", ms=2, label="Euler, NFE 2")
        ax.plot(steps, midpoint, color=theme["red"], marker="s", ms=1.8, label="midpoint, NFE 2")
        ax.axhline(ceiling[-1], color=theme["ink"], lw=0.8, ls=(0, (4, 2)), label="oracle ceiling")
        ax.axvline(best, color=theme["muted"], lw=0.6, ls=(0, (1, 1.5)))
        ax.set_xlim(0, total)
        steps_axis(ax, label=letter == "c")
        panel(ax, letter, title)
    axes[0, 0].legend(loc="center right", bbox_to_anchor=(0.86, 0.36))
    axes[0, 0].text(
        best,
        0.97,
        f"selected\n{best / 1000:g}k ",
        transform=axes[0, 0].get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=6.5,
        style="italic",
        color=theme["muted"],
    )
    ax = axes[1, 1]
    shade_detach(ax, warmup, ramp, theme)
    within, top1 = column(val, "retrieval/within_1"), column(val, "retrieval/top1")
    ax.plot(steps, within, color=theme["green"], marker="o", ms=2, label="within one patch")
    ax.plot(steps, top1, color=theme["yellow"], marker="s", ms=1.8, label="exact patch")
    ax.axhline(1.0 / 1280, color=theme["ink"], lw=0.8, ls=(0, (4, 2)), label="chance, 1/1,280")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, total)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.06))
    steps_axis(ax)
    panel(ax, "d", "Patch retrieval: thermal token to own RGB token")
    fig.subplots_adjust(wspace=0.22, hspace=0.32)
    return fig


def fig_per_frame(data: Mapping, numbers: Mapping, theme: Mapping):
    """Per test frame: sample spread against realized error, and colourfulness against the photo."""
    s = subset(data["per_pair"], "box_aligned")
    stats = numbers["per_frame"]["box_aligned/euler"]
    rmse = lstar_rmse(s["psnr"])
    ratio = stats["colourfulness_ratio_of_means"]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.35))
    panels = (
        (
            s["l_std"],
            rmse,
            r"Sample spread $\sigma_{L^*}$",
            r"Realized $L^*$ RMSE",
            "Brightness",
            rf"$\rho_s = {stats['spearman_l_std_vs_lstar_rmse']:.2f}$",
        ),
        (
            s["ab_std"],
            s["ciede2000"],
            r"Sample spread $\sigma_{ab}$",
            r"Realized $\Delta E_{00}$",
            "Colour",
            rf"$\rho_s = {stats['spearman_ab_std_vs_ciede2000']:.2f}$",
        ),
        (
            s["colorfulness_target"],
            s["colorfulness"],
            "Colourfulness, real photograph",
            "Colourfulness, Dawn",
            "Colourfulness",
            "",
        ),
    )
    for ax, (x, y, xlabel, ylabel, title, note), letter in zip(axes, panels, "abc"):
        ax.scatter(x, y, s=3, color=theme["blue"], alpha=0.45, linewidths=0, rasterized=True)
        square = letter == "c"
        xmax = float(np.percentile(x, 99.5)) * 1.12
        ymax = float(np.percentile(y, 99.5)) * 1.08
        if square:
            xmax = ymax = max(xmax, ymax)
        line = np.array([0.0, max(xmax, ymax)])
        ax.plot(
            line,
            line,
            color=theme["ink"],
            lw=0.8,
            ls=(0, (4, 2)),
            label="spread = error" if not square else r"$y = x$",
        )
        if square:
            ax.plot(line, ratio * line, color=theme["red"], lw=1.1, label=rf"$y = {ratio:.2f}\,x$")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        panel(ax, letter, title)
        ax.text(0.05, 0.94, note, transform=ax.transAxes, va="top", fontsize=8)
        ax.legend(loc="upper left" if square else "lower right")
    fig.subplots_adjust(wspace=0.36)
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
    box, fallback = subset(data["per_pair"], "box_aligned"), subset(data["per_pair"], "fallback")
    oracle = oracle_columns(data["oracle"]) if "oracle" in data else None
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.2))
    groups = [
        (box, theme["blue"], "-", f"Dawn, box-aligned test ({len(box['psnr'])})"),
        (fallback, theme["red"], "-", f"Dawn, fallback test ({len(fallback['psnr'])})"),
    ]
    if oracle is not None:
        groups.append(
            (oracle, theme["ink"], (0, (4, 2)), f"oracle, box-aligned val ({len(oracle['psnr'])})")
        )
    for ax, key, label, letter in (
        (axes[0], "psnr", r"$L^*$ PSNR per frame (dB)", "a"),
        (axes[1], "ciede2000", r"CIEDE2000 $\Delta E_{00}$ per frame", "b"),
    ):
        values = np.concatenate([g[0][key] for g in groups])
        bins = np.linspace(np.percentile(values, 0.5), np.percentile(values, 99.5), 32)
        for s, color, ls, name in groups:
            ax.hist(
                s[key],
                bins=bins,
                density=True,
                histtype="step",
                lw=1.1,
                color=color,
                ls=ls,
                label=name,
            )
            ax.axvline(s[key].mean(), color=color, lw=1.0, ls=ls, ymax=0.09)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_ylim(0, None)
        panel(ax, letter)
    axes[1].legend(loc="upper right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.22)
    return fig


def fig_refiner(data: Mapping, theme: Mapping):
    """Stage 2: what the refiner adds over the unrefined grid, and how far it moves L*."""
    tables = data["refiner"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.0))
    labels = {"val_box_aligned": "box-aligned val", "val_fallback": "fallback val"}
    for name, color in (("val_box_aligned", theme["blue"]), ("val_fallback", theme["red"])):
        if name not in tables:
            continue
        table = tables[name]
        axes[0].plot(
            table["step"], column(table, "psnr_L/improvement"), color=color, label=labels[name]
        )
        axes[1].plot(
            table["step"], column(table, "residual/abs_mean_Lstar"), color=color, label=labels[name]
        )
    axes[0].set_ylabel(r"$\Delta$ $L^*$ PSNR (dB)")
    axes[1].set_ylabel(r"Mean $|\Delta L^*|$ applied")
    panel(axes[0], "a", "Refined minus unrefined (gate: $\\geq 0$)")
    panel(axes[1], "b", "Size of the correction")
    for ax in axes:
        steps_axis(ax)
        ax.set_xlabel(r"Refiner step ($\times 10^3$)")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    axes[0].legend(loc="lower right")
    fig.subplots_adjust(wspace=0.3)
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
    histograms = runs / "baseline" / "histograms.npz"
    if histograms.exists():
        data["histograms"] = load_histograms(histograms)
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


def export(runs: Path, dest: Path, oracle: Path | None, wandb_log: Path | None = None) -> None:
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
    if wandb_log is not None:
        histograms = read_wandb_histograms(wandb_log)
        save_histograms(histograms, dest / "baseline" / "histograms.npz")
        logger.info("extracted %d histogram series from %s", len(histograms), wandb_log)
    logger.info("exported telemetry from %s to %s", runs, dest)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "results",
        help="campaign directory: baseline/, overfit/, refiner/, evaluation/, oracle/ "
        "(default: the committed results/)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs" / "images" / "plots",
        help="where the figures and numbers.json go (default: docs/images/plots)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        metavar="DIR",
        help="first copy the CSV mirrors, run configs and evaluation files from --runs into DIR "
        "(e.g. results), then plot from DIR",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        default=ROOT / "outputs" / "oracle",
        help="with --export: the check_oracle.py output to copy as well (default: outputs/oracle)",
    )
    parser.add_argument(
        "--wandb-log",
        type=Path,
        default=None,
        metavar="FILE",
        help="with --export: the baseline's local run-<id>.wandb log, whose wandb.watch "
        "histograms become baseline/histograms.npz (needs the wandb package)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=sorted(THEMES),
        default=["light", "dark"],
        help="which page themes to render (default: both)",
    )
    parser.add_argument("--format", choices=("svg", "png", "pdf"), default="svg")
    args = parser.parse_args(argv)

    runs = args.runs
    if args.export is not None:
        export(runs, args.export, args.oracle, args.wandb_log)
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
    if "histograms" in data:
        figures["distributions"] = lambda t: fig_distributions(data, t)
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
            path = args.out / f"{name}{suffix}.{args.format}"
            fig.savefig(path)
            plt.close(fig)
            logger.info("wrote %s", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    logger.info("wrote %s", args.out / "numbers.json")


if __name__ == "__main__":
    main()
