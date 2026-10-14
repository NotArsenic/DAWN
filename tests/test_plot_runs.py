"""Tests for scripts/plot_runs.py: the README figures and numbers rebuilt from local telemetry."""

import csv
import importlib.util
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_runs.py"
WARMUP, RAMP, TOTAL = 30, 10, 100


def _load_script():
    spec = importlib.util.spec_from_file_location("plot_runs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flow_run(root: Path, name: str) -> None:
    run = root / name
    run.mkdir(parents=True)
    flow = {"warmup_steps": WARMUP, "warmup_ramp_steps": RAMP, "num_train_steps": TOTAL}
    (run / "run_config.json").write_text(json.dumps({"flow": flow}), encoding="utf-8")
    train = []
    for step in range(0, TOTAL, 10):
        row = {"step": step}
        for i in range(5):
            row[f"gate/gate_{i}"] = step / TOTAL * (i + 1) / 6
        row["gate/gate_5"] = 0.5 - abs(step - 40) / 100  # rises to a peak at step 40, then falls
        for key in ("flow_brightness", "flow_color", "render", "align", "color", "total"):
            row[f"loss/{key}"] = 1.0 / (1 + step)
        row.update(
            {
                "loss/lr_main": 2e-4,
                "loss/lr_lora": 1e-4,
                "loss/ema_decay": 0.9,
                "loss/cond_scale": min(1.0, max(0.0, (step - WARMUP + 1) / RAMP)),
                "grad/encoder": 0.1 if step < WARMUP else 0.2,
            }
        )
        train.append(row)
    _write_csv(run / "metrics" / "train.csv", train)
    val, valloss = [], []
    for step in range(10, TOTAL + 1, 10):
        row = {"step": step}
        prefix = "val_box_aligned/"
        for method, offset in (("euler", 0.0), ("midpoint", 0.5), ("oracle", -3.0)):
            row[f"{prefix}ciede2000/{method}"] = 9.0 - step / 100 + offset
            row[f"{prefix}psnr_L/{method}"] = 18.0 + step / 100 - offset
        for method in ("euler", "midpoint", "oracle"):
            row[f"{prefix}colorfulness/ratio_{method}"] = 0.6
        row[f"{prefix}retrieval/top1"] = 0.4
        row[f"{prefix}retrieval/within_1"] = 0.8
        val.append(row)
        loss_row = {"step": step}
        for key in ("flow_brightness", "flow_color", "render", "align", "color", "total"):
            loss_row[f"valloss_box_aligned/loss/{key}"] = 1.0 / (1 + step)
        valloss.append(loss_row)
    val[-3][f"{prefix}ciede2000/euler"] = 0.0  # the best checkpoint is the third-to-last point
    _write_csv(run / "metrics" / "val_box_aligned.csv", val)
    _write_csv(run / "metrics" / "valloss_box_aligned.csv", valloss)


HIST_STEPS = list(range(10, TOTAL, 10))
TURN = 40  # gate 5's peak in _flow_run


def _histograms(module, root: Path) -> None:
    """Weights that fan out, gradients that shrink, and gate gradients that turn at TURN."""
    steps = np.array(HIST_STEPS)
    grow, shrink = steps / TOTAL, 1.0 - steps / (2 * TOTAL)
    counts = np.tile(np.exp(-(np.linspace(-3, 3, 64) ** 2)), (len(steps), 1)) * 1000
    hist = {}
    for name in module.HIST_TENSORS:
        if name.endswith("dino_gate"):
            opening = -1e-3 * np.ones(len(steps))
            if name == "blocks.5.dino_gate":
                opening = np.where(steps <= TURN, -1e-3, 1e-3)
            for kind, values in (("parameters", grow), ("gradients", opening)):
                hist[f"{kind}/{name}"] = {
                    "steps": steps,
                    "counts": np.ones((len(steps), 64), dtype=np.int64),
                    "bins": np.repeat(values[:, None], 65, axis=1).astype(np.float32),
                }
            continue
        for kind, width in (("parameters", 0.01 * grow), ("gradients", 1e-3 * shrink)):
            hist[f"{kind}/{name}"] = {
                "steps": steps,
                "counts": counts.astype(np.int64),
                "bins": (width[:, None] * np.linspace(-1, 1, 65)).astype(np.float32),
            }
    module.save_histograms(hist, root / "baseline" / "histograms.npz")


def _tree(root: Path, module=None) -> None:
    _flow_run(root, "baseline")
    _flow_run(root, "overfit")
    if module is not None:
        _histograms(module, root)
    pairs = []
    for subset in ("box_aligned", "fallback"):
        for method in ("euler", "midpoint"):
            for i in range(12):
                pairs.append(
                    {
                        "subset": subset,
                        "method": method,
                        "psnr": 25.0 - i,
                        "ciede2000": 3.0 + i,
                        "l_std": 1.0 + 0.1 * i,
                        "ab_std": 0.5 + 0.05 * i,
                        "colorfulness": 10.0,
                        "colorfulness_target": 20.0,
                    }
                )
    _write_csv(root / "evaluation" / "per_pair.csv", pairs)
    refiner = [
        {
            "step": s,
            "val_box_aligned/psnr_L/improvement": s / 1000,
            "val_box_aligned/residual/abs_mean_Lstar": 0.3,
        }
        for s in (100, 200)
    ]
    _write_csv(root / "refiner" / "metrics" / "val_box_aligned.csv", refiner)
    oracle = [
        {"fallback": i % 2 == 1, "psnr_grid_r1": 25.0, "de_grid_only_bilinear": 3.0}
        for i in range(6)
    ]
    (root / "oracle").mkdir()
    (root / "oracle" / "rows_val.jsonl").write_text(
        "\n".join(json.dumps(r) for r in oracle) + "\n", encoding="utf-8"
    )


def test_every_figure_is_written_in_both_themes_with_its_numbers(tmp_path):
    runs, out = tmp_path / "runs", tmp_path / "plots"
    module = _load_script()
    _tree(runs, module)
    module.main(["--runs", str(runs), "--out", str(out)])
    names = (
        "gates",
        "distributions",
        "schedule",
        "losses",
        "validation",
        "per_frame",
        "test_distributions",
        "refiner",
    )
    for name in names:
        assert (out / f"{name}.svg").stat().st_size > 0, name
        assert (out / f"{name}_dark.svg").stat().st_size > 0, name

    numbers = json.loads((out / "numbers.json").read_text(encoding="utf-8"))
    assert numbers["schedule"] == {
        "warmup_steps": WARMUP,
        "ramp_steps": RAMP,
        "num_train_steps": TOTAL,
    }
    assert numbers["gates"]["block_5"]["peak_step"] == 40, (
        "a gate that turns back is found at its peak"
    )
    assert numbers["gates"]["block_5"]["final"] < numbers["gates"]["block_5"]["peak"]
    assert numbers["validation"]["best_step"] == 80, "best is the lowest CIEDE2000, not the last"
    assert numbers["encoder_grad_norm"]["1k_after_ramp"] == pytest.approx(0.2)
    frame = numbers["per_frame"]["box_aligned/euler"]
    assert frame["spearman_l_std_vs_lstar_rmse"] == pytest.approx(1.0), "spread grows with error"
    assert frame["colourfulness_ratio_of_means"] == pytest.approx(0.5)
    turns = numbers["gate_gradient_snapshots"]
    assert turns["turn_step"] == TURN
    assert turns["block_5"] == {"opening_before_turn": "4/4", "closing_after_turn": "5/5"}
    assert turns["block_0"]["closing_after_turn"] == "0/5"
    weights = numbers["weight_distributions"]
    assert weights["adaln_weight_std"][1] > weights["adaln_weight_std"][0], "the weights fan out"
    assert numbers["gradient_shrink"]["adaln"]["overall"] > 1.0, "the gradients shrink"


def _leveldb_log(path: Path, payloads: list[bytes]) -> None:
    """Write payloads in the .wandb framing: 32 KiB blocks, records split across them, padding."""
    block, header = 32768, 7
    out = bytearray(struct.pack("<4sHB", b":W&B", 0xBEE1, 0))
    for payload in payloads:
        first = True
        while True:
            left = block - len(out) % block
            if left < header:
                out.extend(b"\x00" * left)
                left = block
            chunk, payload = payload[: left - header], payload[left - header :]
            kind = (1 if not payload else 2) if first else (3 if payload else 4)
            crc = zlib.crc32(chunk, zlib.crc32(bytes([kind]))) & 0xFFFFFFFF
            out.extend(struct.pack("<IHB", crc, len(chunk), kind) + chunk)
            first = False
            if not payload:
                break
    path.write_bytes(bytes(out))


def test_the_wandb_log_reader_reassembles_records_across_blocks(tmp_path):
    module = _load_script()
    # 32,751 bytes leaves 3 at the end of the first block: too few for a header, so it is padding.
    payloads = [b"a" * 32751, b"small", b"x" * 70000, b"", b"last"]
    path = tmp_path / "run.wandb"
    _leveldb_log(path, payloads)
    assert list(module.wandb_records(path)) == payloads

    corrupt = bytearray(path.read_bytes())
    corrupt[20] ^= 0xFF
    path.write_bytes(bytes(corrupt))
    with pytest.raises(ValueError, match="checksum"):
        list(module.wandb_records(path))


def test_export_copies_only_the_small_files(tmp_path):
    runs, dest = tmp_path / "runs", tmp_path / "results"
    _tree(runs)
    (runs / "baseline" / "flow_best.pt").write_bytes(b"not copied")
    (runs / "evaluation" / "report.md").write_text("# Evaluation\n", encoding="utf-8")
    module = _load_script()
    module.export(runs, dest, oracle=runs / "oracle")
    assert (dest / "baseline" / "metrics" / "train.csv").exists()
    assert (dest / "baseline" / "run_config.json").exists()
    assert (dest / "evaluation" / "report.md").exists()
    assert (dest / "oracle" / "rows_val.jsonl").exists()
    assert not list(dest.rglob("*.pt")), "checkpoints stay out of the committed results"
