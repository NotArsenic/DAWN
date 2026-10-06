"""The oracle (dawn/flow/oracle.py, scripts/check_oracle.py) on a tiny synthetic tree."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")
pytest.importorskip("PIL")

from dawn.data.pairs import AlignedPairs  # noqa: E402
from dawn.flow import oracle  # noqa: E402
from tests.flow_helpers import tiny_data_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("oracle")
    cfg = tiny_data_cfg(root, n_box=3, n_fallback=2)
    cfg["color"]["bgu"].update(bins=8, grid=[8, 10])  # 2 x 2 colour pixels per BGU cell
    dataset = AlignedPairs.from_config(cfg, "val_pairs")
    rows, sweeps = oracle.run_oracle(
        dataset,
        range(len(dataset)),
        cfg,
        "cpu",
        batch_size=2,
        sweep=((4, 4, 5, 1e-3), (8, 8, 10, 1e-3)),
        sweep_pairs=3,
    )
    return cfg, dataset, rows, sweeps


def test_every_pair_gets_every_metric_and_the_pair_kind(tiny):
    _, dataset, rows, _ = tiny
    assert [r["id"] for r in rows] == dataset.image_ids
    assert [r["fallback"] for r in rows] == [False, False, False, True, True]
    for row in rows:
        assert set(oracle.METRICS) <= set(row)
        assert all(torch.isfinite(torch.tensor(row[k])) for k in oracle.METRICS)


def test_the_synthetic_lightness_is_exactly_representable_so_the_grid_oracle_is_high(tiny):
    _, _, rows, _ = tiny
    assert oracle.summarize(rows, "psnr_grid_r1")["mean"] > 35
    # refine and the correction can only help (to within noise) on a representable target.
    assert (
        oracle.summarize(rows, "psnr_grid_r1")["mean"]
        >= oracle.summarize(rows, "psnr_grid_r0")["mean"] - 0.5
    )
    assert (
        oracle.summarize(rows, "psnr_corr_bgu_r1")["mean"]
        >= oracle.summarize(rows, "psnr_grid_r1")["mean"] - 1.0
    )


def test_the_colour_ceiling_of_the_bgu_is_in_the_ballpark_of_bilinear(tiny):
    _, _, rows, _ = tiny
    bgu = oracle.summarize(rows, "ab_err_bgu")["mean"]
    bilinear = oracle.summarize(rows, "ab_err_bilinear")["mean"]
    assert 0 <= bgu < 2 * bilinear + 1.0 and bilinear < 6.0


def test_summaries_and_tables(tiny):
    _, _, rows, sweeps = tiny
    s = oracle.summarize(rows, "psnr_grid_r1")
    assert s["n"] == 5 and s["p10"] <= s["p50"] <= s["p90"]
    assert oracle.summarize([], "psnr_grid_r1")["n"] == 0
    table = oracle.metrics_table(rows)
    assert "box-aligned mean (n=3)" in table and "fallback mean (n=2)" in table
    assert all(oracle.LABELS[k] in table for k in oracle.METRICS)
    text, best, baseline = oracle.sweep_table(sweeps)
    assert best in sweeps and len(best) == 4 and baseline > 0
    assert "mean abs delta a*b*" in text and "bilinear (baseline)" in text
    assert all(len(v) == 3 for v in sweeps.values())  # exactly the first sweep_pairs pairs


def test_worst_ids_picks_the_lowest_psnr_first_and_at_least_one(tiny):
    _, _, rows, _ = tiny
    ordered = sorted(rows, key=lambda r: r["psnr_grid_r1"])
    assert oracle.worst_ids(rows, fraction=0.4) == [r["id"] for r in ordered[:2]]
    assert oracle.worst_ids(rows, fraction=0.0001) == [ordered[0]["id"]]


def test_the_worst_sheets_are_written_as_jpegs(tiny, tmp_path):
    from PIL import Image

    cfg, dataset, rows, _ = tiny
    ids = oracle.worst_ids(rows, fraction=0.6)
    paths = oracle.write_worst_sheets(dataset, ids, rows, cfg, tmp_path, scenes_per_sheet=2)
    assert len(paths) == 2 and all(p.exists() for p in paths)
    with Image.open(paths[0]) as sheet:
        assert sheet.format == "JPEG" and sheet.width > 200


def test_the_probe_ids_are_the_handoffs_seed0_sample(tiny):
    spec = importlib.util.spec_from_file_location(
        "check_oracle_script", ROOT / "scripts/check_oracle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, dataset, _, _ = tiny
    ids = module.probe_ids(dataset)
    import random

    random.seed(0)
    expected = {dataset.image_ids[i] for i in random.sample(range(len(dataset)), 5)}
    assert ids == expected


def test_the_script_runs_end_to_end_and_writes_the_report(tmp_path, monkeypatch, capsys):
    cfg = tiny_data_cfg(tmp_path / "tree", n_box=2, n_fallback=1)
    flow_yaml = tmp_path / "all.yaml"
    omegaconf.OmegaConf.save(omegaconf.OmegaConf.create(cfg), flow_yaml)
    spec = importlib.util.spec_from_file_location(
        "check_oracle_script2", ROOT / "scripts/check_oracle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "oracle_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_oracle.py",
            "--config",
            str(flow_yaml),
            "--data-config",
            str(flow_yaml),
            "--output",
            str(out),
            "--workers",
            "0",
            "--sweep-pairs",
            "2",
            "--train-sample",
            "2",
        ],
    )
    module.main()
    report = (out / "report.md").read_text(encoding="utf-8")
    for needle in (
        "# Brightness and colour oracle",
        "grid only, refine 0",
        "BGU grid sweep",
        "Worst 10% of grid-only scenes",
        "Checks against the handoff",
        "Train sample",
    ):
        assert needle in report, needle
    assert (out / "rows_val.jsonl").exists() and (out / "rows_train.jsonl").exists()
    assert list(out.glob("worst10_grid_only_*.jpg"))
