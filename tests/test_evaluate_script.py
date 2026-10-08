"""scripts/evaluate.py: the plan's metric table, end to end on a tiny aligned tree.

Run against random weights, exactly as the flow phase was tested: this pins the plumbing and the
report shape, not the quality of any model.
"""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("omegaconf")
pytest.importorskip("peft")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.data.pairs import AlignedPairs  # noqa: E402
from dawn.flow.inference import load_checkpoint  # noqa: E402
from dawn.flow.loaders import split_by_pair_kind  # noqa: E402
from dawn.flow.targets import FitSettings, compute_token_stats, prepare_batch  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import FlowTrainer  # noqa: E402
from scripts.evaluate import (  # noqa: E402
    build_report,
    score_subset,
    summarize,
    table,
    vehicle_region,
)
from tests.flow_helpers import TINY_INPUT, tiny_data_cfg, wake_up  # noqa: E402

PCTS = [10, 50, 90]


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    """A tiny aligned tree, a checkpoint trained one step on it, and the loaded bundle."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("aligned")
    cfg = tiny_data_cfg(root, n_box=3, n_fallback=2)
    dataset = AlignedPairs.from_config(cfg, "val_pairs", train=False)
    raw = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    stats = compute_token_stats(
        [prepare_batch(raw, TINY_INPUT, cfg["pairs"])],
        TokenLayout.from_config(cfg),
        FitSettings.from_config(cfg),
    )
    trainer = FlowTrainer.from_config(cfg, stats, "cpu")
    path = trainer.save_checkpoint(tmp_path_factory.mktemp("ck") / "flow.pt")
    bundle = load_checkpoint(path, "cpu", use_ema=False)
    wake_up(bundle.velocity, std=0.1, seed=3)
    return cfg, dataset, bundle


def _ecfg(**over):
    base = dict(
        split="val_pairs", batch_size=2, workers=0, output_dir="x", nfe=2,
        methods=["euler"], num_samples=2, seed_from_image_id=True,
        report_subsets=True, headline_subset="box_aligned",
    )
    return {**base, **over}


def _mcfg(**over):
    return {**dict(lpips=False, lpips_net="alex", vehicle_region="mask",
                   min_vehicle_pixels=4), **over}


def test_score_subset_returns_one_row_per_pair_with_the_plans_metrics(setup):
    cfg, dataset, bundle = setup
    subsets = split_by_pair_kind(dataset, None)
    rows = score_subset(
        bundle, subsets["box_aligned"], cfg, _ecfg(), _mcfg(), "euler", torch.device("cpu"), None
    )
    assert len(rows) == len(subsets["box_aligned"])
    r = rows[0]
    for k in ("psnr", "ssim", "ciede2000", "colorfulness", "colorfulness_target",
              "psnr_vehicles", "ssim_vehicles", "ciede2000_vehicles", "l_std", "ab_std"):
        assert k in r, k
        assert isinstance(r[k], float) and r[k] == r[k], k  # finite, not NaN
    assert r["method"] == "euler" and isinstance(r["fallback"], bool)
    assert {"mean_psnr", "mean_ciede2000", "mean_colorfulness"} <= set(r)


def test_one_sample_drops_the_spread_columns(setup):
    cfg, dataset, bundle = setup
    subsets = split_by_pair_kind(dataset, None)
    rows = score_subset(
        bundle, subsets["box_aligned"], cfg, _ecfg(num_samples=1), _mcfg(), "euler",
        torch.device("cpu"), None,
    )
    assert "l_std" not in rows[0] and "ab_std" not in rows[0]


def test_seeding_from_the_image_id_makes_the_numbers_reproducible(setup):
    cfg, dataset, bundle = setup
    sub = split_by_pair_kind(dataset, None)["box_aligned"]
    a = score_subset(bundle, sub, cfg, _ecfg(), _mcfg(), "euler", torch.device("cpu"), None)
    b = score_subset(bundle, sub, cfg, _ecfg(), _mcfg(), "euler", torch.device("cpu"), None)
    assert [r["psnr"] for r in a] == [r["psnr"] for r in b]
    assert [r["ciede2000"] for r in a] == [r["ciede2000"] for r in b]


def test_vehicle_region_modes(setup):
    cfg, dataset, bundle = setup
    raw = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    dev = torch.device("cpu")
    assert vehicle_region(raw, TINY_INPUT, "none", 4, dev) is None
    m = vehicle_region(raw, TINY_INPUT, "mask", 4, dev)
    assert m.shape == (2, 1, *TINY_INPUT) and m.min() >= 0 and m.max() <= 1
    pytest.importorskip("scipy")
    b = vehicle_region(raw, TINY_INPUT, "boxes", 4, dev)
    assert b.shape == m.shape
    assert set(b.unique().tolist()) <= {0.0, 1.0}, "boxes are hard regions, not fractions"
    assert b.sum() >= (m > 0.5).sum(), "a bounding box covers at least its own component"


def test_an_unknown_vehicle_region_is_a_clear_error(setup):
    cfg, dataset, bundle = setup
    raw = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    with pytest.raises(ValueError, match="mask \\| boxes \\| none"):
        vehicle_region(raw, TINY_INPUT, "crops", 4, torch.device("cpu"))


def test_summarize_reports_a_mean_and_the_requested_percentiles():
    rows = [{"psnr": float(v), "id": "x"} for v in range(1, 11)]
    s = summarize(rows, PCTS)
    assert s["psnr"]["mean"] == pytest.approx(5.5)
    assert s["psnr"]["p10"] <= s["psnr"]["p50"] <= s["psnr"]["p90"]
    assert "id" not in s, "only float columns are summarized"


def test_table_renders_every_metric_as_a_row():
    out = table(summarize([{"psnr": 1.0, "ssim": 0.5}, {"psnr": 3.0, "ssim": 0.7}], PCTS), PCTS)
    assert "| psnr |" in out and "| ssim |" in out
    assert out.splitlines()[0].count("|") == 2 + len(PCTS) + 1


def test_report_names_the_checkpoint_the_masks_and_the_vehicle_caveat(setup):
    cfg, dataset, bundle = setup
    sub = split_by_pair_kind(dataset, None)["box_aligned"]
    rows = score_subset(bundle, sub, cfg, _ecfg(), _mcfg(), "euler", torch.device("cpu"), None)
    args = type("A", (), {"checkpoint": "ck.pt"})()
    text = build_report(args, bundle, _ecfg(), _mcfg(), {("box_aligned", "euler"): rows}, PCTS)
    assert "# Evaluation" in text
    assert "box-aligned" in text.lower()
    assert "vehicles` mask stands in" in text, "the boxes caveat must survive into the report"
    assert "valid & ~moving" in text
    assert "## box_aligned / euler" in text


def test_the_shipped_eval_config_has_the_keys_the_script_reads():
    from dawn.flow.loaders import load_config

    cfg = load_config("configs/flow.yaml", "configs/data.yaml", ["configs/eval.yaml"])
    for key in ("split", "batch_size", "workers", "output_dir", "nfe", "methods",
                "num_samples", "seed_from_image_id", "headline_subset"):
        assert key in cfg["eval"], key
    for key in ("lpips", "lpips_net", "vehicle_region", "min_vehicle_pixels"):
        assert key in cfg["metrics"], key
    assert cfg["metrics"]["vehicle_region"] in ("mask", "boxes", "none")
    assert set(cfg["eval"]["methods"]) <= {"euler", "midpoint"}
    assert "visdrone" not in json.dumps(dict(cfg["eval"])).lower(), "no old-regime leftovers"
