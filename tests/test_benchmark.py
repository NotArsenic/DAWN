"""Tests for component latency and computational cost benchmarking."""


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

from dawn.flow.render import RenderSettings  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import build_models  # noqa: E402
from scripts.benchmark_latency import benchmark, count_macs, sync, table, time_ms  # noqa: E402
from tests.flow_helpers import TINY_INPUT, tiny_cfg  # noqa: E402


def _args(**over):
    base = dict(
        batch_size=1, nfe=2, method="euler", samples=1, precision="fp32",
        no_attn=False, warmup=0, trials=2,
    )
    return type("A", (), {**base, **over})()


@pytest.fixture(scope="module")
def bundle():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    layout = TokenLayout.from_config(cfg)
    encoder, velocity, _ = build_models(cfg, layout, torch.device("cpu"))
    return cfg, {
        "encoder": encoder.eval(),
        "velocity": velocity.eval(),
        "layout": layout,
        "input_size": TINY_INPUT,
        "settings": RenderSettings.from_config(cfg),
    }


def test_count_macs_matches_hand_arithmetic():
    """Verify MAC counting against analytical linear layer cost."""
    m = torch.nn.Linear(64, 128, bias=False)
    x = torch.randn(8, 64)
    assert count_macs(lambda: m(x)) == pytest.approx(8 * 64 * 128 / 1e9, rel=1e-9)


def test_time_ms_reports_ordered_statistics():
    out = time_ms(lambda: torch.randn(256, 256) @ torch.randn(256, 256),
                  torch.device("cpu"), warmup=1, trials=8)
    assert set(out) == {"ms_median", "ms_p90", "ms_min"}
    assert out["ms_min"] <= out["ms_median"] <= out["ms_p90"]
    assert out["ms_min"] > 0


def test_sync_is_a_noop_on_cpu():
    sync(torch.device("cpu"))  # must not raise


def test_benchmark_reports_every_component_plus_a_total(bundle):
    cfg, b = bundle
    rows = benchmark(cfg, b, torch.device("cpu"), _args())
    names = [r["component"] for r in rows]
    assert len(rows) == 4
    assert "encoder" in names[0] and "velocity" in names[1] and "render" in names[2]
    assert names[-1] == "TOTAL per frame"
    for r in rows:
        assert r["gmacs"] >= 0 and r["ms_median"] > 0


def test_the_total_is_the_sum_of_the_components(bundle):
    cfg, b = bundle
    rows = benchmark(cfg, b, torch.device("cpu"), _args())
    for key in ("gmacs", "ms_median"):
        assert rows[-1][key] == pytest.approx(sum(r[key] for r in rows[:-1]), rel=1e-9)


def test_the_encoder_runs_once_however_many_samples_are_drawn(bundle):
    """Verify encoder execution cost does not scale with sample count."""
    cfg, b = bundle
    one = benchmark(cfg, b, torch.device("cpu"), _args(samples=1))
    four = benchmark(cfg, b, torch.device("cpu"), _args(samples=4))
    assert four[0]["gmacs"] == pytest.approx(one[0]["gmacs"], rel=1e-6), "encoder cost is flat"
    assert four[1]["gmacs"] == pytest.approx(4 * one[1]["gmacs"], rel=0.02), "velocity scales"


def test_more_nfe_costs_proportionally_more_velocity(bundle):
    cfg, b = bundle
    two = benchmark(cfg, b, torch.device("cpu"), _args(nfe=2))
    four = benchmark(cfg, b, torch.device("cpu"), _args(nfe=4))
    assert four[1]["gmacs"] == pytest.approx(2 * two[1]["gmacs"], rel=0.02)
    assert four[0]["gmacs"] == pytest.approx(two[0]["gmacs"], rel=1e-6)


def test_midpoint_at_the_same_nfe_costs_the_same_as_euler(bundle):
    """Verify midpoint and Euler solvers have identical cost at matching NFE."""
    cfg, b = bundle
    e = benchmark(cfg, b, torch.device("cpu"), _args(method="euler", nfe=2))
    m = benchmark(cfg, b, torch.device("cpu"), _args(method="midpoint", nfe=2))
    assert m[1]["gmacs"] == pytest.approx(e[1]["gmacs"], rel=0.02)


def test_per_frame_costs_are_divided_by_the_batch(bundle):
    cfg, b = bundle
    one = benchmark(cfg, b, torch.device("cpu"), _args(batch_size=1))
    two = benchmark(cfg, b, torch.device("cpu"), _args(batch_size=2))
    assert two[0]["gmacs"] == pytest.approx(one[0]["gmacs"], rel=0.02), "reported per frame"


def test_table_renders_a_row_per_component_with_fps(bundle):
    cfg, b = bundle
    out = table(benchmark(cfg, b, torch.device("cpu"), _args()))
    assert "GMACs" in out and "FPS" in out
    assert "TOTAL per frame" in out
    assert len(out.splitlines()) == 6  # header, rule, 3 components, total


def test_json_output_is_round_trippable(tmp_path, bundle):
    cfg, b = bundle
    rows = benchmark(cfg, b, torch.device("cpu"), _args())
    p = tmp_path / "bench.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert json.loads(p.read_text())["rows"][-1]["component"] == "TOTAL per frame"
