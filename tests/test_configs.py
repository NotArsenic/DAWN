"""Config sanity: the decided keys exist, and the debug overlay cannot silently do nothing."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.grid import layout  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _load(name):
    return omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(CONFIGS / name))


def _leaf_paths(tree, prefix=()):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from _leaf_paths(value, prefix + (key,))
        else:
            yield prefix + (key,)


def _has(tree, path):
    for key in path:
        if not isinstance(tree, dict) or key not in tree:
            return False
        tree = tree[key]
    return True


@pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
def test_config_parses(name):
    assert isinstance(_load(name), dict)


def test_grid_config_matches_layout():
    grid = _load("flow.yaml")["grid"]
    assert grid["coeffs_per_bin"] == layout.NUM_COEFFS
    assert grid["spatial_size"] * grid["spatial_size"] == 256  # tokens; docs/conventions.md 2.3


def test_flow_config_has_the_decided_keys():
    cfg = _load("flow.yaml")
    assert cfg["encoder"]["backend"] in ("timm", "transformers")
    for table in ("model_ids", "lora_targets"):
        assert set(cfg["encoder"][table]) == {"timm", "transformers"}
    for path in [
        ("encoder", "lora_alpha"),
        ("flow", "align_weight"),
        ("flow", "timesteps_per_image"),
        ("flow", "warmup_steps"),
        ("velocity", "detach_dino_attn"),
        ("logging", "log_attn_gates"),
        ("logging", "wandb_mode"),
    ]:
        assert _has(cfg, path), ".".join(path)


def test_moonlight_color_temperature_is_not_daylight():
    lighting = _load("data.yaml")["degradation"]["lighting"]
    assert lighting["moonlight_temp_range"] == [3800, 4300]


def test_debug_overlay_only_overrides_existing_keys():
    base = {**_load("flow.yaml"), **_load("data.yaml")}
    missing = [".".join(p) for p in _leaf_paths(_load("debug.yaml")) if not _has(base, p)]
    assert not missing, f"debug.yaml overrides keys that do not exist: {missing}"
