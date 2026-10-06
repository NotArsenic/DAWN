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


def test_flow_config_records_the_pair_policy():
    pairs = _load("flow.yaml")["pairs"]
    assert pairs == {
        "fallback_pixel_mask": "vehicles",
        "box_pixel_mask": "moving",
        "fallback_render_weight": 0.5,
        "eval_subset": "box_aligned",
        "report_fallback_separately": True,
    }


def test_data_config_has_the_decided_keys():
    cfg = _load("data.yaml")
    for path in [
        ("dronevehicle", "root"),
        ("dronevehicle", "splits"),
        ("dronevehicle", "aligned_dir"),
        ("dronevehicle", "stats_csv"),
        ("dronevehicle", "manifest_dir"),
        ("dronevehicle", "outputs_dir"),
        ("dronevehicle", "frame_size"),
        ("dronevehicle", "border"),
        ("dronevehicle", "border_check", "min_value"),
        ("dronevehicle", "class_map"),
        ("dronevehicle", "max_pairs"),
        ("day_filter", "min_lum_median"),
        ("align", "max_match_dist_px"),
        ("align", "copy_tol_px"),
        ("align", "exclude_copied_boxes"),
        ("align", "ransac_inlier_px"),
        ("align", "min_matches"),
        ("align", "min_inlier_ratio"),
        ("align", "max_median_residual_px"),
        ("align", "max_scale_deviation"),
        ("align", "max_rotation_deg"),
        ("align", "max_shift_px"),
        ("align", "homography", "min_matches"),
        ("align", "grad_check", "min_gain"),
        ("align", "grad_check", "search_px"),
        ("align", "grad_check", "max_best_shift_px"),
        ("align", "moving", "min_residual_px"),
        ("align", "moving", "dilate_px"),
        ("dedup", "max_hamming"),
        ("masks", "vehicles", "dilate_px"),
        ("masks", "vehicles", "extend_px"),
        ("masks", "vehicles", "static_test", "min_prominence"),
        ("masks", "vehicles", "max_frame_fraction"),
        ("augment", "hflip"),
        ("augment", "vflip"),
        ("augment", "rot90"),
        ("augment", "thermal_contrast", "prob"),
        ("augment", "thermal_contrast", "gamma"),
    ]:
        assert _has(cfg, path), ".".join(path)


def test_data_config_decisions_hold():
    cfg = _load("data.yaml")
    assert cfg["dronevehicle"]["frame_size"] == [712, 840] and cfg["dronevehicle"]["border"] == 100
    assert cfg["augment"]["rot90"] is False  # 90 degree turns would break 640x512 batching
    classes = cfg["dronevehicle"]["class_map"]
    assert classes["truvk"] == "truck" and classes["*"] is None
    assert {classes[k] for k in ("feright_car", "feright car", "feright")} == {"freight_car"}
    assert cfg["align"]["exclude_copied_boxes"] is True


def test_debug_overlay_only_overrides_existing_keys():
    base = {**_load("flow.yaml"), **_load("data.yaml")}
    missing = [".".join(p) for p in _leaf_paths(_load("debug.yaml")) if not _has(base, p)]
    assert not missing, f"debug.yaml overrides keys that do not exist: {missing}"
