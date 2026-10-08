"""Config sanity: the decided keys exist, and the debug overlay cannot silently do nothing."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
omegaconf = pytest.importorskip("omegaconf")

from dawn.flow.tokens import TokenLayout  # noqa: E402

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


def test_the_default_resolution_is_the_plans_and_everything_derives_from_it():
    cfg = _load("flow.yaml")
    assert cfg["encoder"]["input_size"] == [512, 640]  # (H, W) of the 640x512 frame
    layout = TokenLayout.from_config(cfg)
    assert layout.grid == (32, 40) and layout.num_tokens == 1280
    assert layout.color_size == (128, 160) and layout.dim == 48
    assert cfg["encoder"]["grad_checkpointing"] is True
    render = cfg["flow"]["render_size"]
    assert (
        render == [256, 320] and render[0] % layout.grid[0] == 0 and render[1] % layout.grid[1] == 0
    )


def test_flow_config_has_the_decided_keys():
    cfg = _load("flow.yaml")
    assert cfg["encoder"]["backend"] in ("timm", "transformers")
    for table in ("model_ids", "lora_targets"):
        assert set(cfg["encoder"][table]) == {"timm", "transformers"}
    for path in [
        ("encoder", "lora_alpha"),
        ("encoder", "input_size"),
        ("grid", "num_bins"),
        ("grid", "ridge_alpha"),
        ("grid", "refine_steps"),
        ("grid", "stats_cache"),
        ("grid", "stats_num_samples"),
        ("color", "block"),
        ("color", "bin_step"),
        ("color", "soft_k"),
        ("color", "soft_sigma"),
        ("color", "min_patch_fraction"),
        ("color", "upsample"),
        ("color", "bgu", "bins"),
        ("color", "bgu", "grid"),
        ("color", "bgu", "ridge_alpha"),
        ("color", "bgu", "refine_steps"),
        ("brightness_correction", "enabled"),
        ("flow", "velocity_weight"),
        ("flow", "render_weight"),
        ("flow", "align_weight"),
        ("flow", "color_weight"),
        ("flow", "segment_balance"),
        ("flow", "render_size"),
        ("flow", "timesteps_per_image"),
        ("flow", "warmup_steps"),
        ("velocity", "detach_dino_attn"),
        ("loader", "val_batch_size"),
        ("logging", "log_attn_gates"),
        ("logging", "wandb_mode"),
    ]:
        assert _has(cfg, path), ".".join(path)
    assert not _has(cfg, ("flow", "noise_weight")) and not _has(cfg, ("grid", "spatial_size"))


def test_the_decided_values_hold():
    cfg = _load("flow.yaml")
    assert cfg["brightness_correction"]["enabled"] is False  # off until the oracle sheet + GPU gate
    assert cfg["grid"]["num_bins"] == 8 and cfg["grid"]["refine_steps"] == 1
    assert cfg["color"]["block"] == 4 and cfg["color"]["bin_step"] == 10
    # The maintainer chose bilinear at Checkpoint A: the BGU ceiling is worse at every grid swept.
    assert cfg["color"]["upsample"] == "bilinear"
    assert (cfg["color"]["soft_k"], cfg["color"]["soft_sigma"]) == (5, 5.0)
    flow = cfg["flow"]
    assert (flow["velocity_weight"], flow["render_weight"]) == (1.0, 0.1)
    assert (flow["align_weight"], flow["color_weight"]) == (0.5, 0.2)
    # 20K steps (~90 passes over ~3,660 pairs; 40K would be ~175) with a 3K-step detach warmup.
    assert flow["num_train_steps"] == 20000 and flow["warmup_steps"] == 3000
    assert flow["timesteps_per_image"] == 4 and flow["num_inference_steps"] == 2
    assert flow["segment_balance"] == "per_value"  # a stated choice, see flow.yaml
    assert cfg["optim"]["batch_size"] == 16 and cfg["optim"]["precision"] == "bf16"
    assert (
        cfg["optim"]["lr_warmup_steps"] == 1000
    )  # the optimizer's warmup is not flow.warmup_steps


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


OVERLAYS = [
    "debug.yaml",
    "mps.yaml",
    "gpu_probe.yaml",
    "correction.yaml",
    "overfit_gpu.yaml",
    "gpu_5090.yaml",
]


@pytest.mark.parametrize("name", OVERLAYS)
def test_overlays_only_override_existing_keys(name):
    base = {**_load("flow.yaml"), **_load("data.yaml")}
    missing = [".".join(p) for p in _leaf_paths(_load(name)) if not _has(base, p)]
    assert not missing, f"{name} overrides keys that do not exist: {missing}"


def _merged(*overlays):
    cfg = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(CONFIGS / "flow.yaml"),
        omegaconf.OmegaConf.load(CONFIGS / "data.yaml"),
        *[omegaconf.OmegaConf.load(CONFIGS / name) for name in overlays],
    )
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)


@pytest.mark.parametrize("name", OVERLAYS)
def test_overlays_give_a_consistent_resolution(name):
    cfg = _merged(name)
    layout = TokenLayout.from_config(cfg)
    render, color = cfg["flow"]["render_size"], layout.color_size
    assert render[0] % layout.grid[0] == 0 and render[1] % layout.grid[1] == 0
    assert render[0] >= color[0] and render[1] >= color[1]
    assert render[0] <= cfg["encoder"]["input_size"][0]


def test_the_mac_overlay_is_fp32_and_the_gpu_probe_keeps_the_real_precision():
    assert _merged("mps.yaml")["optim"]["precision"] == "fp32"
    assert _merged("debug.yaml")["optim"]["precision"] == "fp32"
    probe = _merged("gpu_probe.yaml")
    assert probe["optim"]["precision"] == "bf16" and probe["optim"]["batch_size"] == 16
    assert probe["encoder"]["input_size"] == [512, 640] and probe["flow"]["warmup_steps"] < 50


def test_the_correction_overlay_widens_the_token_and_the_overfit_overlay_keeps_the_real_size():
    assert TokenLayout.from_config(_merged("correction.yaml")).dim == 64
    gate = _merged("overfit_gpu.yaml")
    assert gate["encoder"]["input_size"] == [512, 640] and gate["optim"]["precision"] == "bf16"
    assert gate["flow"]["warmup_steps"] < gate["flow"]["num_train_steps"]


def test_the_5090_overlay_spends_its_headroom_on_timesteps_not_on_the_batch():
    """32 GB buys velocity samples per optimizer step, not fewer optimizer steps.

    Raising `optim.batch_size` instead would cut steps-per-epoch on a 3,662-pair training set
    until the budget no longer clears `flow.warmup_steps`
    (docs/training_budget_and_changes.md section 1).
    """
    base, gpu = _merged(), _merged("gpu_5090.yaml")
    assert gpu["flow"]["timesteps_per_image"] > base["flow"]["timesteps_per_image"]
    assert gpu["optim"]["batch_size"] == base["optim"]["batch_size"]
    for key in ("num_train_steps", "warmup_steps", "warmup_ramp_steps"):
        assert gpu["flow"][key] == base["flow"][key], "step counts stay put when the batch does"
    assert gpu["optim"]["precision"] == "bf16" and gpu["encoder"]["input_size"] == [512, 640]
    # L_render would otherwise double with the timesteps; the subset keeps it at its old cost.
    subset = gpu["flow"]["render_subset"]
    assert subset == base["optim"]["batch_size"] * base["flow"]["timesteps_per_image"]


def test_the_validation_resolution_and_selection_metric_are_the_decided_ones():
    log = _load("flow.yaml")["logging"]
    steps = _load("flow.yaml")["flow"]["num_train_steps"]
    # 8 validation points over a 20K-step run was not enough to see a divergence begin.
    assert steps / log["val_every"] >= 40
    assert log["select_metric"] == "ciede2000"  # the full Lab difference, not L* alone
    assert log["watch_scope"] == "velocity" and log["watch_freq"] == 500
    assert log["val_retrieval"] is True and log["metrics_csv"] is True


def test_the_schedule_defaults_stay_on_the_baseline():
    """One cycle and the cosine floor are opt-in: the baseline run must be the plan's schedule."""
    optim = _load("flow.yaml")["optim"]
    assert optim["schedule"] == "cosine"
    assert optim["cosine"]["min_factor"] == 0.0
    assert optim["ema_warmup"] is True  # strictly lowers the decay early; no late-run effect


def test_the_conditioning_ramp_lands_well_inside_the_run():
    flow = _load("flow.yaml")["flow"]
    assert 0 < flow["warmup_ramp_steps"] < flow["num_train_steps"] - flow["warmup_steps"]


def test_the_debug_overlay_runs_on_the_cpu_with_random_weights_and_no_wandb():
    cfg = _merged("debug.yaml")
    assert cfg["encoder"]["pretrained"] is False and cfg["logging"]["wandb_mode"] == "disabled"
    assert cfg["dronevehicle"]["max_pairs"] == 16 and cfg["loader"]["num_workers"] == 0


def test_the_old_regime_configs_are_gone():
    assert not (CONFIGS / "camera_calib.yaml").exists()
