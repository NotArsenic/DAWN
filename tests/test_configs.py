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


def test_the_refiner_config_adds_one_section_and_otherwise_only_overrides():
    """`refiner.yaml` is a stage-2 CONFIG, not an overlay, so it is not in `OVERLAYS`.

    It is merged as `flow.yaml + data.yaml + refiner.yaml`, so the same trap applies: a key that
    exists nowhere else is a typo that silently does nothing. The one thing it may legitimately
    add is its own `refiner:` section.
    """
    base = {**_load("flow.yaml"), **_load("data.yaml")}
    stray = [
        ".".join(p)
        for p in _leaf_paths(_load("refiner.yaml"))
        if p[0] != "refiner" and not _has(base, p)
    ]
    assert not stray, f"refiner.yaml sets keys that exist nowhere: {stray}"
    assert "refiner" not in base, "the refiner section belongs to stage 2 only"


def test_the_refiner_config_matches_the_plans_architecture_and_loss():
    """The plan: two depthwise-separable 5x5 blocks, width 24, L + 0.1 tanh(residual),
    256x256 crops, L1 + 0.01 * |residual|."""
    refiner = _load("refiner.yaml")["refiner"]
    assert (refiner["hidden_channels"], refiner["num_blocks"]) == (24, 2)
    assert refiner["kernel_size"] == 5
    assert 1 + refiner["num_blocks"] * (refiner["kernel_size"] - 1) == 9  # the 9x9 field
    assert refiner["residual_scale"] == 0.1  # +/- 10 L* units
    assert refiner["crop_size"] == 256 and refiner["residual_penalty"] == 0.01


def test_the_two_stages_log_to_different_projects_under_one_entity():
    flow, refiner = _load("flow.yaml")["logging"], _load("refiner.yaml")["logging"]
    assert flow["wandb_entity"] == "my-team"
    assert (flow["wandb_project"], refiner["wandb_project"]) == ("dawn", "dawn-refiner")
    # The entity is inherited through the merge, never repeated: one place to change it.
    assert "wandb_entity" not in refiner


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


def test_the_fast_overlay_rescales_every_step_denominated_knob_together():
    """The rescaling trap: a shorter budget that forgot one of these is a different experiment.

    The SHAPE of the schedule must survive -- the detach still lifting at ~15% of the run, the
    LR warmup at ~5% -- or the short run is not a shorter version of the long one.
    """
    full, fast = _merged("gpu_5090.yaml"), _merged("gpu_5090.yaml", "fast.yaml")
    ratio = fast["flow"]["num_train_steps"] / full["flow"]["num_train_steps"]
    assert 0 < ratio < 1, "fast.yaml must actually shorten the budget"
    for section, key in (
        ("flow", "warmup_steps"),
        ("flow", "warmup_ramp_steps"),
        ("optim", "lr_warmup_steps"),
        ("logging", "val_every"),
        ("logging", "save_every"),
    ):
        scaled = full[section][key] * ratio
        assert fast[section][key] == pytest.approx(scaled, rel=0.1), f"{section}.{key} not rescaled"


def test_the_fast_overlay_keeps_the_budget_clear_of_its_own_warmup():
    """Below ~2x the warmup the conditioning detach eats the run (trainer warns, then wastes it)."""
    fast = _merged("gpu_5090.yaml", "fast.yaml")["flow"]
    assert fast["num_train_steps"] > 4 * fast["warmup_steps"]


def test_the_fast_overlay_cuts_time_without_touching_the_batch():
    """Total work per epoch is independent of batch size and linear in timesteps, so the batch
    is NOT the lever and raising it only costs optimizer steps."""
    full, fast = _merged("gpu_5090.yaml"), _merged("gpu_5090.yaml", "fast.yaml")
    assert fast["optim"]["batch_size"] == full["optim"]["batch_size"]
    assert fast["optim"]["lr"] == full["optim"]["lr"]  # no LR rescale needed if the batch is fixed
    assert fast["flow"]["timesteps_per_image"] < full["flow"]["timesteps_per_image"]


def test_the_fast_overlay_must_be_applied_after_the_hardware_one():
    """It puts timesteps_per_image back, so the order in the override list is load-bearing."""
    assert _merged("fast.yaml", "gpu_5090.yaml")["flow"]["timesteps_per_image"] == 8
    assert _merged("gpu_5090.yaml", "fast.yaml")["flow"]["timesteps_per_image"] == 4


def test_the_shipped_campaign_defaults_to_one_stage_1_run():
    """A project ships a model; the ablations are research and are parked behind --ablations."""
    cfg = _load("experiments.yaml")
    assert [r["name"] for r in cfg["runs"]] == ["baseline"]
    assert not cfg["runs"][0].get("overrides"), "the baseline is the plain configuration"
    assert len(cfg["ablations"]) >= 1, "parked, not deleted"
    assert cfg.get("refiner") and cfg.get("evaluation"), "the essentials still produce a result"


def test_every_parked_ablation_still_moves_exactly_one_knob():
    """They are parked, not abandoned: the rule has to hold whenever one is finally run."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    for run in _load("experiments.yaml")["ablations"]:
        assert len(run["overrides"]) == 1, f"{run['name']} moves more than one knob"
        assert run.get("ablates"), f"{run['name']} does not say which key it ablates"
        assert (repo / run["overrides"][0]).exists(), run["overrides"][0]
