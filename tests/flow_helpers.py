"""Test fixture helpers and mock model builders for flow model unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

TIMM_TARGETS = r".*blocks\.\d+\.attn\.(qkv|proj)"
HF_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

TINY_INPUT = (64, 80)
TINY_RENDER = (32, 40)


def require_backend(backend: str):
    """Skip test unless peft and the specified backend library are installed."""
    pytest.importorskip("peft")
    pytest.importorskip(backend)


def tiny_encoder(backend: str = "timm", depth: int = 2, input_size=TINY_INPUT, **kwargs):
    """Construct lightweight random-weight DINOv3 encoder for testing."""
    require_backend(backend)
    from dawn.models.encoder import DINOEncoder

    if backend == "timm":
        model_id, targets = "vit_small_plus_patch16_dinov3.lvd1689m", TIMM_TARGETS
        backbone_kwargs = {"depth": depth}
    else:
        model_id, targets = "facebook/dinov3-vits16plus-pretrain-lvd1689m", HF_TARGETS
        backbone_kwargs = {"num_hidden_layers": depth}
    return DINOEncoder(
        backend=backend,
        model_id=model_id,
        lora_targets=targets,
        pretrained=False,
        input_size=input_size,
        backbone_kwargs=backbone_kwargs,
        **kwargs,
    )


def wake_up(module, std: float = 0.05, seed: int = 0):
    """Perturb parameters of a zero-initialized model to produce nonzero outputs."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 0:
                continue
            param.add_(torch.randn(param.shape, generator=generator) * std)
    return module


def load_configs(*overrides: str) -> dict:
    """Load base configuration files merged with optional override files."""
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(CONFIGS / "flow.yaml"),
        omegaconf.OmegaConf.load(CONFIGS / "data.yaml"),
        *[omegaconf.OmegaConf.load(CONFIGS / name) for name in overrides],
    )
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)


def tiny_cfg(correction: bool = False, **updates) -> dict:
    """Generate minimal test configuration with small spatial dimensions."""
    cfg = load_configs()
    cfg["encoder"].update(
        input_size=list(TINY_INPUT),
        pretrained=False,
        grad_checkpointing=False,
        backbone_kwargs={"depth": 2},
    )
    cfg["flow"].update(
        render_size=list(TINY_RENDER), num_train_steps=6, warmup_steps=2, timesteps_per_image=2
    )
    cfg["optim"].update(batch_size=2, precision="fp32", lr_warmup_steps=1)
    cfg["color"]["bgu"].update(bins=4, grid=[2, 2])
    cfg["brightness_correction"]["enabled"] = correction
    cfg["loader"].update(num_workers=0, pin_memory=False)
    cfg["logging"].update(wandb_mode="disabled", log_every=1, val_every=3, save_every=3)
    for section, values in updates.items():
        cfg[section].update(values)
    return cfg


def tiny_layout(correction: bool = False, num_bins: int = 8):
    """Construct TokenLayout with small grid dimensions for unit testing."""
    from dawn.flow.tokens import TokenLayout

    return TokenLayout(num_bins=num_bins, grid=(4, 5), block=4, correction=correction)


def synthetic_batch(batch: int = 2, size=TINY_INPUT, seed: int = 0, fallback=None) -> dict:
    """Generate synthetic multimodal batch with deterministic lightness mappings."""
    import torch.nn.functional as F

    from dawn.data.color import lab_to_linear

    generator = torch.Generator().manual_seed(seed)
    height, width = size
    coarse = torch.rand(batch, 1, 6, 8, generator=generator)
    thermal = F.interpolate(coarse, size=size, mode="bilinear", align_corners=False)
    thermal = thermal * 0.8 + 0.1
    ys = torch.linspace(-1, 1, height).view(1, 1, height, 1)
    xs = torch.linspace(-1, 1, width).view(1, 1, 1, width)
    lightness = 100 * (0.25 + 0.45 * thermal)
    a_star = 14 * torch.sin(2 * xs + torch.arange(batch).view(-1, 1, 1, 1)) + 0 * ys
    b_star = 12 * torch.cos(2 * ys) + 0 * xs + 6
    lab = torch.cat([lightness, a_star.expand_as(thermal), b_star.expand_as(thermal)], dim=1)
    rgb = lab_to_linear(lab).clamp(0, 1)

    def block(row0, col0, rows, cols):
        m = torch.zeros(batch, 1, height, width, dtype=torch.bool)
        m[..., row0 : row0 + rows, col0 : col0 + cols] = True
        return m

    return {
        "thermal": thermal.float(),
        "rgb": rgb.float(),
        "valid": ~block(0, 0, 5, width),
        "moving": block(height - 13, 0, 13, 17),
        "vehicles": block(height - 21, 0, 21, 31),
        "image_id": [f"train_{seed:02d}{i:03d}" for i in range(batch)],
        "fallback": torch.tensor(fallback if fallback is not None else [False] * batch),
    }


def write_tiny_tree(root, n_box: int = 3, n_fallback: int = 1, size=TINY_INPUT) -> dict:
    """Create mock directory structure of aligned image pairs for testing."""
    import numpy as np
    from PIL import Image

    from dawn.data import dronevehicle as dv
    from dawn.data.color import linear_to_srgb8
    from dawn.data.pairs import write_manifest

    root = Path(root)
    patch = {
        "root": str(root / "raw"),
        "aligned_dir": str(root / "aligned"),
        "stats_csv": str(root / "stats.csv"),
        "manifest_dir": str(root / "manifests"),
        "outputs_dir": str(root / "out"),
    }
    layout = dv.DroneVehicleLayout.from_config(patch)
    fallback_ids = []
    for split_index, split in enumerate(("train", "val")):
        ids = []
        for i in range(n_box + n_fallback):
            image_id = f"{split}_{i:05d}"
            item = synthetic_batch(1, size, seed=10 * split_index + i)
            arrays = {
                "rgb": linear_to_srgb8(item["rgb"][0]).permute(1, 2, 0).numpy(),
                "thermal": (item["thermal"][0, 0] * 255).round().to(torch.uint8).numpy(),
            }
            for kind in ("valid", "moving", "vehicles"):
                arrays[kind] = item[kind][0, 0].numpy().astype(np.uint8) * 255
            for kind, array in arrays.items():
                path = layout.aligned_path(image_id, kind)
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(path)
            ids.append(image_id)
            if i >= n_box:
                fallback_ids.append(image_id)
        write_manifest(layout.manifest(f"{split}_pairs"), ids)
    write_manifest(layout.manifest("fallback_pairs"), fallback_ids)
    return patch


def tiny_data_cfg(root, correction: bool = False, n_box: int = 3, n_fallback: int = 1, **updates):
    """Generate test configuration pointing to temporary aligned dataset directory."""
    cfg = tiny_cfg(correction, **updates)
    cfg["dronevehicle"].update(write_tiny_tree(root, n_box, n_fallback))
    cfg["dronevehicle"]["max_pairs"] = None
    return cfg
