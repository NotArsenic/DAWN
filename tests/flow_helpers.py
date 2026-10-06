"""Builders shared by the flow-stage tests.

Everything here stays offline and CPU-sized: the encoder is a randomly initialized DINOv3 with
`depth` blocks instead of 12, so no gated weights are downloaded and a test is seconds, not
minutes; the images are small synthetic ones whose brightness is an exact function of the thermal.
Shapes, token layout, register count and the LoRA target names are the real ones.

A tiny setup: a 64 x 80 input gives a 4 x 5 token grid, a 16 x 20 colour map and, at render size
32 x 40, 8 x 8 render pixels per... patch of 8 x 8 (render / token grid).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

TIMM_TARGETS = r".*blocks\.\d+\.attn\.(qkv|proj)"
HF_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

TINY_INPUT = (64, 80)  # (H, W) encoder input of the tiny setup
TINY_RENDER = (32, 40)


def require_backend(backend: str):
    """Skip unless peft and the backend's library are installed."""
    pytest.importorskip("peft")
    pytest.importorskip(backend)


def tiny_encoder(backend: str = "timm", depth: int = 2, input_size=TINY_INPUT, **kwargs):
    """A randomly initialized DINOv3 ViT-S+ with `depth` blocks, LoRA attached."""
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
    """Perturb every parameter except the 0-dim gates, so a zero-init net stops being a no-op.

    adaLN-Zero and the zero-initialized output head make an untrained DiT return exactly zero
    for every input, which makes it useless for testing that inputs reach the output. This is
    "one optimizer step happened", not a trained model.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for param in module.parameters():
            if param.ndim == 0:  # dino_gate: the tests set it explicitly
                continue
            param.add_(torch.randn(param.shape, generator=generator) * std)
    return module


def load_configs(*overrides: str) -> dict:
    """flow.yaml + data.yaml + the named overrides (file names under configs/), as a plain dict."""
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(CONFIGS / "flow.yaml"),
        omegaconf.OmegaConf.load(CONFIGS / "data.yaml"),
        *[omegaconf.OmegaConf.load(CONFIGS / name) for name in overrides],
    )
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)


def tiny_cfg(correction: bool = False, **updates) -> dict:
    """The real flow config at the tiny resolution, random-weight 2-block DINO, fp32, no wandb."""
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
    from dawn.flow.tokens import TokenLayout

    return TokenLayout(num_bins=num_bins, grid=(4, 5), block=4, correction=correction)


def synthetic_batch(batch: int = 2, size=TINY_INPUT, seed: int = 0, fallback=None) -> dict:
    """A collated `AlignedPairs`-style batch whose L*/100 is an exact affine map of the thermal.

    thermal  smooth field in [0.1, 0.9]
    rgb      linear RGB of Lab(100 * (0.25 + 0.45 * thermal), a(x), b(y)): the brightness grid can
             reproduce the lightness exactly, and the a*b* vary smoothly (inside the gamut)
    valid / moving / vehicles   bool masks with odd-aligned edges (so a 2x downsample
             gives fractional weights); `moving` and `vehicles` each cover one corner block
    """
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
