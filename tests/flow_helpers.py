"""Builders shared by the stage-1 tests.

Everything here stays offline and CPU-sized: the encoder is a randomly initialized DINOv3 with
`depth` blocks instead of 12, so no gated weights are downloaded and a test is seconds, not
minutes. Shapes, token layout, register count and the LoRA target names are the real ones.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"

TIMM_TARGETS = r".*blocks\.\d+\.attn\.(qkv|proj)"
HF_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def require_backend(backend: str):
    """Skip unless peft and the backend's library are installed."""
    pytest.importorskip("peft")
    pytest.importorskip(backend)


def tiny_encoder(backend: str = "timm", depth: int = 2, **kwargs):
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


def load_configs(*overrides: str) -> tuple[dict, dict]:
    """`(cfg, calib)`: flow.yaml + data.yaml + the named overrides, and camera_calib.yaml."""
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.merge(
        omegaconf.OmegaConf.load(CONFIGS / "flow.yaml"),
        omegaconf.OmegaConf.load(CONFIGS / "data.yaml"),
        *[omegaconf.OmegaConf.load(CONFIGS / name) for name in overrides],
    )
    calib = omegaconf.OmegaConf.load(CONFIGS / "camera_calib.yaml")
    return (
        omegaconf.OmegaConf.to_container(cfg, resolve=True),
        omegaconf.OmegaConf.to_container(calib, resolve=True),
    )


def unit_stats(num_coeffs: int = 12, num_bins: int = 8) -> dict:
    """Identity grid stats, for tests that care about shapes rather than scaling."""
    return {
        "mean": torch.zeros(num_coeffs, num_bins),
        "std": torch.ones(num_coeffs, num_bins),
    }


def night_batch(num_images: int, cfg: dict, calib: dict, level: str | None = "medium") -> dict:
    """A collated `NightPairs` batch of `num_images` synthetic images."""
    from dawn.flow.dataset import NightPairs, SyntheticDay, collate_night_pairs

    source = SyntheticDay(num_images=num_images, size=256)
    pairs = NightPairs(source, cfg["degradation"], calib, level=level)
    return collate_night_pairs([pairs[i] for i in range(num_images)])
