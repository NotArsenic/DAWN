"""Utilities for loading trained model checkpoints for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from dawn.flow.render import RenderSettings
from dawn.flow.tokens import TokenLayout
from dawn.flow.trainer import build_models
from dawn.models.refiner import LocalRefiner


@dataclass(frozen=True)
class InferenceBundle:
    """Container holding models and configuration loaded for inference.

    Args:
        encoder: Vision encoder model with loaded weights.
        velocity: Velocity transformer model with loaded weights.
        layout: Token layout specification.
        mean: Normalization mean vector of shape (D,).
        std: Normalization standard deviation vector of shape (D,).
        settings: Render settings derived from checkpoint configuration.
        input_size: Spatial dimensions (H, W) expected by encoder.
        render_size: Spatial dimensions (h, w) used during training.
        step: Training step index recorded in checkpoint.
        config: Full configuration mapping from checkpoint.
        ema: Whether exponential moving average weights were loaded.
    """

    encoder: nn.Module
    velocity: nn.Module
    layout: TokenLayout
    mean: torch.Tensor
    std: torch.Tensor
    settings: RenderSettings
    input_size: tuple[int, int]
    render_size: tuple[int, int]
    step: int
    config: dict
    ema: bool


def _apply_ema(state: Mapping, modules: Mapping[str, nn.Module]) -> int:
    """Copy EMA shadow weights to target modules."""
    shadow = state["ema"]["shadow"]
    applied = 0
    with torch.no_grad():
        for prefix, module in modules.items():
            for name, param in module.named_parameters():
                key = f"{prefix}.{name}"
                if key in shadow:
                    param.copy_(shadow[key].to(param.dtype).to(param.device))
                    applied += 1
    return applied


def load_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    use_ema: bool = True,
) -> InferenceBundle:
    """Load model weights, layout, and statistics from a flow checkpoint.

    Args:
        path: Path to checkpoint file.
        device: Device to place models and parameters on.
        use_ema: Whether to apply exponential moving average parameter weights.

    Returns:
        InferenceBundle initialized with checkpoint components in evaluation mode.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If file is missing required checkpoint entries.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    missing = {"config", "layout", "stats", "velocity", "lora"} - set(state)
    if missing:
        raise ValueError(
            f"{path} is not a Dawn flow checkpoint: it has no {sorted(missing)} "
            "(expected one written by FlowTrainer.save_checkpoint)"
        )

    cfg = state["config"]
    layout = TokenLayout.from_descriptor(state["layout"])
    device = torch.device(device)
    encoder, velocity, _color_head = build_models(cfg, layout, device)

    velocity.load_state_dict(state["velocity"])
    unexpected = encoder.load_state_dict(state["lora"], strict=False)[1]
    if unexpected:
        raise ValueError(f"checkpoint has unknown encoder keys: {unexpected[:4]}")

    modules = {"encoder": encoder, "velocity": velocity}
    if use_ema:
        if "ema" not in state or not state["ema"].get("shadow"):
            raise ValueError(f"{path} carries no EMA shadow; pass use_ema=False to use it anyway")
        if _apply_ema(state, modules) == 0:
            raise ValueError(
                f"{path}'s EMA shadow matched none of the encoder or velocity parameters"
            )

    for module in modules.values():
        module.eval().requires_grad_(False)

    mean, std = layout.stats_vector(state["stats"])
    return InferenceBundle(
        encoder=encoder,
        velocity=velocity,
        layout=layout,
        mean=mean.to(device),
        std=std.to(device),
        settings=RenderSettings.from_config(cfg),
        input_size=tuple(int(v) for v in state["input_size"]),
        render_size=tuple(int(v) for v in state["render_size"]),
        step=int(state.get("step", 0)),
        config=cfg,
        ema=use_ema,
    )


def load_refiner(
    path: str | Path,
    device: torch.device | str = "cpu",
    *,
    use_ema: bool = True,
) -> LocalRefiner:
    """Load trained LocalRefiner model from checkpoint.

    Args:
        path: Path to refiner checkpoint file.
        device: Device to place model on.
        use_ema: Whether to apply exponential moving average parameter weights.

    Returns:
        LocalRefiner initialized in evaluation mode with gradients disabled.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        ValueError: If checkpoint format is invalid or missing weights.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    missing = {"config", "refiner", "refiner_config"} - set(state)
    if missing:
        raise ValueError(
            f"{path} is not a Dawn refiner checkpoint: it has no {sorted(missing)} "
            "(expected one written by scripts/train_refiner.py)"
        )

    device = torch.device(device)
    refiner = LocalRefiner.from_config(state["refiner_config"]).to(device)
    refiner.load_state_dict(state["refiner"])

    if use_ema:
        if "ema" not in state or not state["ema"].get("shadow"):
            raise ValueError(f"{path} carries no EMA shadow; pass use_ema=False to use it anyway")
        if _apply_ema(state, {"refiner": refiner}) == 0:
            raise ValueError(f"{path}'s EMA shadow matched none of the refiner's parameters")

    return refiner.eval().requires_grad_(False)
