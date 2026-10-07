"""Load a trained checkpoint into the pieces inference needs.

`infer.py`, `evaluate.py` and `benchmark_latency.py` all want the same four things out of a
checkpoint and must not each rebuild them differently: the encoder with its LoRA weights, the
velocity net, the token layout, and the normalization stats. The last one matters most --
**stats come from the checkpoint, never from `grid.stats_cache`** (docs/conventions.md section
5.1); a run that read the cache file instead would silently unstandardize with the wrong numbers
if the cache had been recomputed since.

A checkpoint is tied to the resolution and the layout it was trained at, so the input size, the
render size and the upsampling settings all come out of it too, rather than out of whatever
`flow.yaml` happens to say today.

The frozen DINO backbone is not in the checkpoint (`FlowTrainer.state_dict`): it is rebuilt from
`encoder.model_ids` and the LoRA weights are loaded on top, so the pretrained weights have to be
the ones the run started from, and loading needs the DINOv3 license the same way training did.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from dawn.flow.render import RenderSettings
from dawn.flow.tokens import TokenLayout
from dawn.flow.trainer import build_models


@dataclass(frozen=True)
class InferenceBundle:
    """Everything after `load_checkpoint`, ready for `sample_with_spread`.

    Args:
        encoder: `DINOEncoder` with the checkpoint's LoRA weights, in eval mode.
        velocity: `VelocityTransformer`, in eval mode.
        layout: the checkpoint's token layout.
        mean: `(D,)` standardization mean, from the checkpoint.
        std: `(D,)` standardization std, from the checkpoint.
        settings: `RenderSettings` built from the checkpoint's own config.
        input_size: `(H, W)` the encoder was trained at; thermal must arrive at this size.
        render_size: `(h, w)` the trainer computed L_render at. Inference renders at
            `input_size`, not here; this is carried for reporting and for reproducing a
            training-time number.
        step: the training step the checkpoint was written at.
        config: the full config the run used, as a plain dict.
        ema: whether the EMA weights were applied.
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
    """Copy the EMA shadow over the live weights, by the trainer's `module.param` names."""
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
    """Rebuild the models from a `FlowTrainer` checkpoint and load its weights.

    Args:
        path: a checkpoint written by `FlowTrainer.save_checkpoint`.
        device: where to put the models and the stats.
        use_ema: apply the EMA shadow (`optim.ema_decay`) over the live weights. This is what
            validation used and what should be reported; `False` gives the raw last-step weights
            and is for debugging a run that looks wrong.

    Returns:
        An `InferenceBundle`. Both models are in eval mode with gradients disabled.

    Raises:
        FileNotFoundError: no such checkpoint.
        ValueError: the file is not a Dawn flow checkpoint, or `use_ema` was asked for and the
            checkpoint carries no EMA shadow for the weights that matter.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at {path}")
    # Our own training checkpoint: it carries the run's config dict, not only tensors.
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
