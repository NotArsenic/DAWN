#!/usr/bin/env python3
"""Smoke test for the DINO encoder backend and velocity transformer memory usage."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.models.encoder import DINOEncoder  # noqa: E402
from dawn.models.velocity import VelocityTransformer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _sync(device: torch.device) -> None:
    """Synchronize CUDA device operations if using CUDA."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    """Execute DINO encoder and velocity transformer smoke tests."""
    parser = argparse.ArgumentParser(description="Smoke-test the DINO encoder backend")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Flow config")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["timm", "transformers"],
        help="Override encoder.backend",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch for the VRAM check")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Random weights: checks the plumbing without downloading anything",
    )
    args = parser.parse_args()

    full = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    cfg = full["encoder"]
    if args.backend:
        cfg["backend"] = args.backend
    if args.no_pretrained:
        cfg["pretrained"] = False
    device = torch.device(args.device)
    height, width = cfg["input_size"]

    logger.info("DINO encoder smoke test")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Backend: {cfg['backend']} ({cfg['model_ids'][cfg['backend']]})")
    logger.info(f"  Device: {device}, input {height}x{width} (H x W)")

    # 1. load
    encoder = DINOEncoder.from_config(cfg).to(device).eval()

    # 2. shapes
    gy, gx = encoder.patch_grid
    tokens = gy * gx
    thermal = torch.rand(2, 1, height, width, device=device)
    rgb = torch.rand(2, 3, height, width, device=device) * 0.5
    with torch.no_grad():
        out = encoder(thermal, return_attn=True)
        bundle = encoder.encode(thermal)
        full_attn = encoder.last_layer_attention(thermal)
        frozen = encoder.forward_frozen(rgb)
    assert out.patch_tokens.shape == (2, tokens, encoder.embed_dim), out.patch_tokens.shape
    assert out.cls_token.shape == (2, encoder.embed_dim), out.cls_token.shape
    assert out.patch_attn.shape == (2, tokens, tokens), out.patch_attn.shape
    assert bundle["patch_map"].shape == (2, encoder.embed_dim, gy, gx)
    assert bundle["thumb"].shape == (2, 1, gy, gx)
    assert bundle["attn_bias"].shape == (2, 1, tokens, tokens)
    assert frozen.patch_tokens.shape == out.patch_tokens.shape
    rows = full_attn.sum(dim=-1)
    logger.info(
        "  Shapes OK: %d tokens (%d x %d). Raw attention rows sum to [%.6f, %.6f] over %d",
        tokens,
        gy,
        gx,
        rows.min(),
        rows.max(),
        full_attn.shape[-1],
    )
    logger.info("  Patch attention rows after renormalizing: %.6f", out.patch_attn.sum(-1).mean())

    # 3. where LoRA landed
    names = DINOEncoder.lora_target_names(encoder.backbone)
    logger.info("  LoRA on %d modules, e.g. %s", len(names), names[:4])
    assert not any("patch_embed" in n or "embeddings" in n for n in names)

    # 4. trainable count and the no-op adapters
    total = sum(p.numel() for p in encoder.parameters())
    logger.info(
        "  Trainable %d (%.2fM) / total %d (%.2fM)",
        encoder.num_trainable_params(),
        encoder.num_trainable_params() / 1e6,
        total,
        total / 1e6,
    )
    with torch.no_grad(), encoder.backbone.disable_adapter():
        off = encoder(thermal, return_attn=True)
    delta = (out.patch_tokens - off.patch_tokens).abs().max()
    logger.info(
        "  |adapted - adapters off| on thermal at init: %.3e (LoRA B is zero: must be 0)", delta
    )
    assert delta == 0, "the adapters are not a no-op at init"

    # 5. memory and speed of a forward + backward
    encoder.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch = torch.rand(args.batch_size, 1, height, width, device=device)
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    started = time.perf_counter()
    with autocast:
        features = encoder(batch, return_attn=True)
    features.patch_tokens.float().square().mean().backward()
    _sync(device)
    if device.type != "cuda":
        logger.info(
            "  Forward + backward at batch %d OK (no VRAM figure off CUDA)", args.batch_size
        )
        logger.info("Smoke test passed.")
        return
    logger.info(
        "  Encoder fwd+bwd at batch %d: peak VRAM %.2f GB, %.2f s (grad checkpointing %s)",
        args.batch_size,
        torch.cuda.max_memory_allocated(device) / 2**30,
        time.perf_counter() - started,
        cfg.get("grad_checkpointing"),
    )

    # 5b. the velocity net at batch x timesteps with the DINO attention bias as a float mask
    layout = TokenLayout.from_config(full)
    velocity = VelocityTransformer.from_config(full["velocity"], layout).to(device).train()
    repeats = full["flow"]["timesteps_per_image"]
    count = args.batch_size * repeats
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    z_t = torch.randn(count, layout.num_tokens, layout.dim, device=device)
    t = torch.rand(count, device=device)
    feats = torch.randn(count, tokens, encoder.embed_dim, device=device)
    bias = torch.randn(count, 1, tokens, tokens, device=device)
    thumb = torch.rand(count, 1, gy, gx, device=device)
    with torch.no_grad():
        for block in velocity.blocks:
            block.dino_gate.fill_(0.1)
    with autocast:
        v = velocity(z_t, t, feats, None, thumb, bias)
    v.float().square().mean().backward()
    _sync(device)
    logger.info(
        "  Velocity fwd+bwd at %d x %d = %d samples: peak VRAM %.2f GB, %.2f s",
        args.batch_size,
        repeats,
        count,
        torch.cuda.max_memory_allocated(device) / 2**30,
        time.perf_counter() - started,
    )
    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
