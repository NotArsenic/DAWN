#!/usr/bin/env python3
"""Smoke test for the DINO encoder backend. Run before any training.

For the backend chosen in configs/flow.yaml (override with --backend), checks that:
    1. The weights load (HF token and gated-repo access are fine).
    2. A random (B, 3, 256, 256) linear image gives patch_tokens (B, 256, 384),
       cls_token (B, 384) and patch_attn (B, 256, 256), with CLS + 4 register tokens stripped.
    3. peft attached LoRA to the attention qkv / output projections only: prints the matched
       module names (patch_embed.proj must not be among them).
    4. Trainable parameters are ~0.2M, and forward_frozen (adapters disabled) matches the
       adapted forward at init, where every LoRA B matrix is zero.
    5. A forward + backward at --batch-size fits the 16 GB budget (prints peak VRAM).

Usage:
    python scripts/smoke_dino.py --backend timm
    python scripts/smoke_dino.py --backend transformers --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dawn.models.encoder import DINOEncoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the DINO encoder backend")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Flow config")
    parser.add_argument("--backend", type=str, default=None, choices=["timm", "transformers"],
                        help="Override encoder.backend")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch for the VRAM check")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Random weights: checks the plumbing without downloading anything")
    args = parser.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)["encoder"]
    if args.backend:
        cfg["backend"] = args.backend
    if args.no_pretrained:
        cfg["pretrained"] = False
    device = torch.device(args.device)

    logger.info("DINO encoder smoke test")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Backend: {cfg['backend']} ({cfg['model_ids'][cfg['backend']]})")
    logger.info(f"  Device: {device}")

    # 1. load
    encoder = DINOEncoder.from_config(cfg).to(device).eval()

    # 2. shapes
    size = cfg["input_size"]
    tokens = (size // cfg["patch_size"]) ** 2
    image = torch.rand(2, 3, size, size, device=device)
    with torch.no_grad():
        out = encoder(image, return_attn=True)
        bundle = encoder.encode(image)
        full = encoder.last_layer_attention(image)
    assert out.patch_tokens.shape == (2, tokens, encoder.embed_dim), out.patch_tokens.shape
    assert out.cls_token.shape == (2, encoder.embed_dim), out.cls_token.shape
    assert out.patch_attn.shape == (2, tokens, tokens), out.patch_attn.shape
    assert bundle["patch_map"].shape == (2, encoder.embed_dim, *encoder.patch_grid)
    assert bundle["thumb"].shape == (2, 3, *encoder.patch_grid)
    assert bundle["attn_bias"].shape == (2, 1, tokens, tokens)
    rows = full.sum(dim=-1)
    logger.info("  Shapes OK. Raw attention rows sum to [%.6f, %.6f] over %d tokens",
                rows.min(), rows.max(), full.shape[-1])
    logger.info("  Patch attention rows after renormalizing: %.6f",
                out.patch_attn.sum(-1).mean())

    # 3. where LoRA landed
    names = DINOEncoder.lora_target_names(encoder.backbone)
    logger.info("  LoRA on %d modules, e.g. %s", len(names), names[:4])
    assert not any("patch_embed" in n or "embeddings" in n for n in names)

    # 4. trainable count and the frozen path
    total = sum(p.numel() for p in encoder.parameters())
    logger.info("  Trainable %d (%.2fM) / total %d (%.2fM)",
                encoder.num_trainable_params(), encoder.num_trainable_params() / 1e6,
                total, total / 1e6)
    # Compare the same attention path on both sides: eager and fused SDPA differ by ~1e-6,
    # which has nothing to do with the adapters.
    with torch.no_grad():
        adapted = encoder(image, return_attn=True)
        frozen = encoder.forward_frozen(image, return_attn=True)
    delta = (adapted.patch_tokens - frozen.patch_tokens).abs().max()
    logger.info("  |adapted - frozen| at init: %.3e (LoRA B is zero, so this must be 0)", delta)
    assert delta == 0, "the adapters are not a no-op at init"

    # 5. memory and speed of a forward + backward
    encoder.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    batch = torch.rand(args.batch_size, 3, size, size, device=device)
    autocast = torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )
    with autocast:
        features = encoder(batch, return_attn=True)
    features.patch_tokens.float().square().mean().backward()
    if device.type == "cuda":
        logger.info("  Peak VRAM at batch %d: %.2f GB",
                    args.batch_size, torch.cuda.max_memory_allocated(device) / 2**30)
    else:
        logger.info("  Forward + backward at batch %d OK (no VRAM figure off CUDA)",
                    args.batch_size)
    logger.info("Smoke test passed.")


if __name__ == "__main__":
    main()
