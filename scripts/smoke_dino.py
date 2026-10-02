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

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the DINO encoder backend")
    parser.add_argument("--config", type=str, default="configs/flow.yaml", help="Flow config")
    parser.add_argument("--backend", type=str, default=None, choices=["timm", "transformers"],
                        help="Override encoder.backend")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch for the VRAM check")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    logger.info("DINO encoder smoke test")
    logger.info(f"  Config: {args.config}")
    logger.info(f"  Backend: {args.backend or 'from config'}")
    logger.info(f"  Device: {args.device}")

    # TODO: Implement
    # 1. cfg = OmegaConf.load(args.config); apply --backend; DINOEncoder.from_config(cfg.encoder)
    # 2. Forward a random image; assert the EncoderOutput shapes above
    # 3. Print [n for n, m in encoder.backbone.named_modules() if "lora_" in n]; assert that
    #    each is an attention module and none is under patch_embed
    # 4. Assert num_trainable_params() is ~0.2M; assert forward_frozen == forward at init
    # 5. bf16 autocast forward + backward at --batch-size; report torch.cuda.max_memory_allocated

    logger.warning("Smoke test not yet implemented.")


if __name__ == "__main__":
    main()
