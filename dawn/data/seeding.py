"""Deterministic per-image random seed utilities."""

from __future__ import annotations

import hashlib

import torch

_MASK = (1 << 63) - 1


def seed_from_image_id(image_id: str, *salts: object) -> int:
    """Generate a deterministic 63-bit integer seed from an image ID and optional salts."""
    digest = hashlib.blake2b(image_id.encode("utf-8"), digest_size=8)
    for salt in salts:
        digest.update(b"\x1f")
        digest.update(str(salt).encode("utf-8"))
    return int.from_bytes(digest.digest(), "big") & _MASK


def generator_for(
    image_id: str,
    *salts: object,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    """Create a torch.Generator seeded deterministically from an image ID and optional salts."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_from_image_id(image_id, *salts))
    return generator
