"""Deterministic per-image randomness.

Every random draw that belongs to one image derives its seed from that image's ID, so the same
ID gives the same output on any machine, in any worker, in any order (docs/conventions.md
section 10). Training asks for a fresh draw instead; the fixed val / test sets ask for a seeded
one.

    rng = generator_for("0000287_00759_d_0000759", "night", "hard")

The hash is BLAKE2b over the ID and the salts, so it does not depend on PYTHONHASHSEED and
neighbouring IDs do not give neighbouring seeds.
"""

from __future__ import annotations

import hashlib

import torch

_MASK = (1 << 63) - 1  # torch.Generator.manual_seed wants a non-negative 63-bit int


def seed_from_image_id(image_id: str, *salts: object) -> int:
    """A stable 63-bit seed for `image_id`, optionally split per purpose by `salts`.

    Args:
        image_id: The image ID (or any stable string key).
        salts: Extra keys that separate independent draws for the same image, for example the
            darkness level or the name of the stage.

    Returns:
        A non-negative integer below 2**63.
    """
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
    """A `torch.Generator` seeded from `image_id` (and `salts`). Same ID, same stream."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed_from_image_id(image_id, *salts))
    return generator
