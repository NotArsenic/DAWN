"""Near-duplicate detection across dataset splits using perceptual hashes."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import astuple, dataclass, fields
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from scipy.fft import dctn

from dawn.data.resample import area_downsample

HASH_SIDE = 8
_DCT_SIDE = 32
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(img: torch.Tensor) -> int:
    """Compute a 64-bit perceptual hash (pHash) of an image."""
    if img.ndim == 3:
        img = img[0]
    small = area_downsample(img.float(), (_DCT_SIDE, _DCT_SIDE)).double().numpy()
    low = dctn(small, type=2, norm="ortho")[:HASH_SIDE, :HASH_SIDE].ravel()
    bits = low > np.median(low[1:])
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def to_hex(value: int) -> str:
    """Format a 64-bit integer as a 16-character hexadecimal string."""
    return f"{value:016x}"


def from_hex(text: str) -> int:
    """Parse a hexadecimal string into an integer."""
    return int(text, 16)


def hamming(a: int, b: int) -> int:
    """Compute the Hamming distance between two integers."""
    return bin(a ^ b).count("1")


def _popcount(values: np.ndarray) -> np.ndarray:
    """Compute bit counts per element for a uint64 array."""
    as_bytes = np.ascontiguousarray(values).view(np.uint8).reshape(*values.shape, 8)
    return _POPCOUNT[as_bytes].sum(axis=-1, dtype=np.int32)


def nearest_pairs(
    query_rgb: np.ndarray,
    query_thermal: np.ndarray,
    train_rgb: np.ndarray,
    train_thermal: np.ndarray,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the nearest train pair for each query pair by maximum modal Hamming distance."""
    n_query = len(query_rgb)
    index = np.full(n_query, -1, dtype=np.int64)
    distance = np.full(n_query, 65, dtype=np.int32)
    if len(train_rgb) == 0:
        return index, distance
    for start in range(0, n_query, chunk):
        sl = slice(start, start + chunk)
        d_rgb = _popcount(query_rgb[sl, None] ^ train_rgb[None, :])
        d_thermal = _popcount(query_thermal[sl, None] ^ train_thermal[None, :])
        worst = np.maximum(d_rgb, d_thermal)
        best = worst.argmin(axis=1)
        index[sl] = best
        distance[sl] = worst[np.arange(len(best)), best]
    return index, distance


def counts_at(distance: np.ndarray, thresholds: list[int]) -> dict[int, int]:
    """Count pairs within each Hamming distance threshold."""
    return {int(k): int((np.asarray(distance) <= k).sum()) for k in thresholds}


class HashedPair(Protocol):

    image_id: str
    rgb_hash: str
    thermal_hash: str


@dataclass(frozen=True)
class DedupRow:
    """Near-duplicate comparison result between a query pair and the closest train pair."""

    image_id: str
    matched_id: str
    distance: int
    d_rgb: int
    d_thermal: int
    near_duplicate: bool


def _hashes(pairs: Sequence[HashedPair]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([from_hex(p.rgb_hash) for p in pairs], dtype=np.uint64),
        np.array([from_hex(p.thermal_hash) for p in pairs], dtype=np.uint64),
    )


def cross_split_duplicates(
    candidates: Sequence[HashedPair], references: Sequence[HashedPair], max_hamming: int
) -> list[DedupRow]:
    """Identify near-duplicate pairs across splits within a maximum Hamming distance."""
    if not candidates:
        return []
    q_rgb, q_thermal = _hashes(candidates)
    t_rgb, t_thermal = _hashes(references)
    index, distance = nearest_pairs(q_rgb, q_thermal, t_rgb, t_thermal)
    rows = []
    for k, candidate in enumerate(candidates):
        if index[k] < 0:
            rows.append(DedupRow(candidate.image_id, "", 65, 64, 64, False))
            continue
        ref = references[index[k]]
        d_rgb = hamming(from_hex(candidate.rgb_hash), from_hex(ref.rgb_hash))
        d_thermal = hamming(from_hex(candidate.thermal_hash), from_hex(ref.thermal_hash))
        rows.append(
            DedupRow(
                candidate.image_id,
                ref.image_id,
                int(distance[k]),
                d_rgb,
                d_thermal,
                bool(distance[k] <= max_hamming),
            )
        )
    return rows


_SPLIT_ORDER = ("train", "val", "test")


def _row_key(image_id: str) -> tuple[int, str]:
    split = image_id.split("_", 1)[0]
    return (_SPLIT_ORDER.index(split) if split in _SPLIT_ORDER else len(_SPLIT_ORDER), image_id)


def write_dedup_log(path: str | Path, rows: Iterable[DedupRow]) -> None:
    """Write deduplication results to a CSV log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([field.name for field in fields(DedupRow)])
        for row in sorted(rows, key=lambda r: _row_key(r.image_id)):
            writer.writerow([int(v) if isinstance(v, bool) else v for v in astuple(row)])


def read_dedup_log(path: str | Path) -> dict[str, DedupRow]:
    """Read deduplication results from a CSV log."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = [
            DedupRow(
                r["image_id"],
                r["matched_id"],
                int(r["distance"]),
                int(r["d_rgb"]),
                int(r["d_thermal"]),
                r["near_duplicate"] == "1",
            )
            for r in csv.DictReader(f)
        ]
    return {row.image_id: row for row in rows}
