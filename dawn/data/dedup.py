"""Near-duplicate detection across splits: 64-bit perceptual hashes and Hamming matching.

Consecutive frames of one flight are near-duplicates, and the official DroneVehicle splits are
not flight-disjoint (77 of the 84 train folders also appear in test). A val or test pair is a
near-duplicate of a train pair when its RGB hash **and** its thermal hash are both within
`dedup.max_hamming` bits of the same train pair. The matching is deliberately two-sided: a scene
that only looks alike in one modality (a thermal gain change, a lighting change) is not a leak.

The hash is the classic DCT perceptual hash: area-downsample to 32x32 (docs/conventions.md
section 4a), 2-D DCT, keep the 8x8 low frequencies, one bit per coefficient above their median.
"""

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

HASH_SIDE = 8  # 8 x 8 = 64 bits
_DCT_SIDE = 32
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash(img: torch.Tensor) -> int:
    """64-bit perceptual hash of a single-channel image.

    The median is taken over the 63 AC coefficients (the DC term, the mean brightness, is far
    larger than the rest and would skew it); the DC bit is kept and is almost always 1.

    Args:
        img: `(H, W)` or `(1, H, W)` float image, any range. Callers pass luminance in display
            codes (sRGB-coded luma for RGB, the thermal codes for thermal).

    Returns:
        An integer in `[0, 2**64)`; bit `i` belongs to coefficient `i` in row-major order.
    """
    if img.ndim == 3:
        img = img[0]
    small = area_downsample(img.float(), (_DCT_SIDE, _DCT_SIDE)).double().numpy()
    low = dctn(small, type=2, norm="ortho")[:HASH_SIDE, :HASH_SIDE].ravel()
    bits = low > np.median(low[1:])
    return int(sum(1 << i for i, bit in enumerate(bits) if bit))


def to_hex(value: int) -> str:
    """16 lowercase hex characters."""
    return f"{value:016x}"


def from_hex(text: str) -> int:
    return int(text, 16)


def hamming(a: int, b: int) -> int:
    """Number of differing bits."""
    return bin(a ^ b).count("1")


def _popcount(values: np.ndarray) -> np.ndarray:
    """Per-element bit count of a uint64 array."""
    as_bytes = np.ascontiguousarray(values).view(np.uint8).reshape(*values.shape, 8)
    return _POPCOUNT[as_bytes].sum(axis=-1, dtype=np.int32)


def nearest_pairs(
    query_rgb: np.ndarray,
    query_thermal: np.ndarray,
    train_rgb: np.ndarray,
    train_thermal: np.ndarray,
    chunk: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """For each query pair, the train pair whose worse modality is closest.

    The distance between a query and a train pair is `max(hamming(rgb), hamming(thermal))`, so a
    pair is within `k` bits exactly when both modalities are. Ties go to the lower train index.

    Args:
        query_rgb, query_thermal: `(Q,)` uint64 hashes, one per query pair.
        train_rgb, train_thermal: `(T,)` uint64 hashes of the candidate train pairs.
        chunk: Queries per step (bounds the `chunk x T` distance matrices).

    Returns:
        `(index, distance)`, both `(Q,)` int: the best train index and its distance. With no
        train pairs the index is -1 and the distance 65 (above any real distance).
    """
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
    """How many query pairs have a train pair within each Hamming threshold (inclusive)."""
    return {int(k): int((np.asarray(distance) <= k).sum()) for k in thresholds}


# --- cross-split matching of whole pairs ---------------------------------------------------------


class HashedPair(Protocol):
    """What `cross_split_duplicates` needs of a pair: an ID and the two hex hashes (the rows of
    `dawn.data.day_filter.PairStats` qualify)."""

    image_id: str
    rgb_hash: str
    thermal_hash: str


@dataclass(frozen=True)
class DedupRow:
    """One row of the dedup log: a val / test pair and the train pair closest to it.

    `distance` is the worse of the two modalities (`d_rgb`, `d_thermal`) for that train pair, and
    `near_duplicate` is `distance <= max_hamming`. `matched_id` is "" when there were no train
    pairs to compare against.
    """

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
    """For each candidate pair (an accepted val / test pair), its closest reference pair (a train
    pair that passed the day filter) and whether it is a near-duplicate of it.

    Near-duplicate: RGB **and** thermal hashes both within `max_hamming` bits of the same
    reference pair. The reference set is the train pairs that passed the day filter, which does
    not depend on any alignment threshold, so loosening alignment later cannot sneak leakage back
    in. One row per candidate, in the order given.
    """
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
    """Write the committed dedup log, sorted by (split, image ID)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow([field.name for field in fields(DedupRow)])
        for row in sorted(rows, key=lambda r: _row_key(r.image_id)):
            writer.writerow([int(v) if isinstance(v, bool) else v for v in astuple(row)])


def read_dedup_log(path: str | Path) -> dict[str, DedupRow]:
    """Read the dedup log into `{image_id: DedupRow}`, in file order."""
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
