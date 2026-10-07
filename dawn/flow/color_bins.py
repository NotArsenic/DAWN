"""The a*b* bins of L_color: derivation, soft encoding and per-patch histograms.

The bins are the 261 occupied cells of a 10-unit a*b* grid over the 8-bit sRGB gamut, derived in
code (no download, no file): map every 8-bit sRGB colour to CIELAB, round a* and b* to the grid,
keep the cells that occur, sorted by a* then b* (docs/conventions.md section 6.1). The plan's 313
is Zhang et al.'s set for the full Lab gamut; this is the set we can actually render.

`soft_encode` is Zhang, Isola & Efros (ECCV 2016): the `k` nearest bins with a Gaussian of width
`sigma` over the a*b* distance, normalized to sum to 1. `patch_histograms` averages the soft
encodings of the pixels of each token's patch, masked, at render size (8x8 samples per patch at
the default 256x320) so that the cost stays small: it never builds a dense (pixels, bins) tensor
for the whole batch.
"""

from __future__ import annotations

import functools

import torch

from dawn.data.color import linear_to_lab, srgb8_to_linear
from dawn.data.resample import area_downsample

BIN_STEP = 10  # a*b* units
NUM_BINS = 261  # occupied cells of the 10-unit grid over the sRGB gamut (`ab_bin_centers`)
_CHUNK = 1 << 14  # pixels per distance matrix: (_CHUNK, 261) floats


@functools.lru_cache(maxsize=None)
def ab_bin_centers(step: int = BIN_STEP, code_step: int = 1) -> torch.Tensor:
    """The occupied a*b* bin centres, `(Q, 2)` float32, sorted by a* then b*.

    Args:
        step: Grid spacing in a*b* units.
        code_step: Use every `code_step`-th 8-bit code per channel (1 = all 16.7M colours; the
            set is the same for 2, which is what the unit test checks). Cached per argument.
    """
    codes = torch.arange(0, 256, code_step, dtype=torch.uint8)
    green_blue = torch.cartesian_prod(codes, codes)  # (G, 2)
    cells: list[torch.Tensor] = []
    for red in codes:
        rgb8 = torch.cat([red.expand(green_blue.shape[0], 1), green_blue], dim=1)  # (G, 3)
        lab = linear_to_lab(srgb8_to_linear(rgb8).view(-1, 3, 1, 1)).view(-1, 3)
        index = torch.round(lab[:, 1:] / step).to(torch.int64)
        cells.append(torch.unique(index, dim=0))
    unique = torch.unique(torch.cat(cells), dim=0)  # lexicographic: a* then b*
    return (unique * step).to(torch.float32)


def soft_encode(
    ab: torch.Tensor, k: int = 5, sigma: float = 5.0, centers: torch.Tensor | None = None
) -> torch.Tensor:
    """Soft-encode a*b* values over the bins (Zhang et al.).

    Args:
        ab: `(..., 2)` a*, b* in CIELAB units.
        k: Number of nearest bins that share the mass.
        sigma: Gaussian width over the a*b* distance.
        centers: `(Q, 2)` bin centres; the 261 by default.

    Returns:
        `(..., Q)` non-negative, summing to 1 over the last dimension, `k` nonzeros at most.
    """
    index, weight = _nearest(ab.reshape(-1, 2), k, sigma, centers)
    out = weight.new_zeros(
        index.shape[0], (centers if centers is not None else ab_bin_centers()).shape[0]
    )
    out.scatter_(1, index, weight)
    return out.reshape(*ab.shape[:-1], out.shape[-1])


def _nearest(
    ab: torch.Tensor, k: int, sigma: float, centers: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(index, weight)`, both `(P, k)`: the k nearest bins and their normalized Gaussian mass."""
    centers = (ab_bin_centers() if centers is None else centers).to(ab)
    indices, weights = [], []
    for start in range(0, ab.shape[0], _CHUNK):
        d2 = torch.cdist(ab[start : start + _CHUNK].float(), centers.float()).pow(2)
        nearest = torch.topk(d2, k, dim=1, largest=False)
        w = torch.exp(-nearest.values / (2 * sigma**2))
        indices.append(nearest.indices)
        weights.append(w / w.sum(dim=1, keepdim=True))
    return torch.cat(indices), torch.cat(weights)


def _pair(patch_size: int | tuple[int, int]) -> tuple[int, int]:
    return (patch_size, patch_size) if isinstance(patch_size, int) else tuple(patch_size)


def _check_divides(shape: tuple[int, int], patch: tuple[int, int]) -> None:
    if shape[0] % patch[0] or shape[1] % patch[1]:
        raise ValueError(f"patch size {patch} must divide the map size {tuple(shape)}")


def patch_histograms(
    ab_map: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int | tuple[int, int],
    k: int = 5,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Per-patch soft histograms over the bins: the masked mean of the pixels' soft encodings.

    Args:
        ab_map: `(B, 2, H, W)` a*, b* in CIELAB units, at the size the histograms are built at.
        mask: `(B, 1, H, W)` keep weights in [0, 1] (fractions are fine).
        patch_size: Pixels per patch `(ph, pw)` (or one int) at this size; must divide `(H, W)`.

    Returns:
        `(B, N, Q)`, patches row-major (`n = y * (W / pw) + x`, the token order). Each histogram
        sums to 1 over the bins, or is all zero when the patch has no weight.
    """
    ph, pw = _pair(patch_size)
    batch, _, height, width = ab_map.shape
    _check_divides((height, width), (ph, pw))
    gy, gx = height // ph, width // pw
    rows = torch.arange(height, device=ab_map.device) // ph
    cols = torch.arange(width, device=ab_map.device) // pw
    patch_id = (rows.view(-1, 1) * gx + cols.view(1, -1)).reshape(-1)  # (P,)
    q = ab_bin_centers().shape[0]
    hist = torch.zeros(batch, gy * gx * q, device=ab_map.device, dtype=torch.float32)
    total = torch.zeros(batch, gy * gx, device=ab_map.device, dtype=torch.float32)
    for b in range(batch):
        ab = ab_map[b].reshape(2, -1).t()  # (P, 2)
        m = mask[b].reshape(-1).to(torch.float32)  # (P,)
        index, weight = _nearest(ab, k, sigma, None)
        flat = (patch_id.unsqueeze(1) * q + index).reshape(-1)
        hist[b].index_add_(0, flat, (weight * m.unsqueeze(1)).reshape(-1))
        total[b].index_add_(0, patch_id, m)
    hist = hist.view(batch, gy * gx, q)
    return hist / total.unsqueeze(-1).clamp(min=torch.finfo(torch.float32).tiny)


def patch_mask_fraction(mask: torch.Tensor, patch_size: int | tuple[int, int]) -> torch.Tensor:
    """Kept share of every patch, `(B, N)`: the mask area-averaged to the patch grid."""
    ph, pw = _pair(patch_size)
    height, width = mask.shape[-2:]
    _check_divides((height, width), (ph, pw))
    return area_downsample(mask.to(torch.float32), (height // ph, width // pw)).flatten(1)
