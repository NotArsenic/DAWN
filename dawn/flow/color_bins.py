"""Color bin derivation, soft encoding, and per-patch histogram computation."""

from __future__ import annotations

import functools

import torch

from dawn.data.color import linear_to_lab, srgb8_to_linear
from dawn.data.resample import area_downsample

BIN_STEP = 10  # a*b* units
NUM_BINS = 261  # occupied cells of the 10-unit grid over the sRGB gamut
_CHUNK = 1 << 14  # pixels per distance matrix: (_CHUNK, 261) floats


@functools.lru_cache(maxsize=None)
def ab_bin_centers(step: int = BIN_STEP, code_step: int = 1) -> torch.Tensor:
    """Compute occupied Lab color bin centers within the sRGB gamut.

    Args:
        step: Grid spacing in Lab units.
        code_step: Sampling step across 8-bit color channels.

    Returns:
        Tensor of shape (Q, 2) containing sorted bin centers.
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
    """Soft-encode Lab color coordinates onto discrete bin centers.

    Args:
        ab: Color coordinates of shape (..., 2).
        k: Number of nearest bins.
        sigma: Gaussian kernel bandwidth.
        centers: Optional bin centers tensor of shape (Q, 2).

    Returns:
        Soft assignments of shape (..., Q) summing to 1 along the last dimension.
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
    """Find k nearest bin centers and compute normalized Gaussian weights."""
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
    """Compute masked per-patch soft color histograms across bins.

    Args:
        ab_map: Color map of shape (B, 2, H, W).
        mask: Loss mask of shape (B, 1, H, W).
        patch_size: Spatial patch dimensions (ph, pw) or scalar.
        k: Number of nearest bins per pixel.
        sigma: Gaussian bandwidth.

    Returns:
        Histogram tensor of shape (B, N, Q).
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
    """Compute average mask coverage fraction for each spatial patch."""
    ph, pw = _pair(patch_size)
    height, width = mask.shape[-2:]
    _check_divides((height, width), (ph, pw))
    return area_downsample(mask.to(torch.float32), (height // ph, width // pw)).flatten(1)

