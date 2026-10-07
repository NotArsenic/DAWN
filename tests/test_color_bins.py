"""The 261 a*b* bins, Zhang et al.'s soft encoding and the per-patch histograms
(docs/conventions.md section 6.1)."""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow.color_bins import (  # noqa: E402
    NUM_BINS,
    ab_bin_centers,
    patch_histograms,
    patch_mask_fraction,
    soft_encode,
)


@pytest.fixture(scope="module")
def centers():
    return ab_bin_centers()


def test_there_are_261_bins_on_a_10_unit_grid_sorted_by_a_then_b(centers):
    assert NUM_BINS == 261 and centers.shape == (261, 2) and centers.dtype == torch.float32
    assert torch.equal(centers, torch.round(centers / 10) * 10)
    assert centers[:, 0].min() == -90 and centers[:, 0].max() == 100
    assert centers[:, 1].min() == -110 and centers[:, 1].max() == 90
    assert len({tuple(c) for c in centers.tolist()}) == 261
    order = sorted(map(tuple, centers.tolist()))
    assert [tuple(c) for c in centers.tolist()] == order


def test_every_second_code_gives_the_same_bins(centers):
    assert torch.equal(ab_bin_centers(code_step=2), centers)


def test_the_bins_are_cached():
    assert ab_bin_centers() is ab_bin_centers()


def test_neutral_grey_is_a_bin(centers):
    assert ((centers == 0).all(dim=1)).any()


def test_soft_targets_sum_to_one_and_have_k_nonzero_entries(centers):
    ab = torch.rand(7, 5, 2) * 100 - 30
    target = soft_encode(ab, k=5, sigma=5.0)
    assert target.shape == (7, 5, NUM_BINS)
    assert torch.allclose(target.sum(dim=-1), torch.ones(7, 5), atol=1e-5)
    assert ((target > 0).sum(dim=-1) <= 5).all() and (target >= 0).all()


def test_a_pure_colour_peaks_at_its_bin(centers):
    index = 120
    target = soft_encode(centers[index : index + 1] + 0.0, k=5, sigma=5.0)
    assert int(target.argmax()) == index
    # An off-centre colour still peaks at its nearest bin.
    near = soft_encode(centers[index : index + 1] + torch.tensor([[2.0, -1.0]]), k=5, sigma=5.0)
    assert int(near.argmax()) == index


def test_soft_encode_uses_a_gaussian_of_the_distance(centers):
    ab = centers[100:101] + torch.tensor([[3.0, 0.0]])
    target = soft_encode(ab, k=5, sigma=5.0)[0]
    nearest = torch.topk(target, 5).indices
    d = (centers[nearest] - ab).norm(dim=-1)
    expected = torch.exp(-(d**2) / (2 * 25.0))
    assert torch.allclose(target[nearest], expected / expected.sum(), atol=1e-5)


def _flat(ab_value, size=(16, 16)):
    return torch.tensor(ab_value, dtype=torch.float32).view(1, 2, 1, 1).expand(1, 2, *size).clone()


def test_a_pure_colour_patch_has_a_histogram_peaking_at_its_bin(centers):
    index = 77
    ab_map = _flat(centers[index].tolist())
    hist = patch_histograms(ab_map, torch.ones(1, 1, 16, 16), patch_size=8)
    assert hist.shape == (1, 4, NUM_BINS)
    assert torch.allclose(hist.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
    assert (hist.argmax(dim=-1) == index).all()


def test_the_patch_histogram_is_a_masked_mean_of_the_pixels(centers):
    a_idx, b_idx = 30, 200
    ab_map = _flat(centers[a_idx].tolist())
    ab_map[..., :, 4:] = torch.tensor(centers[b_idx].tolist()).view(
        1, 2, 1, 1
    )  # right half: colour B
    mask = torch.ones(1, 1, 16, 16)
    both = patch_histograms(ab_map, mask, patch_size=16)[0, 0]
    # Colour A fills 4/16 columns and colour B 12/16: their peak bins carry the same ratio.
    assert torch.isclose(both[b_idx] / both[a_idx], torch.tensor(3.0), rtol=1e-4)
    assert set(torch.topk(both, 2).indices.tolist()) == {a_idx, b_idx}
    mask[..., :, 4:] = 0.0  # mask out colour B: only colour A is left
    only_a = patch_histograms(ab_map, mask, patch_size=16)[0, 0]
    assert int(only_a.argmax()) == a_idx and torch.isclose(only_a.sum(), torch.tensor(1.0))
    assert only_a[b_idx] < 1e-3


def test_a_fully_masked_patch_has_an_all_zero_histogram_and_zero_fraction():
    ab_map, mask = _flat([10.0, 10.0]), torch.ones(1, 1, 16, 16)
    mask[..., :8, :8] = 0.0
    hist = patch_histograms(ab_map, mask, patch_size=8)
    assert hist[0, 0].abs().sum() == 0 and torch.isclose(hist[0, 1].sum(), torch.tensor(1.0))
    fraction = patch_mask_fraction(mask, patch_size=8)
    assert torch.allclose(fraction, torch.tensor([[0.0, 1.0, 1.0, 1.0]]))


def test_patch_size_may_be_non_square_and_must_divide_the_map():
    ab_map, mask = _flat([0.0, 0.0], (16, 24)), torch.ones(1, 1, 16, 24)
    assert patch_histograms(ab_map, mask, patch_size=(8, 4)).shape == (1, 2 * 6, NUM_BINS)
    with pytest.raises(ValueError, match="divide"):
        patch_histograms(ab_map, mask, patch_size=7)
