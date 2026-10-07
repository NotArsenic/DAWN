"""dawn/data/dedup.py: 64-bit perceptual hashes and cross-split near-duplicate matching."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from dawn.data import dedup  # noqa: E402


def _scene(seed: int, h: int = 512, w: int = 640) -> torch.Tensor:
    """A smooth random `(1, H, W)` image with a few rectangles: structure for the hash."""
    g = torch.Generator().manual_seed(seed)
    img = torch.nn.functional.avg_pool2d(torch.rand(1, 1, h, w, generator=g), 21, 1, 10)[0]
    for _ in range(6):
        y, x = (
            int(torch.randint(0, h - 100, (1,), generator=g)),
            int(torch.randint(0, w - 100, (1,), generator=g)),
        )
        img[:, y : y + 80, x : x + 90] = float(torch.rand(1, generator=g))
    return (img - img.min()) / (img.max() - img.min())


def test_hash_is_a_64_bit_integer_and_deterministic():
    img = _scene(0)
    h = dedup.phash(img)
    assert isinstance(h, int)
    assert 0 <= h < 2**64
    assert dedup.phash(img) == h


def test_hex_round_trip_is_16_characters():
    for h in (0, 1, 2**64 - 1, dedup.phash(_scene(1))):
        text = dedup.to_hex(h)
        assert len(text) == 16
        assert dedup.from_hex(text) == h


def test_hamming_counts_differing_bits():
    assert dedup.hamming(0b1011, 0b0001) == 2
    assert dedup.hamming(2**64 - 1, 0) == 64


def test_the_same_scene_with_noise_stays_close_and_other_scenes_are_far():
    base = _scene(0)
    noisy = (
        base + 0.02 * torch.randn(base.shape, generator=torch.Generator().manual_seed(9))
    ).clamp(0, 1)
    near = dedup.hamming(dedup.phash(base), dedup.phash(noisy))
    others = [dedup.hamming(dedup.phash(base), dedup.phash(_scene(s))) for s in range(1, 9)]
    assert near <= 4
    assert min(others) > 12


def test_hash_survives_a_monotone_contrast_curve():
    # The thermal camera rescales its gain frame by frame: the hash must not care much.
    base = _scene(3)
    curved = (base.clamp(0, 1) ** 0.7) * 0.8 + 0.1
    assert dedup.hamming(dedup.phash(base), dedup.phash(curved)) <= 6


def test_hash_changes_under_a_flip():
    base = _scene(4)
    assert dedup.hamming(dedup.phash(base), dedup.phash(base.flip(-1))) > 12


def test_hash_accepts_2d_input():
    img = _scene(5)
    assert dedup.phash(img[0]) == dedup.phash(img)


# --- near-duplicate search ------------------------------------------------------------------------


def _arr(*values):
    return np.array(values, dtype=np.uint64)


def test_nearest_pairs_requires_both_modalities_to_be_close():
    train_rgb, train_th = _arr(0b0000, 0b1111_0000), _arr(0b0000, 0b0000)
    query_rgb, query_th = _arr(0b0000), _arr(0b11111111)  # rgb matches pair 0, thermal does not
    index, dist = dedup.nearest_pairs(query_rgb, query_th, train_rgb, train_th)
    assert dist[0] == 8  # the worse of the two modalities, minimised over train pairs
    assert index[0] in (0, 1)


def test_nearest_pairs_picks_the_train_pair_with_the_smallest_worst_case_distance():
    train_rgb = _arr(0b0000_0000, 0b0000_0011, 0xFFFF_FFFF)
    train_th = _arr(0b0000_1111, 0b0000_0001, 0x0)
    query_rgb, query_th = _arr(0b0000_0000), _arr(0b0000_0000)
    index, dist = dedup.nearest_pairs(query_rgb, query_th, train_rgb, train_th)
    # pair 0: (0, 4) -> 4; pair 1: (2, 1) -> 2; pair 2: (32, 0) -> 32
    assert (index[0], dist[0]) == (1, 2)


def test_nearest_pairs_chunks_do_not_change_the_answer():
    rng = np.random.default_rng(0)
    q = rng.integers(0, 2**63, (50,), dtype=np.uint64)
    q2 = rng.integers(0, 2**63, (50,), dtype=np.uint64)
    t = rng.integers(0, 2**63, (70,), dtype=np.uint64)
    t2 = rng.integers(0, 2**63, (70,), dtype=np.uint64)
    a = dedup.nearest_pairs(q, q2, t, t2, chunk=7)
    b = dedup.nearest_pairs(q, q2, t, t2, chunk=1000)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_nearest_pairs_with_no_train_pairs_returns_no_match():
    index, dist = dedup.nearest_pairs(_arr(1), _arr(1), _arr(), _arr())
    assert index[0] == -1
    assert dist[0] > 64


def test_counts_at_thresholds_count_queries_at_or_under_each():
    dist = np.array([0, 3, 5, 7, 12, 100])
    assert dedup.counts_at(dist, [4, 6, 8, 10]) == {4: 2, 6: 3, 8: 4, 10: 4}


# --- cross-split matching of pairs ------------------------------------------------------------


def _stats(image_id, rgb, thermal):
    from dawn.data.day_filter import PairStats

    split, stem = image_id.split("_")
    return PairStats(
        image_id=image_id,
        split=split,
        stem=stem,
        status="ok",
        folder="f",
        rgb_filename="a",
        thermal_filename="b",
        border_rgb="",
        border_thermal="",
        thermal_channels_ok=True,
        lum_median=0.2,
        lum_p05=0.1,
        lum_p95=0.5,
        lum_mean=0.2,
        rgb_hash=dedup.to_hex(rgb),
        thermal_hash=dedup.to_hex(thermal),
    )


def test_cross_split_duplicates_need_both_modalities_within_the_threshold():
    refs = [_stats("train_1", 0b0, 0b0), _stats("train_2", 2**40 - 1, 2**40 - 1)]
    cands = [
        _stats("val_1", 0b0, 0b0),  # identical to train_1
        _stats("val_2", 0b111, 0b11),  # 3 and 2 bits away: within 6
        _stats("val_3", 0b0, 2**10 - 1),  # RGB matches train_1 but thermal is 10 bits off
        _stats("test_1", 2**20 - 1, 2**30 - 1),  # far from both
    ]
    rows = {r.image_id: r for r in dedup.cross_split_duplicates(cands, refs, max_hamming=6)}
    assert (rows["val_1"].matched_id, rows["val_1"].distance, rows["val_1"].near_duplicate) == (
        "train_1",
        0,
        True,
    )
    assert rows["val_2"].near_duplicate and (rows["val_2"].d_rgb, rows["val_2"].d_thermal) == (3, 2)
    assert not rows["val_3"].near_duplicate and rows["val_3"].distance == 10
    assert not rows["test_1"].near_duplicate
    assert rows["val_1"].d_rgb == rows["val_1"].d_thermal == 0


def test_the_threshold_is_inclusive():
    refs = [_stats("train_1", 0, 0)]
    near = dedup.cross_split_duplicates([_stats("val_1", 0b111111, 0)], refs, 6)[0]
    far = dedup.cross_split_duplicates([_stats("val_1", 0b1111111, 0)], refs, 6)[0]
    assert near.near_duplicate and not far.near_duplicate


def test_no_references_means_no_duplicates():
    rows = dedup.cross_split_duplicates([_stats("val_1", 1, 1)], [], 6)
    assert rows[0].matched_id == "" and not rows[0].near_duplicate


def test_dedup_log_round_trips_sorted(tmp_path):
    refs = [_stats("train_1", 0, 0)]
    rows = dedup.cross_split_duplicates([_stats("val_2", 1, 1), _stats("test_1", 0, 0)], refs, 6)
    path = tmp_path / "logs" / "dedup.csv"
    dedup.write_dedup_log(path, rows)
    back = dedup.read_dedup_log(path)
    assert list(back) == ["val_2", "test_1"] or list(back) == ["test_1", "val_2"]
    assert back["test_1"].near_duplicate and back["test_1"].matched_id == "train_1"
    assert path.read_text(encoding="utf-8").splitlines()[0] == (
        "image_id,matched_id,distance,d_rgb,d_thermal,near_duplicate"
    )
