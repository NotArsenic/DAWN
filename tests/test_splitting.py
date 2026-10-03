"""Labels -> manifests: vocabulary, the zero-error auto-day cutoffs, and the group split."""

from pathlib import Path

import pytest

omegaconf = pytest.importorskip("omegaconf")

from dawn.data.splitting import (  # noqa: E402
    FINAL_SUBSETS,
    AutoDayCutoffs,
    assert_groups_disjoint,
    auto_day_ids,
    build_final_manifests,
    fit_auto_day_cutoffs,
    map_labels,
    regroup_train_val,
    write_final_manifests,
)
from dawn.data.visdrone import ImageStats, read_manifest  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data.yaml"


@pytest.fixture(scope="module")
def split_cfg():
    return omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load(CONFIG))["split"]


def _row(image_id, split="train", lum=0.2, highlight=4.0):
    return ImageStats(
        image_id=image_id,
        split=split,
        native_width=1920,
        native_height=1080,
        lum_median=lum,
        lum_mean=lum,
        lum_p05=lum / 4,
        lum_p95=lum * 3,
        num_boxes=10,
        noise=0.01,
        highlight_ratio=highlight,
    )


# --- vocabulary ------------------------------------------------------------------------------


def test_night_becomes_hard_test_and_dusk_is_dropped(split_cfg):
    labels = {"a": "day", "b": "night", "c": "dusk", "d": "exclude", "e": None}
    assert map_labels(labels, split_cfg["label_map"]) == {"a": "day", "b": "hard_test"}


def test_an_unmapped_hand_label_is_an_error_not_a_silent_drop():
    with pytest.raises(ValueError, match="label_map"):
        map_labels({"a": "twilight"}, {"day": "day"})


# --- auto-day cutoffs -------------------------------------------------------------------------


def _separable_corpus():
    """Day frames bright and flat, night frames dark with blown highlights."""
    stats, labels = [], {}
    for i in range(60):
        image_id = f"000{100 + i}_00000_d_000000{i % 10}"
        stats.append(_row(image_id, lum=0.15 + 0.002 * i, highlight=3.0 + 0.05 * i))
        labels[image_id] = "day"
    for i in range(20):
        image_id = f"000{300 + i}_00000_d_000000{i % 10}"
        stats.append(_row(image_id, lum=0.01 + 0.001 * i, highlight=25.0 + i))
        labels[image_id] = "hard_test"
    return stats, labels


def test_the_cutoffs_make_zero_errors_on_the_hand_labelled_frames():
    stats, labels = _separable_corpus()
    cutoffs = fit_auto_day_cutoffs(stats, labels, margin=0.1, min_recall=0.2)
    assert cutoffs.errors == 0
    for row in stats:
        if labels[row.image_id] != "day":
            assert not cutoffs.accepts(row), f"{row.image_id} is night but passes"
    assert cutoffs.recall > 0.9
    assert cutoffs.num_day == 60 and cutoffs.num_other == 20


def test_zero_errors_is_kept_even_when_it_costs_almost_all_the_recall():
    """One night frame hiding inside the day cloud has to push the cutoff past it."""
    stats, labels = _separable_corpus()
    intruder = _row("0009999_00000_d_0000001", lum=0.16, highlight=3.2)
    stats.append(intruder)
    labels[intruder.image_id] = "hard_test"
    cutoffs = fit_auto_day_cutoffs(stats, labels, margin=0.0, min_recall=0.0)
    assert cutoffs.errors == 0
    assert not cutoffs.accepts(intruder)


def test_inseparable_features_fail_loudly_instead_of_returning_a_useless_cutoff():
    stats, labels = [], {}
    for i in range(40):  # day and night drawn from the same distribution
        for kind in ("day", "hard_test"):
            image_id = f"{kind[:1]}{i:06d}_00000_d_0000001"
            stats.append(_row(image_id, lum=0.1 + 0.001 * i, highlight=5.0))
            labels[image_id] = kind
    with pytest.raises(ValueError, match="min_recall"):
        fit_auto_day_cutoffs(stats, labels, min_recall=0.2)


def test_the_margin_makes_the_cutoffs_stricter():
    stats, labels = _separable_corpus()
    loose = fit_auto_day_cutoffs(stats, labels, margin=0.0, min_recall=0.0)
    tight = fit_auto_day_cutoffs(stats, labels, margin=0.3, min_recall=0.0)
    assert tight.min_luminance >= loose.min_luminance
    assert tight.max_highlight_ratio <= loose.max_highlight_ratio
    assert tight.recall <= loose.recall
    assert tight.errors == 0


def test_fitting_needs_both_classes():
    stats, labels = _separable_corpus()
    only_day = {k: v for k, v in labels.items() if v == "day"}
    with pytest.raises(ValueError, match="non-day"):
        fit_auto_day_cutoffs(stats, only_day, min_recall=0.0)


def test_only_still_collections_are_auto_labelled():
    cutoffs = AutoDayCutoffs(0.1, 10.0, 1.0, 1, 1, 0, 0.1)
    stats = [
        _row("9999901_00000_d_0000001", lum=0.3, highlight=4.0),  # still, bright: accepted
        _row("9999901_00000_d_0000002", lum=0.01, highlight=90.0),  # still, dark: rejected
        _row("0000287_00759_d_0000759", lum=0.3, highlight=4.0),  # a video: never auto-labelled
    ]
    assert auto_day_ids(stats, cutoffs) == ["9999901_00000_d_0000001"]


# --- group split -------------------------------------------------------------------------------


def test_a_group_never_straddles_train_and_val():
    image_ids = [f"{seq:07d}_00000_d_{frame:07d}" for seq in range(40) for frame in range(5)]
    train, val = regroup_train_val(image_ids, val_fraction=0.1, seed=7)
    assert not train & val
    assert len(val) == 4 and len(train) == 36


def test_the_group_split_is_reproducible_and_seed_dependent():
    image_ids = [f"{seq:07d}_00000_d_0000001" for seq in range(50)]
    first = regroup_train_val(image_ids, 0.1, seed=1)[1]
    assert first == regroup_train_val(image_ids, 0.1, seed=1)[1]
    assert first != regroup_train_val(image_ids, 0.1, seed=2)[1]


def test_the_group_split_rejects_impossible_requests():
    with pytest.raises(ValueError, match="val_fraction"):
        regroup_train_val(["0000001_a"], 1.5)
    with pytest.raises(ValueError, match="at least 2 groups"):
        regroup_train_val(["0000001_a", "0000001_b"], 0.1)


def test_shared_groups_are_reported_not_ignored():
    ok = {"train": ["0000001_a"], "val": ["0000002_a"]}
    assert assert_groups_disjoint(ok) == {"train": {"0000001"}, "val": {"0000002"}}
    with pytest.raises(ValueError, match="share 1 group"):
        assert_groups_disjoint({"train": ["0000001_a"], "val": ["0000001_b"]})


# --- end to end ---------------------------------------------------------------------------------


def _corpus_with_stills():
    stats, hand = [], {}
    for seq in range(30):  # labelled day videos, split train/val in the raw download
        for frame in range(4):
            image_id = f"{seq:07d}_00000_d_{frame:07d}"
            stats.append(_row(image_id, split="train" if seq < 25 else "val",
                              lum=0.18 + 0.001 * seq, highlight=3.0 + 0.02 * seq))
            hand[image_id] = "day"
    for seq in range(30, 38):  # labelled night videos
        for frame in range(3):
            image_id = f"{seq:07d}_00000_d_{frame:07d}"
            stats.append(_row(image_id, lum=0.015, highlight=40.0))
            hand[image_id] = "night"
    for seq in range(38, 42):  # labelled dusk: dropped entirely
        image_id = f"{seq:07d}_00000_d_0000001"
        stats.append(_row(image_id, lum=0.05, highlight=12.0))
        hand[image_id] = "dusk"
    for i in range(20):  # test-dev day
        image_id = f"{900 + i:07d}_00000_d_0000001"
        stats.append(_row(image_id, split="test-dev", lum=0.2, highlight=3.5))
        hand[image_id] = "day"
    for i in range(40):  # unlabelled still collections, half bright half dark
        bright = i % 2 == 0
        image_id = f"999990{i % 4}_00000_d_{i:07d}"
        stats.append(_row(image_id, lum=0.25 if bright else 0.008,
                          highlight=3.0 if bright else 60.0))
        hand[image_id] = None
    return stats, hand


def test_build_final_manifests_produces_the_four_manifests(split_cfg, tmp_path):
    stats, hand = _corpus_with_stills()
    manifests, cutoffs = build_final_manifests(stats, hand, split_cfg)

    assert set(manifests) == set(FINAL_SUBSETS)
    assert cutoffs is not None and cutoffs.errors == 0
    assert manifests["test_day"] and manifests["hard_test"]
    assert manifests["train_day"] and manifests["val_day"]

    # Dusk never appears anywhere.
    everything = {i for ids in manifests.values() for i in ids}
    assert not any(hand.get(i) == "dusk" for i in everything)
    # Bright stills were picked up as day; dark ones were not.
    stills = [i for i in manifests["train_day"] + manifests["val_day"] if i.startswith("99999")]
    assert stills
    assert all(float(next(r.lum_median for r in stats if r.image_id == i)) > 0.1 for i in stills)
    # test-dev stays whole and keeps away from train and val.
    assert len(manifests["test_day"]) == 20
    assert_groups_disjoint({k: v for k, v in manifests.items() if k != "hard_test"})

    counts = write_final_manifests(manifests, tmp_path)
    for name in FINAL_SUBSETS:
        assert (tmp_path / f"{name}.txt").exists()
        assert read_manifest(tmp_path / f"{name}.txt") == manifests[name]
        assert counts[name] == len(manifests[name])


def test_disabling_auto_day_leaves_the_stills_out(split_cfg):
    stats, hand = _corpus_with_stills()
    cfg = {**split_cfg, "auto_day": {**split_cfg["auto_day"], "enabled": False}}
    manifests, cutoffs = build_final_manifests(stats, hand, cfg)
    assert cutoffs is None
    assert not any(i.startswith("99999") for ids in manifests.values() for i in ids)


def test_validation_holds_out_roughly_the_configured_share_of_groups(split_cfg):
    stats, hand = _corpus_with_stills()
    manifests, _ = build_final_manifests(stats, hand, split_cfg)
    groups = assert_groups_disjoint(
        {k: manifests[k] for k in ("train_day", "val_day", "test_day")}
    )
    pool = len(groups["train_day"]) + len(groups["val_day"])
    share = len(groups["val_day"]) / pool
    assert 0.03 < share < 0.25, f"validation holds out {share:.0%} of groups"
