"""Shared pytest setup."""

import pytest


@pytest.fixture(autouse=True)
def _seed():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def fake_visdrone(tmp_path_factory):
    """Raw fake VisDrone tree (tests/fake_visdrone.py); returns its VisDroneLayout."""
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    from tests.fake_visdrone import make_fake_visdrone

    return make_fake_visdrone(tmp_path_factory.mktemp("visdrone"))


@pytest.fixture(scope="session")
def prepared_visdrone(fake_visdrone):
    """The fake tree after all three prepare phases: stats, labelled manifests, 512-px copies."""
    from dawn.data.prepare import (
        compute_image_stats,
        make_lowres_copy,
        resolve_labels,
        write_manifests,
    )
    from dawn.data.visdrone import write_stats_csv
    from tests.fake_visdrone import FAKE_IMAGES, FAKE_LABELS

    layout = fake_visdrone
    stats = [compute_image_stats(layout, split, image_id) for split, image_id, *_ in FAKE_IMAGES]
    write_stats_csv(layout.stats_csv, stats)
    write_manifests(stats, resolve_labels(stats, FAKE_LABELS), layout.manifest_dir)
    for split, image_id, *_ in FAKE_IMAGES:
        make_lowres_copy(layout, split, image_id)
    return layout
