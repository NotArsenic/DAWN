"""The rectified-flow loss: the target, the bin weighting and the four terms' bookkeeping."""

import pytest

torch = pytest.importorskip("torch")

from dawn.flow.loss import FlowMatchingLoss, bin_weights  # noqa: E402

B, K, GY, GX = 3, 8, 16, 16


def _grids(generator=None):
    shape = (B, 12, K, GY, GX)
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def test_velocity_loss_is_zero_when_the_prediction_is_the_target():
    z_0, z_1 = _grids()
    loss = FlowMatchingLoss(bin_count_weighting=False)
    out = loss(z_1 - z_0, z_0, z_1, torch.rand(B))
    assert float(out["velocity"]) == pytest.approx(0.0, abs=1e-12)
    assert float(out["total"]) == pytest.approx(0.0, abs=1e-12)


def test_the_target_is_z1_minus_z0_not_the_other_way_round():
    """t = 0 is noise and t = 1 is data, so the field points from z_0 to z_1."""
    z_0, z_1 = _grids()
    loss = FlowMatchingLoss(bin_count_weighting=False)
    right = float(loss(z_1 - z_0, z_0, z_1, torch.rand(B))["velocity"])
    wrong = float(loss(z_0 - z_1, z_0, z_1, torch.rand(B))["velocity"])
    assert right < wrong


def test_untrained_velocity_loss_is_about_two():
    """z_1 - z_0 has unit variance in each half, so predicting zero costs about 2."""
    generator = torch.Generator().manual_seed(0)
    z_0, z_1 = _grids(generator)
    out = FlowMatchingLoss(bin_count_weighting=False)(
        torch.zeros_like(z_0), z_0, z_1, torch.rand(B, generator=generator)
    )
    assert 1.7 < float(out["velocity"]) < 2.3


def test_bin_weights_have_mean_one_per_sample():
    counts = torch.rand(B, K, GY, GX) * 100
    weights = bin_weights(counts)
    assert weights.shape == (B, 1, K, GY, GX)
    assert torch.allclose(weights.mean(dim=(1, 2, 3, 4)), torch.ones(B), atol=1e-5)


def test_bin_weighting_ignores_nodes_no_pixel_reached():
    """An empty node is fitted from its ridge prior alone; matching it teaches nothing."""
    z_0, z_1 = _grids()
    counts = torch.ones(B, K, GY, GX)
    counts[:, 0] = 0.0  # the darkest bin is empty everywhere
    loss = FlowMatchingLoss(bin_count_weighting=True)

    v_pred = torch.zeros_like(z_0)
    base = float(loss(v_pred, z_0, z_1, torch.rand(B), node_weights=counts)["velocity"])
    wrecked = z_1.clone()
    wrecked[:, :, 0] += 50.0  # a huge error, but only in the empty bin
    after = float(loss(v_pred, z_0, wrecked, torch.rand(B), node_weights=counts)["velocity"])
    assert after == pytest.approx(base, rel=1e-5)


def test_bin_weighting_is_off_when_asked():
    z_0, z_1 = _grids()
    counts = torch.rand(B, K, GY, GX)
    off = FlowMatchingLoss(bin_count_weighting=False)
    assert float(off(torch.zeros_like(z_0), z_0, z_1, torch.rand(B), node_weights=counts)[
        "velocity"
    ]) == pytest.approx(float((z_1 - z_0).pow(2).mean()), rel=1e-6)


def test_missing_terms_are_reported_as_zero_not_dropped():
    z_0, z_1 = _grids()
    out = FlowMatchingLoss()(torch.zeros_like(z_0), z_0, z_1, torch.rand(B))
    assert set(out) == {"total", "velocity", "apply", "noise", "align"}
    for key in ("apply", "noise", "align"):
        assert float(out[key]) == 0.0


def test_total_is_the_weighted_sum():
    z_0, z_1 = _grids()
    loss = FlowMatchingLoss(
        velocity_weight=1.0, apply_weight=0.1, noise_weight=1.0, align_weight=0.5,
        bin_count_weighting=False,
    )
    out = loss(
        v_pred=torch.zeros_like(z_0),
        z_0=z_0,
        z_1=z_1,
        t=torch.rand(B),
        noise_pred=torch.zeros(B, 2),
        noise_gt=torch.ones(B, 2),
        applied_pred=torch.zeros(B, 3, 8, 8),
        day_image=torch.full((B, 3, 8, 8), 0.5),
        feat_adapted=torch.randn(B, 16, 4),
        feat_frozen=torch.randn(B, 16, 4),
    )
    expected = (
        out["velocity"] + 0.1 * out["apply"] + out["noise"] + 0.5 * out["align"]
    )
    assert float(out["total"]) == pytest.approx(float(expected), rel=1e-6)
    assert float(out["apply"]) == pytest.approx(0.5)
    assert float(out["noise"]) == pytest.approx(1.0)


def test_align_is_zero_for_identical_features_and_two_for_opposite_ones():
    z_0, z_1 = _grids()
    loss = FlowMatchingLoss(bin_count_weighting=False)
    feat = torch.randn(B, 16, 32)
    same = loss(torch.zeros_like(z_0), z_0, z_1, torch.rand(B),
                feat_adapted=feat, feat_frozen=feat)
    flipped = loss(torch.zeros_like(z_0), z_0, z_1, torch.rand(B),
                   feat_adapted=feat, feat_frozen=-feat)
    assert float(same["align"]) == pytest.approx(0.0, abs=1e-6)
    assert float(flipped["align"]) == pytest.approx(2.0, abs=1e-6)


def test_align_never_pushes_gradient_into_the_frozen_target():
    z_0, z_1 = _grids()
    adapted = torch.randn(B, 16, 32, requires_grad=True)
    frozen = torch.randn(B, 16, 32, requires_grad=True)
    out = FlowMatchingLoss(bin_count_weighting=False)(
        torch.zeros_like(z_0), z_0, z_1, torch.rand(B),
        feat_adapted=adapted, feat_frozen=frozen,
    )
    out["total"].backward()
    assert adapted.grad is not None
    assert frozen.grad is None


def test_shape_mismatches_are_rejected():
    z_0, z_1 = _grids()
    with pytest.raises(ValueError):
        FlowMatchingLoss()(torch.zeros(B, 12, K, GY, GX - 1), z_0, z_1, torch.rand(B))


def test_from_config_reads_the_flow_section():
    import omegaconf

    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load("configs/flow.yaml"))
    loss = FlowMatchingLoss.from_config(cfg["flow"])
    assert loss.apply_weight == 0.1
    assert loss.align_weight == 0.5
    assert loss.velocity_weight == 1.0
    assert loss.noise_weight == 1.0
