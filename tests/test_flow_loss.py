"""Tests for flow matching training loss functions and weighting schemes."""

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from dawn.flow.color_bins import NUM_BINS  # noqa: E402
from dawn.flow.loss import FlowLoss  # noqa: E402
from tests.flow_helpers import load_configs, tiny_layout  # noqa: E402

B = 2


def _loss(correction=False, **kwargs):
    return FlowLoss(tiny_layout(correction), **kwargs)


def _tokens(layout, seed=0):
    g = torch.Generator().manual_seed(seed)
    shape = (B, layout.num_tokens, layout.dim)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def _weights(layout, seed=0):
    g = torch.Generator().manual_seed(seed)
    gy, gx = layout.grid
    node = torch.rand(B, layout.num_bins, gy, gx, generator=g) + 0.1
    fraction = torch.rand(B, 1, *layout.color_size, generator=g)
    return layout.weights(node, fraction)


@pytest.fixture(params=[False, True], ids=["correction_off", "correction_on"])
def correction(request):
    return request.param


# --- L_flow ---------------------------------------------------------------------------------


def test_flow_loss_is_zero_when_the_prediction_is_the_target(correction):
    loss = _loss(correction)
    z_0, z_1 = _tokens(loss.layout)
    out = loss.flow(z_1 - z_0, z_0, z_1, _weights(loss.layout))
    assert out["flow"] == 0 and all(float(v) == 0 for v in out.values())


def test_the_target_is_z1_minus_z0_not_the_other_way_round():
    loss = _loss()
    z_0, z_1 = _tokens(loss.layout)
    assert (
        loss.flow(z_1 - z_0, z_0, z_1, None)["flow"] < loss.flow(z_0 - z_1, z_0, z_1, None)["flow"]
    )


def test_untrained_velocity_loss_is_about_two_with_unit_noise_and_unit_data():
    loss = _loss()
    z_0, z_1 = _tokens(loss.layout)
    value = loss.flow(torch.zeros_like(z_0), z_0, z_1, None)["flow"]
    assert 1.6 < float(value) < 2.4  # E|z1 - z0|^2 = 2 for independent unit Gaussians


def test_flow_is_logged_per_segment_and_the_segments_match_the_layout(correction):
    loss = _loss(correction)
    z_0, z_1 = _tokens(loss.layout)
    out = loss.flow(torch.zeros_like(z_0), z_0, z_1, None)
    expected = {"flow", "flow_brightness", "flow_color"} | (
        {"flow_correction"} if correction else set()
    )
    assert set(out) == expected
    target = (z_1 - z_0).pow(2)
    for name, seg in loss.layout.segments.items():
        assert torch.isclose(out[f"flow_{name}"], target[..., seg].mean(), rtol=1e-5)


def test_per_value_balance_is_the_plain_mean_over_the_token_so_colour_counts_double(correction):
    """Verify per-value balance weighs segments proportionally to their dimensionality."""
    loss = _loss(correction, segment_balance="per_value")
    z_0, z_1 = _tokens(loss.layout)
    out = loss.flow(torch.zeros_like(z_0), z_0, z_1, None)
    layout = loss.layout
    assert torch.isclose(out["flow"], (z_1 - z_0).pow(2).mean(), rtol=1e-5)
    sizes = {n: s.stop - s.start for n, s in layout.segments.items()}
    expected = sum(sizes[n] * out[f"flow_{n}"] for n in sizes) / layout.dim
    assert torch.isclose(out["flow"], expected, rtol=1e-5)
    assert sizes["color"] == 2 * sizes["brightness"]


def test_equal_balance_counts_every_segment_the_same(correction):
    loss = _loss(correction, segment_balance="equal")
    z_0, z_1 = _tokens(loss.layout)
    out = loss.flow(torch.zeros_like(z_0), z_0, z_1, None)
    parts = [out[f"flow_{n}"] for n in loss.layout.segments]
    assert torch.isclose(out["flow"], torch.stack(parts).mean(), rtol=1e-5)


def test_segment_balance_must_be_known():
    with pytest.raises(ValueError, match="segment_balance"):
        _loss(segment_balance="loudest")


def test_weights_act_per_value_and_zero_weight_values_do_not_count(correction):
    loss = _loss(correction)
    z_0, z_1 = _tokens(loss.layout)
    v = torch.zeros_like(z_0)
    weights = _weights(loss.layout)
    plain = loss.flow(v, z_0, z_1, None)["flow"]
    weighted = loss.flow(v, z_0, z_1, weights)["flow"]
    assert not torch.isclose(plain, weighted)
    # Zero out the weight of every value of sample 1: only sample 0 contributes.
    cut = weights.clone()
    cut[1] = 0
    only_first = loss.flow(v, z_0, z_1, cut)["flow"]
    first = loss.flow(v[:1], z_0[:1], z_1[:1], weights[:1])["flow"]
    assert torch.isclose(only_first, first / B, rtol=1e-4)


def test_flow_loss_rejects_mismatched_shapes():
    loss = _loss()
    z_0, z_1 = _tokens(loss.layout)
    with pytest.raises(ValueError, match="v_pred"):
        loss.flow(z_0[:, :-1], z_0, z_1, None)
    with pytest.raises(ValueError, match="layout"):
        loss.flow(z_0[..., :-1], z_0[..., :-1], z_1[..., :-1], None)


# --- L_render ---------------------------------------------------------------------------------


def _lab(seed=0, size=(8, 10)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, 3, *size, generator=g) * 30 + 40


def test_render_loss_is_zero_at_the_target_and_is_l1_of_lab_over_100():
    loss = _loss()
    lab = _lab()
    weight, sample = torch.ones(B, 1, 8, 10), torch.ones(B)
    assert loss.render(lab, lab, weight, sample) == 0
    shifted = lab + torch.tensor([10.0, 0.0, -5.0]).view(1, 3, 1, 1)
    assert torch.isclose(loss.render(shifted, lab, weight, sample), torch.tensor(0.05), rtol=1e-5)


def test_render_loss_masks_by_fraction_and_down_weights_fallback_pairs():
    loss = _loss()
    lab = _lab()
    wrong = lab.clone()
    wrong[1] += 30.0  # sample 1 is off by 0.3 everywhere
    ones = torch.ones(B, 1, 8, 10)
    base = loss.render(wrong, lab, ones, torch.ones(B))
    assert torch.isclose(base, torch.tensor(0.15), rtol=1e-4)  # mean of 0 and 0.3
    half = loss.render(wrong, lab, ones, torch.tensor([1.0, 0.5]))  # sample 1 is a fallback pair
    assert torch.isclose(half, torch.tensor(0.075), rtol=1e-4)  # mean of 0 and 0.5 * 0.3
    masked = ones.clone()
    masked[1] = 0.0  # the bad sample is fully masked: contributes nothing, and no NaN
    out = loss.render(wrong, lab, masked, torch.ones(B))
    assert torch.isfinite(out) and out == 0
    partial = ones.clone()
    partial[1, :, :4] = 0.0  # half of the bad sample masked: its masked MEAN is unchanged
    assert torch.isclose(loss.render(wrong, lab, partial, torch.ones(B)), base, rtol=1e-4)


# --- L_align ------------------------------------------------------------------------------


def test_align_is_zero_for_identical_features_and_two_for_opposite_ones():
    loss = _loss()
    feats = torch.randn(B, 20, 384)
    weight = torch.ones(B, 20)
    assert torch.allclose(loss.align(feats, feats.clone(), weight), torch.zeros(()), atol=1e-6)
    assert torch.allclose(loss.align(feats, -feats, weight), torch.tensor(2.0), atol=1e-5)


def test_align_never_pushes_gradient_into_the_frozen_target():
    loss = _loss()
    adapted = torch.randn(B, 20, 384, requires_grad=True)
    frozen = torch.randn(B, 20, 384, requires_grad=True)
    loss.align(adapted, frozen, torch.ones(B, 20)).backward()
    assert adapted.grad is not None and frozen.grad is None


def test_align_is_a_masked_mean_over_patches():
    loss = _loss()
    adapted, frozen = torch.randn(1, 4, 16), torch.randn(1, 4, 16)
    cos = F.cosine_similarity(adapted, frozen, dim=-1)[0]
    weight = torch.tensor([[1.0, 0.0, 0.5, 0.0]])
    expected = ((1 - cos) * weight[0]).sum() / weight.sum()
    assert torch.isclose(loss.align(adapted, frozen, weight), expected, rtol=1e-5)


# --- L_color ----------------------------------------------------------------------------------


def test_color_loss_is_the_cross_entropy_against_the_soft_target_and_ignores_small_patches():
    loss = _loss(min_patch_fraction=0.5)
    logits = torch.randn(1, 3, NUM_BINS)
    target = torch.softmax(torch.randn(1, 3, NUM_BINS), dim=-1)
    weight = torch.tensor([[1.0, 0.4, 0.8]])  # patch 1 is under min_patch_fraction: counts 0
    ce = -(target * F.log_softmax(logits, dim=-1)).sum(-1)[0]
    expected = (ce[0] * 1.0 + ce[2] * 0.8) / 1.8
    assert torch.isclose(loss.color(logits, target, weight), expected, rtol=1e-5)
    # Perfect, certain prediction of a one-hot target costs ~0.
    hot = torch.zeros(1, 1, NUM_BINS)
    hot[..., 7] = 1.0
    sure = hot * 50.0
    assert loss.color(sure, hot, torch.ones(1, 1)) < 1e-6


def test_color_loss_with_no_usable_patch_is_zero_not_nan():
    loss = _loss(min_patch_fraction=0.5)
    out = loss.color(
        torch.randn(1, 2, NUM_BINS), torch.rand(1, 2, NUM_BINS), torch.full((1, 2), 0.1)
    )
    assert out == 0 and torch.isfinite(out)


# --- the total -----------------------------------------------------------------------------------


def test_total_is_the_weighted_sum_and_missing_terms_are_reported_as_zero(correction):
    loss = _loss(
        correction, velocity_weight=1.0, render_weight=0.1, align_weight=0.5, color_weight=0.2
    )
    layout = loss.layout
    z_0, z_1 = _tokens(layout)
    v = torch.zeros_like(z_0)
    lab, tgt = _lab(1), _lab(2)
    feats = torch.randn(B, layout.num_tokens, 384)
    logits = torch.randn(B, layout.num_tokens, NUM_BINS)
    soft = torch.softmax(torch.randn(B, layout.num_tokens, NUM_BINS), dim=-1)
    full = loss(
        v_pred=v,
        z_0=z_0,
        z_1=z_1,
        token_weights=None,
        lab_pred=lab,
        lab_target=tgt,
        pixel_weight=torch.ones(B, 1, 8, 10),
        sample_weight=torch.ones(B),
        feat_adapted=feats,
        feat_frozen=feats + 1.0,
        patch_weight=torch.ones(B, layout.num_tokens),
        color_logits=logits,
        color_target=soft,
        color_patch_weight=torch.ones(B, layout.num_tokens),
    )
    expected = 1.0 * full["flow"] + 0.1 * full["render"] + 0.5 * full["align"] + 0.2 * full["color"]
    assert torch.isclose(full["total"], expected, rtol=1e-5)
    only_flow = loss(v_pred=v, z_0=z_0, z_1=z_1)
    assert only_flow["render"] == only_flow["align"] == only_flow["color"] == 0
    assert torch.isclose(only_flow["total"], only_flow["flow"])


def test_from_config_reads_the_flow_section():
    cfg = load_configs()
    loss = FlowLoss.from_config(cfg, tiny_layout())
    flow = cfg["flow"]
    assert (loss.velocity_weight, loss.render_weight) == (
        flow["velocity_weight"],
        flow["render_weight"],
    )
    assert (loss.align_weight, loss.color_weight) == (flow["align_weight"], flow["color_weight"])
    assert loss.segment_balance == flow["segment_balance"] == "per_value"
    assert loss.min_patch_fraction == cfg["color"]["min_patch_fraction"]
