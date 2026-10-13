"""Tests for VelocityTransformer architecture, conditioning streams, zero-initialization, and attention gating."""

import pytest

torch = pytest.importorskip("torch")

from dawn.models.encoder import attention_to_bias  # noqa: E402
from dawn.models.velocity import VelocityTransformer  # noqa: E402
from tests.flow_helpers import load_configs, tiny_layout, wake_up  # noqa: E402

B, DINO = 2, 384


def _make(correction=False, **kwargs):
    return VelocityTransformer(tiny_layout(correction), **kwargs)


@pytest.fixture(params=[False, True], ids=["correction_off", "correction_on"])
def layout(request):
    return tiny_layout(request.param)


@pytest.fixture
def inputs(layout):
    n, d = layout.num_tokens, layout.dim
    gy, gx = layout.grid
    generator = torch.Generator().manual_seed(11)
    attn = torch.rand(B, n, n, generator=generator)
    return {
        "z_t": torch.randn(B, n, d, generator=generator),
        "t": torch.rand(B, generator=generator),
        "dino_features": torch.randn(B, n, DINO, generator=generator),
        "dino_attn": attn / attn.sum(dim=-1, keepdim=True),
        "thumb": torch.randn(B, 1, gy, gx, generator=generator),
    }


def _call(model, inputs, **overrides):
    args = {**inputs, **overrides}
    return model(
        args["z_t"],
        args["t"],
        args["dino_features"],
        args.get("dino_attn"),
        args.get("thumb"),
        args.get("attn_bias"),
    )


def test_output_shape_and_model_sizes_come_from_the_layout(layout, inputs):
    model = VelocityTransformer(layout)
    assert model.num_tokens == layout.num_tokens and model.token_dim == layout.dim
    assert _call(model, inputs).shape == inputs["z_t"].shape


def test_parameter_count_is_about_seven_million_at_the_plans_size():
    """Verify model parameter count is approximately seven million with full token layout."""
    from dawn.flow.tokens import TokenLayout

    model = VelocityTransformer(TokenLayout(num_bins=8, grid=(32, 40), block=4))
    assert 6e6 < model.num_params() < 9e6, model.num_params()


def test_velocity_is_exactly_zero_at_init(layout, inputs):
    """Verify velocity output is identically zero at initialization."""
    out = _call(VelocityTransformer(layout), inputs)
    assert torch.equal(out, torch.zeros_like(out))


def test_every_block_starts_with_a_zero_attention_gate(layout):
    model = VelocityTransformer(layout)
    assert model.attn_gates() == {f"gate_{i}": 0.0 for i in range(6)}


def test_with_a_zero_gate_the_attention_bias_changes_nothing(layout, inputs):
    """Verify attention bias has no effect when all attention gates are zero."""
    model = wake_up(VelocityTransformer(layout))
    other = torch.rand_like(inputs["dino_attn"])
    other = other / other.sum(dim=-1, keepdim=True)
    with torch.no_grad():
        first = _call(model, inputs)
        second = _call(model, inputs, dino_attn=other)
        none = _call(model, inputs, dino_attn=None)
    assert torch.equal(first, second)
    # Same arithmetic as passing no mask at all, up to the kernel SDPA picks for each.
    assert torch.allclose(first, none, atol=1e-5)
    assert first.abs().max() > 1e-3, "the model is still a no-op, so this proves nothing"


def test_a_nonzero_gate_makes_the_attention_bias_matter(layout, inputs):
    model = wake_up(VelocityTransformer(layout))
    with torch.no_grad():
        before = _call(model, inputs)
        for block in model.blocks:
            block.dino_gate.fill_(0.5)
        after = _call(model, inputs)
    assert not torch.allclose(before, after)


def test_the_gate_takes_gradient(layout, inputs):
    model = wake_up(VelocityTransformer(layout))
    _call(model, inputs).square().mean().backward()
    assert all(block.dino_gate.grad is not None for block in model.blocks)


def test_a_precomputed_bias_and_the_raw_attention_agree(layout, inputs):
    model = wake_up(VelocityTransformer(layout))
    with torch.no_grad():
        for block in model.blocks:
            block.dino_gate.fill_(0.3)
        from_attn = _call(model, inputs)
        from_bias = _call(
            model, inputs, dino_attn=None, attn_bias=attention_to_bias(inputs["dino_attn"])
        )
    assert torch.equal(from_attn, from_bias)


def test_dino_attn_wins_when_both_attention_forms_are_passed(layout, inputs):
    """Verify raw attention takes precedence when both attention forms are provided."""
    model = wake_up(VelocityTransformer(layout))
    n = layout.num_tokens
    with torch.no_grad():
        for block in model.blocks:
            block.dino_gate.fill_(0.3)
        both = _call(model, inputs, attn_bias=torch.zeros(B, 1, n, n))
        assert torch.equal(both, _call(model, inputs))


def test_detach_dino_attn_keeps_gradient_out_of_the_bias(layout, inputs):
    for detach, expect_grad in ((True, False), (False, True)):
        model = wake_up(VelocityTransformer(layout, detach_dino_attn=detach))
        with torch.no_grad():
            for block in model.blocks:
                block.dino_gate.fill_(0.4)
        attn = inputs["dino_attn"].clone().requires_grad_(True)
        _call(model, inputs, dino_attn=attn).square().mean().backward()
        assert (attn.grad is not None) is expect_grad


def test_each_conditioning_stream_reaches_the_output(layout, inputs):
    model = wake_up(VelocityTransformer(layout))
    base = _call(model, inputs, dino_attn=None)
    for key in ("z_t", "t", "dino_features", "thumb"):
        changed = torch.randn_like(inputs[key]) if key != "t" else 1.0 - inputs["t"]
        out = _call(model, inputs, dino_attn=None, **{key: changed})
        assert not torch.allclose(base, out), f"{key} does not affect the output"


def test_the_thumbnail_is_optional(layout, inputs):
    model = wake_up(VelocityTransformer(layout))
    assert _call(model, inputs, thumb=None).shape == inputs["z_t"].shape


def test_the_thumbnail_is_added_per_token_in_row_major_order(layout, inputs):
    """Verify thumbnail spatial coordinates correspond to token ordering."""
    model = wake_up(VelocityTransformer(layout))
    gy, gx = layout.grid
    perm = torch.randperm(layout.num_tokens, generator=torch.Generator().manual_seed(1))
    scrambled = inputs["thumb"].flatten(2)[..., perm].reshape(B, 1, gy, gx)
    with torch.no_grad():
        base = _call(model, inputs, dino_attn=None)
        assert not torch.allclose(base, _call(model, inputs, dino_attn=None, thumb=scrambled))


def test_from_config_reads_the_velocity_section_of_flow_yaml(layout):
    cfg = load_configs()
    model = VelocityTransformer.from_config(cfg["velocity"], layout)
    assert len(model.blocks) == cfg["velocity"]["num_blocks"]
    assert model.hidden_dim == cfg["velocity"]["hidden_dim"]


# --- the colour head ----------------------------------------------------------------------


def test_color_head_gives_one_logit_per_bin_per_patch():
    from dawn.flow.color_bins import NUM_BINS
    from dawn.models.color_head import ColorHead

    head = ColorHead()
    logits = head(torch.randn(B, 20, DINO))
    assert logits.shape == (B, 20, NUM_BINS) == (B, 20, 261)


def test_color_head_trains_the_tokens_it_reads():
    from dawn.models.color_head import ColorHead

    tokens = torch.randn(B, 5, DINO, requires_grad=True)
    ColorHead()(tokens).square().mean().backward()
    assert tokens.grad is not None and tokens.grad.abs().sum() > 0
