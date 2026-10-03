"""The velocity transformer: shapes, the zero-init guarantee and the DINO attention gate.

The three conditioning streams (grid tokens, DINO patch tokens, log thumbnail) are added per
token, so the only thing that can silently go wrong is an ordering mismatch; `test_layout.py`
pins the reshape and these tests pin that the model uses it.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.grid.layout import grid_to_tokens, tokens_to_grid  # noqa: E402
from dawn.models.noise_head import LOG_PRIOR, NoiseHead, log_noise_target  # noqa: E402
from dawn.models.velocity import VelocityTransformer  # noqa: E402
from tests.flow_helpers import wake_up  # noqa: E402

B, K, GY, GX, D = 2, 8, 16, 16, 384


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(11)
    attn = torch.rand(B, GY * GX, GY * GX, generator=generator)
    return {
        "z_t": torch.randn(B, 12, K, GY, GX, generator=generator),
        "t": torch.rand(B, generator=generator),
        "dino_features": torch.randn(B, GY * GX, D, generator=generator),
        "dino_attn": attn / attn.sum(dim=-1, keepdim=True),
        "thumb": torch.randn(B, 3, GY, GX, generator=generator),
    }


def test_output_shape_matches_the_input_grid(inputs):
    model = VelocityTransformer()
    out = model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
                inputs["thumb"])
    assert out.shape == inputs["z_t"].shape


def test_parameter_count_is_about_seven_million():
    model = VelocityTransformer()
    assert 6e6 < model.num_params() < 9e6, model.num_params()


def test_velocity_is_exactly_zero_at_init(inputs):
    """adaLN-Zero plus a zero-initialized output head: the flow starts as the identity ODE."""
    model = VelocityTransformer()
    out = model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
                inputs["thumb"])
    assert torch.equal(out, torch.zeros_like(out))


def test_every_block_starts_with_a_zero_attention_gate():
    model = VelocityTransformer()
    assert model.attn_gates() == {f"gate_{i}": 0.0 for i in range(6)}


def test_with_a_zero_gate_the_attention_bias_changes_nothing(inputs):
    """Two different attention maps must give bit-identical output while every gate is 0."""
    model = wake_up(VelocityTransformer())
    other = torch.rand_like(inputs["dino_attn"])
    other = other / other.sum(dim=-1, keepdim=True)

    with torch.no_grad():
        first = model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
                      inputs["thumb"])
        second = model(inputs["z_t"], inputs["t"], inputs["dino_features"], other,
                       inputs["thumb"])
        none = model(inputs["z_t"], inputs["t"], inputs["dino_features"], None, inputs["thumb"])
    assert torch.equal(first, second)
    # Same arithmetic as passing no mask at all, up to the kernel SDPA picks for each.
    assert torch.allclose(first, none, atol=1e-5)
    assert first.abs().max() > 1e-3, "the model is still a no-op, so this proves nothing"


def test_a_nonzero_gate_makes_the_attention_bias_matter(inputs):
    model = wake_up(VelocityTransformer())
    with torch.no_grad():
        before = model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
                       inputs["thumb"])
        for block in model.blocks:
            block.dino_gate.fill_(0.5)
        after = model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
                      inputs["thumb"])
    assert not torch.allclose(before, after)


def test_the_gate_takes_gradient(inputs):
    model = wake_up(VelocityTransformer())
    out = model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
                inputs["thumb"])
    out.square().mean().backward()
    assert all(block.dino_gate.grad is not None for block in model.blocks)


def test_a_precomputed_bias_and_the_raw_attention_agree(inputs):
    from dawn.models.encoder import attention_to_bias

    model = wake_up(VelocityTransformer())
    with torch.no_grad():
        for block in model.blocks:
            block.dino_gate.fill_(0.3)
        from_attn = model(inputs["z_t"], inputs["t"], inputs["dino_features"],
                          inputs["dino_attn"], inputs["thumb"])
        from_bias = model(inputs["z_t"], inputs["t"], inputs["dino_features"], None,
                          inputs["thumb"], attention_to_bias(inputs["dino_attn"]))
    assert torch.equal(from_attn, from_bias)


def test_passing_both_attention_forms_is_an_error(inputs):
    model = VelocityTransformer()
    with pytest.raises(ValueError, match="not both"):
        model(inputs["z_t"], inputs["t"], inputs["dino_features"], inputs["dino_attn"],
              inputs["thumb"], torch.zeros(B, 1, 256, 256))


def test_detach_dino_attn_keeps_gradient_out_of_the_bias(inputs):
    model = wake_up(VelocityTransformer(detach_dino_attn=True))
    attn = inputs["dino_attn"].clone().requires_grad_(True)
    with torch.no_grad():
        for block in model.blocks:
            block.dino_gate.fill_(0.4)
    model(inputs["z_t"], inputs["t"], inputs["dino_features"], attn,
          inputs["thumb"]).square().mean().backward()
    assert attn.grad is None

    leaky = wake_up(VelocityTransformer(detach_dino_attn=False))
    with torch.no_grad():
        for block in leaky.blocks:
            block.dino_gate.fill_(0.4)
    attn = inputs["dino_attn"].clone().requires_grad_(True)
    leaky(inputs["z_t"], inputs["t"], inputs["dino_features"], attn,
          inputs["thumb"]).square().mean().backward()
    assert attn.grad is not None


def test_each_conditioning_stream_reaches_the_output(inputs):
    model = wake_up(VelocityTransformer())
    base = model(inputs["z_t"], inputs["t"], inputs["dino_features"], None, inputs["thumb"])
    for key in ("z_t", "t", "dino_features", "thumb"):
        changed = dict(inputs)
        changed[key] = torch.randn_like(inputs[key]) if key != "t" else 1.0 - inputs["t"]
        out = model(changed["z_t"], changed["t"], changed["dino_features"], None,
                    changed["thumb"])
        assert not torch.allclose(base, out), f"{key} does not affect the output"


def test_the_thumbnail_is_optional(inputs):
    model = wake_up(VelocityTransformer())
    out = model(inputs["z_t"], inputs["t"], inputs["dino_features"])
    assert out.shape == inputs["z_t"].shape


def test_token_round_trip_is_the_layout_module(inputs):
    """The model must not re-derive the reshape (docs/conventions.md section 8, rule 3)."""
    grid = inputs["z_t"]
    assert torch.equal(tokens_to_grid(grid_to_tokens(grid), GY, GX, K), grid)


@pytest.mark.parametrize(
    "bad",
    [
        {"z_t": torch.randn(B, 11, K, GY, GX)},
        {"z_t": torch.randn(B, 12, 4, GY, GX)},
        {"t": torch.rand(B + 1)},
        {"dino_features": torch.randn(B, 100, D)},
    ],
)
def test_shape_errors_are_reported(inputs, bad):
    model = VelocityTransformer()
    args = {**inputs, **bad}
    with pytest.raises(ValueError):
        model(args["z_t"], args["t"], args["dino_features"], args["dino_attn"], args["thumb"])


# --- noise head -------------------------------------------------------------------------


def test_noise_head_accepts_cls_and_patch_tokens():
    head = NoiseHead()
    assert head(torch.randn(B, D)).shape == (B, 2)
    assert head(torch.randn(B, 256, D)).shape == (B, 2)


def test_noise_head_starts_near_the_prior():
    """Not at zero: a 14-nat bias would eat the whole gradient clip for a few hundred steps."""
    head = NoiseHead()
    with torch.no_grad():
        out = head(torch.randn(64, D))
    assert torch.allclose(out.mean(dim=0), torch.tensor(LOG_PRIOR), atol=0.5)
    assert float((out - torch.tensor(LOG_PRIOR)).pow(2).mean()) < 1.0


def test_log_noise_target_folds_quantization_into_the_read_variance():
    params = {"a": torch.full((2, 3), 0.01), "b": torch.full((2, 3), 1e-5),
              "quant_var": torch.full((2,), 1e-5)}
    target = log_noise_target(params)
    assert target.shape == (2, 2)
    assert torch.allclose(target[:, 0], torch.log(torch.tensor(0.01)))
    assert torch.allclose(target[:, 1], torch.log(torch.tensor(2e-5)))


def test_log_noise_target_accepts_a_single_unbatched_sample():
    target = log_noise_target({"a": torch.full((3,), 0.02), "b": torch.full((3,), 1e-4)})
    assert target.shape == (1, 2)
