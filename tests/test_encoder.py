"""Tests for DINO encoder feature extraction, token layout, LoRA adaptation, and attention."""

import pytest

torch = pytest.importorskip("torch")

from dawn.data.color import linear_to_srgb  # noqa: E402
from dawn.data.resample import area_downsample  # noqa: E402
from dawn.models.encoder import ATTN_FLOOR, DINOEncoder, attention_to_bias  # noqa: E402
from tests.flow_helpers import TINY_INPUT, load_configs, tiny_encoder  # noqa: E402

BACKENDS = ["timm", "transformers"]
GY, GX = 4, 5
N = GY * GX


@pytest.fixture(scope="module", params=BACKENDS)
def encoder(request):
    return tiny_encoder(request.param).eval()


@pytest.fixture(scope="module")
def thermal():
    generator = torch.Generator().manual_seed(3)
    return torch.rand(2, 1, *TINY_INPUT, generator=generator) * 0.6 + 0.2


@pytest.fixture(scope="module")
def rgb():
    generator = torch.Generator().manual_seed(4)
    return torch.rand(2, 3, *TINY_INPUT, generator=generator) * 0.5


def test_the_patch_grid_is_the_input_size_over_the_patch_size(encoder):
    assert encoder.input_size == TINY_INPUT and encoder.patch_grid == (GY, GX)


def test_forward_shapes_match_the_conventions(encoder, thermal):
    out = encoder(thermal, return_attn=True)
    assert out.patch_tokens.shape == (2, N, 384)
    assert out.cls_token.shape == (2, 384)
    assert out.patch_attn.shape == (2, N, N)
    assert out.patch_tokens.dtype == torch.float32


def test_attention_is_only_computed_when_asked_for(encoder, thermal):
    assert encoder(thermal).patch_attn is None


def test_attention_rows_sum_to_one_before_slicing(encoder, thermal):
    """Verify raw attention softmax rows sum to one and normalized patch block sums to one."""
    full = encoder.last_layer_attention(thermal)
    assert full.shape == (2, 6, 5 + N, 5 + N)
    assert torch.allclose(full.sum(dim=-1), torch.ones(2, 6, 5 + N), atol=1e-5)

    patch = encoder(thermal, return_attn=True).patch_attn
    assert torch.allclose(patch.sum(dim=-1), torch.ones(2, N), atol=1e-5)
    assert (patch >= 0).all()


def test_patch_attention_is_the_head_average_of_the_patch_block(encoder, thermal):
    full = encoder.last_layer_attention(thermal)
    expected = full[:, :, 5:, 5:].mean(dim=1)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(encoder(thermal, return_attn=True).patch_attn, expected, atol=1e-6)


def test_attention_is_detached_from_the_encoder(encoder, thermal):
    assert encoder(thermal, return_attn=True).patch_attn.requires_grad is False


def test_lora_on_and_off_give_identical_tokens_at_init(encoder, thermal):
    """Verify zero-initialized LoRA adapter is an exact identity at initialization."""
    gray = thermal.expand(-1, 3, -1, -1)
    with torch.no_grad():
        adapted = encoder(thermal, return_attn=True)
        frozen_same_pixels = encoder.forward_frozen(gray, return_attn=True)
    with torch.no_grad(), encoder.backbone.disable_adapter():
        off = encoder(thermal, return_attn=True)
    assert torch.equal(adapted.patch_tokens, off.patch_tokens)
    assert torch.equal(adapted.cls_token, off.cls_token)
    assert frozen_same_pixels.patch_tokens.shape == adapted.patch_tokens.shape


def test_a_nonzero_adapter_changes_the_tokens(encoder, thermal):
    """Verify non-zero LoRA parameters alter output features."""
    target = next(param for name, param in encoder.named_parameters() if "lora_B" in name)
    with torch.no_grad():
        target.add_(0.05)
        try:
            adapted = encoder(thermal).patch_tokens
            with encoder.backbone.disable_adapter():
                frozen = encoder(thermal).patch_tokens
        finally:
            target.sub_(0.05)
    assert not torch.allclose(adapted, frozen)


def test_frozen_features_carry_no_gradient_and_take_linear_rgb(encoder, rgb):
    out = encoder.forward_frozen(rgb)
    assert out.patch_tokens.requires_grad is False
    assert out.patch_tokens.shape == (2, N, 384)


def test_encode_bundles_the_conditioning_tensors(encoder, thermal):
    bundle = encoder.encode(thermal)
    assert set(bundle) == {"patch_tokens", "patch_map", "cls", "thumb", "attn_bias"}
    assert bundle["patch_map"].shape == (2, 384, GY, GX)
    assert bundle["cls"].shape == (2, 384)
    assert bundle["thumb"].shape == (2, 1, GY, GX)
    assert bundle["attn_bias"].shape == (2, 1, N, N)
    assert set(encoder.encode(thermal, return_attn=False)) == {
        "patch_tokens",
        "patch_map",
        "cls",
        "thumb",
    }


def test_patch_map_is_the_row_major_reshape_of_the_tokens(encoder, thermal):
    """Verify patch map retains row-major correspondence with output tokens."""
    bundle = encoder.encode(thermal, return_attn=False)
    tokens, patch_map = bundle["patch_tokens"], bundle["patch_map"]
    for y, x in ((0, 0), (0, GX - 1), (2, 3), (GY - 1, GX - 1)):
        assert torch.equal(patch_map[:, :, y, x], tokens[:, y * GX + x])


def test_a_marker_in_the_thermal_lands_in_the_right_row_major_token(encoder):
    """Verify spatial perturbations map to corresponding row-major tokens."""
    base = torch.full((1, 1, *TINY_INPUT), 0.4)
    marked = base.clone()
    marked[..., 16:32, 48:64] = 0.95  # patch row 1, patch column 3  -> token 1 * 5 + 3
    with torch.no_grad():
        delta = (encoder(marked).patch_tokens - encoder(base).patch_tokens).norm(dim=-1)[0]
    assert int(delta.argmax()) == 1 * GX + 3


def test_thumbnail_is_the_plain_area_average_of_the_thermal(encoder, thermal):
    thumb = encoder.thumbnail(thermal)
    assert thumb.shape == (2, 1, GY, GX)
    assert torch.allclose(thumb, area_downsample(thermal, (GY, GX)), atol=1e-6)


def test_attention_bias_is_the_log_of_the_attention(encoder, thermal):
    patch_attn = encoder(thermal, return_attn=True).patch_attn
    bias = attention_to_bias(patch_attn)
    assert bias.shape == (2, 1, N, N)
    assert torch.allclose(bias[:, 0], torch.log(patch_attn + ATTN_FLOOR), atol=1e-6)
    assert torch.isfinite(bias).all()


def test_the_bias_of_zero_attention_is_finite():
    """Verify zero attention values produce finite log-bias without NaNs."""
    bias = attention_to_bias(torch.zeros(1, 4, 4))
    assert torch.isfinite(bias).all()
    assert torch.allclose(bias, torch.full_like(bias, float(torch.log(torch.tensor(ATTN_FLOOR)))))
    assert torch.equal(0.0 * bias, torch.zeros_like(bias))


def test_the_thermal_path_does_not_use_srgb(encoder):
    """Verify thermal preprocessing applies ImageNet normalization without sRGB transfer."""
    thermal = torch.rand(1, 1, *TINY_INPUT)
    processed = encoder._preprocess_thermal(thermal)
    expected = (thermal.expand(-1, 3, -1, -1) - encoder.imagenet_mean) / encoder.imagenet_std
    assert torch.allclose(processed, expected, atol=1e-6)
    assert not torch.allclose(
        processed,
        (linear_to_srgb(thermal).expand(-1, 3, -1, -1) - encoder.imagenet_mean)
        / encoder.imagenet_std,
        atol=1e-3,
    )


def test_the_rgb_path_is_srgb_encoded_not_linear(encoder):
    """Verify RGB preprocessing applies sRGB conversion prior to normalization."""
    rgb = torch.rand(1, 3, *TINY_INPUT)
    processed = encoder._preprocess_rgb(rgb)
    expected = (linear_to_srgb(rgb) - encoder.imagenet_mean) / encoder.imagenet_std
    assert torch.allclose(processed, expected, atol=1e-6)
    assert not torch.allclose(
        processed, (rgb - encoder.imagenet_mean) / encoder.imagenet_std, atol=1e-3
    )


def test_the_encoder_rejects_a_frame_of_the_wrong_size(encoder):
    """Verify encoder validates expected frame dimensions."""
    with pytest.raises(ValueError, match="64x80"):
        encoder(torch.rand(1, 1, 80, 64))


def test_lora_targets_only_the_attention_projections(encoder):
    names = DINOEncoder.lora_target_names(encoder.backbone)
    assert names, "peft matched nothing"
    assert all("attn" in name or "attention" in name for name in names), names
    assert not any("patch_embed" in name or "mlp" in name for name in names), names


def test_only_lora_parameters_train(encoder):
    trainable = [name for name, p in encoder.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    assert all("lora_" in name for name in trainable), trainable[:4]
    assert encoder.num_trainable_params() == sum(
        p.numel() for p in encoder.parameters() if p.requires_grad
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_from_config_reads_the_encoder_section_of_flow_yaml(backend, thermal):
    """Verify encoder instantiation from configuration dictionary."""
    pytest.importorskip("peft")
    pytest.importorskip(backend)
    cfg = load_configs()["encoder"] | {
        "backend": backend,
        "pretrained": False,
        "input_size": list(TINY_INPUT),
        "backbone_kwargs": {"depth": 2} if backend == "timm" else {"num_hidden_layers": 2},
    }
    encoder = DINOEncoder.from_config(cfg, detach_attn=False).eval()
    assert encoder.input_size == TINY_INPUT
    assert encoder.num_prefix_tokens == 1 + cfg["num_register_tokens"]
    assert encoder.detach_attn is False
    assert encoder(thermal, return_attn=True).patch_tokens.shape == (2, N, 384)
    assert len(DINOEncoder.lora_target_names(encoder.backbone)) >= 2


def test_the_default_input_size_gives_the_plans_1280_tokens():
    """Verify default resolution produces expected patch grid dimensions."""
    cfg = load_configs()["encoder"]
    h, w = cfg["input_size"]
    assert (h // cfg["patch_size"], w // cfg["patch_size"]) == (32, 40)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_checkpointing_still_delivers_lora_gradients(backend, thermal):
    """Verify gradient checkpointing propagates gradients to LoRA parameters."""
    encoder = tiny_encoder(backend, grad_checkpointing=True)
    encoder.train()
    encoder(thermal).patch_tokens.square().mean().backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert grads and any(g is not None and float(g.abs().sum()) > 0 for g in grads)


@pytest.mark.parametrize("backend", BACKENDS)
def test_checkpointed_backward_works_while_the_attention_is_captured(backend, thermal):
    """Verify gradient checkpointing backward pass operates with attention capture enabled."""
    encoder = tiny_encoder(backend, grad_checkpointing=True)
    encoder.train()
    out = encoder(thermal, return_attn=True)
    (out.patch_tokens.square().mean() + out.patch_attn.sum() * 0.0).backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert grads and any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    bundle = tiny_encoder(backend, grad_checkpointing=True).train().encode(thermal)
    bundle["patch_tokens"].square().mean().backward()


def test_the_attention_with_and_without_the_capture_flag_gives_the_same_tokens(encoder, thermal):
    """Verify attention capture flag does not alter output token representations."""
    with torch.no_grad():
        plain = encoder(thermal).patch_tokens
        with_attn = encoder(thermal, return_attn=True).patch_tokens
    assert torch.equal(plain, with_attn)


def test_rope_position_augmentation_is_off(encoder, thermal):
    """Verify deterministic forward passes without stochastic position augmentation."""
    encoder.train()
    try:
        with torch.no_grad():
            first, second = encoder(thermal).patch_tokens, encoder(thermal).patch_tokens
    finally:
        encoder.eval()
    assert torch.equal(first, second)
