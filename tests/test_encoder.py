"""The DINO encoder's contract: shapes, where LoRA landed, and the attention it hands on.

Nothing here downloads weights. The backbone is randomly initialized and two blocks deep;
every shape, the [CLS, 4 registers, patches] layout and the peft target names are the real
ones, which is what the rest of the pipeline depends on.
"""

import pytest

torch = pytest.importorskip("torch")

from dawn.models.encoder import ATTN_FLOOR, DINOEncoder, attention_to_bias  # noqa: E402
from tests.flow_helpers import tiny_encoder  # noqa: E402

BACKENDS = ["timm", "transformers"]


@pytest.fixture(scope="module", params=BACKENDS)
def encoder(request):
    return tiny_encoder(request.param).eval()


@pytest.fixture(scope="module")
def image():
    generator = torch.Generator().manual_seed(3)
    return torch.rand(2, 3, 256, 256, generator=generator) * 0.2


def test_forward_shapes_match_the_conventions(encoder, image):
    out = encoder(image, return_attn=True)
    assert out.patch_tokens.shape == (2, 256, 384)
    assert out.cls_token.shape == (2, 384)
    assert out.patch_attn.shape == (2, 256, 256)
    assert out.patch_tokens.dtype == torch.float32


def test_attention_is_only_computed_when_asked_for(encoder, image):
    assert encoder(image).patch_attn is None


def test_attention_rows_sum_to_one_before_slicing(encoder, image):
    """The raw softmax rows cover all 261 tokens; the patch block is renormalized after."""
    full = encoder.last_layer_attention(image)
    assert full.shape == (2, 6, 261, 261)
    assert torch.allclose(full.sum(dim=-1), torch.ones(2, 6, 261), atol=1e-5)

    patch = encoder(image, return_attn=True).patch_attn
    assert torch.allclose(patch.sum(dim=-1), torch.ones(2, 256), atol=1e-5)
    assert (patch >= 0).all()


def test_patch_attention_is_the_head_average_of_the_patch_block(encoder, image):
    full = encoder.last_layer_attention(image)
    expected = full[:, :, 5:, 5:].mean(dim=1)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert torch.allclose(encoder(image, return_attn=True).patch_attn, expected, atol=1e-6)


def test_attention_is_detached_from_the_encoder(encoder, image):
    assert encoder(image, return_attn=True).patch_attn.requires_grad is False


def test_lora_on_and_off_give_identical_tokens_at_init(encoder, image):
    """peft initializes B at zero, so the adapter is a bit-exact no-op before any step."""
    with torch.no_grad():
        adapted = encoder(image, return_attn=True)
        frozen = encoder.forward_frozen(image, return_attn=True)
    assert torch.equal(adapted.patch_tokens, frozen.patch_tokens)
    assert torch.equal(adapted.cls_token, frozen.cls_token)


def test_a_nonzero_adapter_changes_the_tokens(encoder, image):
    """And the same comparison must fail once B is not zero, or the test above proves nothing."""
    target = next(
        param for name, param in encoder.named_parameters() if "lora_B" in name
    )
    with torch.no_grad():
        target.add_(0.05)
        try:
            adapted = encoder(image).patch_tokens
            frozen = encoder.forward_frozen(image).patch_tokens
        finally:
            target.sub_(0.05)
    assert not torch.allclose(adapted, frozen)


def test_frozen_features_carry_no_gradient(encoder, image):
    assert encoder.forward_frozen(image).patch_tokens.requires_grad is False


def test_encode_bundles_the_four_conditioning_tensors(encoder, image):
    bundle = encoder.encode(image)
    assert set(bundle) == {"patch_tokens", "patch_map", "cls", "thumb", "attn_bias"}
    assert bundle["patch_map"].shape == (2, 384, 16, 16)
    assert bundle["cls"].shape == (2, 384)
    assert bundle["thumb"].shape == (2, 3, 16, 16)
    assert bundle["attn_bias"].shape == (2, 1, 256, 256)


def test_patch_map_is_the_row_major_reshape_of_the_tokens(encoder, image):
    """Token n, DINO patch n and grid cell n must be the same region (conventions 2.3)."""
    bundle = encoder.encode(image, return_attn=False)
    tokens, patch_map = bundle["patch_tokens"], bundle["patch_map"]
    for y, x in ((0, 0), (0, 15), (7, 3), (15, 15)):
        assert torch.equal(patch_map[:, :, y, x], tokens[:, y * 16 + x])


def test_thumbnail_is_the_log_of_the_area_average(encoder):
    from dawn.data.resample import area_downsample

    image = torch.rand(2, 3, 256, 256)
    expected = torch.log(area_downsample(image, (16, 16)) + 1e-3)
    assert torch.allclose(encoder.thumbnail(image), expected, atol=1e-6)


def test_attention_bias_is_the_log_of_the_attention(encoder, image):
    patch_attn = encoder(image, return_attn=True).patch_attn
    bias = attention_to_bias(patch_attn)
    assert bias.shape == (2, 1, 256, 256)
    assert torch.allclose(bias[:, 0], torch.log(patch_attn + ATTN_FLOOR), atol=1e-6)
    assert torch.isfinite(bias).all()


def test_the_bias_of_zero_attention_is_finite():
    """A zero-attention entry must not become -inf: a zero gate would then give NaN."""
    bias = attention_to_bias(torch.zeros(1, 4, 4))
    assert torch.isfinite(bias).all()
    assert torch.allclose(bias, torch.full_like(bias, float(torch.log(torch.tensor(ATTN_FLOOR)))))
    assert torch.equal(0.0 * bias, torch.zeros_like(bias))


def test_the_input_is_srgb_encoded_not_linear(encoder):
    """DINO is pretrained on sRGB; feeding it linear values would be a silent 2.2-gamma bug."""
    from dawn.data.color import linear_to_srgb

    image = torch.rand(1, 3, 256, 256)
    processed = encoder._preprocess(image)
    expected = (linear_to_srgb(image) - encoder.imagenet_mean) / encoder.imagenet_std
    assert torch.allclose(processed, expected, atol=1e-6)
    assert not torch.allclose(
        processed, (image - encoder.imagenet_mean) / encoder.imagenet_std, atol=1e-3
    )


def test_the_encoder_rejects_the_wrong_input_size(encoder):
    with pytest.raises(ValueError, match="256x256"):
        encoder(torch.rand(1, 3, 224, 224))


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
def test_from_config_reads_the_encoder_section_of_flow_yaml(backend, image):
    """Guards the config key names: a rename here is otherwise only caught on a GPU run."""
    import omegaconf

    pytest.importorskip("peft")
    pytest.importorskip(backend)
    cfg = omegaconf.OmegaConf.to_container(omegaconf.OmegaConf.load("configs/flow.yaml"))
    cfg = cfg["encoder"] | {
        "backend": backend,
        "pretrained": False,
        "backbone_kwargs": {"depth": 2} if backend == "timm" else {"num_hidden_layers": 2},
    }
    encoder = DINOEncoder.from_config(cfg, detach_attn=False).eval()
    assert encoder.input_size == cfg["input_size"]
    assert encoder.num_prefix_tokens == 1 + cfg["num_register_tokens"]
    assert encoder.detach_attn is False
    assert encoder(image, return_attn=True).patch_tokens.shape == (2, 256, 384)
    assert len(DINOEncoder.lora_target_names(encoder.backbone)) >= 2


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_checkpointing_still_delivers_lora_gradients(backend, image):
    """Reentrant checkpointing would drop them: the block inputs do not require grad."""
    encoder = tiny_encoder(backend, grad_checkpointing=True)
    encoder.train()
    encoder(image).patch_tokens.square().mean().backward()
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert grads and any(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_rope_position_augmentation_is_off(encoder, image):
    """Two forwards of one image must agree, in train mode too, or L_align targets drift."""
    encoder.train()
    try:
        with torch.no_grad():
            first, second = encoder(image).patch_tokens, encoder(image).patch_tokens
    finally:
        encoder.eval()
    assert torch.equal(first, second)
