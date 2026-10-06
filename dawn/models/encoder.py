"""DINOv3 ViT-S+/16 encoder with peft LoRA adapters, behind one interface for two backends.

Backends (config: encoder.backend):
    timm          timm.create_model(model_id); needs timm >= 1.0.20.
    transformers  AutoModel.from_pretrained(model_id); needs the `transformers` extra and
                  approved access to the gated facebook/dinov3-* repo.

Both are wrapped so `DINOEncoder.forward` returns the same `EncoderOutput`; nothing outside
this module knows which backend is in use.

LoRA comes from peft only (no hand-rolled LoRA layer). That also gives the L_align target for
free: `disable_adapter()` runs the same network with the adapters off, i.e. the frozen
pretrained features (see `forward_frozen`).

Backend libraries are imported lazily inside the loaders so this module imports without them.

What the flow stage actually consumes is the dict from `encode`, which bundles the four
conditioning tensors of docs/conventions.md section 7:

    patch_map   (B, 384, gy, gx)  patch tokens as a feature map, row-major like the grid cells
    cls         (B, 384)          the CLS token, the noise head's input
    attn_bias   (B, 1, N, N)      log(A + 1e-6) of the last layer's patch-to-patch attention
    thumb       (B, 3, gy, gx)    log(area_downsample(night_linear) + 1e-3)

RoPE position augmentation (random shift / jitter / rescale) is a pretraining augmentation and
is switched OFF here on both backends. It would otherwise make two forward passes of the same
image disagree, which would turn the L_align target into noise (timm ships it off already; the
transformers config defaults to `pos_embed_rescale = 2.0`).
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Mapping, NamedTuple

import torch
import torch.nn as nn

from dawn.data.color import linear_to_srgb
from dawn.data.resample import area_downsample

logger = logging.getLogger(__name__)

Backend = Literal["timm", "transformers"]

# DINO is pretrained on ImageNet-normalized sRGB (timm's pretrained_cfg for the DINOv3 models).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ATTN_FLOOR = 1e-6  # attn_bias = log(attention + ATTN_FLOOR)
THUMB_FLOOR = 1e-3  # thumb = log(area_downsample(linear) + THUMB_FLOOR)

# Architecture of DINOv3 ViT-S+/16, for the `pretrained=False` path where no hub config is
# downloaded. Matches timm's vit_small_plus_patch16_dinov3 (28.7M parameters).
VITS_PLUS = {
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_hidden_layers": 12,
    "num_attention_heads": 6,
    "num_register_tokens": 4,
    "patch_size": 16,
    "use_gated_mlp": True,
    "layerscale_value": 1.0,
    "rope_theta": 100.0,
}


class EncoderOutput(NamedTuple):
    """Backend-independent encoder result.

    N = (H / 16) * (W / 16) patches in row-major (y, x) order; C = 384 for ViT-S+.
    CLS and register tokens are stripped from the patch outputs (layout: [CLS, 4 registers,
    patches]).
    """

    patch_tokens: torch.Tensor  # (B, N, C)
    cls_token: torch.Tensor  # (B, C)
    patch_attn: torch.Tensor | None  # (B, N, N) last layer, head-averaged; only if requested


class DINOEncoder(nn.Module):
    """DINOv3 ViT-S+/16 with LoRA on the attention qkv and output projections.

    The pretrained weights are frozen; only the LoRA parameters train (peft freezes the rest).
    """

    def __init__(
        self,
        backend: Backend = "timm",
        model_id: str = "vit_small_plus_patch16_dinov3.lvd1689m",
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        lora_targets: str | list[str] | None = None,
        pretrained: bool = True,
        input_size: int = 256,
        patch_size: int = 16,
        num_register_tokens: int = 4,
        detach_attn: bool = True,
        grad_checkpointing: bool = False,
        backbone_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.model_id = model_id
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_prefix_tokens = 1 + num_register_tokens  # CLS + registers
        self.detach_attn = detach_attn
        self._backbone_kwargs = dict(backbone_kwargs or {})

        backbone = self._load_backbone(backend, model_id, pretrained)
        self.embed_dim = self._infer_embed_dim(backbone)
        if grad_checkpointing:
            self._enable_grad_checkpointing(backbone)
        self.backbone: nn.Module = self._apply_lora(
            backbone, lora_rank, lora_alpha, lora_dropout, lora_targets
        )

        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

        logger.info(
            "DINOEncoder(%s, %s): %d trainable / %d total parameters (LoRA rank %d, alpha %g)",
            backend,
            model_id,
            self.num_trainable_params(),
            sum(p.numel() for p in self.parameters()),
            lora_rank,
            lora_alpha,
        )

    @classmethod
    def from_config(cls, cfg: Mapping, detach_attn: bool = True) -> "DINOEncoder":
        """Build from the `encoder:` section of configs/flow.yaml.

        `detach_attn` belongs to the velocity net's `velocity.detach_dino_attn`: the attention
        bias is produced here, so it is also cut here.
        """
        backend = cfg["backend"]
        targets = cfg["lora_targets"][backend]
        return cls(
            detach_attn=detach_attn,
            backend=backend,
            model_id=cfg["model_ids"][backend],
            lora_rank=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            lora_targets=targets if isinstance(targets, str) else list(targets),
            pretrained=cfg.get("pretrained", True),
            input_size=cfg.get("input_size", 256),
            patch_size=cfg.get("patch_size", 16),
            num_register_tokens=cfg.get("num_register_tokens", 4),
            grad_checkpointing=cfg.get("grad_checkpointing", False),
            backbone_kwargs=cfg.get("backbone_kwargs"),
        )

    # --- construction ---------------------------------------------------------------------

    def _load_backbone(self, backend: Backend, model_id: str, pretrained: bool) -> nn.Module:
        """Load the raw (not yet LoRA-wrapped) backbone for the chosen backend."""
        if backend == "timm":
            return self._load_timm(model_id, pretrained)
        if backend == "transformers":
            return self._load_transformers(model_id, pretrained)
        raise ValueError(f"Unknown encoder backend {backend!r}; expected 'timm' or 'transformers'.")

    def _load_timm(self, model_id: str, pretrained: bool) -> nn.Module:
        """timm.create_model(model_id, pretrained=pretrained), 256x256 input, no classifier."""
        import timm

        model = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=0,
            img_size=self.input_size,
            **self._backbone_kwargs,
        )
        prefix = getattr(model, "num_prefix_tokens", None)
        if prefix is not None and prefix != self.num_prefix_tokens:
            raise ValueError(
                f"{model_id} has {prefix} prefix tokens, config says {self.num_prefix_tokens} "
                "(1 CLS + encoder.num_register_tokens)"
            )
        # timm's RotaryEmbeddingDinoV3 ships with the coordinate augmentation off; make sure.
        rope = getattr(model, "rope", None)
        for name in ("shift_coords", "jitter_coords", "rescale_coords"):
            if getattr(rope, name, None) is not None:
                setattr(rope, name, None)
                rope.aug_active = False
        return model

    def _load_transformers(self, model_id: str, pretrained: bool) -> nn.Module:
        """AutoModel.from_pretrained(model_id); attention maps need the eager attention path."""
        from transformers import AutoModel
        from transformers.models.dinov3_vit import DINOv3ViTConfig, DINOv3ViTModel

        no_rope_aug = {"pos_embed_shift": None, "pos_embed_jitter": None, "pos_embed_rescale": None}
        overrides = {**no_rope_aug, **self._backbone_kwargs}
        if pretrained:
            model = AutoModel.from_pretrained(
                model_id, dtype=torch.float32, attn_implementation="sdpa", **overrides
            )
        else:
            config = DINOv3ViTConfig(**{**VITS_PLUS, "image_size": self.input_size, **overrides})
            config._attn_implementation = "sdpa"
            model = DINOv3ViTModel(config)
        registers = getattr(model.config, "num_register_tokens", 0)
        if 1 + registers != self.num_prefix_tokens:
            raise ValueError(
                f"{model_id} has {registers} register tokens, config says "
                f"{self.num_prefix_tokens - 1} (encoder.num_register_tokens)"
            )
        # Give the last block's attention its own config object so it can switch to the eager
        # path (the only one that materializes attention weights) without slowing the other 11.
        last = self._transformers_layers(model)[-1].attention
        last.config = copy.deepcopy(last.config)
        return model

    def _apply_lora(
        self,
        backbone: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
        targets: str | list[str] | None,
    ) -> nn.Module:
        """Wrap the backbone with peft: get_peft_model(backbone, LoraConfig(...)).

        `targets` is peft's target_modules: a str is a regex (full match), a list matches
        module-name suffixes. They must hit only the attention qkv / output projections of
        every block, NOT patch_embed.proj. peft freezes all non-LoRA parameters.
        """
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            bias="none",
            init_lora_weights=True,  # A ~ Kaiming, B = 0: the adapter starts as a no-op
        )
        model = get_peft_model(backbone, config)
        adapted = self.lora_target_names(model)
        if not adapted:
            raise ValueError(f"peft matched no module with target_modules={targets!r}")
        bad = [name for name in adapted if "patch_embed" in name or "embeddings" in name]
        if bad:
            raise ValueError(f"LoRA must stay off the patch embedding, but matched {bad}")
        return model

    @staticmethod
    def lora_target_names(peft_model: nn.Module) -> list[str]:
        """Names of the modules peft actually wrapped, without the peft prefixes."""
        names = []
        for name, _ in peft_model.named_modules():
            if name.endswith(".lora_A"):
                names.append(name.removeprefix("base_model.model.").removesuffix(".lora_A"))
        return names

    def _enable_grad_checkpointing(self, backbone: nn.Module) -> None:
        """Recompute block activations in the backward pass (16 GB card, batch 32).

        Non-reentrant checkpointing only. The frozen patch embedding produces an input that
        does not require grad, and reentrant checkpointing skips the backward of a block whose
        inputs are all leaf-free, which would silently drop every LoRA gradient.
        """
        if self.backend == "timm":
            backbone.set_grad_checkpointing(True)  # timm defaults to use_reentrant=False
        else:
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _infer_embed_dim(self, backbone: nn.Module) -> int:
        if self.backend == "timm":
            return int(backbone.embed_dim)
        return int(backbone.config.hidden_size)

    @staticmethod
    def _transformers_layers(backbone: nn.Module) -> nn.ModuleList:
        return backbone.model.layer

    # --- forward --------------------------------------------------------------------------

    @property
    def patch_grid(self) -> tuple[int, int]:
        """`(gy, gx)` patch tokens for the configured input size."""
        side = self.input_size // self.patch_size
        return side, side

    def _preprocess(self, image_linear: torch.Tensor) -> torch.Tensor:
        """Linear RGB -> sRGB -> ImageNet mean/std, the only place DINO's input is built."""
        if image_linear.ndim != 4 or image_linear.shape[1] != 3:
            raise ValueError(f"image must be (B, 3, H, W), got {tuple(image_linear.shape)}")
        if image_linear.shape[2:] != (self.input_size, self.input_size):
            raise ValueError(
                f"encoder input must be {self.input_size}x{self.input_size}, got "
                f"{tuple(image_linear.shape[2:])}: area-downsample it first"
            )
        srgb = linear_to_srgb(image_linear.clamp(0.0, 1.0))
        return (srgb - self.imagenet_mean.to(srgb.dtype)) / self.imagenet_std.to(srgb.dtype)

    def _base_model(self) -> nn.Module:
        """The wrapped backbone itself (LoRA layers are substituted in place inside it)."""
        return self.backbone.get_base_model()

    def _tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """All tokens of the last layer, `(B, prefix + N, C)`."""
        model = self._base_model()
        if self.backend == "timm":
            return model.forward_features(pixel_values)
        return model(pixel_values=pixel_values).last_hidden_state

    @contextmanager
    def _capture_attention(self) -> Iterator[dict[str, torch.Tensor]]:
        """Make the last block compute explicit softmax attention, and capture it.

        Both backends default to a fused kernel that never materializes the attention matrix.
        This switches the LAST block only (the other 11 keep the fast path) and restores it
        afterwards, so `return_attn=False` costs nothing.
        """
        captured: dict[str, torch.Tensor] = {}
        model = self._base_model()
        if self.backend == "timm":
            attn = model.blocks[-1].attn
            previous, attn.fused_attn = attn.fused_attn, False

            def pre_hook(_module, args):  # args[0] is the softmaxed attention
                captured["attn"] = args[0]

            handle = attn.attn_drop.register_forward_pre_hook(pre_hook)
            try:
                yield captured
            finally:
                handle.remove()
                attn.fused_attn = previous
        else:
            attn = self._transformers_layers(model)[-1].attention
            previous = attn.config._attn_implementation
            attn.config._attn_implementation = "eager"

            def hook(_module, _args, output):  # (attn_output, attn_weights)
                captured["attn"] = output[1]

            handle = attn.register_forward_hook(hook)
            try:
                yield captured
            finally:
                handle.remove()
                attn.config._attn_implementation = previous

    def _patch_attention(self, full_attn: torch.Tensor) -> torch.Tensor:
        """`(B, heads, T, T)` -> `(B, N, N)`: drop the prefix, average heads, renormalize rows.

        Dropping the CLS and register columns takes mass out of every row, so the rows are
        divided by what is left; the result is a proper distribution over patches again.
        Detached when `detach_attn` is set: the bias conditions the velocity net, it is not a
        gradient path into the encoder.
        """
        if full_attn is None:
            raise RuntimeError("the attention hook captured nothing; the backend changed shape")
        prefix = self.num_prefix_tokens
        attn = full_attn[:, :, prefix:, prefix:].mean(dim=1).to(torch.float32)
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny)
        return attn.detach() if self.detach_attn else attn

    def forward(
        self,
        image_linear: torch.Tensor,
        return_attn: bool = False,
    ) -> EncoderOutput:
        """Extract patch features (and optionally attention) from a linear-RGB image.

        Converts to sRGB and applies ImageNet mean/std internally, because DINO is pretrained
        on sRGB. Callers pass linear RGB like everywhere else in the pipeline
        (docs/conventions.md section 1).

        Args:
            image_linear: (B, 3, 256, 256) linear RGB in [0, 1], already resized by the caller.
            return_attn: Also return the last-layer patch-to-patch attention, head-averaged,
                with CLS and register rows/columns stripped (needs an eager attention path).

        Returns:
            EncoderOutput with patch_tokens (B, 256, 384), cls_token (B, 384) and
            patch_attn (B, 256, 256) or None.
        """
        pixel_values = self._preprocess(image_linear)
        if return_attn:
            with self._capture_attention() as captured:
                tokens = self._tokens(pixel_values)
            patch_attn = self._patch_attention(captured.get("attn"))
        else:
            tokens = self._tokens(pixel_values)
            patch_attn = None
        prefix = self.num_prefix_tokens
        return EncoderOutput(tokens[:, prefix:], tokens[:, 0], patch_attn)

    def forward_frozen(
        self,
        image_linear: torch.Tensor,
        return_attn: bool = False,
    ) -> EncoderOutput:
        """Same network with the LoRA adapters disabled: the pretrained features.

        This is the L_align target. No gradients.
        """
        with torch.no_grad(), self.backbone.disable_adapter():
            return self.forward(image_linear, return_attn)

    def encode(
        self,
        image_linear: torch.Tensor,
        return_attn: bool = True,
    ) -> dict[str, torch.Tensor]:
        """The conditioning bundle the flow stage consumes (module docstring).

        Args:
            image_linear: (B, 3, 256, 256) linear RGB night image in [0, 1].
            return_attn: Compute `attn_bias`. `False` leaves it out of the dict.

        Returns:
            dict with 'patch_tokens' (B, N, C), 'patch_map' (B, C, gy, gx), 'cls' (B, C),
            'thumb' (B, 3, gy, gx) and, if asked for, 'attn_bias' (B, 1, N, N).
        """
        out = self.forward(image_linear, return_attn=return_attn)
        grid_h, grid_w = self.patch_grid
        batch, num_patches, channels = out.patch_tokens.shape
        if num_patches != grid_h * grid_w:
            raise RuntimeError(f"expected {grid_h * grid_w} patch tokens, got {num_patches}")
        bundle = {
            "patch_tokens": out.patch_tokens,
            "patch_map": out.patch_tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w),
            "cls": out.cls_token,
            "thumb": self.thumbnail(image_linear),
        }
        if out.patch_attn is not None:
            bundle["attn_bias"] = attention_to_bias(out.patch_attn)
        return bundle

    def thumbnail(self, image_linear: torch.Tensor) -> torch.Tensor:
        """`log(area_downsample(image, patch_grid) + 1e-3)`, `(B, 3, gy, gx)`.

        Area averaging is the project's only downsample (docs/conventions.md section 4a); at
        256 -> 16 it is exactly a 16x16 average pool. The log is what makes a lamp and a dark
        road comparable numbers. Autocast is switched off: averaging a dark night frame in
        bf16 loses the small values the log is there to resolve.
        """
        linear = image_linear.to(torch.float32).clamp(0.0, 1.0)
        with torch.autocast(device_type=linear.device.type, enabled=False):
            small = area_downsample(linear, self.patch_grid)
        return torch.log(small + THUMB_FLOOR)

    def last_layer_attention(self, image_linear: torch.Tensor) -> torch.Tensor:
        """Raw per-head attention of the last block, `(B, heads, T, T)`, `T = prefix + N`.

        Rows sum to 1 here; `forward(return_attn=True)` is this, sliced to patches, averaged
        over heads and renormalized. Exposed for tests and for attention visualizations.
        """
        pixel_values = self._preprocess(image_linear)
        with self._capture_attention() as captured:
            self._tokens(pixel_values)
        return captured["attn"]

    def num_trainable_params(self) -> int:
        """Return count of trainable parameters (~0.2M: rank 8 on qkv + proj, 12 blocks)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def attention_to_bias(patch_attn: torch.Tensor, floor: float = ATTN_FLOOR) -> torch.Tensor:
    """`(B, N, N)` attention -> `(B, 1, N, N)` additive logit bias `log(A + floor)`.

    The leading 1 is the head axis: the same bias is added to every head of every velocity
    block, scaled by that block's learned gate. The floor keeps zero attention finite
    (log 1e-6 = -13.8), so a zero gate multiplies a finite number and cannot produce NaNs.
    """
    if patch_attn.ndim != 3:
        raise ValueError(f"patch_attn must be (B, N, N), got {tuple(patch_attn.shape)}")
    return torch.log(patch_attn.to(torch.float32) + floor).unsqueeze(1)
