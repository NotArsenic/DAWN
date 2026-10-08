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

The input is THERMAL, `(B, 1, H, W)` = code / 255 at `input_size = (H, W)` (640x512 in the plan,
given as (512, 640)): it is copied to three channels and normalized with the ImageNet mean/std.
Thermal is camera codes, not light, so it never goes through `linear_to_srgb` (docs/conventions.md
section 1). The paired RGB goes through `forward_frozen` (adapters disabled) for the L_align
target, and THAT path is linear -> sRGB -> ImageNet mean/std, because DINO was pretrained on sRGB.

What the flow stage actually consumes is the dict from `encode`, which bundles the conditioning
tensors of docs/conventions.md section 7:

    patch_tokens (B, N, 384)      patch tokens, row-major: token n = y * gx + x
    patch_map   (B, 384, gy, gx)  the same tokens as a feature map
    cls         (B, 384)          the CLS token
    attn_bias   (B, 1, N, N)      log(A + 1e-6) of the last layer's patch-to-patch attention
    thumb       (B, 1, gy, gx)    area_downsample(thermal, (gy, gx)): one value per patch

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
# The thermal path uses the same mean/std but skips the sRGB encoding (thermal is not light).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ATTN_FLOOR = 1e-6  # attn_bias = log(attention + ATTN_FLOOR)

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

    N = (H / 16) * (W / 16) patches in row-major (y, x) order (1,280 at 512 x 640); C = 384.
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
        input_size: tuple[int, int] | int = (512, 640),
        patch_size: int = 16,
        num_register_tokens: int = 4,
        detach_attn: bool = True,
        grad_checkpointing: bool = False,
        backbone_kwargs: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.model_id = model_id
        self.input_size = _as_size(input_size)
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
            input_size=cfg.get("input_size", (512, 640)),
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
        """timm.create_model(model_id, ...) at `input_size` (H, W), no classifier head."""
        import timm

        model = timm.create_model(
            model_id,
            pretrained=pretrained,
            num_classes=0,
            img_size=tuple(self.input_size),
            **self._backbone_kwargs,
        )
        prefix = getattr(model, "num_prefix_tokens", None)
        if prefix is not None and prefix != self.num_prefix_tokens:
            raise ValueError(
                f"{model_id} has {prefix} prefix tokens, config says {self.num_prefix_tokens} "
                "(1 CLS + encoder.num_register_tokens)"
            )
        # The LAST block always computes explicit softmax attention (so a hook can capture it).
        # It must not depend on whether this call asks for the attention: under gradient
        # checkpointing the backward pass recomputes the block, and a different attention path
        # there than in the forward raises a CheckpointError. The other 11 blocks stay fused.
        model.blocks[-1].attn.fused_attn = False
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
            config = DINOv3ViTConfig(
                **{**VITS_PLUS, "image_size": max(self.input_size), **overrides}
            )
            config._attn_implementation = "sdpa"
            model = DINOv3ViTModel(config)
        registers = getattr(model.config, "num_register_tokens", 0)
        if 1 + registers != self.num_prefix_tokens:
            raise ValueError(
                f"{model_id} has {registers} register tokens, config says "
                f"{self.num_prefix_tokens - 1} (encoder.num_register_tokens)"
            )
        # Give the last block's attention its own config object and keep it on the eager path (the
        # only one that materializes attention weights) for good, without slowing the other 11:
        # under gradient checkpointing the recompute must take the same path as the forward.
        last = self._transformers_layers(model)[-1].attention
        last.config = copy.deepcopy(last.config)
        last.config._attn_implementation = "eager"
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
        """Recompute block activations in the backward pass (16 GB card, 1,280 tokens, batch 16).

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
        """`(gy, gx)` patch tokens for the configured input size: (32, 40) at 512 x 640."""
        return self.input_size[0] // self.patch_size, self.input_size[1] // self.patch_size

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.imagenet_mean.to(image.dtype)) / self.imagenet_std.to(image.dtype)

    def _check_size(self, image: torch.Tensor) -> None:
        if tuple(image.shape[2:]) != self.input_size:
            h, w = self.input_size
            raise ValueError(
                f"encoder input must be {h}x{w} (H x W), got "
                f"{image.shape[2]}x{image.shape[3]}: area-downsample it first"
            )

    def _preprocess_thermal(self, thermal: torch.Tensor) -> torch.Tensor:
        """Thermal -> three channels -> ImageNet mean/std. No sRGB: thermal is not light."""
        self._check_size(thermal)
        return self._normalize(thermal.clamp(0.0, 1.0).expand(-1, 3, -1, -1))

    def _preprocess_rgb(self, image_linear: torch.Tensor) -> torch.Tensor:
        """Linear RGB -> sRGB -> ImageNet mean/std: the L_align target's input (DINO saw sRGB)."""
        self._check_size(image_linear)
        return self._normalize(linear_to_srgb(image_linear.clamp(0.0, 1.0)))

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
        """Capture the last block's softmax attention with a hook.

        The last block computes explicit attention whether or not it is captured (set at load time:
        a fused kernel never materializes the matrix, and the path must not change between a
        forward and its checkpoint recompute), so asking for the attention only adds the hook.
        """
        captured: dict[str, torch.Tensor] = {}
        model = self._base_model()
        if self.backend == "timm":

            def pre_hook(_module, args):  # args[0] is the softmaxed attention
                captured["attn"] = args[0]

            handle = model.blocks[-1].attn.attn_drop.register_forward_pre_hook(pre_hook)
        else:

            def hook(_module, _args, output):  # (attn_output, attn_weights)
                captured["attn"] = output[1]

            handle = self._transformers_layers(model)[-1].attention.register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

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

    def _run(self, pixel_values: torch.Tensor, return_attn: bool) -> EncoderOutput:
        if return_attn:
            with self._capture_attention() as captured:
                tokens = self._tokens(pixel_values)
            patch_attn = self._patch_attention(captured.get("attn"))
        else:
            tokens = self._tokens(pixel_values)
            patch_attn = None
        prefix = self.num_prefix_tokens
        return EncoderOutput(tokens[:, prefix:], tokens[:, 0], patch_attn)

    def forward(
        self,
        thermal: torch.Tensor,
        return_attn: bool = False,
    ) -> EncoderOutput:
        """Extract patch features (and optionally attention) from a thermal frame.

        Args:
            thermal: (B, 1, H, W) = code / 255 in [0, 1] at `input_size`, already resized by the
                caller (area averaging). Copied to 3 channels, ImageNet mean/std; no sRGB.
            return_attn: Also return the last-layer patch-to-patch attention, head-averaged,
                with CLS and register rows/columns stripped.

        Returns:
            EncoderOutput with patch_tokens (B, N, 384), cls_token (B, 384) and
            patch_attn (B, N, N) or None.
        """
        return self._run(self._preprocess_thermal(thermal), return_attn)

    def forward_frozen(
        self,
        image_linear: torch.Tensor,
        return_attn: bool = False,
    ) -> EncoderOutput:
        """The paired RGB through the same network with the LoRA adapters disabled.

        The pretrained features of the daylight image are the L_align target. DINO is pretrained
        on sRGB, so the linear RGB is converted to sRGB plus ImageNet mean/std here and callers
        never pre-convert (docs/conventions.md section 1). No gradients.

        Args:
            image_linear: (B, 3, H, W) linear RGB in [0, 1] at `input_size`.
        """
        pixel_values = self._preprocess_rgb(image_linear)
        with torch.no_grad(), self.backbone.disable_adapter():
            return self._run(pixel_values, return_attn)

    def encode(
        self,
        thermal: torch.Tensor,
        return_attn: bool = True,
    ) -> dict[str, torch.Tensor]:
        """The conditioning bundle the flow stage consumes (module docstring).

        Args:
            thermal: (B, 1, H, W) thermal in [0, 1] at `input_size`.
            return_attn: Compute `attn_bias`. `False` leaves it out of the dict.

        Returns:
            dict with 'patch_tokens' (B, N, C), 'patch_map' (B, C, gy, gx), 'cls' (B, C),
            'thumb' (B, 1, gy, gx) and, if asked for, 'attn_bias' (B, 1, N, N).
        """
        out = self.forward(thermal, return_attn=return_attn)
        grid_h, grid_w = self.patch_grid
        batch, num_patches, channels = out.patch_tokens.shape
        if num_patches != grid_h * grid_w:
            raise RuntimeError(f"expected {grid_h * grid_w} patch tokens, got {num_patches}")
        bundle = {
            "patch_tokens": out.patch_tokens,
            "patch_map": out.patch_tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w),
            "cls": out.cls_token,
            "thumb": self.thumbnail(thermal),
        }
        if out.patch_attn is not None:
            bundle["attn_bias"] = attention_to_bias(out.patch_attn)
        return bundle

    def thumbnail(self, thermal: torch.Tensor) -> torch.Tensor:
        """`area_downsample(thermal, patch_grid)`, `(B, 1, gy, gx)`: one value per patch.

        Area averaging is the project's only downsample (docs/conventions.md section 4a). One
        channel, no log: thermal is not photon-starved. Autocast is switched off so the average
        is exact.
        """
        thermal = thermal.to(torch.float32).clamp(0.0, 1.0)
        with torch.autocast(device_type=thermal.device.type, enabled=False):
            return area_downsample(thermal, self.patch_grid)

    def last_layer_attention(self, thermal: torch.Tensor) -> torch.Tensor:
        """Raw per-head attention of the last block, `(B, heads, T, T)`, `T = prefix + N`.

        Rows sum to 1 here; `forward(return_attn=True)` is this, sliced to patches, averaged
        over heads and renormalized. Exposed for tests and for attention visualizations.
        """
        pixel_values = self._preprocess_thermal(thermal)
        with self._capture_attention() as captured:
            self._tokens(pixel_values)
        return captured["attn"]

    def num_trainable_params(self) -> int:
        """Return count of trainable parameters (0.22M: rank 8 on qkv + proj, 12 blocks)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def attention_to_bias(patch_attn: torch.Tensor, floor: float = ATTN_FLOOR) -> torch.Tensor:
    """`(B, N, N)` attention -> `(B, 1, N, N)` additive logit bias `log(A + floor)`.

    The leading 1 is the head axis: the same bias is added to every head of every velocity
    block, scaled by that block's learned gate. The floor keeps zero attention finite
    (log 1e-6 = -13.8), so a zero gate multiplies a finite number and cannot produce NaNs.
    """
    return torch.log(patch_attn.to(torch.float32) + floor).unsqueeze(1)


def _as_size(size: tuple[int, int] | list[int] | int) -> tuple[int, int]:
    """`(H, W)` from a pair, or a square from one int."""
    if isinstance(size, int):
        return size, size
    height, width = size
    return int(height), int(width)
