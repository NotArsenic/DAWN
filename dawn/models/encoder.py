"""DINOv3 ViT-S+/16 encoder with LoRA adapters for thermal conditioning."""

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ATTN_FLOOR = 1e-6  # attn_bias = log(attention + ATTN_FLOOR)

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
    """Container for encoder outputs."""

    patch_tokens: torch.Tensor  # (B, N, C)
    cls_token: torch.Tensor  # (B, C)
    patch_attn: torch.Tensor | None  # (B, N, N)


class DINOEncoder(nn.Module):
    """DINOv3 ViT-S+/16 encoder with parameter-efficient fine-tuning via LoRA."""

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
        """Instantiate encoder from configuration mapping."""
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

    # construction
    def _load_backbone(self, backend: Backend, model_id: str, pretrained: bool) -> nn.Module:
        """Load vision backbone for the specified backend."""
        if backend == "timm":
            return self._load_timm(model_id, pretrained)
        if backend == "transformers":
            return self._load_transformers(model_id, pretrained)
        raise ValueError(f"Unknown encoder backend {backend!r}; expected 'timm' or 'transformers'.")

    def _load_timm(self, model_id: str, pretrained: bool) -> nn.Module:
        """Load and configure timm backbone."""
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
        # Compute explicit softmax attention in last block for hook capture.
        model.blocks[-1].attn.fused_attn = False
        # Disable coordinate augmentation in rotary embeddings.
        rope = getattr(model, "rope", None)
        for name in ("shift_coords", "jitter_coords", "rescale_coords"):
            if getattr(rope, name, None) is not None:
                setattr(rope, name, None)
                rope.aug_active = False
        return model

    def _load_transformers(self, model_id: str, pretrained: bool) -> nn.Module:
        """Load and configure transformers backbone."""
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
        # Use eager attention on the last block to materialize attention weights.
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
        """Attach LoRA adapters to backbone attention projection layers."""
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=targets,
            bias="none",
            init_lora_weights=True,
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
        """Return names of adapted modules without adapter prefixes."""
        names = []
        for name, _ in peft_model.named_modules():
            if name.endswith(".lora_A"):
                names.append(name.removeprefix("base_model.model.").removesuffix(".lora_A"))
        return names

    def _enable_grad_checkpointing(self, backbone: nn.Module) -> None:
        """Enable non-reentrant gradient checkpointing on backbone."""
        if self.backend == "timm":
            backbone.set_grad_checkpointing(True)
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

    # forward
    @property
    def patch_grid(self) -> tuple[int, int]:
        """Return (grid_h, grid_w) patch token counts for input size."""
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
        """Preprocess thermal tensor by repeating channels and normalizing."""
        self._check_size(thermal)
        return self._normalize(thermal.clamp(0.0, 1.0).expand(-1, 3, -1, -1))

    def _preprocess_rgb(self, image_linear: torch.Tensor) -> torch.Tensor:
        """Convert linear RGB to sRGB and normalize."""
        self._check_size(image_linear)
        return self._normalize(linear_to_srgb(image_linear.clamp(0.0, 1.0)))

    def _base_model(self) -> nn.Module:
        """Return underlying base model unwrapped from LoRA."""
        return self.backbone.get_base_model()

    def _tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract all tokens from final layer, shape (B, prefix + N, C)."""
        model = self._base_model()
        if self.backend == "timm":
            return model.forward_features(pixel_values)
        return model(pixel_values=pixel_values).last_hidden_state

    @contextmanager
    def _capture_attention(self) -> Iterator[dict[str, torch.Tensor]]:
        """Context manager to capture attention weights from final block via forward hook."""
        captured: dict[str, torch.Tensor] = {}
        model = self._base_model()
        if self.backend == "timm":

            def pre_hook(_module, args):
                captured["attn"] = args[0]

            handle = model.blocks[-1].attn.attn_drop.register_forward_pre_hook(pre_hook)
        else:

            def hook(_module, _args, output):
                captured["attn"] = output[1]

            handle = self._transformers_layers(model)[-1].attention.register_forward_hook(hook)
        try:
            yield captured
        finally:
            handle.remove()

    def _patch_attention(self, full_attn: torch.Tensor) -> torch.Tensor:
        """Strip prefix tokens, average across heads, and renormalize patch attention rows."""
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
        """Extract patch features and optional attention from thermal input.

        Args:
            thermal: Thermal tensor of shape (B, 1, H, W).
            return_attn: Whether to return last-layer patch attention.

        Returns:
            EncoderOutput containing patch tokens, CLS token, and optional attention.
        """
        return self._run(self._preprocess_thermal(thermal), return_attn)

    def forward_frozen(
        self,
        image_linear: torch.Tensor,
        return_attn: bool = False,
    ) -> EncoderOutput:
        """Extract features with adapters disabled for alignment targets.

        Args:
            image_linear: Linear RGB image of shape (B, 3, H, W).
            return_attn: Whether to return last-layer patch attention.

        Returns:
            EncoderOutput containing frozen patch tokens, CLS token, and optional attention.
        """
        pixel_values = self._preprocess_rgb(image_linear)
        with torch.no_grad(), self.backbone.disable_adapter():
            return self._run(pixel_values, return_attn)

    def encode(
        self,
        thermal: torch.Tensor,
        return_attn: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Extract conditioning tensor dictionary from thermal input.

        Args:
            thermal: Thermal tensor of shape (B, 1, H, W).
            return_attn: Whether to compute and include attention bias.

        Returns:
            Dictionary containing patch_tokens, patch_map, cls, thumb, and optional attn_bias.
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
        """Downsample thermal image to patch grid resolution via area averaging.

        Args:
            thermal: Thermal tensor of shape (B, 1, H, W).

        Returns:
            Downsampled tensor of shape (B, 1, gy, gx).
        """
        thermal = thermal.to(torch.float32).clamp(0.0, 1.0)
        with torch.autocast(device_type=thermal.device.type, enabled=False):
            return area_downsample(thermal, self.patch_grid)

    def last_layer_attention(self, thermal: torch.Tensor) -> torch.Tensor:
        """Extract raw multi-head attention weights from last transformer block.

        Args:
            thermal: Thermal tensor of shape (B, 1, H, W).

        Returns:
            Attention tensor of shape (B, heads, num_tokens, num_tokens).
        """
        pixel_values = self._preprocess_thermal(thermal)
        with self._capture_attention() as captured:
            self._tokens(pixel_values)
        return captured["attn"]

    def num_trainable_params(self) -> int:
        """Return total count of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def attention_to_bias(patch_attn: torch.Tensor, floor: float = ATTN_FLOOR) -> torch.Tensor:
    """Convert attention weights of shape (B, N, N) to additive logit bias of shape (B, 1, N, N)."""
    return torch.log(patch_attn.to(torch.float32) + floor).unsqueeze(1)


def _as_size(size: tuple[int, int] | list[int] | int) -> tuple[int, int]:
    """Convert size specification to (height, width) tuple."""
    if isinstance(size, int):
        return size, size
    height, width = size
    return int(height), int(width)
