"""Flow matching losses and multi-term training objectives."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from dawn.flow.tokens import TokenLayout

SEGMENT_BALANCES = ("per_value", "equal")
_TINY = torch.finfo(torch.float32).tiny


class FlowLoss(nn.Module):
    """Loss module combining flow matching, rendering, feature alignment, and color classification.

    Args:
        layout: TokenLayout defining token segmentation.
        velocity_weight: Weight for flow matching velocity loss.
        render_weight: Weight for rendered Lab L1 loss.
        align_weight: Weight for DINO feature alignment cosine loss.
        color_weight: Weight for color head cross-entropy loss.
        segment_balance: Strategy for balancing flow loss across segments ('per_value' or 'equal').
        min_patch_fraction: Minimum valid patch fraction threshold for color loss.
        fallback_render_weight: Relative weight for fallback-aligned pairs in render loss.
    """

    def __init__(
        self,
        layout: TokenLayout,
        velocity_weight: float = 1.0,
        render_weight: float = 0.1,
        align_weight: float = 0.5,
        color_weight: float = 0.2,
        segment_balance: str = "per_value",
        min_patch_fraction: float = 0.5,
        fallback_render_weight: float = 0.5,
    ):
        super().__init__()
        if segment_balance not in SEGMENT_BALANCES:
            raise ValueError(
                f"flow.segment_balance must be one of {SEGMENT_BALANCES}, got {segment_balance!r}"
            )
        self.layout = layout
        self.velocity_weight = velocity_weight
        self.render_weight = render_weight
        self.align_weight = align_weight
        self.color_weight = color_weight
        self.segment_balance = segment_balance
        self.min_patch_fraction = min_patch_fraction
        self.fallback_render_weight = fallback_render_weight

    @classmethod
    def from_config(cls, cfg: Mapping, layout: TokenLayout) -> FlowLoss:
        """Instantiate FlowLoss from configuration mapping and TokenLayout."""
        flow = cfg["flow"]
        return cls(
            layout,
            velocity_weight=flow["velocity_weight"],
            render_weight=flow["render_weight"],
            align_weight=flow["align_weight"],
            color_weight=flow["color_weight"],
            segment_balance=flow["segment_balance"],
            min_patch_fraction=cfg["color"]["min_patch_fraction"],
            fallback_render_weight=cfg["pairs"]["fallback_render_weight"],
        )

    def sample_weight(self, fallback: torch.Tensor) -> torch.Tensor:
        """Compute sample-level loss weights based on pair alignment type."""
        ones = torch.ones(fallback.shape, dtype=torch.float32, device=fallback.device)
        return torch.where(fallback.bool(), ones * self.fallback_render_weight, ones)

    def flow(
        self,
        v_pred: torch.Tensor,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Compute weighted MSE flow velocity loss across token segments.

        Args:
            v_pred: Predicted velocity tokens of shape (B, N, D).
            z_0: Source noise tokens of shape (B, N, D).
            z_1: Target data tokens of shape (B, N, D).
            weights: Optional token weights of shape (B, N, D).

        Returns:
            Dictionary containing overall 'flow' loss and per-segment losses.
        """
        if v_pred.shape != z_0.shape or v_pred.shape != z_1.shape:
            raise ValueError(
                f"v_pred {tuple(v_pred.shape)}, z_0 {tuple(z_0.shape)} and z_1 "
                f"{tuple(z_1.shape)} must all be the same (B, N, D)"
            )
        if v_pred.ndim != 3 or v_pred.shape[1:] != (self.layout.num_tokens, self.layout.dim):
            raise ValueError(
                f"tokens must be (B, {self.layout.num_tokens}, {self.layout.dim}) for this "
                f"layout, got {tuple(v_pred.shape)}"
            )
        squared = (v_pred.float() - (z_1.float() - z_0.float())).pow(2)
        if weights is not None:
            squared = squared * weights.to(squared.dtype)
        segments = {name: squared[..., seg].mean() for name, seg in self.layout.segments.items()}
        if self.segment_balance == "per_value":
            total = (
                sum(
                    (seg.stop - seg.start) * segments[name]
                    for name, seg in self.layout.segments.items()
                )
                / self.layout.dim
            )
        else:
            total = torch.stack(list(segments.values())).mean()
        return {"flow": total, **{f"flow_{name}": value for name, value in segments.items()}}

    def render(
        self,
        lab_pred: torch.Tensor,
        lab_target: torch.Tensor,
        pixel_weight: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked L1 loss between predicted and target Lab images.

        Args:
            lab_pred: Predicted Lab image of shape (B, 3, H, W).
            lab_target: Target Lab image of shape (B, 3, H, W).
            pixel_weight: Pixel mask weights of shape (B, 1, H, W).
            sample_weight: Sample weights of shape (B,).

        Returns:
            Scalar render loss tensor.
        """
        l1 = (lab_pred.float() - lab_target.float()).abs().mean(dim=1, keepdim=True) / 100.0
        weight = pixel_weight.to(torch.float32)
        per_pair = (l1 * weight).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp(min=_TINY)
        return (per_pair * sample_weight.to(per_pair.dtype)).mean()

    def align(
        self, adapted: torch.Tensor, frozen: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine distance alignment loss between adapted and frozen features.

        Args:
            adapted: Adapted patch tokens of shape (B, N, C).
            frozen: Frozen target patch tokens of shape (B, N, C).
            weight: Patch validity weights of shape (B, N).

        Returns:
            Scalar alignment loss tensor.
        """
        cosine = F.cosine_similarity(adapted.float(), frozen.float().detach(), dim=-1)
        weight = weight.to(cosine.dtype)
        return ((1.0 - cosine) * weight).sum() / weight.sum().clamp(min=_TINY)

    def color(
        self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss between predicted bin logits and target histograms.

        Args:
            logits: Predicted bin logits of shape (B, N, Q).
            target: Target soft histograms of shape (B, N, Q).
            weight: Patch validity weights of shape (B, N).

        Returns:
            Scalar color loss tensor.
        """
        weight = weight.to(torch.float32)
        weight = torch.where(weight >= self.min_patch_fraction, weight, torch.zeros_like(weight))
        cross_entropy = -(target.float() * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
        return (weight * cross_entropy).sum() / weight.sum().clamp(min=_TINY)

    def forward(
        self,
        v_pred: torch.Tensor,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        token_weights: torch.Tensor | None = None,
        lab_pred: torch.Tensor | None = None,
        lab_target: torch.Tensor | None = None,
        pixel_weight: torch.Tensor | None = None,
        sample_weight: torch.Tensor | None = None,
        feat_adapted: torch.Tensor | None = None,
        feat_frozen: torch.Tensor | None = None,
        patch_weight: torch.Tensor | None = None,
        color_logits: torch.Tensor | None = None,
        color_target: torch.Tensor | None = None,
        color_patch_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all active loss terms and their weighted total.

        Returns:
            Dictionary mapping loss names to scalar tensors.
        """
        out = self.flow(v_pred, z_0, z_1, token_weights)
        zero = v_pred.new_zeros((), dtype=torch.float32)
        out["render"] = (
            self.render(lab_pred, lab_target, pixel_weight, sample_weight)
            if lab_pred is not None
            else zero
        )
        out["align"] = (
            self.align(feat_adapted, feat_frozen, patch_weight)
            if feat_adapted is not None
            else zero
        )
        out["color"] = (
            self.color(color_logits, color_target, color_patch_weight)
            if color_logits is not None
            else zero
        )
        out["total"] = (
            self.velocity_weight * out["flow"]
            + self.render_weight * out["render"]
            + self.align_weight * out["align"]
            + self.color_weight * out["color"]
        )
        return out

