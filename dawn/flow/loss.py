"""Flow matching losses.

Conditional flow matching / rectified flow, in STANDARDIZED token space
(docs/conventions.md section 5):
    z_0 ~ N(0, I)
    z_1 = standardize(pack(target parts))
    z_t = (1 - t) * z_0 + t * z_1        (t = 0 noise, t = 1 data)
    v_target = z_1 - z_0

Loss = velocity_weight * L_flow     weighted MSE of the velocity, per segment of the token
     + render_weight   * L_render   L1 of Lab/100 after one-step estimate and render
     + align_weight    * L_align    1 - cosine of adapted (thermal) vs frozen (RGB) DINO tokens
     + color_weight    * L_color    cross-entropy of the colour head over the 261 a*b* bins

Every term is a MASKED mean: the pixel mask and its fractions arrive as weights, never thresholded,
and a term with no weight at all is 0, not NaN. L_render arrives as an already rendered image: the
caller unstandardizes and renders before this module, so it never sees stats.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from dawn.flow.tokens import TokenLayout

SEGMENT_BALANCES = ("per_value", "equal")
_TINY = torch.finfo(torch.float32).tiny


class FlowLoss(nn.Module):
    """The four losses of stage 1, with the token layout deciding the L_flow segments.

    Args:
        layout: The token layout; its segments are brightness, color and (optionally) correction.
        velocity_weight, render_weight, align_weight, color_weight: Loss weights.
        segment_balance: How the segments of L_flow are summed. `per_value`: each segment counts
            in proportion to its number of values, i.e. the plain mean over the token (colour's 32
            values weigh twice brightness's 16). `equal`: each segment's own mean counts the same.
        min_patch_fraction: L_color ignores patches whose kept share is below this.
        fallback_render_weight: Weight of a fallback-aligned pair in L_render (`sample_weight`).
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
        """Build from the parsed flow.yaml (`flow`, `color.min_patch_fraction`, `pairs`)."""
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
        """`(B,)` L_render weight per pair: `fallback_render_weight` for fallback pairs, else 1."""
        ones = torch.ones(fallback.shape, dtype=torch.float32, device=fallback.device)
        return torch.where(fallback.bool(), ones * self.fallback_render_weight, ones)

    # --- the four terms -------------------------------------------------------------------------

    def flow(
        self,
        v_pred: torch.Tensor,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """L_flow: weighted MSE of `v_pred` against `z_1 - z_0`, summed over the segments.

        Args:
            v_pred, z_0, z_1: (B, N, D) standardized tokens.
            weights: (B, N, D) from `TokenLayout.weights` (each segment mean 1 per sample), or
                None for unweighted.

        Returns:
            `flow` (the balanced sum) and `flow_<segment>` (each segment's own weighted mean),
            for the segments the layout has.
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
        """L_render: L1 of `Lab / 100`, a masked mean per pair, times the pair's weight.

        Args:
            lab_pred, lab_target: (B, 3, H, W) L*, a*, b* at the render size.
            pixel_weight: (B, 1, H, W) mask fractions at the render size.
            sample_weight: (B,) per-pair weight (`sample_weight(fallback)`).
        """
        l1 = (lab_pred.float() - lab_target.float()).abs().mean(dim=1, keepdim=True) / 100.0
        weight = pixel_weight.to(torch.float32)
        per_pair = (l1 * weight).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3)).clamp(min=_TINY)
        return (per_pair * sample_weight.to(per_pair.dtype)).mean()

    def align(
        self, adapted: torch.Tensor, frozen: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """L_align: `1 - cosine` per patch, a masked mean; no gradient into `frozen`.

        Args:
            adapted: (B, N, C) patch tokens of the LoRA-adapted encoder on thermal.
            frozen: (B, N, C) patch tokens of the frozen encoder on the paired RGB.
            weight: (B, N) kept share of every patch.
        """
        cosine = F.cosine_similarity(adapted.float(), frozen.float().detach(), dim=-1)
        weight = weight.to(cosine.dtype)
        return ((1.0 - cosine) * weight).sum() / weight.sum().clamp(min=_TINY)

    def color(
        self, logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """L_color: cross-entropy of the colour head against the soft histogram, per patch.

        Patches whose kept share is below `min_patch_fraction` count 0; the rest are weighted by
        their share.

        Args:
            logits: (B, N, Q) from `ColorHead`.
            target: (B, N, Q) soft histograms (`patch_histograms`), each summing to 1.
            weight: (B, N) kept share of every patch.
        """
        weight = weight.to(torch.float32)
        weight = torch.where(weight >= self.min_patch_fraction, weight, torch.zeros_like(weight))
        cross_entropy = -(target.float() * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1)
        return (weight * cross_entropy).sum() / weight.sum().clamp(min=_TINY)

    # --- all together ---------------------------------------------------------------------------

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
        """All four terms and their weighted `total`. A term whose inputs are missing is 0.

        Returns:
            dict: `total`, `flow`, `flow_<segment>` per segment, `render`, `align`, `color`.
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
