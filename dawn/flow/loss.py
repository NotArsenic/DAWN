"""Flow matching loss computation.

Conditional flow matching / rectified flow, in STANDARDIZED grid space
(docs/conventions.md section 5):
    z_0 ~ N(0, I)
    z_1 = standardize(fitted grid)
    z_t = (1 - t) * z_0 + t * z_1        (t = 0 noise, t = 1 data)
    v_target = z_1 - z_0

Loss = λ_flow  * ||v_pred - v_target||^2                   (L_flow: velocity MSE)
     + λ_apply * L1(render(unstandardize(z_1_hat)), day)   (L_render: apply loss)
     + λ_noise * MSE(noise_pred, noise_gt)                 (L_noise: noise head)
     + λ_align * dist(feat_adapted, feat_frozen)           (L_align: LoRA stays near DINO)

The render term arrives as an already rendered image: the caller un-standardizes the grid
before slicing, so this module never sees the grid stats.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def bin_weights(node_weights: torch.Tensor) -> torch.Tensor:
    """Per-node pixel counts -> weights with mean 1, `(B, 1, K, gy, gx)`.

    `fit_grid` returns how much trilinear weight each node actually collected. A node that no
    pixel reached is fitted from its ridge prior alone, so matching its coefficients teaches
    the flow nothing; a node holding a tenth of the image should count ten times a node holding
    a hundredth. Normalizing to mean 1 per sample keeps the loss on the same scale as a plain
    MSE, so `flow.velocity_weight` means the same thing whether or not weighting is on.

    Args:
        node_weights: (B, K, gy, gx) from `fit_grid`.

    Returns:
        (B, 1, K, gy, gx), ready to broadcast over the 12 coefficients.
    """
    if node_weights.ndim != 4:
        raise ValueError(f"node_weights must be (B, K, gy, gx), got {tuple(node_weights.shape)}")
    weights = node_weights.to(torch.float32).clamp(min=0.0)
    mean = weights.mean(dim=(1, 2, 3), keepdim=True)
    return (weights / mean.clamp(min=torch.finfo(torch.float32).tiny)).unsqueeze(1)


class FlowMatchingLoss(nn.Module):
    """Combined loss for bilateral grid flow matching.

    Components:
        1. Velocity MSE: ||v_pred - (z_1 - z_0)||^2
        2. Apply (render) L1: |slice_and_apply(unstandardize(z_1_hat), night_clean) - day|
           (skip at early steps)
        3. Noise MSE: ||noise_pred - (log_a, log_b)||^2
        4. Align: distance between LoRA-adapted and frozen (adapters disabled) DINO patch
           tokens; the frozen target comes from `DINOEncoder.forward_frozen`.
        5. Optional bin-count weighting (per-node pixel counts from `fit_grid`).
    """

    def __init__(
        self,
        velocity_weight: float = 1.0,
        apply_weight: float = 0.1,
        noise_weight: float = 1.0,
        align_weight: float = 0.5,
        bin_count_weighting: bool = True,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.apply_weight = apply_weight
        self.noise_weight = noise_weight
        self.align_weight = align_weight
        self.bin_count_weighting = bin_count_weighting

    @classmethod
    def from_config(cls, cfg) -> "FlowMatchingLoss":
        """Build from the `flow:` section of configs/flow.yaml."""
        return cls(
            velocity_weight=cfg["velocity_weight"],
            apply_weight=cfg["apply_weight"],
            noise_weight=cfg["noise_weight"],
            align_weight=cfg["align_weight"],
            bin_count_weighting=cfg["bin_count_weighting"],
        )

    def forward(
        self,
        v_pred: torch.Tensor,
        z_0: torch.Tensor,
        z_1: torch.Tensor,
        t: torch.Tensor,
        noise_pred: torch.Tensor | None = None,
        noise_gt: torch.Tensor | None = None,
        applied_pred: torch.Tensor | None = None,
        day_image: torch.Tensor | None = None,
        node_weights: torch.Tensor | None = None,
        feat_adapted: torch.Tensor | None = None,
        feat_frozen: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute losses.

        Args:
            v_pred: (B, 12, K, gy, gx) predicted velocity, standardized.
            z_0: (B, 12, K, gy, gx) source noise.
            z_1: (B, 12, K, gy, gx) target grid coefficients, standardized.
            t: (B,) time values.
            noise_pred: (B, 2) predicted (log_a, log_b).
            noise_gt: (B, 2) ground-truth (log_a, log_b).
            applied_pred: (B, 3, H, W) linear RGB rendered from the UN-standardized
                predicted grid.
            day_image: (B, 3, H, W) ground-truth linear RGB day image.
            node_weights: (B, K, gy, gx) pixel counts from `fit_grid`, for bin-count
                weighting (broadcast over the coefficient axis).
            feat_adapted: (B, 256, C) patch tokens from the LoRA-adapted encoder.
            feat_frozen: (B, 256, C) patch tokens from `forward_frozen` (no gradient).

        Returns:
            dict with 'total', 'velocity', 'apply', 'noise', 'align' losses.
        """
        del t  # the target is t-independent; kept in the signature for schedulable weightings
        if v_pred.shape != z_0.shape or v_pred.shape != z_1.shape:
            raise ValueError(
                f"v_pred {tuple(v_pred.shape)}, z_0 {tuple(z_0.shape)} and z_1 "
                f"{tuple(z_1.shape)} must all be (B, 12, K, gy, gx)"
            )
        zero = v_pred.new_zeros(())

        squared = (v_pred - (z_1 - z_0)).pow(2)
        if self.bin_count_weighting and node_weights is not None:
            squared = squared * bin_weights(node_weights).to(squared.dtype)
        velocity = squared.mean()

        if applied_pred is not None and day_image is not None:
            apply_loss = F.l1_loss(applied_pred.to(torch.float32), day_image.to(torch.float32))
        else:
            apply_loss = zero

        if noise_pred is not None and noise_gt is not None:
            noise = F.mse_loss(noise_pred.to(torch.float32), noise_gt.to(torch.float32))
        else:
            noise = zero

        if feat_adapted is not None and feat_frozen is not None:
            cosine = F.cosine_similarity(
                feat_adapted.to(torch.float32), feat_frozen.to(torch.float32).detach(), dim=-1
            )
            align = (1.0 - cosine).mean()
        else:
            align = zero

        total = (
            self.velocity_weight * velocity
            + self.apply_weight * apply_loss
            + self.noise_weight * noise
            + self.align_weight * align
        )
        return {
            "total": total,
            "velocity": velocity,
            "apply": apply_loss,
            "noise": noise,
            "align": align,
        }
