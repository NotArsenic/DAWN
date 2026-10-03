"""Noise parameter prediction head.

Small MLP appended to the DINO encoder that predicts the Poisson-Gaussian
noise parameters (log a, log b) used for analytic uncertainty propagation.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

LOG_FLOOR = 1e-12  # keeps log() finite if a degradation ever reports a zero variance

# Where the output starts. Measured over 200 training draws of configs/data.yaml's degradation:
# log a = -8.3 +- 1.7, log b = -14.5 +- 3.1. A head initialized at zero would spend its first
# few hundred steps walking a 14-nat bias, and with a global gradient clip of 1.0 that one
# gradient crowds out the flow, render and align terms. Starting at the prior leaves only the
# spread to learn, which is the part that depends on the image.
LOG_PRIOR = (-8.0, -14.0)


class NoiseHead(nn.Module):
    """Predicts Poisson-Gaussian noise parameters from DINO features.

    The sensor noise model is: Var[pixel] = a * pixel + b
    where a is the Poisson scale and b is the readout variance.

    We predict (log a, log b) for positivity.

    Input: the CLS token (B, dino_dim), or patch features (B, num_patches, dino_dim), which
    are mean-pooled first.
    Output: (B, 2) with (log_a, log_b), at NATIVE resolution (docs/conventions.md section 6).
    """

    def __init__(
        self,
        dino_dim: int = 384,
        hidden_dim: int = 128,
        log_prior: tuple[float, float] = LOG_PRIOR,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.mlp[-1].bias.copy_(torch.tensor(log_prior, dtype=torch.float32))

    def forward(self, dino_features: torch.Tensor) -> torch.Tensor:
        """Predict noise parameters.

        Args:
            dino_features: (B, dino_dim) CLS token, or (B, num_patches, dino_dim) patch
                features (mean-pooled over the patches).

        Returns:
            (B, 2) with (log_a, log_b).
        """
        if dino_features.ndim == 3:
            dino_features = dino_features.mean(dim=1)
        elif dino_features.ndim != 2:
            raise ValueError(
                f"expected (B, C) or (B, N, C) features, got {tuple(dino_features.shape)}"
            )
        return self.mlp(dino_features)


def log_noise_target(noise_params: Mapping[str, torch.Tensor | float]) -> torch.Tensor:
    """`(log a, log b)` ground truth from a degradation's `noise_params`, `(B, 2)`.

    `a` and `b` are per channel at native resolution; the head predicts one pair per image, so
    both are averaged over the channels. Quantization adds a flat variance on top of the read
    noise and is indistinguishable from it downstream, so it is folded into `b`: what the head
    is asked for is the EFFECTIVE `Var = a*x + b` of the frame it is looking at, before the
    `1/n` of area averaging (docs/conventions.md section 6).

    Args:
        noise_params: dict with 'a' and 'b' `(B, 3)` or `(3,)` tensors, and optionally
            'quant_var' (scalar or `(B,)`).

    Returns:
        (B, 2) float32 `(log a, log b)`.
    """
    a = torch.as_tensor(noise_params["a"], dtype=torch.float32)
    b = torch.as_tensor(noise_params["b"], dtype=torch.float32)
    if a.ndim == 1:
        a, b = a.unsqueeze(0), b.unsqueeze(0)
    quant = noise_params.get("quant_var", 0.0)
    quant = torch.as_tensor(quant, dtype=torch.float32, device=b.device).reshape(-1, 1)
    return torch.log(
        torch.stack([a.mean(dim=-1), (b + quant).mean(dim=-1)], dim=-1).clamp(min=LOG_FLOOR)
    )
