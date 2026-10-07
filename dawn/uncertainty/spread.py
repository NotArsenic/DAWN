"""Sample-spread uncertainty: brightness and colour maps from K flow samples.

The flow is stochastic in `z_0`, so K draws for the same frame give K renders that agree where
the conditioning pins the answer down and disagree where it does not. Their spread is the model's
own statement of where it is unsure, and it needs no extra head, no calibration target and no
second network: `sample_parts` draws the K members in one batch off a single DINO pass, so the
whole map costs `nfe * K` velocity passes (docs/conventions.md section 11).

Units are CIELAB, so the maps read against the metrics directly:

    l_std   L* units (the image's L* spans 0-100), comparable to the L* error
    ab_std  a*b* units, the RMS distance from the mean colour in the a*b* plane, so it is on the
            same scale as a chroma CIEDE2000 and can be ranked against it

**The spread is a disagreement, not an error bar.** It says the model had several answers here,
which is not the same as being wrong here: a region the model is confidently wrong about has a
small spread. `dawn/eval/calibration.py` measures how well the two actually rank together; until
that number exists, read these maps as "generated, not measured", which is what the README
promises.

With `num_samples == 1` there is nothing to disagree with: both maps come back as zeros rather
than NaN, so a K=1 run still produces a well-formed result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from dawn.data.color import lab_to_linear
from dawn.flow.render import RenderSettings, render
from dawn.flow.sampler import Method, sample_parts
from dawn.flow.tokens import TokenLayout


@dataclass(frozen=True)
class SpreadResult:
    """One frame's point estimate, ensemble mean and the two uncertainty maps.

    `sample_rgb` and `rgb` answer different questions. A single member is what the model actually
    generates: sharp, saturated, one plausible colouring. The ensemble mean is the average of K
    colourings, so it is smoother and less colourful (the regression-to-brown the flow exists to
    avoid) but closer in CIEDE2000. Show `sample_rgb`; score whichever the evaluation says.

    Args:
        sample_rgb: `(B, 3, h, w)` linear RGB of member 0, the point estimate to display.
        rgb: `(B, 3, h, w)` linear RGB of the ensemble mean, clamped to the gamut.
        lab: `(B, 3, h, w)` the ensemble mean in L*, a*, b*.
        l_std: `(B, 1, h, w)` standard deviation of L* across members, in L* units.
        ab_std: `(B, 1, h, w)` RMS distance from the mean colour in the a*b* plane,
            `sqrt(var(a*) + var(b*))`, in a*b* units.
        num_samples: K, the number of members the spread was taken over.
    """

    sample_rgb: torch.Tensor
    rgb: torch.Tensor
    lab: torch.Tensor
    l_std: torch.Tensor
    ab_std: torch.Tensor
    num_samples: int


def spread_from_parts(
    parts: Mapping[str, torch.Tensor],
    thermal: torch.Tensor,
    size: tuple[int, int],
    settings: RenderSettings,
) -> SpreadResult:
    """Render K sampled members and reduce them to a mean and two spread maps.

    The members are rendered one at a time and reduced by Welford's algorithm, so peak memory is
    that of two renders whatever K is. Rendering is the expensive half of a K-sample inference
    (the BGU especially), and it is unavoidable: the spread is defined on rendered L*a*b*, not on
    tokens, because a token's variance says nothing about how much of it survives the grid slice.

    Args:
        parts: raw parts with a leading `(S, B)`, exactly as `sample_parts` returns them:
            `grid (S, B, 2, K, gy, gx)`, `ab (S, B, 2, hc, wc)` and, with the correction,
            `corr (S, B, 1, hc, wc)`.
        thermal: `(B, 1, H, W)` thermal, `code / 255`, at least `size`.
        size: `(h, w)` to render at (docs/conventions.md section 6.4).
        settings: layout and upsampling settings, as `render` takes them.

    Returns:
        A `SpreadResult` at `size`.

    Raises:
        ValueError: `parts` is empty, the members disagree on `S`, or `S` is 0.
    """
    if not parts:
        raise ValueError("parts is empty: expected the dict sample_parts returns")
    counts = {name: int(value.shape[0]) for name, value in parts.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"parts disagree on the sample count S: {counts}")
    num_samples = next(iter(counts.values()))
    if num_samples < 1:
        raise ValueError(f"parts carry {num_samples} samples, need at least 1")

    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    first: torch.Tensor | None = None
    for index in range(num_samples):
        member = render(
            {name: value[index] for name, value in parts.items()}, thermal, size, settings
        )
        lab = member["lab"].to(torch.float32)
        if first is None:
            first, mean, m2 = member["rgb"], torch.zeros_like(lab), torch.zeros_like(lab)
        delta = lab - mean
        mean = mean + delta / (index + 1)
        m2 = m2 + delta * (lab - mean)

    variance = m2 / (num_samples - 1) if num_samples > 1 else torch.zeros_like(m2)
    return SpreadResult(
        sample_rgb=first,
        rgb=lab_to_linear(mean).clamp(0.0, 1.0),
        lab=mean,
        l_std=variance[:, 0:1].sqrt(),
        ab_std=(variance[:, 1:2] + variance[:, 2:3]).sqrt(),
        num_samples=num_samples,
    )


@torch.no_grad()
def sample_with_spread(
    velocity_model,
    encoded: Mapping[str, torch.Tensor],
    layout: TokenLayout,
    mean: torch.Tensor,
    std: torch.Tensor,
    thermal: torch.Tensor,
    size: tuple[int, int],
    settings: RenderSettings,
    *,
    num_samples: int = 8,
    method: Method = "euler",
    nfe: int = 2,
    generator: torch.Generator | None = None,
    image_ids: Sequence[str] | None = None,
) -> SpreadResult:
    """One DINO pass in, a rendered estimate and its uncertainty maps out.

    The whole inference path of docs/conventions.md section 6.4 after the encoder: draw K members,
    unstandardize once, render each, reduce. `encoded` is what `DINOEncoder.encode` returned for
    this same thermal frame, so the encoder is never run twice.

    Args:
        velocity_model: `VelocityTransformer`, in eval mode.
        encoded: `DINOEncoder.encode` output: needs `patch_tokens` and `thumb`, uses `attn_bias`
            when the encoder was asked for it.
        layout: the checkpoint's token layout.
        mean, std: `(D,)` stats from the checkpoint, never from the cache file (section 5.1).
        thermal: `(B, 1, H, W)` thermal, `code / 255`.
        size: `(h, w)` to render at.
        settings: layout and upsampling settings.
        num_samples: K members. 4-8 is the plan's range; 1 skips the spread.
        method: `"euler"` or `"midpoint"`.
        nfe: velocity evaluations per member.
        generator: RNG for `z_0`. Ignored when `image_ids` is given.
        image_ids: one per image, so the draw is reproducible from the ID (section 10.3).

    Returns:
        A `SpreadResult` at `size`.
    """
    parts = sample_parts(
        velocity_model,
        encoded["patch_tokens"],
        layout,
        mean,
        std,
        attn_bias=encoded.get("attn_bias"),
        thumb=encoded.get("thumb"),
        method=method,
        nfe=nfe,
        num_samples=num_samples,
        generator=generator,
        image_ids=image_ids,
    )
    return spread_from_parts(parts, thermal, size, settings)
