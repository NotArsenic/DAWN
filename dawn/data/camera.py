"""Camera model: sensor noise + ISP.

The other half of the synthetic night degradation (the first half is the lighting,
lighting.py). Turns the noise-free `night_clean` into the image the model actually sees:

    night_clean (linear)
      -> Poisson-Gaussian sensor noise, Var[y | x] = a*x + b
      -> clip to [0, 1]
      -> linear_to_srgb (ISP gamma) -> 8-bit quantize -> JPEG round trip
      -> srgb_to_linear
      -> night (linear, float32, [0, 1])

The ISP ends by decoding back to linear, so `night` looks exactly like a linearized real night
JPEG and the rest of the pipeline stays linear (docs/conventions.md section 1). The noise
parameters are defined in section 6: `b` is a read-noise VARIANCE, not a standard deviation.

Resolution (section 6). `(a, b)` are native-resolution values. A training image is a 512-px copy
in which every pixel area-averages n = W*H / (w*h) native pixels (`native_size` comes with every
VisDroneDataset item), so the sensor must simulate binned pixels: Var = (a/n)*x + b/n. That is
exact, since Poisson counts sum to a Poisson count and read-noise variances add. Adding
native-strength noise to the 512-px copy instead would make DINO see far noisier inputs in
training than at inference, where the full frame is area-downsampled to 256x256.

JPEG artifacts and in-camera denoising mostly vanish after a 4x downsample, so simulating them
matters for the native-resolution path (denoiser crops) and little for the 256x256 flow path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CameraParams:
    """Randomly sampled camera parameters for one image."""

    noise_a: float  # shot-noise gain (Poisson scale)
    noise_b: float  # read-noise variance (NOT std)
    jpeg_quality: int


def sample_camera_params(
    cfg: dict,
    device: torch.device = torch.device("cpu"),
) -> CameraParams:
    """Sample random camera parameters from config ranges.

    Args:
        cfg: The `degradation.camera` section of data.yaml.
        device: Tensor device.

    Returns:
        CameraParams with all random values filled in.
    """
    raise NotImplementedError


def apply_camera(
    night_clean: torch.Tensor,
    params: CameraParams,
) -> torch.Tensor:
    """Run the sensor and ISP on a noise-free linear night image.

    Args:
        night_clean: (3, H, W) linear RGB in [0, 1].
        params: Sampled camera parameters.

    Returns:
        night: (3, H, W) linear RGB in [0, 1]: noisy, quantized and JPEG-compressed.
    """
    raise NotImplementedError
