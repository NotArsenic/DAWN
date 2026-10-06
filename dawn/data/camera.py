"""Camera model: sensor, auto exposure and ISP. Scene radiance in, night image out.

The other half of the synthetic night degradation (the first half is lighting.py). lighting.py
says how much light fell on the scene; this module says what the photo of it looks like.

    radiance (ambient white = 1, unbounded)
      -> raw = radiance * N_white / full_well           what the sensor records, saturates at 1
      -> Poisson photon count + Gaussian read noise     the only randomness
      -> auto-exposure digital gain g, capped           a dark scene STAYS dark when g caps out
      -> partial white balance (per channel)
      -> 12-bit quantize
      -> [native path only] linear denoise blur, linear sharpen
      -> linear_to_srgb -> 8-bit -> JPEG -> back to linear
      -> night (linear, float32, [0, 1])

The ISP ends by decoding back to linear, so `night` is exactly what a linearized real night JPEG
looks like and the rest of the pipeline stays linear (docs/conventions.md section 1).

Photons and resolution
----------------------
`N_white` (log-uniform, sampled per image) is the number of electrons a white pixel at relative
radiance 1 collects at NATIVE resolution. It is the one knob that decides how dark and how noisy
the frame is, and it already has the exposure baked in: a bright moonlit frame draws thousands
of electrons, starlight draws tens.

`n` is how many native pixels one simulated pixel area-averages, `W*H / (w*h)`
(docs/conventions.md section 4a). A simulated pixel therefore collects `Poisson(u*N_white*n)`
photons and `n*sigma_r^2` of read-noise variance, then divides by `n` -- exactly a sensor with
`n` times bigger pixels. Adding native-strength noise to a downsampled copy instead would make
training far noisier than inference.

Noise bookkeeping (docs/conventions.md section 6)
-------------------------------------------------
With `a_raw = 1 / full_well` and `b_raw = sigma_r^2 / full_well^2`, the normalized raw signal
`x` has `Var[x] = (a_raw * x + b_raw) / n`. Everything after the sensor is a per-channel linear
scale `s_c = g * wb_c`, so the OUTPUT obeys

    Var[y_c] = (a_c * y_c + b_c + quant_var) / n,   a_c = s_c * a_raw,  b_c = s_c^2 * b_raw

and `a_c`, `b_c` are reported at native resolution (`n = 1`), per channel, and so is the
quantization variance `step^2 / 12`: the ADC digitizes native pixels, so averaging `n` of them
divides that term by `n` too. On the native path both filters are linear, so they multiply
`a_c`, `b_c` and `quant_var` by the sum of the squared weights of their composite kernel.
JPEG and the final 8-bit step are nonlinear and are NOT in the budget; see
`CameraResult.jpeg_quality`.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image

from dawn.data.color import linear_to_srgb8, luminance, srgb8_to_linear

CHANNELS = 3


@dataclass
class CameraParams:
    """Sampled camera settings for one image. `a_raw` and `b_raw` come from the calibration."""

    u: float
    target: float
    photon_window_u: float
    read_noise: float  # sigma_r, electrons per native pixel
    wb_gains: torch.Tensor  # (3,) partial white-balance gains, green == 1
    jpeg_quality: int
    chroma_sigma: float
    chroma_gain: float
    a_raw: float  # 1 / full_well
    b_raw: float  # sigma_r^2 / full_well^2
    ae_percentile: float
    gain_min: float
    gain_max: float
    n_white_range: tuple[float, float]
    quant_bits: int
    denoise_sigma: float
    denoise_radius: int
    sharpen_amount: float
    sharpen_sigma: float
    sharpen_radius: int
    n_white: float | None = None
    gain: float | None = None
    n_white_clamped: bool | None = None


@dataclass
class CameraResult:
    """One rendered night image and its exact noise budget."""

    night: torch.Tensor  # (3, H, W) linear RGB, float32, [0, 1]
    gain: float  # the auto-exposure gain actually applied
    gain_capped: bool  # True when AE wanted more than `gain_max`: the frame stays dark
    noise_a: torch.Tensor  # (3,) shot term at NATIVE resolution (pre-ISP)
    noise_b: torch.Tensor  # (3,) read variance at NATIVE resolution (pre-ISP)
    quant_var: float  # quantization variance at NATIVE resolution; divide by n like a and b
    n: float  # native pixels per simulated pixel
    S: torch.Tensor  # (3, 3) ISP variance mixing matrix
    M: torch.Tensor  # (3, 3) ISP DC colour matrix
    saturated: float  # fraction of samples clipped at 1.0
    jpeg_quality: int | None  # None when the JPEG round trip was skipped


def _uniform(lo: float, hi: float, generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator) * (hi - lo) + lo)


def _log_uniform_at(lo: float, hi: float, u: float) -> float:
    """The point at quantile `u` of LogUniform[lo, hi]. `u = 0` is the darkest."""
    return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))


def sample_camera_params(
    cfg: dict,
    calib: dict,
    mode: str = "moonlit",
    level: str | None = None,
    ambient_gains: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    device: torch.device = torch.device("cpu"),
) -> CameraParams:
    """Sample camera settings for one image.

    Args:
        cfg: The `degradation.camera` section of data.yaml.
        calib: Parsed configs/camera_calib.yaml.
        mode: Lighting mode, which picks the `n_white_range` band.
        level: `None` for a fresh uniform draw over the whole range (training), or one of
            `cfg["darkness_levels"]` ("easy" / "medium" / "hard") to restrict the draw to that
            band of the log-uniform range (the fixed val and test sets).
        ambient_gains: (3,) illuminant gains from LightingParams, used to aim the partial white
            balance. `None` means a neutral illuminant.
        generator: Seeded RNG; `None` uses the global one (training's fresh draw).
        device: Device for the sampled tensors.

    Returns:
        CameraParams with every random value filled in.

    Raises:
        ValueError: `mode` has no `n_white_range` entry, or `level` is not a configured band.
    """
    ranges = cfg["n_white_range"]
    if mode not in ranges:
        raise ValueError(f"no n_white_range for mode {mode!r}; have {sorted(ranges)}")

    if level is None:
        quantile = _uniform(0.0, 1.0, generator)
    else:
        bands = cfg["darkness_levels"]
        if level not in bands:
            raise ValueError(f"unknown darkness level {level!r}; have {sorted(bands)}")
        q_lo, q_hi = (float(v) for v in bands[level])
        quantile = _uniform(q_lo, q_hi, generator)

    target_log10_quantiles = calib["exposure"]["target_log10_quantiles"]
    idx = quantile * 20.0
    idx_floor = int(math.floor(idx))
    if idx_floor >= 20:
        target = float(10**target_log10_quantiles[20])
    else:
        frac = idx - idx_floor
        log10_T = target_log10_quantiles[idx_floor] * (1.0 - frac) + target_log10_quantiles[idx_floor + 1] * frac
        target = float(10**log10_T)

    photon_window_u = _uniform(0.0, 1.0, generator)
    read_noise = _uniform(*(float(v) for v in cfg["read_noise_range"]), generator)
    a_raw = float(calib["measured"]["noise_a"])
    strength = _uniform(*(float(v) for v in cfg["white_balance_strength"]), generator)

    if ambient_gains is None:
        wb = torch.ones(CHANNELS, dtype=torch.float32, device=device)
    else:
        wb = ambient_gains.to(device=device, dtype=torch.float32).clamp(min=1e-3).pow(-strength)
        wb = wb / wb[1].clamp(min=1e-6)

    jpeg_quality = int(
        torch.randint(
            int(cfg["jpeg_quality_range"][0]),
            int(cfg["jpeg_quality_range"][1]) + 1,
            (),
            generator=generator,
        )
    )

    chroma_sigma = _uniform(*(float(v) for v in cfg["chroma"]["sigma_range"]), generator)
    chroma_gain = _uniform(*(float(v) for v in cfg["chroma"]["gain_range"]), generator)

    isp = calib["isp"]
    return CameraParams(
        u=quantile,
        target=target,
        photon_window_u=photon_window_u,
        read_noise=read_noise,
        wb_gains=wb,
        jpeg_quality=jpeg_quality,
        chroma_sigma=chroma_sigma,
        chroma_gain=chroma_gain,
        a_raw=a_raw,
        b_raw=read_noise**2 * a_raw**2,
        ae_percentile=float(calib["exposure"]["percentile"]),
        gain_min=float(cfg["gain_min"]),
        gain_max=float(calib["exposure"]["gain_max"]),
        n_white_range=(float(ranges[mode][0]), float(ranges[mode][1])),
        quant_bits=int(cfg["quant_bits"]),
        denoise_sigma=float(isp["denoise_sigma"]),
        denoise_radius=int(isp["denoise_radius"]),
        sharpen_amount=float(isp["sharpen_amount"]),
        sharpen_sigma=float(isp["sharpen_sigma"]),
        sharpen_radius=int(isp["sharpen_radius"]),
    )


def _gaussian_2d(sigma: float, radius: int, device: torch.device) -> torch.Tensor:
    """A normalized (2r+1, 2r+1) Gaussian. `sigma <= 0` gives the identity (a delta)."""
    size = 2 * radius + 1
    if sigma <= 0.0 or radius == 0:
        kernel = torch.zeros(size, size, device=device)
        kernel[radius, radius] = 1.0
        return kernel
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    line = torch.exp(-offsets.pow(2) / (2 * sigma**2))
    line = line / line.sum()
    return line.view(-1, 1) * line.view(1, -1)


def isp_kernel(params: CameraParams, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """The composite native-path filter: denoise blur followed by unsharp sharpening.

    Both stages are linear and sum to 1, so the composite sums to 1 (unit DC gain: it moves no
    energy into or out of a flat patch) and multiplies any white noise variance by the sum of
    its squared weights.

    Returns:
        (k, k) float32, summing to 1.
    """
    blur = _gaussian_2d(params.denoise_sigma, params.denoise_radius, device)
    sharp_blur = _gaussian_2d(params.sharpen_sigma, params.sharpen_radius, device)
    sharpen = -params.sharpen_amount * sharp_blur
    sharpen[params.sharpen_radius, params.sharpen_radius] += 1.0 + params.sharpen_amount
    # Full convolution, so no weight is truncated and the composite still sums to exactly 1.
    pad = sharpen.shape[0] - 1
    return F.conv2d(
        blur.view(1, 1, *blur.shape), sharpen.view(1, 1, *sharpen.shape), padding=pad
    )[0, 0]


def jpeg_roundtrip(image_linear: torch.Tensor, quality: int) -> torch.Tensor:
    """Encode a linear image as an 8-bit sRGB JPEG and decode it back to linear.

    Args:
        image_linear: (3, H, W) linear RGB in [0, 1].
        quality: PIL JPEG quality, 1-95.

    Returns:
        (3, H, W) linear RGB in [0, 1], float32.
    """
    codes = linear_to_srgb8(image_linear).permute(1, 2, 0).contiguous().numpy()
    buffer = io.BytesIO()
    Image.fromarray(codes, mode="RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    decoded = torch.from_numpy(
        torch.frombuffer(
            bytearray(Image.open(buffer).convert("RGB").tobytes()), dtype=torch.uint8
        ).numpy()
    ).view(image_linear.shape[1], image_linear.shape[2], CHANNELS)
    return srgb8_to_linear(decoded.permute(2, 0, 1).contiguous())


def resolve_exposure(radiance: torch.Tensor, params: CameraParams) -> None:
    """Fills n_white and gain in params from the radiance."""
    if params.n_white is not None:
        return
    device = radiance.device
    wb_signal = radiance * params.a_raw * params.wb_gains.view(3, 1, 1).to(device)
    l1_map = luminance(wb_signal)
    L1 = float(torch.quantile(l1_map.flatten().to(torch.float32), params.ae_percentile))
    L1 = max(L1, 1e-12)

    T = params.target
    win_min = T / (params.gain_max * L1)
    win_max = T / (params.gain_min * L1)

    cfg_min, cfg_max = params.n_white_range
    inter_min = max(win_min, cfg_min)
    inter_max = min(win_max, cfg_max)

    clamped = False
    if inter_min >= inter_max:
        clamped = True
        inter_min, inter_max = win_min, win_max

    n_white = _log_uniform_at(inter_min, inter_max, params.photon_window_u)
    gain = T / (n_white * L1)
    gain = min(max(gain, params.gain_min), params.gain_max)

    params.n_white = n_white
    params.gain = gain
    params.n_white_clamped = clamped


def apply_camera(
    radiance: torch.Tensor,
    params: CameraParams,
    n: float = 1.0,
    noise: bool = True,
    native_path: bool = False,
    jpeg: bool | None = None,
    generator: torch.Generator | None = None,
) -> CameraResult:
    """Run the sensor, auto exposure and ISP on a linear scene radiance.

    With `noise=False` and `jpeg=False` this is the expected value of the noisy render: the same
    gain, the same white balance, the same quantization and the same filters, with the random
    draws replaced by their means. That is `night_clean` (degrade.py), and it is NOT the scene
    radiance.

    Args:
        radiance: (3, H, W) linear scene radiance from `apply_lighting`, >= 0, unbounded above.
        params: From `sample_camera_params`.
        n: Native pixels area-averaged into one pixel of `radiance`. 1.0 on the native path.
        noise: Draw photon and read noise. False gives the noise-free expectation.
        native_path: Run the in-camera denoise blur and sharpening. These only survive at native
            resolution; a 4x downsample erases them (docs/conventions.md section 4a).
        jpeg: Run the JPEG round trip. Defaults to `noise`, so a noise-free render is also
            artifact-free and `night == night_clean` exactly.
        generator: Seeded RNG; `None` uses the global one.

    Returns:
        CameraResult: the image plus the exact per-channel `(a, b)` at native resolution.

    Raises:
        ValueError: `radiance` is not (3, H, W), or `n` is not positive.
    """
    if radiance.ndim != 3 or radiance.shape[0] != CHANNELS:
        raise ValueError(f"radiance must be (3, H, W), got {tuple(radiance.shape)}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    jpeg = noise if jpeg is None else jpeg
    device = radiance.device
    radiance = radiance.to(torch.float32).clamp(min=0.0)

    resolve_exposure(radiance, params)

    # --- sensor ---------------------------------------------------------------------------
    scale = params.n_white * params.a_raw
    raw_mean = radiance * scale
    if noise:
        rate = (radiance * params.n_white * n).clamp(min=0.0)
        electrons = torch.poisson(rate, generator=generator)
        if params.read_noise > 0:
            electrons = electrons + torch.randn(
                radiance.shape, generator=generator, device=device
            ) * (params.read_noise * math.sqrt(n))
        raw = electrons * (params.a_raw / n)
    else:
        raw = raw_mean
    raw = raw.clamp(0.0, 1.0)

    # --- auto exposure and partial white balance ---------------------------------------------
    channel_scale = params.gain * params.wb_gains.to(device)  # (3,)
    signal = (raw * channel_scale.view(CHANNELS, 1, 1)).clamp(0.0, 1.0)
    saturated = float((signal >= 1.0).to(torch.float32).mean())

    # --- 12-bit quantization -------------------------------------------------------------
    levels = float(2**params.quant_bits - 1)
    step = 1.0 / levels
    signal = torch.round(signal * levels * n) / (levels * n)
    quant_var = step**2 / 12.0

    noise_a = channel_scale * params.a_raw
    noise_b = channel_scale.pow(2) * params.b_raw

    # --- in-camera ISP, native resolution only ------------------------------------------------
    S = torch.eye(3, device=device)
    M = torch.eye(3, device=device)
    if native_path:
        chroma_radius = int(math.ceil(3 * params.chroma_sigma))
        k_c = 2 * chroma_radius + 1
        chroma_blur = _gaussian_2d(params.chroma_sigma, chroma_radius, device)
        K_m = torch.zeros(3, k_c, k_c, device=device)
        K_m[0, chroma_radius, chroma_radius] = 1.0
        K_m[1] = chroma_blur * params.chroma_gain
        K_m[2] = chroma_blur * params.chroma_gain

        T = torch.tensor([
            [0.2126, 0.7152, 0.0722],
            [-0.2126, -0.7152, 0.9278],
            [0.7874, -0.7152, -0.0722],
        ], device=device)
        Tinv = torch.linalg.inv(T)

        W_chroma = torch.einsum('im, myx, mj -> ijyx', Tinv, K_m, T)

        K_luma = isp_kernel(params, device)
        k_l = K_luma.shape[0]

        W_flat = W_chroma.reshape(9, 1, k_c, k_c)
        W_full = F.conv2d(W_flat, K_luma.view(1, 1, k_l, k_l), padding=k_l - 1)
        k_full = W_full.shape[-1]
        W_full = W_full.view(3, 3, k_full, k_full)

        pad = k_full // 2
        padded = F.pad(signal.unsqueeze(0), (pad, pad, pad, pad), mode="replicate")
        signal = F.conv2d(padded, W_full)[0].clamp(0.0, 1.0)

        S = W_full.pow(2).sum(dim=(2, 3))
        M = W_full.sum(dim=(2, 3))

    # --- output transfer function ----------------------------------------------------------
    if jpeg:
        night = jpeg_roundtrip(signal, params.jpeg_quality)
    else:
        night = signal
    return CameraResult(
        night=night.clamp(0.0, 1.0).to(torch.float32),
        gain=params.gain,
        gain_capped=params.n_white_clamped,
        noise_a=noise_a.to(torch.float32).cpu(),
        noise_b=noise_b.to(torch.float32).cpu(),
        quant_var=quant_var,
        n=float(n),
        S=S.cpu(),
        M=M.cpu(),
        saturated=saturated,
        jpeg_quality=int(params.jpeg_quality) if jpeg else None,
    )
