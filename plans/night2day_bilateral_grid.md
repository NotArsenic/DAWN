---
name: Night2Day Bilateral Grid
overview: Build a flow-matched bilateral grid system for night-to-day drone footage enhancement on a 5060 Ti in 4 weeks. DINOv3 ViT-S encoder, velocity transformer predicts affine color grids, bounded denoiser, analytic uncertainty. Train on VisDrone with synthetic night degradation.
todos:
  - id: dinov3-access
    content: Accept DINOv3 license on HuggingFace, set up HF token. Check if DINOv2 (Apache 2.0) is a simpler fallback.
    status: pending
  - id: week1-data-grid
    content: "Week 1: Write degradation pipeline, grid fit/slice/guidance, color utils, both CPU check scripts. All laptop, no GPU."
    status: pending
  - id: check-oracle
    content: "Run check_oracle.py: fitted grid PSNR must be >30dB or increase grid to 32x32."
    status: pending
  - id: check-degradation
    content: "Run check_degradation.py: synthetic night stats must overlap real VisDrone night frames."
    status: pending
  - id: week2-models-training
    content: "Week 2: Write encoder (DINOv3+LoRA), velocity transformer, flow loss/sampler, train stage 1 on 5060 Ti."
    status: pending
  - id: week3-denoiser-uncertainty
    content: "Week 3: Write denoiser, analytic uncertainty, train stage 2, build infer.py for full pipeline."
    status: pending
  - id: week4-eval-polish
    content: "Week 4: Metrics, calibration, visualizations, latency benchmark, README, clean up repo."
    status: pending
isProject: false
---

# Night-to-Day via Flow-Matched Bilateral Grids

## Constraints
- **Hardware**: RTX 5060 Ti (16GB VRAM)
- **Timeline**: 4 weeks
- **Goal**: Working repo, not a paper submission

## Architecture Summary

```
night image (full res)
  ├── linearize -> guidance: blur luminance -> log -> bin index   (full res, cheap)
  └── resize 256x256 -> DINOv3 ViT-S/16 (+LoRA) -> patch tokens  (once)
                      └── 16x16 log thumbnail ──┘
                                │
               noise z0 -> velocity transformer x 2 steps -> grid (16x16x8x12)
                                │
         trilinear slice at full res -> out_lin = A*x + b -> clamp [0,1]
                                │
                  Var_out = A * Sigma_in * A^T   (uncertainty, arithmetic only)
                                │
                local denoiser (9x9 RF, bounded residual) -> re-gamma -> day image
```

**Key property**: output = affine transform of input pixels. Cannot add or remove objects.

## Week 1: Data + Grid Infrastructure (laptop, no GPU)

**Files to write:**
- `n2d/data/visdrone.py` -- dataset class, download VisDrone-DET train (~6.5K images)
- `n2d/data/degrade.py` -- synthetic night degradation pipeline:
  - sRGB -> linear
  - Exposure reduction: k ~ LogUniform(0.01, 0.15)
  - Illumination field: random 4x4 grid + Gaussian "light pools"
  - Color cast: color temp ~ U(2000K, 4500K)
  - Poisson-Gaussian noise: y = a*Poisson(x/a) + N(0,b)
  - Re-gamma, 8-bit quantize, JPEG compress
- `n2d/data/color.py` -- sRGB/linear conversions, color temp utilities
- `n2d/grid/fit.py` -- weighted least squares grid fitting (night_clean -> day)
  - 16x16 spatial x 8 log-luminance bins x 12 coefficients (3x4 affine)
  - Ridge regularization: sparse bins -> cell fit -> global fit
  - `torch.linalg.solve` on 2,048 small 4x4 systems per image
- `n2d/grid/slice.py` -- trilinear interpolation at full resolution
- `n2d/grid/guidance.py` -- blurred log-luminance for bin indexing
- `n2d/grid/normalize.py` -- compute dataset mean/std for grid coefficients
- `scripts/check_degradation.py` -- compare synthetic vs real night histograms
- `scripts/check_oracle.py` -- fitted grid PSNR ceiling (must be >30dB or grid is too coarse)

**Week 1 gates:**
- `check_oracle.py` shows PSNR > 30dB on fitted grids
- `check_degradation.py` shows synthetic night stats overlap real VisDrone night frames
- Both run on CPU, no GPU needed

## Week 2: Encoder + Velocity Transformer + Stage 1 Training (GPU)

**Files to write:**
- `n2d/models/encoder.py` -- DINOv3 ViT-S/16 with LoRA rank 8 on qkv+proj
  - Input: 256x256 night image -> 256 patch tokens (16x16 map)
  - Outputs: patch tokens (16x16x384), CLS token, patch-to-patch attention
  - Attention slicing: `attn[5:, 5:]` to skip CLS + 4 register tokens
- `n2d/models/noise_head.py` -- MLP on CLS token predicting (log a, log b) sensor noise params
- `n2d/models/velocity.py` -- velocity transformer (6 DiT blocks, hidden 256, 4 heads)
  - Input tokens: 256, each is 96 values (8 bins x 12 coefficients), standardized
  - Conditioning: DINO features (Linear 384->256) + log thumbnail (Linear 3->256) + 2D pos embed
  - adaLN-Zero for timestep conditioning
  - Attention bias: `QK^T/sqrt(d) + gate * log(A_dino + 1e-6)`, gate learned per block, init 0
  - Output: Linear 256->96, zero-initialized
  - ~7M params, ~1.2 GMACs per step
- `n2d/flow/loss.py` -- rectified flow loss:
  - `L_flow = ||v - (z1 - z0)||^2`, weighted per bin by pixel count
  - `L_render = 0.1 * L1(apply(z_hat_1, night_clean_256), day_256)`
  - `L_noise = MSE(predicted_log_ab, true_log_ab)`
- `n2d/flow/sampler.py` -- Euler sampler, 2 steps
- `scripts/train_flow.py` -- stage 1 training loop
  - AdamW, LR 2e-4 (LoRA 1e-4), batch 64 (or smaller for VRAM), bf16, EMA 0.999
  - ~100K steps
  - Grid fitting on-the-fly (fresh degradation each step)

**VRAM considerations for 5060 Ti (16GB):**
- DINOv3 ViT-S: ~21M params = ~84MB in bf16
- Velocity transformer: ~7M params = ~28MB in bf16
- Batch 64 at 256x256 might be tight -- start with batch 32, gradient accumulation 2x
- Grid fitting is scatter_add + linalg.solve, memory-light

**Week 2 gate:** Loss converges, visual quality at 2 NFE looks reasonable on synthetic val set

## Week 3: Denoiser + Uncertainty + Inference (GPU)

**Files to write:**
- `n2d/models/denoiser.py` -- lightweight full-resolution denoiser
  - Input: [transformed image (3ch), log-variance map (1ch)]
  - Architecture: 1x1 conv 4->24, 2x [depthwise 5x5 -> pointwise 24->24 -> GELU], 1x1 24->3
  - Output: `y + 0.1 * tanh(residual)` -- bounded, 9x9 receptive field
  - ~2.5K MACs/pixel, ~7.5 GMACs at 2000x1500
- `n2d/uncertainty/propagate.py` -- analytic uncertainty:
  - `Sigma_in = diag(a*x + b)` per pixel (Poisson-Gaussian noise model)
  - `Var_out = A * Sigma_in * A^T`
  - Scale by gamma curve slope for display units
  - Output: per-pixel standard deviation map
- `scripts/train_denoiser.py` -- stage 2 training
  - Input: fitted grid applied to noisy night image + log-variance map
  - Loss: `L1(out, day) + 0.01 * |residual|`
  - Random 256x256 crops at native resolution
  - Optional: finetune on flow-predicted grids (closes train/test gap)
  - ~1-3 A100-hours equivalent = ~5-15 hours on 5060 Ti
- `scripts/infer.py` -- full inference pipeline
  - Load night image, linearize, extract DINO features, run velocity transformer 2 steps
  - Slice grid at full res, apply transform, compute uncertainty, run denoiser, re-gamma

**Week 3 gate:** End-to-end inference works at full resolution, uncertainty maps are visually meaningful

## Week 4: Evaluation + Polish

**Files to write:**
- `n2d/eval/metrics.py` -- PSNR, SSIM, LPIPS on synthetic test pairs
  - Overall and on object crops using VisDrone bounding boxes
- `n2d/eval/calibration.py` -- predicted std vs actual error scatter plot
- `n2d/eval/visualize.py` -- image grids, uncertainty map overlays, before/after comparisons
- `scripts/evaluate.py` -- run full eval suite
- `scripts/benchmark_latency.py` -- time each component on 5060 Ti
- `README.md` -- installation, usage, results, limitations

**Evaluation plan (no baselines needed for a project repo):**
- PSNR/SSIM/LPIPS on synthetic test pairs
- Object-crop metrics using VisDrone detection boxes
- Uncertainty calibration: predicted std vs actual error
- Image grids on real VisDrone night frames (qualitative)
- Latency breakdown per component

**Honest limitations to document in README:**
- Output is relighting/enhancement, not true day translation (can't change sky/shadows)
- Denoiser can soften small objects (watch object-crop metrics)
- Sim-to-real gap: real night images have camera-specific processing
- Color errors possible (red car -> orange car)
- Grid cells are large at native res (~125x94 px per cell)

## Repo Structure

```
night2day/
├── configs/
│   ├── data.yaml
│   ├── flow.yaml
│   └── denoiser.yaml
├── n2d/
│   ├── data/
│   │   ├── visdrone.py
│   │   ├── degrade.py
│   │   └── color.py
│   ├── grid/
│   │   ├── fit.py
│   │   ├── slice.py
│   │   ├── guidance.py
│   │   └── normalize.py
│   ├── models/
│   │   ├── encoder.py
│   │   ├── velocity.py
│   │   ├── denoiser.py
│   │   └── noise_head.py
│   ├── flow/
│   │   ├── loss.py
│   │   └── sampler.py
│   ├── uncertainty/
│   │   └── propagate.py
│   └── eval/
│       ├── metrics.py
│       ├── calibration.py
│       └── visualize.py
├── scripts/
│   ├── check_degradation.py
│   ├── check_oracle.py
│   ├── train_flow.py
│   ├── train_denoiser.py
│   ├── infer.py
│   ├── evaluate.py
│   └── benchmark_latency.py
├── requirements.txt
└── README.md
```

## Dependencies
- `torch`, `torchvision` -- core
- `timm` -- DINOv3 ViT-S model loading
- `peft` -- LoRA for DINO fine-tuning
- `lpips` -- perceptual metrics
- `wandb` -- experiment tracking
- `einops` -- tensor ops
- `huggingface_hub` -- DINOv3 weight download (needs HF token + license acceptance)

## DINOv3 Access (do this today)
1. Go to the DINOv3 model page on Hugging Face
2. Accept Meta's license agreement
3. Generate an HF token with read access
4. `huggingface-cli login` on your machine
5. Note: DINOv3 weights have a Meta license (not Apache 2.0 like DINOv2). Read it before releasing checkpoints.

## Pre-GPU Sanity Checks (run before any training)
1. `python scripts/check_oracle.py` -- fitted grid PSNR ceiling. If below ~30dB, increase grid to 32x32 before training anything.
2. `python scripts/check_degradation.py` -- synthetic vs real night brightness/noise/color histograms. If they don't overlap, fix degradation params first.
