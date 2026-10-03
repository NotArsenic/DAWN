# Dawn

**Hallucination-free nighttime drone image enhancement via flow-matched bilateral grids.**

Dawn takes a night drone image, predicts a grid of affine color transforms using conditional flow matching, and applies them to the original pixels at full resolution. Because the output is always `A·x + b` of the input, the model **cannot add or remove objects by construction**. A per-pixel uncertainty map comes from analytic noise propagation at zero extra inference cost.

> **Honest framing.** This is nighttime image *enhancement*, not night-to-day *translation*. The output is a well-exposed, denoised night image. It does not invent sun shadows, blue sky, or daytime content. That's the no-hallucination guarantee doing its job.

## Method overview

```
night image (full res)
  ├── linearize → guidance (log-luminance bins)           full res, cheap
  └── resize 256² → DINOv3 ViT-S+/16 (+LoRA) → tokens    once, ~8.1 GMACs
                  └── 16×16 log thumbnail ─┘
                              │
         noise z₀ → velocity transformer × 2 steps → grid (16×16×8×12)
                              │
       slice grid at full res → out = A·x + b              ~0.5 GMACs
                              → Var = A·Σ·Aᵀ (uncertainty)  arithmetic
                              │
       bounded denoiser (9×9 receptive field) → day image   ~7.5 GMACs

Total: ~18 GMACs per frame. Real-time on A100, feasible on Jetson Orin.
```

### Key properties

- **No structural hallucination.** Output = affine transform of input pixels. Proved and audited.
- **Analytic uncertainty.** `Var_out = A · Σ_in · Aᵀ` — zero neural compute.
- **Fast.** ~18 GMACs at 2000×1500. Two flow steps.
- **Lightweight.** ~36M total parameters (29M DINOv3 ViT-S+ + 7M velocity transformer).

## Installation

```bash
git clone https://github.com/rzx01/dawn.git
cd dawn
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA ≥ 11.8

DINOv3 weights are gated. Accept the [license on HuggingFace](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m) and set:

```bash
export HF_TOKEN=your_token_here
```

The encoder (DINOv3 ViT-S+/16) loads through `timm` by default. To use the `transformers` backend instead, install the extra and set `encoder.backend: transformers` in `configs/flow.yaml`:

```bash
pip install -e ".[transformers]"
```

DINOv2 (Apache 2.0) works as a drop-in fallback if licensing is a concern.

## Data

We train on synthetic nighttime degradations of [VisDrone-DET](https://github.com/VisDrone/VisDrone-Dataset) daytime images.

```bash
# Download VisDrone-DET (extracts to <root>/VisDrone2019-DET-<split>/; data.yaml: visdrone.root)
python scripts/download_visdrone.py --root data/VisDrone_dataset --splits train val test

# Phase 1, slow, once: size, luminance, box count, noise level, highlight ratio per image
python scripts/prepare_visdrone.py stats
# Labelling sheets (one row per video sequence) + data/labels/visdrone_labels.csv to fill in
python scripts/prepare_visdrone.py label-sheets
# Fill it in from the terminal, one keypress per sequence (images inline in Ghostty / kitty / iTerm2)
python scripts/label_visdrone.py
# Phase 2, instant: hand labels -> day / night / dusk manifests (data/manifests, committed),
# a luminance histogram, a noise-vs-highlight scatter and contact sheets
python scripts/prepare_visdrone.py split
# Phase 3: 512-px PNG copies, area-averaged in linear space
python scripts/prepare_visdrone.py resize

# Verify degradation pipeline matches real night statistics
python scripts/check_degradation.py --data_dir data/visdrone
```

## Quick start

### CPU sanity checks (run before any GPU spend)

```bash
# 1. Check grid fitting ceiling (PSNR upper bound)
python scripts/check_oracle.py --data_dir data/visdrone --grid_size 16

# 2. Check synthetic vs real night statistics
python scripts/check_degradation.py --data_dir data/visdrone

# 3. Check the DINO encoder backend loads, LoRA attaches and a batch fits in VRAM
python scripts/smoke_dino.py --backend timm
```

### Tests

```bash
pytest
```

### Training

```bash
# Tiny end-to-end debug run (overlay config merged on top of flow.yaml)
python scripts/train_flow.py --config configs/flow.yaml --override configs/debug.yaml --device cpu

# Stage 1: flow model (encoder LoRA + velocity transformer)
python scripts/train_flow.py --config configs/flow.yaml

# Stage 2: denoiser
python scripts/train_denoiser.py --config configs/denoiser.yaml
```

### Inference

```bash
python scripts/infer.py \
    --input path/to/night_image.jpg \
    --checkpoint checkpoints/dawn.pt \
    --output_dir results/ \
    --save_uncertainty
```

### Evaluation

```bash
python scripts/evaluate.py \
    --config configs/eval.yaml \
    --checkpoint checkpoints/dawn.pt
```

## Project structure

```
dawn/
├── configs/                 # YAML configuration files
│   ├── data.yaml            # Dataset paths, degradation parameters
│   ├── flow.yaml            # Stage 1: flow model training
│   ├── denoiser.yaml        # Stage 2: denoiser training
│   ├── eval.yaml            # Evaluation settings
│   └── debug.yaml           # Overlay for a tiny end-to-end debug run
├── docs/
│   └── conventions.md       # Grid layout, coordinates, color space: read before touching grid code
├── dawn/                    # Main package
│   ├── data/                # Dataset and degradation
│   │   ├── visdrone.py      # VisDrone layout, file formats, box filter, dataset (3 modes)
│   │   ├── prepare.py       # The three prepare_visdrone.py phases: stats, split, resize
│   │   ├── resample.py      # area_downsample: the only downsampling in the project
│   │   ├── degrade.py       # Synthetic night degradation: composes lighting + camera
│   │   ├── lighting.py      # Moonlit / streetlit scene illumination
│   │   ├── camera.py        # Sensor noise + ISP (gamma, 8-bit, JPEG)
│   │   └── color.py         # sRGB ↔ linear, luminance, color temperature utilities
│   ├── grid/                # Bilateral grid operations
│   │   ├── layout.py        # Canonical grid layout constants and token reshapes
│   │   ├── fit.py           # Weighted least-squares grid fitting
│   │   ├── slice.py         # Trilinear grid slicing at full resolution
│   │   ├── guidance.py      # Log-luminance guidance computation
│   │   └── normalize.py     # Coefficient standardization
│   ├── models/              # Neural network modules
│   │   ├── encoder.py       # DINOv3 ViT-S+/16 with peft LoRA (timm or transformers backend)
│   │   ├── velocity.py      # DiT-style velocity transformer
│   │   ├── denoiser.py      # Bounded full-resolution denoiser
│   │   └── noise_head.py    # Sensor noise parameter predictor
│   ├── flow/                # Flow matching training and sampling
│   │   ├── loss.py          # Rectified flow loss + apply, noise and align losses
│   │   └── sampler.py       # Euler ODE solver (1–4 steps)
│   ├── uncertainty/         # Uncertainty estimation
│   │   └── propagate.py     # Analytic Var = A·Σ·Aᵀ propagation
│   └── eval/                # Evaluation utilities
│       ├── metrics.py       # PSNR, SSIM, LPIPS
│       ├── calibration.py   # Uncertainty calibration analysis
│       └── visualize.py     # Image grids, uncertainty maps
├── scripts/                 # Runnable scripts
│   ├── check_degradation.py # CPU: synthetic vs real night comparison
│   ├── check_oracle.py      # CPU: grid fitting PSNR ceiling
│   ├── download_visdrone.py # Dataset download helper
│   ├── prepare_visdrone.py  # stats CSV, luminance split into manifests, 512-px PNG copies
│   ├── smoke_dino.py        # DINO backend / LoRA / VRAM smoke test
│   ├── train_flow.py        # Stage 1 training
│   ├── train_denoiser.py    # Stage 2 training
│   ├── infer.py             # Single-image or batch inference
│   ├── evaluate.py          # Full evaluation pipeline
│   └── benchmark_latency.py # Latency profiling
├── tests/                   # pytest suite; scaffold-stub tests skip until implemented
├── pyproject.toml
└── README.md
```

## Training cost

| Phase | A100-hours |
|-------|-----------|
| Stage 1 (flow model) | 6–10 |
| Stage 2 (denoiser) | 1–3 |
| Tuning and ablations | ~5–10 |
| **Total** | **~12–23** |

## Known limitations

1. **Not night-to-day translation.** Output is enhanced night, not synthetic daytime. No sun shadows, no blue sky. This is the no-hallucination guarantee, not a bug.
2. **Synthetic training data.** Trains on degraded VisDrone daytime images. The Poisson-Gaussian noise model approximates real sensor noise. Synthetic metrics will be optimistic; real-night results are the honest test.
3. **Color ambiguity.** The affine transform can shift hues (a red car might appear orange). Flow sample diversity captures this ambiguity.
4. **Denoiser can soften small objects.** Bounded residual (9×9 receptive field) prevents structural changes but may blur fine detail.
5. **Grid resolution.** Lighting that changes faster than a grid cell (~125×94 px at 2000×1500) is handled only through brightness bins.

## Citation

```bibtex
@misc{dawn2026,
  title   = {Dawn: Hallucination-Free Nighttime Drone Image Enhancement
             via Flow-Matched Bilateral Grids},
  author  = {},
  year    = {2026},
  url     = {https://github.com/rzx01/dawn}
}
```

## License

MIT
