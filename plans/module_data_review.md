# Module/Data Branch — What Happened & What's Next

## Summary of Changes (7 commits, scaffold → Module/Data)

This branch is a **complete implementation of Week 1**: the data infrastructure layer. It replaced the original scaffold stubs with production-quality code, added a conventions document, a comprehensive test suite, and hand-labelled 272 VisDrone video sequences.

---

## Commit-by-Commit Breakdown

### 1. `b98f337` — Modified Scaffolding
- Restructured the project fundamentals:
  - Renamed concept from "night-to-day translation" to "nighttime enhancement" (matching the no-hallucination decision).
  - Fixed UTF-16 encoding issue from the original scaffold (all files now UTF-8 LF).
  - Rewrote `CLAUDE.md` as the authoritative AI-facing project guide.
  - Created `docs/conventions.md` — the single source of truth for every layout, coordinate system, color space, and shape convention. This is the project's most important document; code that disagrees with it is wrong.
  - Updated `pyproject.toml` (added `peft`, `omegaconf` deps; `transformers` as optional).
  - Updated all config YAMLs to match the new conventions (grid shape `(B, 12, K, gy, gx)`, degradation split into lighting + camera subsections, VisDrone layout paths).

### 2. `4954d1f` — Graphify init and ignored
- Updated `.gitignore` for the data directory structure.
- Cleaned up `__init__.py` files.

### 3. `8200095` — Implement: color.py and supplementary files
- **`dawn/data/color.py`** — Fully implemented (not a stub anymore):
  - `srgb_to_linear` / `linear_to_srgb` (IEC 61966-2-1, exact piecewise).
  - `srgb8_to_linear` — 256-entry LUT for uint8 input (the file-input boundary).
  - `linear_to_srgb8` — the file-output boundary.
  - `luminance()` — Rec. 709 weights, keeps channel dim.
  - `color_temp_to_illuminant_rgb()` — Planckian locus via CIE 1931 chromaticity, XYZ→sRGB matrix. Proper physics, not Tanner Helland approximation.
  - `color_temp_to_gains()` — per-channel gains relative to 6500K reference.
  - `srgb_gamma_slope()` — for variance propagation to display units.
- **`dawn/data/resample.py`** — Fully implemented:
  - `area_downsample()` — exact area-averaging via overlap matrix multiplication. Works for any factor (integer or not), preserves mean, differentiable.
  - `fit_long_side()` — aspect-preserving resize, never upsamples.
- **`dawn/data/lighting.py`** — New module (stub). Split from the old monolithic `degrade.py`. Handles scene illumination: moonlit vs streetlit regimes, exposure, color temp, light pools, halos. Returns `night_clean`.
- **`dawn/data/camera.py`** — New module (stub). The other half of degradation: Poisson-Gaussian sensor noise, gamma, 8-bit quantization, JPEG compression. Returns `night`.
- **`dawn/data/degrade.py`** — Refactored to be a thin composition layer that calls `lighting.py` then `camera.py`. No longer monolithic.
- **`dawn/grid/layout.py`** — New module, fully implemented:
  - Canonical grid shape constants (`NUM_COEFFS`, `MATRIX`, `OFFSET`, `IDENTITY_COEFFS`).
  - `grid_to_tokens()` / `tokens_to_grid()` — the one-and-only reshape between 5D grid `(B, 12, K, gy, gx)` and velocity transformer tokens `(B, 256, 96)`.
- Updated stubs across `dawn/grid/`, `dawn/models/`, `dawn/flow/`, `dawn/uncertainty/`, `dawn/eval/` to match the new conventions (shape signatures, docstrings referencing `docs/conventions.md`).
- Added `configs/debug.yaml` — overlay config for tiny CPU end-to-end runs (16 images, 50 steps, no wandb).

### 4. `1472499` — Added and ignored VisDrone dataset
- Downloaded the raw VisDrone-DET dataset to `data/VisDrone_dataset/`.
- Updated `.gitignore` to exclude it (large binary files).

### 5. `8e08260` — Updated gitignore
- Refined ignore patterns for data artifacts.

### 6. `935b345` — Implement VisDrone dataset loading, multi-phase preparation pipeline, and tests
**The big commit.** Implements the full data preparation pipeline and dataset class:

- **`dawn/data/visdrone.py`** — Fully implemented (~425 lines):
  - `VisDroneLayout` dataclass: maps the 4-directory VisDrone layout (raw, stats, manifests, prepared) from `configs/data.yaml`.
  - `ImageStats` dataclass: per-image metadata (luminance percentiles, box count, noise level, highlight ratio).
  - `BoxFilter`: annotation filtering (drop categories 0/11, score-0 boxes, min-size 16px).
  - `read_annotations()`: VisDrone-DET annotation parser.
  - `read_stats_csv()` / `write_stats_csv()`: stats CSV I/O.
  - `read_labels()` / `set_labels()` / `read_manifest()` / `write_manifest()`: file format handlers.
  - `sequence_id()` / `is_still_collection()`: video-sequence logic (99999xx = stills).
  - `VisDroneDataset(Dataset)`: three modes:
    - `lowres`: 512-px copies, variable aspect, optional RAM cache. For flow training.
    - `native_crops`: random 256×256 crops of raw images. For denoiser training.
    - `eval`: whole raw image + filtered detection boxes. For evaluation.
  - All images returned as linear RGB float32 [0,1] — sRGB never leaves this module.

- **`dawn/data/prepare.py`** — Fully implemented (~600 lines):
  - Phase 1 (`stats`): per-image luminance, noise level (high-pass MAD-based), highlight ratio (p99.5/median). Multiprocessing via `ProcessPoolExecutor`.
  - Phase 2 (`split`): hand labels → manifests. Luminance histogram, noise-vs-highlight scatter plot, contact sheets.
  - Phase 3 (`resize`): area-downsample in linear space → 8-bit sRGB PNG copies.
  - Labelling support: `write_labels_template()`, `save_sequence_sheets()` (darkest/median/brightest frame per video), `preview_image()`.
  - Beautiful matplotlib charts with a consistent color palette.

- **`scripts/prepare_visdrone.py`** — CLI for all 3 phases: `stats`, `label-sheets`, `split`, `resize`.
- **`scripts/label_visdrone.py`** — Interactive terminal labeller (~200 lines):
  - One-keypress-per-sequence workflow (d/n/u/x/s/b/o/q).
  - Inline image previews via kitty, iTerm2, and VS Code terminal protocols.
  - Saves every keypress immediately. Pre-fetches next image in background.
- **`scripts/smoke_dino.py`** — Stub for verifying DINOv3/v2 encoder backend.

- **Tests** — 1,582 lines across 16 test files:
  - `tests/fake_visdrone.py`: generates a fake VisDrone-DET directory structure for testing.
  - `tests/conftest.py`: shared fixtures.
  - `tests/helpers.py`: `skip_until_implemented` decorator for stub modules.
  - `test_color.py` (281 lines): sRGB roundtrips, LUT accuracy, luminance, color temp physics, gamma slope.
  - `test_resample.py` (85 lines): mean preservation, integer/fractional factors, dtype/device.
  - `test_visdrone.py` (362 lines): layout paths, annotations, box filtering, dataset modes, stats CSV, manifests, labels.
  - `test_prepare.py` (337 lines): all 3 phases, label resolution, noise level, thumbnail generation.
  - `test_configs.py` (71 lines): config consistency (debug.yaml keys ⊆ flow.yaml ∪ data.yaml).
  - `test_grid_rules.py` (111 lines): layout constants, token reshapes, identity coefficients.
  - `test_guidance.py`, `test_layout.py`, `test_normalize.py`, `test_propagate.py`: convention enforcement tests.
  - `test_degrade_contract.py`: shape/type contracts for degradation (skips until lighting/camera implemented).
  - `test_imports.py`: every subpackage imports cleanly.

### 7. `b05d356` — Modified labels for visdrone
- **`data/labels/visdrone_labels.csv`** — 272 video sequences hand-labelled:
  - Night: ~55 sequences (darkest, high noise, high highlight ratio).
  - Dusk: ~12 sequences (intermediate brightness, moderate noise).
  - Day: ~205 sequences (bright, low noise).
  - Sorted darkest-first with helper columns (sheet_row, split, frames, log10_lum, noise, highlight_ratio).

---

## Architecture Decisions Made in This Branch

1. **Degradation split into lighting + camera** — cleaner separation of concerns. Lighting produces `night_clean` (grid fitting target), camera adds noise (model input).
2. **Grid shape `(B, 12, K, gy, gx)`** — matches PyTorch's 5D convention for `F.grid_sample`. Not the `(B, gy, gx, K, 12)` from the original scaffold.
3. **Tokens `(B, 256, 96)` = `(B, gy*gx, K*12)`** — bin-major, coefficient-minor. One token per spatial cell, each holding all 8 bins × 12 coefficients.
4. **Area downsampling as the only resize** — averaging in linear space, exact for any factor, preserves mean. No PIL resize or `F.interpolate` anywhere.
5. **sRGB only at boundaries** — the entire pipeline is linear RGB internally. DINO encoder handles its own sRGB conversion.
6. **Three dataset modes** — `lowres` (flow training), `native_crops` (denoiser), `eval` (metrics + boxes). Not three separate dataset classes.
7. **`noise_b` is a variance, not std** — consistent with `Var = a*x + b` everywhere.

---

## What's Implemented vs Still Stub

### ✅ Fully Implemented
| Module | Status |
|--------|--------|
| `dawn/data/color.py` | Complete — all color space conversions, color temp physics |
| `dawn/data/resample.py` | Complete — area downsampling |
| `dawn/data/visdrone.py` | Complete — dataset class, all file formats, 3 modes |
| `dawn/data/prepare.py` | Complete — all 3 preparation phases, labelling support |
| `dawn/data/degrade.py` | Thin composition layer (delegates to lighting + camera) |
| `dawn/grid/layout.py` | Complete — canonical constants and token reshapes |
| `dawn/grid/guidance.py` | Complete — log-luminance guidance computation |
| `docs/conventions.md` | Complete — authoritative reference |
| `scripts/prepare_visdrone.py` | Complete — CLI for data prep |
| `scripts/label_visdrone.py` | Complete — interactive labeller |
| All test files | Complete — 1,582 lines |
| Hand labels | 272 sequences labelled |

### 🔲 Still Stubs (raise NotImplementedError)
| Module | What it needs |
|--------|---------------|
| `dawn/data/lighting.py` | `sample_lighting_params`, `apply_lighting` |
| `dawn/data/camera.py` | `sample_camera_params`, `apply_camera` |
| `dawn/grid/fit.py` | `fit_grid`, weighted least squares |
| `dawn/grid/slice.py` | `slice_grid`, `trilinear_weights`, `apply_affine` |
| `dawn/grid/normalize.py` | `compute_grid_stats` (the `standardize`/`unstandardize` helpers are done) |
| `dawn/models/encoder.py` | DINOv3 ViT-S backbone + LoRA |
| `dawn/models/velocity.py` | Velocity transformer (DiT blocks) |
| `dawn/models/denoiser.py` | Bounded residual denoiser |
| `dawn/models/noise_head.py` | Noise parameter MLP |
| `dawn/flow/loss.py` | Flow matching loss |
| `dawn/flow/sampler.py` | Euler/midpoint ODE samplers |
| `dawn/uncertainty/propagate.py` | Analytic variance propagation |
| `dawn/eval/metrics.py` | PSNR, SSIM, LPIPS |
| `dawn/eval/calibration.py` | ECE, calibration curves |
| `dawn/eval/visualize.py` | Comparison grids, uncertainty coloring |
| `scripts/train_flow.py` | Stage 1 training loop |
| `scripts/train_denoiser.py` | Stage 2 training loop |
| `scripts/infer.py` | Full inference pipeline |
| `scripts/evaluate.py` | Evaluation suite |
| `scripts/smoke_dino.py` | DINO backend verification |

---

## Recommended Next Steps

### Immediate (complete Week 1 gates)
1. **Implement `lighting.py`** — `sample_lighting_params` and `apply_lighting`. The config structure and `LightingParams` dataclass are already defined. Needs: exposure reduction, illumination field (4×4 grid + Gaussian light pools), color cast via `color_temp_to_gains`, halo around lights. Return `night_clean` in linear [0,1].
2. **Implement `camera.py`** — `sample_camera_params` and `apply_camera`. Needs: Poisson-Gaussian noise (`Var = a*x + b`), sRGB gamma, 8-bit quantize, JPEG roundtrip, decode back to linear. Handle the resolution scaling (`a/n, b/n` for binned pixels).
3. **Implement `grid/fit.py`** — Weighted least-squares grid fitting. Use `slice.trilinear_weights` for node weights (don't reimplement). Solve 2,048 small 4×4 systems via `torch.linalg.solve`. Ridge regularization: sparse bin → cell fit → global fit.
4. **Implement `grid/slice.py`** — `trilinear_weights` (the canonical implementation all other grid code depends on), `slice_grid`, `apply_affine`. Follow the coordinate convention in `docs/conventions.md` §3 exactly.
5. **Run `check_oracle.py`** — fitted grid PSNR must be >30 dB or increase grid to 32×32.
6. **Run `check_degradation.py`** — synthetic night stats must overlap real VisDrone night frames.

### Then (Week 2)
7. Implement `models/encoder.py` — DINOv3 ViT-S+ with `peft` LoRA.
8. Implement `models/velocity.py` — 6 DiT blocks, adaLN-Zero, DINO attention bias.
9. Implement `flow/loss.py` and `flow/sampler.py`.
10. Implement `scripts/train_flow.py` — Stage 1 training loop.

### Open Decisions (from CLAUDE.md)
- **`flow.warmup_steps`**: What does the 5K-step warmup gate? (render-loss ramp-in, LoRA unfreeze, or L_align ramp).
- **Still collections**: The 49 `99999xx` groups (6,497 images, day+night interleaved) need per-image labels. Strategy: pre-sort by noise/highlight features, or label by hand?
