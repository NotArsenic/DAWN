# Dawn conventions

The single source of truth for every layout, coordinate system, color space and definition that more than one module relies on. **If code and this file disagree, this file wins**: fix the code, or change this file in the same commit. Read it before touching `dawn/grid/`, `dawn/models/velocity.py`, `dawn/flow/`, `dawn/data/` or `dawn/uncertainty/`.

Notation: `B` batch, `H, W` image rows/columns, `K` brightness bins (8), `gy, gx` grid cell rows/columns (16, 16). Sizes come from `configs/flow.yaml: grid.*`; the coefficient constants and token reshapes live in `dawn/grid/layout.py`.

## At a glance

| Topic | Convention |
|---|---|
| Images | `(B, 3, H, W)`, RGB, float32, **linear**, `[0, 1]`. sRGB only at the boundaries in §1 |
| Grid (raw *and* standardized) | `(B, 12, K, gy, gx)` = `(coefficient, bin, cell row, cell col)` |
| 12 coefficients | `c = 3i + j` is `M[i, j]`, `c = 9 + i` is `t[i]`; `out_i = Σ_j M[i,j]·x_j + t_i` |
| Velocity tokens | `(B, 256, 96)`; token `n = y·gx + x`, value `v = k·12 + c` |
| Grid coordinates | node values sit at cell **centers**; coordinates outside the outer centers clamp |
| Guidance | `compute_guidance`: luminance → 5×5 box blur (replicate pad) → clamp ≥ 1e-4 → log10 → `(· + 4) / 4` |
| Raw vs standardized | stats `(12, K)`; pixels see raw, flow sees standardized; convert only via `normalize.py` |
| Noise | `Var[y \| x] = a·x + b`; `b` is a **variance** |
| Flow time | `t = 0` noise, `t = 1` data |

## 1. Color space

- Inside the pipeline every image is **linear RGB, float32, `[0, 1]`**, channel order R, G, B, layout `(B, 3, H, W)` (`(3, H, W)` for one sample). That covers `day`, `night_clean`, `night`, the output of `apply_affine`, and the denoiser's input and output. Clamp to `[0, 1]` after anything that can leave the range (noise, `apply_affine`, the denoiser).
- sRGB exists only at these boundaries. Nothing else calls `srgb_to_linear` / `linear_to_srgb`:
  1. File input: decode, then `srgb_to_linear` (dataset, `infer.py`).
  2. File or display output: `linear_to_srgb`, then save (`infer.py`, `visualize.py`, scripts).
  3. `camera.py`: it simulates a camera ISP (gamma → 8-bit → JPEG) and ends by decoding back to linear, so `night` is exactly what a linearized real night JPEG looks like.
  4. `DINOEncoder.forward`: DINO is pretrained on sRGB, so the encoder takes **linear** RGB and converts to sRGB plus ImageNet mean/std itself. Callers never pre-convert.
  5. Display-unit uncertainty (`srgb_gamma_slope`).
- All image tensors and grids are float32. bf16 autocast is for network forward passes only; grid fitting, slicing, `apply_affine` and variance propagation run in float32 (`torch.linalg.solve` in `fit_grid` needs it).

## 2. Grid layout

### 2.1 Canonical shape: `(B, 12, K, gy, gx)`

| axis | name | size | meaning |
|---|---|---|---|
| 0 | `b` | | batch |
| 1 | `c` | 12 | affine coefficient (§2.2) |
| 2 | `k` | K = 8 | brightness bin; 0 is the darkest (lowest guidance) |
| 3 | `y` | gy = 16 | cell row; 0 is the top |
| 4 | `x` | gx = 16 | cell column; 0 is the left |

This is PyTorch's 5-D `(N, C, D, H, W)` order (`K` is the depth axis), so `F.grid_sample` slices it directly (§3). Raw and standardized grids use the same layout. Per-pixel coefficient maps (output of `slice_grid`) are `(B, 12, H, W)` with the same channel order.

### 2.2 Coefficient order

| `c` | 0 1 2 | 3 4 5 | 6 7 8 | 9 10 11 |
|---|---|---|---|---|
| meaning | `M[0,0] M[0,1] M[0,2]` | `M[1,0] M[1,1] M[1,2]` | `M[2,0] M[2,1] M[2,2]` | `t[0] t[1] t[2]` |

`out_i = Σ_j M[i,j]·x_j + t_i`, `i` = output channel (R, G, B = 0, 1, 2), `j` = input channel. So `M = coeffs[:9].reshape(3, 3)` (row = output channel) and `t = coeffs[9:12]`: nine matrix entries row by row, then the three offsets. This is **not** the row-major flattening of the 3×4 matrix `[M | t]`. Identity is `[1,0,0, 0,1,0, 0,0,1, 0,0,0]` (`layout.IDENTITY_COEFFS`).

Least-squares trap: with `Φ = [x, 1]` (N×4) and `Y` (N×3), `X = (ΦᵀWΦ + λI)⁻¹ ΦᵀWY` is 4×3, so `M = X[:3].T` and `t = X[3]`. Transpose before flattening.

### 2.3 Tokens for the velocity transformer: `(B, 256, 96)`

```python
tokens = rearrange(grid,   "b c k y x -> b (y x) (k c)")                        # (B, 256, 96)
grid   = rearrange(tokens, "b (y x) (k c) -> b c k y x", y=gy, x=gx, k=K, c=12)
```

- Token index `n = y·gx + x` (row-major). This is also the order of DINO's 16×16 patch tokens and of the 16×16 thumbnail, so token `n`, patch `n` and cell `n` cover the same image region.
- Value index inside a token `v = k·12 + c` (bin-major, coefficient-minor).
- Implemented once, in `dawn/grid/layout.py` (`grid_to_tokens`, `tokens_to_grid`). Only `VelocityTransformer` works in tokens; `z_0`, `z_1`, `z_t`, `v`, the samplers and the losses all stay in the 5-D layout.
- The one-to-one match of DINO patches and grid cells holds for a 16×16 grid and a 256×256 encoder input. A 32×32 grid (the oracle-check fallback) means 1,024 tokens, so revisit the conditioning before switching.

### 2.4 Node weights ("pixel counts"): `(B, K, gy, gx)`

For each node, the sum of the trilinear weights of every pixel that contributes to it. Returned by `fit_grid`, used by the flow loss's bin-count weighting (broadcast over the coefficient axis with `.unsqueeze(1)`).

## 3. Coordinates: cell centers

Grid values sit at cell **centers** (PyTorch's `align_corners=False`). For pixel `(r, c)` of an `H×W` image (integer indices) with guidance `g ∈ [0, 1]`:

```
yf = (r + 0.5) * gy / H - 0.5        # node i sits at coordinate i
xf = (c + 0.5) * gx / W - 0.5
zf = g * K - 0.5
```

Clamp each coordinate to `[0, N-1]` (`N` = `gy`, `gx`, `K`), then `i0 = floor(f)`, `i1 = min(i0 + 1, N-1)`, `w1 = f - i0`, `w0 = 1 - w1`. A pixel's 8 corner nodes get the weights `w_y · w_x · w_z`, which sum to 1. Outside the outer cell centers the edge node wins entirely (replicate).

- Bin `k` represents guidance `(k + 0.5) / K`. Never bin with `floor(g·K)` or `g·(K-1)`.
- The grid spans whichever image tensor it was fit on or sliced on (fractions of `H` and `W`), so a grid fit at 256² slices at 2000×1500. A training crop is its own image.
- The equivalent `F.grid_sample(grid, coords, mode="bilinear", padding_mode="border", align_corners=False)` (`"bilinear"` is trilinear for 5-D input) takes `coords = (x_norm, y_norm, z_norm) = ((2c+1)/W - 1, (2r+1)/H - 1, 2g - 1)`: the image's own normalized pixel centers, and `2g - 1` for guidance.
- **Single implementation:** `dawn.grid.slice.trilinear_weights(guidance, (K, gy, gx)) -> (node_index, weight)`, both `(B, 8, H, W)`. `node_index` is int64, flat over `(K, gy, gx)` as `k·gy·gx + y·gx + x`; `weight` is float32 and sums to 1 over dim 1; corner `n = 4·dz + 2·dy + dx`. `slice_grid` may use `F.grid_sample` for speed at full resolution, as long as `tests/test_grid_rules.py` (agreement with `trilinear_weights`) passes.

## 4. Guidance

`dawn.grid.guidance.compute_guidance(image_linear) -> (B, 1, H, W)` in `[0, 1]`:

1. `Y = 0.2126 R + 0.7152 G + 0.0722 B` (Rec. 709 weights, on **linear** RGB).
2. 5×5 box blur (mean filter, stride 1, same size) with **replicate** padding. Zero padding darkens the borders and puts border pixels in the wrong bin.
3. `Y = clamp(Y, 1e-4, 1)`.
4. `log10(Y)`.
5. Map `[-4, 0]` to `[0, 1]`: `g = (log10(Y) + 4) / 4`.

Guidance is computed from the image that is being indexed. Fitting, rendering in the flow loss, and `check_oracle.py` use `compute_guidance(night_clean)`; inference and stage-2 training use `compute_guidance(night)`.

## 5. Raw vs standardized grids

One layout (§2.1), two spaces:

| | raw | standardized (`_std`) |
|---|---|---|
| what | physical affine coefficients | `(raw - mean) / std` per (coefficient, bin) |
| produced by | `fit_grid` | `standardize(raw)` |
| consumed by | `slice_grid`, `apply_affine`, `propagate_variance`, rendering, the denoiser: anything touching pixels | `z_0`, `z_1`, `z_t`, `v`, `VelocityTransformer`, the samplers, the velocity loss |
| names | `grid`, `grid_raw`, `coeffs` (per pixel) | `grid_std`, `z_*`, `v` |

- Stats `mean` and `std` are float32 `(12, K)` (same `(c, k)` order as the grid) and broadcast as `(1, 12, K, 1, 1)`. They are computed once over raw fitted grids (`compute_grid_stats`) and cached at `grid.stats_cache`. **A checkpoint embeds the stats it was trained with**; inference reads them from the checkpoint, never from the cache file.
- Conversion happens only through `standardize` / `unstandardize` (`dawn/grid/normalize.py`), and only here:
  1. Training target: `z_1 = standardize(fit_grid(...)[0])`.
  2. Render loss: `z1_hat = z_t + (1 - t)·v_pred` is standardized, so `unstandardize`, then `slice_grid`, then `apply_affine`, then compare with `day`. `FlowMatchingLoss` only receives the rendered image (`applied_pred`); it never sees stats.
  3. Inference: the sampler output is standardized; `unstandardize` once, then slice.
  4. Stage 2: fitted grids are already raw; flow-predicted grids go through `unstandardize` first.
- Nothing else multiplies or divides by stats. A docstring that takes a grid states raw or standardized.
- Flow time: `t = 0` is noise, `t = 1` is data. `z_t = (1 - t)·z_0 + t·z_1`, `v_target = z_1 - z_0`, samplers integrate 0 → 1.

## 6. Noise model

`Var[y | x] = a·x + b` per pixel and channel, in linear `[0, 1]` units. `a` is the shot-noise gain, `b` the read-noise **variance** (σ², not a standard deviation). Both are positive. `camera.py` produces `(a, b)` as floats; the dataset exposes `noise_params = {"a": ..., "b": ...}`; `NoiseHead` predicts the natural logs `(log a, log b)`; `sensor_variance` takes raw `a`, `b`.

## 7. Shapes at module boundaries

| API | In → out |
|---|---|
| `compute_guidance(image_linear)` | `(B,3,H,W)` → `(B,1,H,W)` |
| `trilinear_weights(guidance, (K,gy,gx))` | `(B,1,H,W)` → `node_index`, `weight`: `(B,8,H,W)` |
| `fit_grid(night_clean, day, guidance)` | `(B,3,H,W)` ×2, `(B,1,H,W)` → raw `grid (B,12,K,gy,gx)`, `node_weights (B,K,gy,gx)` |
| `slice_grid(grid, guidance)` | raw grid, `(B,1,H,W)` → `coeffs (B,12,H,W)` |
| `apply_affine(coeffs, image)` | `(B,12,H,W)`, `(B,3,H,W)` → `(B,3,H,W)`, clamped |
| `standardize` / `unstandardize(grid, mean, std)` | grid `(B,12,K,gy,gx)`, stats `(12,K)` |
| `DINOEncoder.forward(image_linear)` | `(B,3,256,256)` → `patch_tokens (B,256,384)`, `cls_token (B,384)`, and with `return_attn` also `patch_attn (B,256,256)`: last layer, head-averaged, CLS and the 4 register tokens stripped |
| `VelocityTransformer(z_t, t, dino_features, dino_attn)` | `(B,12,K,gy,gx)`, `(B,)`, `(B,256,384)`, `(B,256,256)` → `v (B,12,K,gy,gx)` |
| `euler_sample(velocity_fn, z_0, n)` | `(B,12,K,gy,gx)` → same, standardized |
| `NoiseHead(dino_features)` | `(B,256,384)` → `(B,2)` |
| `propagate_variance(coeffs, sigma_sq_in)` | `(B,12,H,W)`, `(B,3,H,W)` → `(B,3,H,W)` |
| `BoundedDenoiser(enhanced, log_variance)` | `(B,3,H,W)` linear, `(B,1,H,W)` → `(B,3,H,W)` linear |
| `VisDroneDataset[i]` | dict: `night`, `night_clean`, `day` (each `(3,h,w)` linear), `noise_params`, `meta` |

## 8. Rules

1. **One trilinear implementation.** `fit.py` gets node indices and weights from `slice.trilinear_weights`. It never computes `floor`, corner weights or coordinates itself.
2. **One guidance function.** Fitting, slicing, rendering, inference, stage 2 and every script call `compute_guidance`. No inline luminance or `log10` code anywhere else.
3. **One layout module.** Token reshapes and coefficient-order constants come from `dawn/grid/layout.py`; no ad-hoc `permute` / `view` of grids.
4. **One conversion point for standardization** (§5).
5. **Conventions change here first.** If one has to change, edit this file in the same commit as the code and update the affected docstrings.

## 9. Known limitations

- The guidance blur is a fixed 5×5 **pixels**. At 256² training 5 px is 2% of the width; at 2000×1500 inference it is 0.25%, so the effective smoothing differs. If bins flicker at full resolution, scale the kernel with resolution and update §4.
- The 16×16 grid, 256-token layout and DINO patch correspondence in §2.3 are coupled; changing the grid size means changing the conditioning.
