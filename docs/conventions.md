# Dawn conventions

The single source of truth for every layout, coordinate system, color space and definition that more than one module relies on. **If code and this file disagree, this file wins**: fix the code, or change this file in the same commit. Read it before touching `dawn/grid/`, `dawn/models/velocity.py`, `dawn/flow/`, `dawn/data/` or `dawn/uncertainty/`.

> **Status, 2026-10-06.** The project moved from synthetic night RGB (VisDrone) to real **day thermal → RGB** (DroneVehicle). Current: §1 (RGB, thermal and CIELAB conventions), the image rows of "At a glance", the data rows of §7, §4a, §8 rules 7 and 8, and **all of §10**. §2, §3, §4 (log-luminance guidance), §5, §6 and the model rows of §7 describe the **parked** old-regime grid, noise and model code (`dawn/grid/`, `dawn/models/`, `dawn/flow/`, `dawn/uncertainty/`) until the flow/model phase rewrites them. They do not apply to new data code, and the plan (`docs/Dawn Night-Vision to Day Colorization for Drone Imagery.md`) wins over them where the two disagree.

Notation: `B` batch, `H, W` image rows/columns, `K` brightness bins (8), `gy, gx` grid cell rows/columns (16, 16). Sizes come from `configs/flow.yaml: grid.*`; the coefficient constants and token reshapes live in `dawn/grid/layout.py`.

## At a glance

| Topic | Convention |
|---|---|
| RGB images | `(B, 3, H, W)`, RGB, float32, **linear**, `[0, 1]`. sRGB only at the boundaries in §1 |
| Thermal images | `(B, 1, H, W)`, float32, `code / 255` in `[0, 1]`. **Not sRGB**: never through `srgb_to_linear` (§1) |
| Frames | raw 840×712 with a 100-px white border; content crop `[100:612, 100:740]` = **640×512** (§10.1) |
| Pixel coordinates | pixel-index: x = column, y = row, pixel `(r, c)` has its centre at `(c, r)`; boxes and warps share them (§10.2) |
| Warp | thermal onto RGB, bilinear inverse mapping, `T: thermal → RGB`; RGB is only cropped, never resampled (§10.2) |
| Masks | `valid` excluded from **every** loss, `moving` from the **pixel** losses (§10.2) |
| Image IDs | `f"{split}_{stem}"`, e.g. `train_00001`; the three manifests are `{train,val,test}_pairs` (§10.1) |
| Grid (raw *and* standardized, parked) | `(B, 12, K, gy, gx)` = `(coefficient, bin, cell row, cell col)` |
| 12 coefficients (parked) | `c = 3i + j` is `M[i, j]`, `c = 9 + i` is `t[i]`; `out_i = Σ_j M[i,j]·x_j + t_i` |
| Velocity tokens (parked) | `(B, 256, 96)`; token `n = y·gx + x`, value `v = k·12 + c` |
| Grid coordinates (parked) | node values sit at cell **centers**; coordinates outside the outer centers clamp |
| Guidance (parked) | `compute_guidance`: luminance → 5×5 box blur (replicate pad) → clamp ≥ 1e-4 → log10 → `(· + 4) / 4` |
| Raw vs standardized (parked) | stats `(12, K)`; pixels see raw, flow sees standardized; convert only via `normalize.py` |
| Downsampling | always area averaging of **linear** values, only through `area_downsample` (§4a) |
| Noise (parked) | `Var[y \| x] = a·x + b`; `b` is a **variance**; `(a, b)` at native resolution, `(a/n, b/n)` after averaging `n` native pixels |
| Flow time | `t = 0` noise, `t = 1` data |

## 1. Color space

- **RGB** inside the pipeline is **linear RGB, float32, `[0, 1]`**, channel order R, G, B, layout `(B, 3, H, W)` (`(3, H, W)` for one sample). That covers the aligned `rgb` target and anything rendered from it. Clamp to `[0, 1]` after anything that can leave the range (an affine render, CIELAB back to RGB).
- **Thermal is not sRGB.** Thermal values are the camera's 8-bit codes: `thermal = code / 255`, float32 in `[0, 1]`, one channel, `(1, H, W)`. Thermal **never** goes through `srgb_to_linear` or `linear_to_srgb`, and no gamma is applied to it anywhere. Its random contrast curves in training act on these code values. The encoder copies it to three channels itself (DINO input, parked).
- sRGB exists only at these boundaries. Nothing else calls `srgb_to_linear` / `linear_to_srgb`:
  1. File input: decode, then `srgb8_to_linear` (`AlignedPairs`; the raw readers `dronevehicle.read_rgb8` return the codes untouched). 8-bit files use `srgb8_to_linear`, bit-identical to `srgb_to_linear(code / 255)`.
  2. File or display output: `linear_to_srgb`, then save (`infer.py`, `visualize.py`, scripts, contact sheets). 8-bit files use `linear_to_srgb8`. The aligned RGB PNG is the lossless 8-bit crop of the decoded JPEG, so it is written without any conversion.
  3. `DINOEncoder.forward` (parked): DINO is pretrained on sRGB, so the encoder takes **linear** RGB and converts to sRGB plus ImageNet mean/std itself. Callers never pre-convert.
  4. Display-unit uncertainty (`srgb_gamma_slope`).
- **CIELAB** (`dawn.data.color.linear_to_lab`, `lab_to_linear`): from **linear** sRGB primaries, D65 white, `(..., 3, H, W)` float32, channels L\*, a\*, b\*. L\* is in `[0, 100]`; white is exactly `(100, 0, 0)` and black `(0, 0, 0)`. The CIE linear segment below `(6/29)^3` keeps gradients finite at 0. Neither direction clamps: clamp RGB to `[0, 1]` before `linear_to_lab` if it can leave the range, and after `lab_to_linear` (out-of-gamut a\*b\* give channels outside `[0, 1]`).
- All image tensors are float32. bf16 autocast is for network forward passes only; grid fitting, slicing and variance propagation (parked) run in float32.

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

## 4a. Downsampling: area averaging

Every downsample in the project is area averaging, in training, inference, grid fitting and data preparation alike, and it goes through one function: `dawn.data.resample.area_downsample(img, (h, w))`.

- Each output pixel is the mean of the input over its footprint, `[i·s, (i+1)·s)` with `s = H_in / h_out` (likewise for width). Input pixels that straddle a footprint edge count by their overlap. For an integer factor this is the plain block mean; for any factor it preserves the image mean. (PIL's `BOX` filter is not this: it weights pixels 0 or 1 by where their center falls.)
- Downsample **linear** images. Averaging sRGB values makes bright lights come out too dim.
- Use `fit_long_side(H, W, long_side)` for the output size of an aspect-preserving resize. It never upsamples.
- Why it matters beyond looks: an area-averaged pixel is exactly what a sensor with `n` times bigger pixels records, so the noise of a downsampled image follows from the native noise (§6). The 2000×1500 → 256×256 DINO input averages about 46 native pixels per input pixel, and training must see the same noise reduction, not noise added after downsampling.

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
  3. Inference: `dawn.flow.sampler.sample_grid` integrates in standardized space and `unstandardize`s once before returning, so what a caller gets back is already raw and goes straight to `slice_grid`.
  4. Stage 2: fitted grids are already raw; flow-predicted grids go through `unstandardize` first.
- Nothing else multiplies or divides by stats. A docstring that takes a grid states raw or standardized.
- Flow time: `t = 0` is noise, `t = 1` is data. `z_t = (1 - t)·z_0 + t·z_1`, `v_target = z_1 - z_0`, samplers integrate 0 → 1.
- **NFE is velocity evaluations, not steps.** Euler spends one per step, midpoint two, so `nfe = 2` is two Euler steps but a single midpoint step. `flow.num_inference_steps` is an NFE budget; `steps_for_nfe` converts.

## 6. Noise model

`Var[y | x] = a·x + b` per pixel and channel, in linear `[0, 1]` units. `a` is the shot-noise gain, `b` the read-noise **variance** (σ², not a standard deviation). Both are positive. `NoiseHead` predicts the natural logs `(log a, log b)`; `sensor_variance` takes raw `a`, `b`.

**Per channel.** `(a, b)` are `(3,)` tensors, not scalars. Everything after the sensor is a per-channel linear scale `s_c = gain · wb_c`, and a scale `s` takes `a → s·a` but `b → s²·b`, so the channel the white balance pushes hardest is disproportionately the noisiest. `CameraResult` and `DegradationResult` both carry `(3,)`; a single number is only ever a summary for reporting.

**Resolution.** `(a, b)` are recorded at **native** resolution (the raw camera frame). A pixel that area-averages `n` native pixels (§4a) has `Var = (a/n)·x + b/n`: a sum of Poisson counts is Poisson and read-noise variances add, so this is exact. The camera model therefore simulates binned pixels: a 512-px training pixel stands for `n = W·H / (w·h)` native pixels, which is why `VisDroneDataset` returns `native_size` with every image.

**Quantization.** The ADC digitizes native pixels, so `quant_var = step²/12` is reported at native resolution like `a` and `b` and divides by `n` with them. A simulated pixel that averages `n` 12-bit samples lands on a lattice `n` times finer. The full budget at the returned resolution is computed by `DegradationResult.variance()`, which uses the ISP matrices. Off the native path (where S and M are identity), it simplifies to:

```
Var[y_c] = (a_c · y_c + b_c + quant_var) / n
```

JPEG and the final 8-bit step are nonlinear and are **not** in the budget.

**The ISP.** On the native path only, the ISP applies a colour-specific denoise blur (on C1, C2) and a composite denoise and unsharp mask. Because these are linear filters that mix channels, the variance of the output `y` involves a mixing matrix `S` and a DC colour matrix `M`:

```
x = M⁻¹ y
Var[y] = S (a · x + b + quant_var) / n
```

where `S[i, j] = Σ_tap W[i, j, tap]²` and `M[i, j] = Σ_tap W[i, j, tap]` for the composite 3×3 channel kernel `W`. `CameraResult` reports `S` and `M` directly, and `DegradationResult.variance()` evaluates this exact formula. Off the native path, `S` and `M` are the identity.

## 7. Shapes at module boundaries

| API | In → out |
|---|---|
| `compute_guidance(image_linear)` | `(B,3,H,W)` → `(B,1,H,W)` |
| `trilinear_weights(guidance, (K,gy,gx))` | `(B,1,H,W)` → `node_index`, `weight`: `(B,8,H,W)` |
| `fit_grid(night_clean, day, guidance)` | `(B,3,H,W)` ×2, `(B,1,H,W)` → raw `grid (B,12,K,gy,gx)`, `node_weights (B,K,gy,gx)` |
| `slice_grid(grid, guidance)` | raw grid, `(B,1,H,W)` → `coeffs (B,12,H,W)` |
| `apply_affine(coeffs, image)` | `(B,12,H,W)`, `(B,3,H,W)` → `(B,3,H,W)`, clamped |
| `standardize` / `unstandardize(grid, mean, std)` | grid `(B,12,K,gy,gx)`, stats `(12,K)` |
| `DINOEncoder.forward(image_linear)` | `(B,3,256,256)` → `patch_tokens (B,256,384)`, `cls_token (B,384)`, and with `return_attn` also `patch_attn (B,256,256)`: last layer, head-averaged, CLS and the 4 register tokens stripped, rows renormalized to sum to 1, detached |
| `DINOEncoder.encode(image_linear)` | `(B,3,256,256)` → dict: `patch_tokens (B,256,384)`, `patch_map (B,384,16,16)` (the same tokens, row-major), `cls (B,384)`, `thumb (B,3,16,16)` = `log(area_downsample(linear) + 1e-3)`, and with `return_attn` also `attn_bias (B,1,256,256)` = `log(patch_attn + 1e-6)` |
| `VelocityTransformer(z_t, t, dino_features, dino_attn, thumb)` | `(B,12,K,gy,gx)`, `(B,)`, `(B,256,384)`, `(B,256,256)`, `(B,3,gy,gx)` → `v (B,12,K,gy,gx)`. `attn_bias (B,1,256,256)` may be passed instead of `dino_attn` |
| `euler_sample(velocity_fn, z_0, n)` | `(B,12,K,gy,gx)` → same, standardized |
| `sample_grid(velocity_model, dino_features, mean, std, …)` | `(B,256,384)`, stats `(12,K)` → `(num_samples,B,12,K,gy,gx)` **raw** |
| `NoiseHead(cls_token)` | `(B,384)`, or `(B,256,384)` patch tokens which are mean-pooled first → `(B,2)` = `(log a, log b)` at native resolution |
| `propagate_variance(coeffs, sigma_sq_in)` | `(B,12,H,W)`, `(B,3,H,W)` → `(B,3,H,W)` |
| `BoundedDenoiser(enhanced, log_variance)` | `(B,3,H,W)` linear, `(B,1,H,W)` → `(B,3,H,W)` linear |
| `area_downsample(img, (h, w))` | `(..., H, W)` float → `(..., h, w)`, same dtype and device |
| `read_rgb8(path)` / `read_thermal8(path)` | JPEG path → `(3, 712, 840)` / `(1, 712, 840)` uint8 raw codes (thermal: one channel, after checking that a 3-channel file's channels are identical) |
| `crop_border(img)` | `(..., 712, 840)` → `(..., 512, 640)` |
| `parse_annotation(path, class_map)` | XML → boxes (class, `(4, 2)` polygon in **cropped** pixel-index coordinates, source kind), folder, filename |
| `warp_thermal(thermal, T, (512, 640))` | `(1, H_t, W_t)` float, `T (3, 3)` thermal → RGB → `(1, 512, 640)` float32 warped, `(1, 512, 640)` bool valid |
| `moving_mask(...)` | boxes, matches, `T` → `(1, 512, 640)` bool, count |
| `linear_to_lab` / `lab_to_linear` | `(..., 3, H, W)` ↔ `(..., 3, H, W)` float32 |
| `AlignedPairs[i]` | dict: `thermal (1, 512, 640)` float32 = code / 255, `rgb (3, 512, 640)` linear float32, `valid`, `moving` and `vehicles` `(1, 512, 640)` bool, `image_id` str, `fallback` bool (aligned by edge correlation, see §10.2). Training adds flips (all four tensors alike) and a thermal-only monotone contrast curve; validation and test are deterministic |

## 8. Rules

1. **One trilinear implementation.** `fit.py` gets node indices and weights from `slice.trilinear_weights`. It never computes `floor`, corner weights or coordinates itself.
2. **One guidance function.** Fitting, slicing, rendering, inference, stage 2 and every script call `compute_guidance`. No inline luminance or `log10` code anywhere else.
3. **One layout module.** Token reshapes and coefficient-order constants come from `dawn/grid/layout.py`; no ad-hoc `permute` / `view` of grids.
4. **One conversion point for standardization** (§5).
5. **One downsampling function.** `area_downsample` on linear images (§4a); no `F.interpolate`, PIL `resize` or `thumbnail` to shrink an image anywhere else.
6. **Conventions change here first.** If one has to change, edit this file in the same commit as the code and update the affected docstrings.
7. **One thermal resample.** `align.warp_thermal` is the only place thermal is resampled: bilinear inverse mapping onto the RGB frame. It interpolates at arbitrary positions rather than shrinking an image, so it is the one sanctioned resample that is not an area average (§4a is about downsampling). The RGB image is never resampled.
8. **Thermal is never sRGB** (§1), and the RGB target is never touched beyond the crop.

## 9. Known limitations

- The guidance blur is a fixed 5×5 **pixels**. At 256² training 5 px is 2% of the width; at 2000×1500 inference it is 0.25%, so the effective smoothing differs. If bins flicker at full resolution, scale the kernel with resolution and update §4.
- The 16×16 grid, 256-token layout and DINO patch correspondence in §2.3 are coupled; changing the grid size means changing the conditioning.
- **Splat-and-solve is biased.** `fit_grid` solves each node against the pixels that touch it, as if that node alone applied to them; the renderer then blends 8 nodes per pixel. The two disagree wherever the grid changes quickly from node to node, so a fitted grid does not re-render its own target exactly even with no noise anywhere. The error is small on smooth grids and is what `refine_steps` buys down, by fitting the render residual and adding the correction. Judge a fit on the **rendered** image, never on coefficient error against a reference grid.

## 10. Data: DroneVehicle pairs, alignment, determinism

Code: `dawn/data/{dronevehicle,day_filter,align,dedup,pairs,report,audit}.py`, `scripts/prepare_dronevehicle.py`, `scripts/audit_alignment.py`. Every threshold lives in `configs/data.yaml` and is commented there with its meaning and unit; this section says what they mean for the data, not their values.

### 10.1 The data, the day filter and the three manifests

- **Source.** DroneVehicle (McCheng/DroneVehicle on Hugging Face): `<root>/<split>/{<split>img, <split>imgr, <split>label, <split>labelr}/<stem>.*` = RGB JPEG, thermal JPEG, RGB XML, thermal XML. The file stem pairs the modalities. Stems repeat across splits, so the global **image ID** is `f"{split}_{stem}"` (`train_00001`), used in every manifest, log, file name and seed.
- **Frames.** Every raw frame is 840×712 with a 100-px white border on both modalities. The content is `[100:612, 100:740]`, **640×512**. The border is checked on every image (`dronevehicle.check_border`: a dark border pixel is `dark`, white leaking into the content's outer ring is `wide`) and failures are logged, not dropped silently. Thermal JPEGs are stored as one channel or as three identical channels; one channel is taken after checking they match.
- **Labels.** VOC-like XML. Objects carry a `<polygon>` (oriented quad), a `<bndbox>` (read as a rectangle), or only a `<point>` (skipped and counted). Coordinates are in the 840×712 frame; `parse_annotation` subtracts the border. Class names go through `dronevehicle.class_map` (typos mapped, `*` dropped). `<size><depth>` is wrong in this dataset and ignored.
- **Stats** (`prepare_dronevehicle.py stats`): per stem, the existence, frame, border and thermal-channel checks, RGB luminance percentiles on the cropped frame, folder and file names, and a 64-bit perceptual hash of each modality. Pairs missing a modality or with an unreadable (e.g. truncated) JPEG get a status and drop out at the day stage.
- **Day filter.** A pair is day when the **median linear luminance of the cropped RGB frame** is at least `day_filter.min_lum_median` (log10 of it is what the histogram and sheets show). The distribution has a bright day mode and a continuum of dusk and night below it with no clean gap, so the cutoff is a human decision made from the histogram and contact sheets, and errs high (the plan excludes dusk and anything borderline). Median luminance alone does not separate day from night perfectly: night frames auto-exposed bright reach about log10 -0.73.
- **The three manifests** are `data/manifests/{train,val,test}_pairs.txt`, one image ID per line, sorted, committed. They hold the pairs that survive every stage below. `fallback_pairs.txt` lists, among them, the pairs that were aligned by the edge-correlation fallback of §10.2 instead of by boxes. The official splits are kept (they are not flight-disjoint; see Dedup). Alongside them, `data/manifests/logs/{day,align,dedup}.csv` and `summary.md` record every decision (floats to 3 decimals).
- **Storage.** Aligned pairs are written once, as PNG, under `dronevehicle.aligned_dir/<split>/` (gitignored), so training never warps on the fly: `<id>_rgb.png` (8-bit sRGB, the lossless crop of the decoded JPEG), `<id>_thermal.png` (8-bit L, the warped thermal; the 8-bit requantization noise sits around 59 dB, far below the JPEG noise), `<id>_valid.png`, `<id>_moving.png` and `<id>_vehicles.png` (L, 0/255).
- **Dedup.** Consecutive frames of a flight are near-duplicates and the official splits share flights, so a val or test pair is **dropped when it near-duplicates a train pair**: its RGB hash **and** its thermal hash are both within `dedup.max_hamming` bits of the same train pair. The reference set is the train pairs that passed the *day filter* (not the aligned ones), so loosening alignment later cannot let leakage back in. Train keeps everything. The hash is the DCT perceptual hash of luma (RGB: sRGB-coded luma; thermal: the codes), computed on the crop after area averaging to 32×32 (§4a); it always has 32 set bits, so distances are even.

### 10.2 Alignment and masks

The thermal and RGB cameras sit a few centimetres apart and expose at slightly different times, so a pair is a similarity warp (plus moving vehicles) from aligned. Each vehicle is boxed separately in both modalities, and the box centres give correspondences without any cross-modal feature matching.

- **Coordinates.** Pixel-index: x is the column, y the row, and pixel `(r, c)` has its centre at `(c, r)`. XML boxes (after the crop) and every transform use them, so an integer shift `(dx, dy)` moves pixel `(c, r)` to `(c + dx, r + dy)` exactly. A box centre is the mean of its four vertices.
- **Transforms** are 3×3 matrices acting on `(x, y, 1)` that map **thermal onto RGB**. The thermal is warped onto the RGB frame; the RGB is only cropped. A similarity is logged as 6 numbers (its first two rows), a homography as 9, so `write` never refits.
- **Copied boxes are not evidence.** About 17% of pairs have identical RGB and thermal label sets and, in the rest, a median of 25% of RGB boxes are exact copies of a thermal box. A copy has zero residual by construction. A box counts as copied when its normalized class matches and all four vertices are equal within `align.copy_tol_px`. Copied boxes are excluded from matching and fitting (`align.exclude_copied_boxes`, a flag the maintainer may reverse; the dry run `align --include-copied` shows what it costs). A pair with identical label sets gets the reason `labels_copied`.
- **Matching.** Boxes of the same class, **mutual** nearest neighbours by centre distance, within `align.max_match_dist_px` (a gate the plan does not state).
- **Fit.** A similarity (scale, rotation, translation) by RANSAC with 2-match minimal sets: every pair is tried when there are at most `ransac_enumerate_max` matches (deterministic, no RNG), otherwise sets are drawn from `generator_for(image_id, "ransac")` (§10.3). It is refit on the inliers (residual within `ransac_inlier_px`). A homography is tried only when the similarity fails on the residual alone and there are at least `align.homography.min_matches` matches, with the same thresholds; the model used is logged.
- **Transform parameters.** scale; rotation in degrees; and **shift = the displacement of the frame centre (W/2, H/2) = (320, 256) under `T`**, not the raw translation, which depends on the origin. For a homography, scale and rotation are those of the local Jacobian at the frame centre.
- **Edge-correlation fallback** (plan step 7). The boxes cannot judge a pair when the thermal labels are copies of the RGB ones (`labels_copied`) or there are fewer than `min_matches` usable matches (`few_matches`): that is "no evidence", not "misaligned". For exactly those pairs `align.edge_translation` estimates a global integer **translation** from the gradient-magnitude correlation of thermal and RGB over every shift within `fallback.search_px`, then again inside each image quadrant, and `fallback_accepts` takes it when the best shift is not on the window edge (saturated), the correlation peak beats the best shift at least 2 px away by `fallback.min_prominence`, and every quadrant's own best shift is within `fallback.max_quadrant_dev_px` of the global one. Such a pair is logged with `model = "edge_translation"` and `reason = ok`, with the estimate in the `fb_*` columns; a pair it rejects keeps its box reason. These pairs are **less trustworthy** than box-aligned ones: against box-fitted pairs the estimate is within 2 px 84% of the time and within 3 px 96% of the time, a pure translation ignores the small scale and rotation, and no boxes were matched, so **no moving mask exists** (`moving` is empty and unknown). They are listed in `fallback_pairs.txt` and flagged `fallback` by `AlignedPairs`; keep them out of the pixel losses and use them for the patch-level ones. The fallback is audited on its own sheets (`fallback_accepted_*`, `fallback_rejected_*`).
- **Acceptance**, in this order, each failure with its own reason code: `few_matches` (or `labels_copied` when the label sets are identical), `inlier_ratio`, `residual` (the **median inlier residual**; the median over all matches is logged too), `scale`, `rotation`, `shift`; then the independent edge check, `edge_check`. Boundaries: matches, inlier ratio, residual and scale deviation are inclusive; rotation and shift are strict "under".
- **Edge check.** Normalized correlation of gradient magnitudes (light blur, then Sobel) between the thermal and the sRGB-coded RGB luma, over the pixels where the warp is valid, before (identity) and after the warp. With near-identity warps "must improve" is noisy, so the gain `after - before` is logged together with the best integer shift of the warped pair within `±search_px`. A pair passes when `gain >= grad_check.min_gain` and that best shift is at most `grad_check.max_best_shift_px` on each axis.
- **Warp.** `align.warp_thermal` resamples thermal onto the 640×512 RGB grid by bilinear inverse mapping (§8 rule 7). The validity mask is false wherever the source falls outside the thermal frame (beyond the pixel extent, `[-0.5, W_t - 0.5]` and likewise in y), eroded by one pixel for the bilinear edge. Outside the RGB frame counts as valid, so an identity warp is valid everywhere.
- **Vehicles mask** (`<id>_vehicles.png`, for **every** pair): every vehicle, wherever it may be, with a margin chosen **per vehicle** (`align.vehicle_mask`; settings under `masks.vehicles` in data.yaml). It is the stand-in for `moving` where `moving` is unknown: **the loss code uses `moving` for box-aligned pairs and `vehicles` for `fallback` pairs**, and eval uses it for vehicle-crop metrics. For each RGB box, as an oriented polygon (never its bounding box):
  1. *Matched* (the vehicle has a thermal box of its own, `match_boxes`): both positions are known, so the RGB polygon and the thermal polygon pushed through the pair's transform get only `dilate_px` (3 px, the fallback's alignment error).
  2. *Static*: no thermal box of its own (copied labels, or unmatched), but a local shift search around the RGB box (the same FFT correlation as the fallback, on a window of the box plus `window_margin_px`, residual shifts within `search_px` after the pair's warp) has a clear peak (`min_prominence`) within `max_residual_px` of no shift: the vehicle did not move, `dilate_px` only.
  3. *Moved or unsure* (a clear shift, no clear peak, a window under `min_window_px`, a window touching unfilled warp pixels, or no images): the polygon is stretched `extend_px` (17.4 px, the p95 of the displacement of vehicles that moved, measured on box-aligned pairs by `prepare_dronevehicle.py vehicle-dilation`) each way along its **long** axis, plus `dilate_px` all round. Vehicles move along their heading, so an isotropic margin of the same reach would waste most of its area.

  Thermal boxes with no RGB counterpart add their warped polygon (`thermal_only`); a thermal box that is a copy of an RGB box adds nothing (it only repeats the RGB position, and warping it would put it in the wrong place). No boxes means an empty mask. The mask is recomputed on every `write` run (it is deterministic) so that `data/manifests/logs/vehicles.csv` always has, per pair, the share of the frame it covers, a `dense` flag above `masks.vehicles.max_frame_fraction` (frames to look at, shown on `outputs/data/audit/vehicles_dense_01.jpg`, not to drop silently) and how many vehicles took each branch.
- **Masks.** `valid` is excluded from **every** loss: its pixels are not real thermal data. `moving` is excluded from the **pixel** losses (render, brightness fit): a matched box whose own residual under `T` exceeds `align.moving.min_residual_px` moved between the exposures, and its mask is the union of its RGB polygon and its thermal polygon pushed through `T`, dilated by `align.moving.dilate_px` as a Euclidean disk. Unmatched and copied boxes are only counted, never masked. A fallback-aligned pair has an empty `moving` mask because nothing is known about which vehicles moved, not because none did.
- **Thresholds are starting values until the audit.** The ones marked in `data.yaml` came from the plan and are revisited after looking at `scripts/audit_alignment.py` (sheets of accepted and rejected pairs with the thermal edges drawn over the RGB, and histograms of the fit metrics). Do not tune them in code.

### 10.3 Determinism: the seed comes from the image ID

Every random draw that belongs to one image derives its seed from that image's ID, through `dawn.data.seeding.seed_from_image_id(image_id, *salts)`. The same ID gives the same output on any machine, in any worker process, in any iteration order, with any batch size.

- The hash is BLAKE2b over the ID and the salts, so it does not depend on `PYTHONHASHSEED`, and neighbouring IDs do not give neighbouring seeds.
- Salts separate independent draws for the same image: the darkness level, the stage name. Use one rather than reusing a generator across purposes.
- `generator_for(...)` returns a seeded `torch.Generator`. Pass it explicitly into every sampling call; never reseed the global RNG inside a dataset or a transform, which would make the result depend on how the loader happened to interleave workers.
- A fresh draw is requested by passing no ID, not by passing a changing one.
