# Dawn conventions

The single source of truth for every layout, coordinate system, color space and definition that more than one module relies on. **If code and this file disagree, this file wins**: fix the code, or change this file in the same commit. Read it before touching `dawn/grid/`, `dawn/models/`, `dawn/flow/`, `dawn/data/` or `dawn/eval/`.

> **Status, 2026-10-06.** The project moved from synthetic night RGB (VisDrone) to real **day thermal → RGB** (DroneVehicle), and the flow phase rebuilt the grid, token, model, flow and metric code for it. Everything here is current. The plan (`docs/Dawn Night-Vision to Day Colorization for Drone Imagery.md`) is the spec this file implements; where this file is more specific than the plan (the 261 a\*b\* bins instead of its 313, `L = L*/100` coefficients, a switchable brightness correction, bilinear colour upsampling instead of the BGU, the resolution as one knob) this file wins.

Notation: `B` batch, `H, W` image rows/columns, `K` brightness bins (8), `gy, gx` token-grid rows/columns (32, 40 at the full 512×640 encoder input), `N = gy·gx` tokens, `D` values per token, `Q` a\*b\* bins (261). Sizes come from `configs/flow.yaml` (`encoder.input_size`, `grid.*`, `color.*`, `brightness_correction.enabled`); the token layout lives in `dawn/flow/tokens.py` and the coefficient constants in `dawn/grid/layout.py`. Nothing else hard-codes 40, 32, 1,280, 160, 128, 16, 32, 48 or 64: code and tests ask the layout. The one exception is `tests/test_tokens.py: test_layout_matches_the_plan`, which pins the plan's numbers as the spec.

## At a glance

| Topic | Convention |
|---|---|
| RGB images | `(B, 3, H, W)`, RGB, float32, **linear**, `[0, 1]`. sRGB only at the boundaries in §1 |
| Thermal images | `(B, 1, H, W)`, float32, `code / 255` in `[0, 1]`. **Not sRGB**: never through `srgb_to_linear` (§1) |
| Frames | raw 840×712 with a 100-px white border; content crop `[100:612, 100:740]` = **640×512** (§10.1) |
| Pixel coordinates | pixel-index: x = column, y = row, pixel `(r, c)` has its centre at `(c, r)`; boxes and warps share them (§10.2) |
| Warp | thermal onto RGB, bilinear inverse mapping, `T: thermal → RGB`; RGB is only cropped, never resampled (§10.2) |
| Masks | `valid` excluded from **every** loss, `moving` (box pairs) or `vehicles` (fallback pairs) from the **pixel** losses (§10.2, §5.3) |
| Image IDs | `f"{split}_{stem}"`, e.g. `train_00001`; the three manifests are `{train,val,test}_pairs` (§10.1) |
| Resolution | one knob: `encoder.input_size = (H, W)`, default `(512, 640)`. Token grid = input / 16; colour map = 4 × token grid; render size = `flow.render_size` (§3) |
| Brightness grid | raw `(B, 2, K, gy, gx)` = (coefficient, bin, cell row, cell col); coefficient 0 = gain, 1 = offset; `L = gain · thermal + offset`, **`L = L*/100`** (§2) |
| Colour map | raw `(B, 2, 4gy, 4gx)` a\*, b\* in CIELAB units, one value per colour pixel; optional correction `(B, 1, 4gy, 4gx)` in `L` units (§2.3); upsampled **bilinearly** by default, bilateral guided upsampling (BGU) behind `color.upsample` (§6.3) |
| Token | `n = y·gx + x`, values `[brightness 2K \| colour 32 \| correction 16 (if enabled)]` = 48 or 64 (§2.3) |
| Grid coordinates | node values sit at cell **centers**; coordinates outside the outer centers clamp (§3) |
| Guidance | `compute_guidance(thermal)`: 5×5 replicate box blur, clamp to `[0, 1]`. No log, no stretch (§4) |
| Raw vs standardized | per-segment stats, flattened to a `(D,)` mean and std by the layout; pixels see raw, flow sees standardized; convert only via `normalize.py` (§5) |
| Downsampling | always area averaging, only through `area_downsample` (§4a); masks become fractions, targets masked means |
| Colour bins | the 261 occupied cells of a 10-unit a\*b\* grid over the 8-bit sRGB gamut, sorted by a\* then b\* (§6) |
| Render | `flow.render.render`: the one path from raw parts to L\*a\*b\* and linear RGB (§6.4) |
| Flow time | `t = 0` noise, `t = 1` data |

## 1. Color space

- **RGB** inside the pipeline is **linear RGB, float32, `[0, 1]`**, channel order R, G, B, layout `(B, 3, H, W)` (`(3, H, W)` for one sample). That covers the aligned `rgb` target and anything rendered from it. Clamp to `[0, 1]` after anything that can leave the range (an affine render, CIELAB back to RGB).
- **Thermal is not sRGB.** Thermal values are the camera's 8-bit codes: `thermal = code / 255`, float32 in `[0, 1]`, one channel, `(1, H, W)`. Thermal **never** goes through `srgb_to_linear` or `linear_to_srgb`, and no gamma is applied to it anywhere. Its random contrast curves in training act on these code values. `DINOEncoder.forward` copies it to three channels and applies ImageNet mean/std itself; it never converts it to sRGB.
- sRGB exists only at these boundaries. Nothing else calls `srgb_to_linear` / `linear_to_srgb`:
  1. File input: decode, then `srgb8_to_linear` (`AlignedPairs`; the raw readers `dronevehicle.read_rgb8` return the codes untouched). 8-bit files use `srgb8_to_linear`, bit-identical to `srgb_to_linear(code / 255)`.
  2. File or display output: `linear_to_srgb`, then save (`infer.py`, `visualize.py`, scripts, contact sheets). 8-bit files use `linear_to_srgb8`. The aligned RGB PNG is the lossless 8-bit crop of the decoded JPEG, so it is written without any conversion.
  3. `DINOEncoder.forward_frozen`: DINO is pretrained on sRGB, so for the L_align target the encoder takes the paired **linear** RGB and converts to sRGB plus ImageNet mean/std itself. Callers never pre-convert. (The thermal input of `forward` is not sRGB and takes no conversion, §1 thermal bullet.)
  4. Display-unit uncertainty (`srgb_gamma_slope`).
- **CIELAB** (`dawn.data.color.linear_to_lab`, `lab_to_linear`): from **linear** sRGB primaries, D65 white, `(..., 3, H, W)` float32, channels L\*, a\*, b\*. L\* is in `[0, 100]`; white is exactly `(100, 0, 0)` and black `(0, 0, 0)`. The CIE linear segment below `(6/29)^3` keeps gradients finite at 0. Neither direction clamps: clamp RGB to `[0, 1]` before `linear_to_lab` if it can leave the range, and after `lab_to_linear` (out-of-gamut a\*b\* give channels outside `[0, 1]`).
- All image tensors are float32. bf16 autocast is for network forward passes only; target building, grid fitting, slicing, rendering and the losses run in float32. bf16 is for CUDA only: MPS and CPU runs use fp32 (`optim.precision`).

## 2. Brightness grid and tokens

### 2.1 Canonical shape: `(B, 2, K, gy, gx)`

| axis | name | size | meaning |
|---|---|---|---|
| 0 | `b` | | batch |
| 1 | `c` | 2 | coefficient: 0 = gain, 1 = offset (§2.2) |
| 2 | `k` | K = 8 | thermal-intensity bin; 0 is the darkest (lowest guidance) |
| 3 | `y` | gy | cell row; 0 is the top |
| 4 | `x` | gx | cell column; 0 is the left |

This is PyTorch's 5-D `(N, C, D, H, W)` order (`K` is the depth axis), so `F.grid_sample` slices it directly (§3). The cells are the DINO patches: `(gy, gx) = input_size / 16`. A general one-input affine grid with `C` outputs (the bilateral guided upsampling of §6.3) has `2C` coefficient channels, output `c` at channels `2c` (gain) and `2c + 1` (offset): `out_c = coeffs[2c] · x + coeffs[2c + 1]`. Per-pixel coefficient maps (output of `slice_grid`) are `(B, 2C, H, W)` with the same channel order.

### 2.2 Coefficients and units

`L = gain · thermal + offset`, then clamped to `[0, 1]`. **`L` is `L*/100`**, not linear luminance: the grid maps thermal straight to CIELAB lightness divided by 100, so the rendered `L*` is `100 · L`. This measured slightly better than fitting linear luminance (24.72 vs 24.52 dB on the oracle) and optimizes the quantity the plan evaluates. a\* and b\* stay in CIELAB units.

### 2.3 Tokens for the velocity transformer: `(B, N, D)`

One layout, defined only in `dawn/flow/tokens.py` (`TokenLayout`). The velocity net's input and output width, the stats vector, the loss weights and the render path all derive from it.

- **Token `n = y·gx + x`**, row-major. It is DINO patch `n`, brightness cell `n`, the thermal thumbnail pixel `n` and the 4×4 block of the colour map that those pixels cover.
- **Values, in this order:**

| segment | count | value index `v` |
|---|---|---|
| brightness | `2K` = 16 | `v = k·2 + c`, bin-major, `c` ∈ {gain, offset} |
| colour | 32 | `v = 2K + ch·16 + py·4 + px`: the token's 4×4 block covers colour pixels `(4y + py, 4x + px)`; `ch` = a\* then b\* |
| correction (only when `brightness_correction.enabled`) | 16 | `v = 2K + 32 + py·4 + px`, the same 4×4 block, in `L` units |

  The correction is appended, so the base layout is identical in both modes. This is the plan's "16 brightness values, then 32 color values". `D` is 48 with the correction off and 64 with it on.
- **Parts** (raw, what the pixel side sees): `grid (B, 2, K, gy, gx)`, `ab (B, 2, 4gy, 4gx)` and, only with the correction, `corr (B, 1, 4gy, 4gx)`. `TokenLayout.pack(parts)` gives `(B, N, D)`, `unpack` inverts it.
- Only `VelocityTransformer` works on tokens, and the flow side (`z_0`, `z_1`, `z_t`, `v`, samplers, losses) stays in token tensors; parts exist at the boundaries (targets in, render out).
- The layout is part of every checkpoint (`TokenLayout.descriptor()`); loading checks it.

### 2.4 Node weights ("pixel counts"): `(B, K, gy, gx)`

For each node, the sum of the trilinear weights of every pixel that contributes to it, each pixel counted by its fit weight (the mask). Returned by `fit_affine_grid`, used by the flow loss's brightness weighting (broadcast over the gain/offset values).

## 3. Resolution and coordinates: cell centers

**Resolution is one knob.** `encoder.input_size = (H, W)` (default `(512, 640)`) fixes everything: the token grid is `(H/16, W/16)` = `(32, 40)`, the colour map is 4× that, `(128, 160)`, and `flow.render_size` (default `(256, 320)`) is where L_render is computed. Nothing hard-codes these. Below the stored 512×640 the trainer area-downsamples thermal, rgb and the masks (as fractions) first (§4a). A checkpoint is tied to its resolution and loading checks it. The render size must be at least the colour-map size, and the colour map at most the input size.

Grid values sit at cell **centers** (PyTorch's `align_corners=False`). For pixel `(r, c)` of an `H×W` image (integer indices) with guidance `g ∈ [0, 1]`:

```
yf = (r + 0.5) * gy / H - 0.5        # node i sits at coordinate i
xf = (c + 0.5) * gx / W - 0.5
zf = g * K - 0.5
```

Clamp each coordinate to `[0, N-1]` (`N` = `gy`, `gx`, `K`), then `i0 = floor(f)`, `i1 = min(i0 + 1, N-1)`, `w1 = f - i0`, `w0 = 1 - w1`. A pixel's 8 corner nodes get the weights `w_y · w_x · w_z`, which sum to 1. Outside the outer cell centers the edge node wins entirely (replicate).

- Bin `k` represents guidance `(k + 0.5) / K`. Never bin with `floor(g·K)` or `g·(K-1)`.
- The grid spans whichever image tensor it was fit on or sliced on (fractions of `H` and `W`), so a grid fit at 256×320 slices at 512×640. A training crop is its own image.
- The equivalent `F.grid_sample(grid, coords, mode="bilinear", padding_mode="border", align_corners=False)` (`"bilinear"` is trilinear for 5-D input) takes `coords = (x_norm, y_norm, z_norm) = ((2c+1)/W - 1, (2r+1)/H - 1, 2g - 1)`: the image's own normalized pixel centers, and `2g - 1` for guidance.
- **Single implementation:** `dawn.grid.slice.trilinear_weights(guidance, (K, gy, gx)) -> (node_index, weight)`, both `(B, 8, H, W)`. `node_index` is int64, flat over `(K, gy, gx)` as `k·gy·gx + y·gx + x`; `weight` is float32 and sums to 1 over dim 1; corner `n = 4·dz + 2·dy + dx`. `slice_grid` uses `F.grid_sample` for speed, as long as `tests/test_grid_rules.py` (agreement with `trilinear_weights`) passes. `fit_affine_grid` and `bilateral_guided_upsample` take their weights from `trilinear_weights` only.

## 4. Guidance

`dawn.grid.guidance.compute_guidance(thermal) -> (B, 1, H, W)` in `[0, 1]`:

1. 5×5 box blur (mean filter, stride 1, same size) with **replicate** padding of the one-channel thermal (`code / 255`). Zero padding darkens the borders and puts border pixels in the wrong bin.
2. Clamp to `[0, 1]`.

That is all: no log (thermal is not photon-starved like a dark photo) and no per-frame stretch (a p1–p99 stretch measured +0.06 dB because thermal already spans most of `[0, 1]`). Guidance is computed from the thermal that is being indexed, at the resolution it is being indexed at; the blur is a fixed 5×5 **pixels** (§9). It is the only guidance implementation; the bilateral guided upsampling of §6.3 passes `L` as its own guide and calls nothing else.

## 4a. Downsampling: area averaging

Every downsample in the project is area averaging, in training, inference, grid fitting and data preparation alike, and it goes through one function: `dawn.data.resample.area_downsample(img, (h, w))`.

- Each output pixel is the mean of the input over its footprint, `[i·s, (i+1)·s)` with `s = H_in / h_out` (likewise for width). Input pixels that straddle a footprint edge count by their overlap. For an integer factor this is the plain block mean; for any factor it preserves the image mean. (PIL's `BOX` filter is not this: it weights pixels 0 or 1 by where their center falls.)
- Downsample **linear** RGB images. Averaging sRGB values makes bright lights come out too dim. Thermal codes are averaged as they are (§1).
- **Lab channels are averaged as values.** The a\*b\* colour target, the correction target and the BGU's low-resolution guide are area averages of L\* / a\* / b\* values, not of linear RGB, because the model predicts in Lab and the target must be the quantity it predicts. §4a's linear-RGB rule is about images; the Lab maps are targets, and the RGB image that L_render compares against is area-downsampled in linear light first and only then converted to Lab.
- **Masks are fractions.** A boolean mask is downsampled to the fraction of its footprint that is kept, and used as a **weight**, never thresholded.
- **A masked low-resolution target is the masked mean** `area(x · m) / area(m)`. Where `area(m) == 0` the target is filled with a neutral value (0) and its weight is 0.
- Use `fit_long_side(H, W, long_side)` for the output size of an aspect-preserving resize. It never upsamples.

## 5. Raw vs standardized tokens

Raw parts (§2.3) are physical numbers; the flow works in standardized tokens.

### 5.1 Stats

Per **segment**, so a bin-0 offset and a bin-7 offset never share a mean (they live on different scales), and the colour and correction values pool over pixels:

| segment | stats shape | pooled over |
|---|---|---|
| brightness | `(2, K)`, `[c, k]` | cells, not bins |
| colour | `(2,)`, a\* then b\* | colour pixels, weighted by the mask fraction |
| correction | `(1,)` | colour pixels, weighted by the mask fraction |

`mean` and `std` for each segment are float32. `TokenLayout.stats_vector(stats)` flattens them into the `(D,)` mean and std that `standardize` and `unstandardize` take: brightness `v = k·2 + c`, each colour channel repeated over its 16 block pixels, the correction repeated likewise. `compute_token_stats` fits the targets (§5.3) of `grid.stats_num_samples` unaugmented train pairs, once; the result is cached at `grid.stats_cache`, keyed by the layout and the fit settings, and recomputed on a mismatch. **A checkpoint embeds the stats and the layout descriptor it was trained with**; inference reads them from the checkpoint, never from the cache file.

### 5.2 Conversion points

`dawn.grid.normalize.standardize(tokens, mean, std)` and `unstandardize` are the only places tokens change space (`std` is floored at 1e-6 by both, so they stay exact inverses):

1. Training target: `z_1 = standardize(pack(targets))`.
2. Render loss: `ẑ_1 = z_t + (1 - t)·v_pred` is standardized, so `unstandardize`, `unpack`, then `render`, then compare with the RGB. `FlowLoss` only receives the rendered image; it never sees stats.
3. Sampling: `dawn.flow.sampler.sample_parts` integrates in standardized space and unstandardizes **once** before returning, so what a caller gets back is already raw parts that go straight to `render`.

Nothing else multiplies or divides by stats. A docstring that takes tokens or parts states raw or standardized.

### 5.3 Targets and the pixel mask

Targets are built on the device in float32 with no gradient, from the thermal, the RGB and the pixel mask `M` (`flow.targets.build_targets`):

- **Pixel mask** `M = valid & ~moving` for box-aligned pairs and `valid & ~vehicles` for `fallback` pairs (`configs/flow.yaml: pairs`), as a float fraction at the working resolution. It zeroes the fit weights and every pixel loss.
- `L = L*/100` of the RGB. Brightness grid and node weights: `fit_affine_grid(thermal, L, compute_guidance(thermal), weights=M, ...)` at `grid.refine_steps` (1). Oracle `L_grid` = that grid rendered.
- Colour target: the masked area average of a\*b\* to the colour-map size; weight = the area-averaged mask. Correction target (only when enabled): the masked area average of `L - L_grid` (the **fitted target grid**, at the configured `refine_steps`), same weight.
- **Weights** (`TokenLayout.weights`, normalized to mean 1 per segment per sample, all zero when a segment has no weight): brightness values take the node weights broadcast over gain/offset; colour and correction values take the mask fraction of their pixel.

### 5.4 Flow time and NFE

`t = 0` is noise, `t = 1` is data. `z_t = (1 - t)·z_0 + t·z_1`, `v_target = z_1 - z_0`, samplers integrate 0 → 1. **NFE is velocity evaluations, not steps.** Euler spends one per step, midpoint two, so `nfe = 2` is two Euler steps but a single midpoint step. `flow.num_inference_steps` is an NFE budget; `steps_for_nfe` converts.

## 6. Colour map, a\*b\* bins, upsampling and the render path

### 6.1 The 261 a\*b\* bins

`dawn.flow.color_bins` derives them in code, deterministically: map every 8-bit sRGB colour to CIELAB (§1), round a\* and b\* to a 10-unit grid, keep the occupied cells. That gives **261** bins (a\* −90…100, b\* −110…90), sorted by a\* then b\*, identical when only every second code is used. This replaces the plan's 313 (Zhang et al.'s set is for the full Lab gamut, ours is the sRGB gamut we can render). Soft encoding (`soft_encode`) is Zhang's: the `color.soft_k` (5) nearest bins with a Gaussian of `color.soft_sigma` (5) over the distance in a\*b\* units, normalized to sum to 1.

`patch_histograms(ab_map, mask, patch_size)` gives the per-token soft histogram `(B, N, Q)`: the masked mean of the soft encodings of the a\*b\* pixels in each patch. It is built from the a\*b\* at **render size** (8×8 samples per patch at the default 256×320), not at full resolution, to keep the cost down; `patch_mask_fraction` gives the patch's kept fraction, and L_color counts a patch 0 below `color.min_patch_fraction`.

### 6.2 The correction is optional

`brightness_correction.enabled` (default **false**) adds a generated `L` correction at the colour-map size. Its target is `L - L_grid(fitted target grid)` as a masked area average (§5.3); it is upsampled by the same function as colour (§6.3) and added to the grid's `L`. Whether to turn it on is the maintainer's call after the worst-10% oracle sheet and the GPU gate: on the 345 val pairs at 640×512 the grid alone reaches 25.1 dB of L\* PSNR at `refine_steps: 1` (24.6 at 0) and the correction, bilinearly upsampled, 26.9.

### 6.3 Upsampling the generated maps: bilinear by default, BGU as a switch

`dawn.grid.upsample.upsample_map(low, guide_lo, guide_hi, mode, ...)` is the one entry point; `color.upsample` selects the mode for both the colour map and the correction.

- **`bilinear`** (the default): `F.interpolate(..., mode="bilinear", align_corners=False)` to the render size. The maintainer chose it at the oracle checkpoint: on the real val pairs the **colour ceiling** (the true a\*b\* area-averaged to the colour size and brought back up, mean |Δa\*b\*| against the truth) is **0.90** bilinear against **1.28** for the best BGU found (`bins 8, 64×80 cells, ridge 0.1, refine 2`) and 2.1 at 16×20 cells, the BGU being worse at every grid swept; the assembled oracle CIEDE2000 is 3.33 against 3.50 with that best BGU (3.9 at 16×20). A stronger ridge does not help; more refine passes help a little (`scripts/check_oracle.py`, `outputs/oracle/report.md`). The BGU's per-cell affine model of colour on `L` smooths away colour detail that is not a function of `L` inside a cell.
- **`bgu`**: bilateral guided upsampling (Chen et al., 2016): at low resolution fit a bilateral grid of `low ≈ gain · guide + offset` per (cell, bin) with `fit_affine_grid`, slice it at the high-resolution guide and apply it to it, so edges follow the guide's. The guides are `L` in `[0, 1]` and are constants (no gradient through the bin assignment). It stays implemented and tested (exact on affine data, sharper than bilinear on a guide step edge) for cases where the guide explains the colour; its grid, bins, ridge and refine steps are `color.bgu`.

Both modes are differentiable with respect to `low`, which is how L_render reaches the colour tokens.

### 6.4 The render path

`dawn.flow.render.render(parts, thermal, size, settings) -> {"lab", "rgb", "L_grid"}`, float32, is the only path from raw parts to pixels. L_render (at `flow.render_size`), validation and the oracle (at the input size) and inference all use it.

1. Area-downsample `thermal` to `size` if it is larger.
2. `g = compute_guidance(thermal_s)`; `L_grid = clamp(gain · thermal_s + offset, 0, 1)` from `slice_grid(grid, g)`.
3. With the correction: `L = clamp(L_grid + upsample(corr), 0, 1)`; otherwise `L = L_grid`.
4. `ab = upsample(ab_lo)`; in `bgu` mode the guide is the **output** `L`, so colour edges follow brightness edges.
5. `lab = (100 · L, ab)`; `rgb = clamp(lab_to_linear(lab), 0, 1)` (clipped to the gamut).

## 7. Shapes at module boundaries

| API | In → out |
|---|---|
| `compute_guidance(thermal)` | `(B,1,H,W)` → `(B,1,H,W)` |
| `trilinear_weights(guidance, (K,gy,gx))` | `(B,1,H,W)` → `node_index`, `weight`: `(B,8,H,W)` |
| `fit_affine_grid(x, y, guidance, weights, grid_size, ridge_alpha, refine_steps)` | `x (B,1,H,W)`, `y (B,C,H,W)`, `guidance (B,1,H,W)`, `weights (B,1,H,W)` or None → raw `grid (B,2C,K,gy,gx)`, `node_weights (B,K,gy,gx)` |
| `slice_grid(grid, guidance)` | raw grid `(B,C',K,gy,gx)`, `(B,1,H,W)` → `coeffs (B,C',H,W)` |
| `apply_gain_offset(coeffs, x, clamp)` | `(B,2C,H,W)`, `(B,1,H,W)` → `(B,C,H,W)`, clamped to `[0,1]` when `clamp` |
| `upsample_map(low, guide_lo, guide_hi, mode)` / `bilateral_guided_upsample(low, guide_lo, guide_hi, weights_lo)` | `(B,C,h,w)`, `(B,1,h,w)`, `(B,1,H,W)` (, `(B,1,h,w)` or None) → `(B,C,H,W)` |
| `TokenLayout.pack(parts)` / `unpack(tokens)` | parts `grid (B,2,K,gy,gx)`, `ab (B,2,4gy,4gx)`, `corr (B,1,4gy,4gx)` ↔ `(B,N,D)` |
| `TokenLayout.stats_vector(stats)` / `weights(node_weights, fraction)` | per-segment stats → `(D,)` mean and std; `(B,K,gy,gx)`, `(B,1,4gy,4gx)` → `(B,N,D)` |
| `standardize` / `unstandardize(tokens, mean, std)` | tokens `(B,N,D)`, stats `(D,)` |
| `render(parts, thermal, size, settings)` | raw parts, `(B,1,H,W)`, `(h,w)` → `lab (B,3,h,w)`, `rgb (B,3,h,w)`, `L_grid (B,1,h,w)` |
| `DINOEncoder.forward(thermal)` | `(B,1,H,W)` → `patch_tokens (B,N,384)`, `cls_token (B,384)`, and with `return_attn` also `patch_attn (B,N,N)`: last layer, head-averaged, CLS and the 4 register tokens stripped, rows renormalized to sum to 1, detached |
| `DINOEncoder.forward_frozen(rgb_linear)` | `(B,3,H,W)` linear, adapters disabled, no gradient → the same `EncoderOutput`: the L_align target |
| `DINOEncoder.encode(thermal)` | `(B,1,H,W)` → dict: `patch_tokens (B,N,384)`, `patch_map (B,384,gy,gx)`, `cls (B,384)`, `thumb (B,1,gy,gx)` = `area_downsample(thermal, (gy,gx))`, and with `return_attn` also `attn_bias (B,1,N,N)` = `log(patch_attn + 1e-6)` |
| `VelocityTransformer(z_t, t, dino_features, dino_attn, thumb)` | `(B,N,D)`, `(B,)`, `(B,N,384)`, `(B,N,N)`, `(B,1,gy,gx)` → `v (B,N,D)`. `attn_bias (B,1,N,N)` may be passed instead of `dino_attn` |
| `ColorHead(patch_tokens)` | `(B,N,384)` → logits `(B,N,Q)` (training only) |
| `soft_encode(ab)` / `patch_histograms(ab_map, mask, patch_size)` | `(...,2)` → `(...,Q)`; `(B,2,h,w)`, `(B,1,h,w)` → `(B,N,Q)` |
| `euler_sample(velocity_fn, z_0, n)` | `(B,N,D)` → same, standardized |
| `sample_parts(velocity_model, dino_features, layout, mean, std, …)` | `(B,N,384)`, `(D,)` stats → dict of **raw** parts, each with leading `(num_samples, B)` |
| `area_downsample(img, (h, w))` | `(..., H, W)` float → `(..., h, w)`, same dtype and device |
| `read_rgb8(path)` / `read_thermal8(path)` | JPEG path → `(3, 712, 840)` / `(1, 712, 840)` uint8 raw codes (thermal: one channel, after checking that a 3-channel file's channels are identical) |
| `crop_border(img)` | `(..., 712, 840)` → `(..., 512, 640)` |
| `parse_annotation(path, class_map)` | XML → boxes (class, `(4, 2)` polygon in **cropped** pixel-index coordinates, source kind), folder, filename |
| `warp_thermal(thermal, T, (512, 640))` | `(1, H_t, W_t)` float, `T (3, 3)` thermal → RGB → `(1, 512, 640)` float32 warped, `(1, 512, 640)` bool valid |
| `moving_mask(...)` | boxes, matches, `T` → `(1, 512, 640)` bool, count |
| `linear_to_lab` / `lab_to_linear` | `(..., 3, H, W)` ↔ `(..., 3, H, W)` float32 |
| `masked_psnr(pred, target, mask, peak)` / `masked_mean(value, mask)` | `(B,C,H,W)` ×2, `(B,1,H,W)` → `(B,)` dB; `(B,H,W)`, `(B,1,H,W)` → `(B,)` |
| `ciede2000(lab_1, lab_2)` | `(..., 3, H, W)` ×2 → `(..., H, W)` |
| `colorfulness(srgb)` | `(B,3,H,W)` in 0–255 → `(B,)` |
| `AlignedPairs[i]` | dict: `thermal (1, 512, 640)` float32 = code / 255, `rgb (3, 512, 640)` linear float32, `valid`, `moving` and `vehicles` `(1, 512, 640)` bool, `image_id` str, `fallback` bool (aligned by edge correlation, see §10.2). Training adds flips (all four tensors alike) and a thermal-only monotone contrast curve; validation and test are deterministic |

## 8. Rules

1. **One trilinear implementation.** `fit.py` and `upsample.py` get node indices and weights from `slice.trilinear_weights`. They never compute `floor`, corner weights or coordinates themselves.
2. **One guidance function.** Fitting, slicing, rendering, inference and every script call `compute_guidance`. No inline blur or clamp code anywhere else.
3. **One token layout.** `dawn/flow/tokens.py` is the only place that packs, unpacks, slices or counts token values; the coefficient constants come from `dawn/grid/layout.py`. No ad-hoc `permute` / `view` of parts, no 16 / 32 / 48 / 64 literal outside the one layout test.
4. **One conversion point for standardization** (§5.2).
5. **One downsampling function.** `area_downsample` (§4a); no `F.interpolate`, PIL `resize` or `thumbnail` to shrink an image anywhere else. The one sanctioned bilinear *up*sample is `upsample_map` (§6.3), which the colour-ceiling baseline in `check_oracle.py` calls too.
6. **Conventions change here first.** If one has to change, edit this file in the same commit as the code and update the affected docstrings.
7. **One thermal resample.** `align.warp_thermal` is the only place thermal is resampled: bilinear inverse mapping onto the RGB frame. It interpolates at arbitrary positions rather than shrinking an image, so it is the one sanctioned resample that is not an area average (§4a is about downsampling). The RGB image is never resampled.
8. **Thermal is never sRGB** (§1), and the RGB target is never touched beyond the crop.
9. **One render path.** `flow.render.render` (§6.4). L_render, validation, the oracle and inference do not re-implement any step of it.
10. **`fit.py` and `upsample.py` use `slice.py`'s weights** (rule 1): the fitted grid and the rendered grid cannot drift apart. `upsample_map` is the only upsampling the render path calls.
11. **Lab is averaged as values** (§4a): targets are masked area averages in Lab, masks are fractions.

## 9. Known limitations

- The guidance blur is a fixed 5×5 **pixels**. At render size (256×320) 5 px is 1.6% of the width; at the 640×512 input it is 0.8%, so the effective smoothing differs between L_render and validation. If bins flicker at full resolution, scale the kernel with resolution and update §4.
- **Splat-and-solve is biased.** `fit_affine_grid` solves each node against the pixels that touch it, as if that node alone applied to them; the renderer then blends 8 nodes per pixel. The two disagree wherever the grid changes quickly from node to node, so a fitted grid does not re-render its own target exactly even with no noise anywhere. The error is small on smooth grids and is what `refine_steps` buys down (about +0.5 dB), by fitting the render residual and adding the correction. Judge a fit on the **rendered** image, never on coefficient error against a reference grid.
- **BGU smoothing.** Even at 2×2 colour pixels per cell the BGU's per-(cell, bin) affine model of colour on `L` loses colour detail that is not a function of `L`, which is why bilinear is the default (§6.3). In `bgu` mode, empty (cell, bin) nodes take their cell's fit through the ridge, and the guide used at fit time is the area-averaged `L` while the slice sees the full-resolution `L`.
- The token layout, the DINO patch grid and the brightness cells are one grid (§2.3); changing the encoder resolution changes all three together, and a checkpoint cannot be loaded at another resolution.
- The 261 bins come from the sRGB gamut, so a\*b\* cells the model could generate outside it do not exist as classes; the render clips to the gamut anyway.

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
