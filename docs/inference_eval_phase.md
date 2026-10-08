# The inference and evaluation phase

What was built after the flow phase, why each piece is the shape it is, and what is still open.
Operational notes, not conventions: `docs/conventions.md` remains the only authoritative file for
layouts and units, and the plan
(`docs/Dawn Night-Vision to Day Colorization for Drone Imagery.md`) is the spec this implements.

Written while **nothing is trained yet**. Every module here was built and tested against random
weights, exactly as the flow phase was. They are waiting on a checkpoint, not on more code.

---

## 1. What the phase delivers

Against the plan's evaluation table:

| plan's question | measure | where |
|---|---|---|
| Is the structure right? | PSNR **and SSIM** on L\* | `masked_psnr`, `masked_ssim` |
| Are the colours close? | mean CIEDE2000 | `ciede2000` + `masked_mean` |
| Are the colours vivid, not brown? | colorfulness (Hasler & Suesstrunk) | `colorfulness` |
| Does it look like a photo? | LPIPS | `lpips_distance` |
| Is the uncertainty honest? | rank correlation of spread against error | `eval/calibration.py` |
| Do vehicles survive? | all of the above over the vehicle region | `evaluate_batch(vehicle_mask=…)` |
| Is it fast? | latency and GMACs per component | `scripts/benchmark_latency.py` |

Against the plan's repo structure:

| spec path | state |
|---|---|
| `dawn/uncertainty/spread.py` | **done** — K rendered samples reduced to two maps |
| `dawn/eval/metrics.py` | **done** — SSIM and LPIPS added to the existing three |
| `dawn/eval/calibration.py` | **done** |
| `dawn/eval/visualize.py` | **done** |
| `scripts/infer.py` | **done** |
| `scripts/evaluate.py`, `configs/eval.yaml` | **done** — rewritten off the VisDrone originals |
| `scripts/benchmark_latency.py` | **done** |
| `dawn/models/refiner.py`, `scripts/train_refiner.py`, `configs/refiner.yaml` | **not built** — the plan's own first thing to cut |

There are now **no `NotImplementedError` stubs left** in `dawn/` or `scripts/`, and `ruff check
dawn scripts tests` is clean.

One file is **not** in the plan's structure: `dawn/flow/inference.py`. It holds `load_checkpoint`,
which `infer`, `evaluate` and `benchmark_latency` all need; without it each would rebuild the
models and, worse, could each get the stats from somewhere different. If the structure should
match the plan exactly, fold it into one of the named modules.

---

## 2. The modules

### `dawn/uncertainty/spread.py`

K samples in, one image and two maps out. `sample_parts` already draws the K members off a single
DINO pass, so this renders each and reduces them.

- **Units are CIELAB.** `l_std` is the standard deviation of L\* across members, in L\* units;
  `ab_std` is `sqrt(var(a*) + var(b*))`, the RMS distance from the mean colour **in the a\*b\*
  plane**. The second is a chroma *distance*, on the same scale as a CIEDE2000, which is what
  makes the calibration row well posed. A per-channel a\*b\* spread would not be.
- **The spread is taken on rendered L\*a\*b\*, never on tokens.** A token's variance says nothing
  about how much of it survives the grid slice and the colour upsampling.
- Members are reduced by **Welford**, so peak memory is two renders whatever K is.
- `SpreadResult` carries member 0's Lab as well as its RGB. `render` clamps RGB to the gamut, so
  deriving Lab back from it would quietly score the clipped colour instead of the generated one.
- `num_samples == 1` returns zeros, not NaN.

### `dawn/flow/inference.py`

`load_checkpoint` → encoder (+LoRA), velocity net, layout, stats, render settings, sizes, step.

The rule it exists to enforce is §5.1: **stats come from the checkpoint, never from
`grid.stats_cache`**. A run that read the cache would unstandardize with whatever the cache holds
today — plausible-looking wrongness rather than a crash. `tests/test_inference_load.py` pins it
with two checkpoints that differ only in their stats.

EMA weights are applied by default, because that is what validation used.

### `scripts/infer.py`

The plan's inference path: read thermal → DINO once → K samples at 2 NFE → render each → reduce →
write.

- Raw 840×712 frames lose the 100-px border; anything larger than the model's input is area
  averaged down; anything **smaller is refused**, because upsampling thermal would invent the
  texture the brightness grid is guided by.
- Uncertainty maps use a **fixed** `[0, scale]` ramp (`--l-scale`, `--ab-scale`), not per-image
  normalization: two frames' maps are only comparable on a shared scale.
- `--seed-from-name` draws `z_0` from the file stem, so a frame renders identically anywhere.

### `dawn/eval/metrics.py`

`masked_ssim` and `lpips_distance` are new; the colour difference and colorfulness were already
there from the flow phase.

- **SSIM matches `skimage` to 1e-8 on the interior.** skimage discards the `window_size // 2`
  border; this keeps it (reflect-padded) and lets the mask decide, so an all-ones mask differs by
  about 6e-4 — that gap is the border and nothing else, which the test proves by masking the
  border out and showing exact agreement. Default `peak=100`, because the plan scores L\*.
- **LPIPS converts linear → sRGB first.** The released nets are trained on display-referred
  images; feeding linear RGB in scores a different, darker pair than a person sees. There is
  deliberately **no masked LPIPS**: zeroing pixels introduces edges a conv stack reads as
  structure.
- `MetricResults` was still the VisDrone container (crop metrics on detection boxes, a YOLO
  appeared/disappeared audit). Replaced with the thermal table.

### `scripts/evaluate.py` + `configs/eval.yaml`

The whole table on `test_pairs`, against the real paired RGB, both samplers at the same NFE.

- **The model side is not configurable.** Layout, resolution, upsampling and stats all come out
  of the checkpoint. `eval.yaml` only chooses the split, the budget and the reporting.
- **The headline is box-aligned pairs** (`eval.headline_subset`). Fallback pairs were aligned by
  edge correlation alone, so they are reported separately as a robustness number.
- **The single sample is the headline, the ensemble mean is reported beside it.** Averaging K
  colourings helps CIEDE2000 and hurts colorfulness — exactly the trade the plan warns about — so
  reporting only one of them would flatter the model on whichever metric was chosen.
- Writes `report.md`, `summary.json` and a **per-pair `per_pair.csv`**, so the worst cases can be
  pulled up rather than hidden behind a mean. Percentiles sit beside every mean.

### `dawn/eval/calibration.py`

The plan's row is a **rank** correlation, and that word is doing real work.

The spread of K samples is a *disagreement*, not a predictive standard deviation: nothing trains
it to come out in L\* units and nothing makes `E[|error|] == spread`. A raw ECE, which compares
magnitudes, would therefore report a large number for a perfectly useful map. Spearman's rho asks
only the answerable question — *where the model disagrees with itself more, is it in fact more
wrong?* — and is invariant to any monotone rescaling.

- `rank_correlation` returns 0.0, never NaN, when it is undefined (constant map, <2 pixels).
- `calibration_curve` bins by **quantile**, not equal width: the spread is heavily skewed, so
  equal-width bins would put nearly everything in the first one. It also reports `monotonicity`,
  the share of adjacent bins where the error rises — 1.0 for a perfectly ordered curve, ~0.5 for
  noise.
- `expected_calibration_error` is still there, but it **fits and reports the scale** it used
  (`fit_scale`: least squares `|error| ~ a · spread`, through the origin). An ECE without its
  scale measures the arbitrary unit choice, not calibration. Fit the scale on validation and pass
  it in for an honest number; `scale=None` fits in-sample and is optimistic.

### `dawn/eval/visualize.py`

Sheets of `thermal | generated | real RGB | L* spread | a*b* spread`, one row per pair.

- Grids leave as **sRGB** — the output boundary. Thermal is shown as repeated grey, not false
  colour, because it is not sRGB and carries no temperature scale this project can claim.
- Ramps are **fixed**, matching `infer.py`'s defaults, so a grid and a saved map mean the same.
- Sheets are written with a sibling `.txt` of image IDs, so a bad row can be traced to its frame.

### `scripts/benchmark_latency.py`

Per component, not just end to end, because the architecture's central claim is that expensive
work runs once. MACs come from torch's own `FlopCounterMode` — no new dependency, no
hand-maintained op table — reported as `flops / 2`, the plan's convention. Timings are medians
over `--trials`, after `--warmup`, with explicit CUDA/MPS synchronisation (without it every
timing is of the enqueue, not the work).

**It needs no trained model.** Cost depends on architecture and shapes, not weight values.

---

## 3. Measured: the plan's cost estimates look right

Run on CPU with random weights at the **half-resolution** `mps.yaml` setting (256×320, a 16×20
token grid = 320 tokens):

| component | GMACs |
|---|---|
| encoder (DINOv3 + LoRA), once | 9.45 |
| velocity × 2 NFE × 1 sample | 3.10 |
| render (slice + bilinear + Lab) | 0.01 |
| **total per frame** | **12.56** |

At the real 640×512 that is 4× the tokens, with attention quadratic on top, which puts the
encoder near the plan's ~45 GMACs and the velocity near its ~10. Nothing here contradicts the
plan's ~65 GMACs at 2 NFE. **Confirm it in one command on the real hardware** — the numbers above
are not a substitute:

```bash
python scripts/benchmark_latency.py --device cuda --output outputs/bench/real.json
python scripts/benchmark_latency.py --device cuda --samples 8    # the uncertainty path
python scripts/benchmark_latency.py --device cuda --no-attn      # what the attention bias costs
```

That last one is worth running. The DINO attention reaches the velocity net as an additive float
mask, which stops SDPA using a flash kernel; `--no-attn` prices exactly that choice.

---

## 4. Open: how the vehicle rows are scored

`metrics.vehicle_region` in `configs/eval.yaml`. **The maintainer picks; the code does not guess.**

The plan asks for metrics on *"vehicle crops from the RGB boxes"*, but boxes are **not persisted**
with the aligned pairs — `ASSET_KINDS` is `rgb/thermal/valid/moving/vehicles`, all PNGs.
Re-deriving true boxes needs the raw 14 GB DroneVehicle download plus the per-pair transform in
`data/manifests/logs/align.csv`.

| option | what it scores | cost |
|---|---|---|
| `mask` (default) | the `vehicles` mask as the region | nothing extra, but the mask is dilated 3 px and stretched 17.4 px for vehicles that may have moved, so it is larger than the true boxes and includes background |
| `boxes` | bounding boxes of that mask's connected components | closer to the plan's "crops"; touching vehicles merge into one component |
| `none` | skips the vehicle rows | — |

`report.md` prints which was used, so a number can never be read without its definition.

---

## 5. Running it

```bash
# the ceiling, before anything is trained
python scripts/check_oracle.py

# after a checkpoint exists
python scripts/evaluate.py --checkpoint checkpoints/run1/flow_best.pt --device cuda
python scripts/evaluate.py --checkpoint ... --split val_pairs --max-pairs 40 --device cpu  # quick
python scripts/infer.py --checkpoint ... --input frame.jpg --samples 8 --save-raw
python scripts/benchmark_latency.py --device cuda
```

`evaluate.py` writes `report.md`, `summary.json` and `per_pair.csv` under `eval.output_dir`.

---

## 6. What is left

1. **Training.** Nothing is trained; every number in the repo is still an oracle ceiling. See
   `docs/training_budget_and_changes.md` for the epoch budget and the one-cycle schedule.
2. **The refiner** — `dawn/models/refiner.py`, `scripts/train_refiner.py`, `configs/refiner.yaml`.
   The plan lists it first in its own cut order. `dawn/models/denoiser.py` survives only because
   its `DepthwiseSeparableConv(k=5)` is the right starting point; `BoundedDenoiser`'s
   `(enhanced, log_variance)` signature belongs to the retired noise model.
   `noise_head.py`, `train_denoiser.py` and `denoiser.yaml` were deleted as dead.
3. **Wire `visualize.py` into `evaluate.py`.** `report.num_examples` exists in `eval.yaml` and
   nothing reads it yet; the grids are written by hand today.
4. **Decide `metrics.vehicle_region`** (§4).
5. **A floor baseline.** There is a ceiling (24.6 dB oracle) and no floor, so a trained 21 dB has
   no context. Two cheap ones: a single global thermal→L\* curve with a\*=b\*=0, and per-image
   mean colour. If the model cannot beat "paint everything the average colour", that is worth
   knowing on day one.

---

## 7. Testing

Everything was built against random weights and a two-block random-init encoder, the way the flow
phase was; **no test downloads DINOv3 weights or reads `data/DroneVehicle`**. `evaluate.py` is
exercised end to end on a tiny aligned tree written to `tmp_path`.

| file | tests |
|---|---|
| `test_metrics.py` | 20 |
| `test_calibration.py` | 19 |
| `test_visualize.py` | 15 |
| `test_spread.py` | 15 |
| `test_benchmark.py` | 11 |
| `test_inference_load.py` | 9 |
| `test_evaluate_script.py` | 9 |
| `test_infer_script.py` | 8 |

**892 tests in the suite, all passing**, CPU only and offline, in about 55 seconds.
