# Review of seven proposed additions

Written 8 October 2026, against the repo at that date: the data and flow phases done, the
inference/eval phase done except the refiner, and **nothing trained yet**. That last fact decides
most of what follows.

Not a conventions document. `docs/conventions.md` remains the only authoritative file for layouts
and units; where this note touches one it points at the section rather than restating it. It is a
record of what was proposed, what was measured against it, and what was decided, so the same
suggestions do not have to be re-litigated from scratch.

> **Acted on, 8 October 2026.** The recommendations in section 6 are now in the code; section 7
> lists exactly what changed and what it cost. **None of the seven proposals themselves was
> implemented** — what went in is the instrumentation that will decide three of them from a real
> run instead of from argument.

The proposals came in three batches: two review notes (overfitting, colour bleeding, hallucination
framing), a LUPI block of three auxiliary losses, and a perceptual-loss suggestion with a reply
arguing against it.

---

## 1. Premises that do not hold

Four factual claims underpinning the advice are wrong about this repo. They are listed first
because some of the recommendations rest on them.

| Claim as made | What the repo actually is |
|---|---|
| "your RTX 5090 budget" | `CLAUDE.md` says **A100**; `README.md` says RTX 5060 Ti. No 5090 anywhere, so the VRAM arithmetic in that advice does not bind |
| "the `one_cycle` schedule keeps the LR near its peak for a prolonged period" | `configs/flow.yaml` sets `optim.schedule: cosine`. One cycle is opt-in, and `docs/training_budget_and_changes.md` §2 already says to run cosine first for the baseline |
| "5,385 pairs is extremely small" | 5,385 is all three splits. Train is **3,662** — and `docs/training_budget_and_changes.md` §1 counts only about **1,049 independent runs of consecutive frames**, so the distinct-scene count is nearer a thousand. The concern is right and understated |
| "robust augmentations (flips, rotations, contrast curves)" | Rotations are **off** and `AugmentConfig.__post_init__` raises on `rot90` (`dawn/data/pairs.py`): a 90° rotation turns 640×512 into 512×640. Flips and a monotone thermal contrast curve only |
| "introduce the aux loss during the first 5,000 steps, while DINO is detached" | The detach gate is `flow.warmup_steps` = **3,000**. 5,000 is the plan's number; the implementation moved |

### The schedule claim is also quantitatively false

Even if `one_cycle` were selected, it does not sit near its peak longer than `cosine` does. Over
the configured 20,000 steps, with `pct_start: 0.2`, `div_factor: 25`, `final_div_factor: 1e4`
against `lr_warmup_steps: 1000`:

```
lr >= 0.9x peak:   one_cycle 20.6%   cosine 20.0%
lr >= 0.8x peak:   one_cycle 29.6%   cosine 29.0%
lr >= 0.5x peak:   one_cycle 50.3%   cosine 50.0%
mean lr factor:    one_cycle 0.504   cosine 0.500
```

(Computed directly from `one_cycle_schedule` and `cosine_warmup_schedule` in
`dawn/flow/trainer.py`.) The two differ by under a percentage point on every threshold. The real
differences are the ramp length — 4,000 steps against 1,000 — and the tail floor, which is what
§2 of the training-budget note already says.

### One premise that does hold

**`L_align` is already LUPI.** `FlowLoss.align` pulls the LoRA-adapted thermal patch tokens toward
`DINOEncoder.forward_frozen(rgb)` — privileged RGB features available only in training. So is
`L_color`: `ColorHead` is a 385-parameter probe over the 261 a\*b\* bins, discarded at inference.
The paradigm is already in use on two channels; the question is only whether a third helps.

---

## 2. Verdicts

| # | Proposal | Verdict |
|---|---|---|
| A | Watch for overfitting on a tiny dataset | **Real risk, wrong lever.** Act on observability instead |
| B | Colour bleeding from bilinear a\*b\* upsampling | **Reasonable theory, already measured against — but not at edges.** One cheap re-measurement |
| C | Hallucination framing | **Already in `README.md`, stated more carefully.** No action |
| 1 | Contrastive (InfoNCE) upgrade to `L_align` | **No.** Its negatives are the wrong negatives for this data |
| 2 | Bounding-box semantic head on DINO tokens | **Most feasible; data already on disk. Low value, and the available mask is the wrong shape.** Shelved behind a measurement |
| 3 | High-frequency Sobel edge distillation | **No.** Token grid too coarse, and the benefit cannot route to the output |
| 4 | LPIPS / ResNet perceptual metric inside `L_render` | **No** — though nearly every argument offered against it is also wrong |

---

## 3. The three review notes

### A. Overfitting — the risk is real, the suggested mitigation is not available

87 epochs over roughly a thousand distinct scenes is thin, and `configs/flow.yaml` already calls
40K steps "well into overfitting". Two things were missed in each direction.

- **Best-checkpoint selection already exists.** `scripts/train_flow.py` saves `flow_best.pt` on the
  box-aligned `psnr_L/euler`, so memorization costs the last checkpoint, not the run.
- **"Cut the epoch budget" contradicts the schedule it was offered against.** A one cycle is a
  fixed-budget anneal: `docs/training_budget_and_changes.md` §2 says not to early-stop one, because
  most of the gain is in the final low-LR stretch. Under `one_cycle` the only way to shorten is to
  re-run shorter. Under `cosine` — the default — cutting is fine.
- **The actual gap is observability.** `logging.val_every: 2500` over 20,000 steps gives **eight
  validation points**. That is not the resolution needed to see a divergence begin. This is the one
  item from the whole set worth acting on directly.
- **Selection has a colour blind spot.** `flow_best.pt` is chosen on L\* PSNR alone, and L\* comes
  mostly from the brightness grid fitted on thermal — the half of the model least able to memorize.
  Colour can degrade while PSNR climbs.

**Action.** Measure seconds/step in the 50-step `configs/gpu_probe.yaml` run, then set `val_every`
so validation costs a few percent of wall-clock (likely 500–1000, giving 20–40 points). Log
`ciede2000/euler` beside the selection metric; consider selecting on it.

### B. Colour bleeding — test it at edges, do not revert on it

`color.upsample: bilinear` was not a matter of taste. On the 345 val pairs the colour ceiling is
**0.90** mean |Δa\*b\*| bilinear against **1.28** for the best BGU swept, and the assembled oracle
CIEDE2000 is **3.33** against **3.50** (`conventions.md` §6.3, `outputs/oracle/report.md`). Two
things soften the warning further: the upsample is only **4×** (128×160 → 512×640), and BGU here is
guided by the generated `L`, which on a hot vehicle against cold road has *more* contrast than the
RGB does — the guide can manufacture an edge that is not a colour edge.

The objection is not fully answered, though, because **mean |Δa\*b\*| over the frame is not an edge
metric**. BGU could be better at boundaries and worse across the flat asphalt that dominates these
frames, and still lose on the mean.

**Action.** Add one number to `scripts/check_oracle.py`: mean |Δa\*b\*| restricted to pixels in the
top decile of RGB gradient magnitude, bilinear against BGU. It settles the question permanently,
`color.upsample` is already a switch, and it subsumes proposal 3 entirely — edge-aware colour
placement is the mechanism BGU implements.

### C. Hallucination framing — nothing to add

`README.md` and `conventions.md` §11 already say this, with a sharper point than the note made:
*a spread is a disagreement, not an error bar — a region the model is confidently wrong about has a
small spread*. `dawn/eval/calibration.py` exists precisely so the rank correlation against realized
error is measured rather than assumed. The note's "the K-sample spread maps correctly highlight this
uncertainty mathematically" is the weaker claim of the two.

---

## 4. The LUPI block

### 1. Contrastive spatial distillation (InfoNCE instead of cosine) — no

Cheap to build: 1,280 × 1,280 similarities per image is small beside the attention bias already
carried. The problem is that the proposed negatives are wrong for this dataset, twice over.

- **"Push away spatially adjacent patches."** Pairs are aligned to 1–2 px by boxes and 2–3 px by the
  edge-correlation fallback, and the fallback is **2,374 of 5,385** pairs (44%). A patch is 16 px,
  so at 3 px of residual the true correspondence genuinely spills into the neighbouring patch.
  Cosine-to-own-patch degrades gracefully under that; InfoNCE trains against it. `conventions.md`
  §10.2 already lists residual misalignment as a hazard, and this amplifies it.
- **"Push away visually similar patches."** Aerial frames are dominated by large uniform regions —
  asphalt, fields, roofs, parking lots. Dozens of patches per frame are *correctly* near-identical.
  Separating them teaches the encoder to encode position and JPEG noise, not semantics.

The collapse it guards against is also weak here: the target is the **frozen** RGB feature, fixed
and different per patch, so collapsing the adapted features toward a constant *raises* `1 − cos`.
There is no symmetric-collapse failure mode of the kind InfoNCE is designed for. LoRA rank 8 on
attention qkv/proj has little capacity to reorganize the representation in any case.

**Instead, measure the premise.** In `FlowTrainer.validate`, report patch retrieval: for each
adapted thermal token, is the nearest frozen-RGB token its own? One 1,280 × 1,280 argmax, nearly
free. If top-1 retrieval is already decent, "blurry semantic soup" is not the problem and a
contrastive loss buys nothing. If it is near chance, the plan's own cheaper fallback — raise the
LoRA rank to 16 — comes first.

### 2. Bounding-box semantic head — feasible, shelved behind a measurement

Credit where due: **the data is already on disk and already in every batch.** `<id>_vehicles.png`
is loaded by `AlignedPairs.__getitem__`, and `patch_mask_fraction` already turns a mask into
per-patch fractions. The head would be `384 → 1`, dropped at inference. Feasibility is not the
issue.

Three reasons not to do it now:

1. **The mask is the wrong shape for the claim.** The `vehicles` mask is deliberately
   over-inclusive (`conventions.md` §10.2): 3 px dilation, and **17.4 px stretched along the
   heading** for any vehicle that may have moved, covering **12% of a frame on average**. A head
   trained on that learns "a vehicle is smeared somewhere around here" — the blurry-blob signal the
   proposal set out to eliminate. Delivering what it claims needs a tight-polygon mask variant:
   another `write` pass, cheap but not free.
2. **It contradicts its own mask.** For the 44% fallback pairs, `vehicles` *is* the exclusion mask
   (`pairs.fallback_pixel_mask: vehicles`). The head would be asked to predict exactly the pixels
   every other loss deletes.
3. **`L_color` already subsumes it for the objective that matters.** A per-patch cross-entropy over
   261 a\*b\* bins against the true histogram already teaches "this patch contains saturated red,
   not asphalt grey". For a colouriser, vehicle *class* identity is strictly less information than
   vehicle *colour distribution*. The plan's decision log already rejected a segmentation loss; its
   stated reason ("no masks exist") is now outdated, but the conclusion survives on a better one.

**Action.** Shelve behind `metrics.vehicle_region`, the open decision in
`docs/inference_eval_phase.md`. If run 1 shows vehicles painted road-grey — low colourfulness and
high CIEDE2000 on vehicle crops — this is the right fix, with a tight mask. Do not pay for it
speculatively.

### 3. High-frequency edge distillation — no

Two structural problems.

- **The head would predict at 32 × 40**: one value per 16 × 16 patch. In a dense aerial urban frame,
  "does this patch contain an edge" is true nearly everywhere — a near-saturated target carrying
  almost no signal. Making it informative needs a higher-resolution prediction, which needs a
  decoder, which is real capacity and a real conflict with `L_flow`.
- **The benefit cannot route to the output.** The claim is that it helps the velocity net generate
  colour transitions that respect structural boundaries. But the colour path is a 128 × 160 a\*b\*
  map bilinearly upsampled: it is *structurally incapable* of placing a colour discontinuity more
  precisely than that grid allows. Knowing where an edge is provides no mechanism to put a colour
  edge there. The mechanism that would is BGU — see B above, where it was measured and rejected.

Secondarily, daylight Sobel edges are dominated by shadows, lane paint and surface texture that
thermal physically cannot see, so a large share of the target is unlearnable. `L_align` already
distills RGB structure at the abstraction level DINO is good at.

---

## 5. Perceptual loss inside `L_render` — right conclusion, wrong reasoning

The reply arguing against it reaches "do not", but three of its four arguments do not hold.

- **"Gradients backpropagating through the ODE step will bend your trajectories."** There is no ODE
  integration in training. `FlowTrainer.compute_losses` computes `z1_hat = z_t + (1 - t) * v_pred`,
  a single algebraic extrapolation, and `L_render` is *already* a pixel loss on exactly that
  estimate. Swapping L1 for LPIPS changes the metric, not the category. Rectified-flow straightness
  comes from the coupling, not from the loss metric.
- **"Architectural redundancy with `L_align`."** No. `L_align` constrains the **encoder's features**;
  a perceptual loss constrains the **output image**. `L_align` never sees anything the model
  generated. Different objectives entirely, so "downgrading your feature-matching signal" does not
  describe what would happen.
- **"A severe throughput bottleneck."** Order of magnitude: AlexNet-LPIPS over the 64 rendered
  samples per step at 256 × 320 is a low single-digit percentage of a step already dominated by two
  ViT-S+ passes over 1,280 tokens. VGG-LPIPS is roughly twenty times that and *would* be material —
  so the claim is right only for the backbone it did not recommend.

The reasons that do hold:

1. **LPIPS is the headline eval metric.** `configs/eval.yaml` sets `lpips: true`, `lpips_net: alex`,
   and the plan's evaluation table uses it for "does it look like a photo?". Training on it destroys
   the one number that answers that question. This is decisive on its own.
2. **Perceptual losses buy texture this architecture cannot produce.** Colour is a 128 × 160 map,
   bilinearly upsampled; there is no high-frequency colour degree of freedom to reward. The gradient
   would be absorbed by whatever can move — the brightness grid coefficients — in ways nobody asked
   for.
3. **`ẑ₁` is garbage over most of the `t` range.** At small `t` it is a one-step jump from near-pure
   noise. L1 on garbage gives a bounded, smooth gradient; a deep feature stack on garbage gives
   whatever it gives.

Worth noting on the other side: `L_render` at weight 0.1 on the one-step estimate is effectively a
mild one-step-distillation term, which is well matched to the 2-NFE inference budget. It is not
fighting `L_flow`; it insures the quantity actually sampled.

---

## 6. What to do instead

The common flaw across all seven is that they are new knobs for a project with four hand-weighted
losses, **zero trained runs**, and eight validation points per run. Everything here is unattributable
until a baseline exists, which is what `docs/training_budget_and_changes.md` §2 already says:
changing two things in one run makes the result unattributable.

In order:

1. **Run the `cosine` baseline.** Every item above is speculation about curves nobody has seen.
2. **Raise validation resolution** (`logging.val_every`, after the probe gives seconds/step) and log
   `ciede2000/euler` beside the PSNR used for selection. The only item from the critique to act on
   directly.
3. **Add three measurements that test the proposals' premises rather than accepting them**, all
   small:
   - patch-retrieval top-1 in `FlowTrainer.validate` → tests proposal 1's "semantic soup"
   - edge-restricted |Δa\*b\*| in `scripts/check_oracle.py` → tests the bleeding warning *and*
     proposal 3's goal
   - vehicle-crop CIEDE2000 and colourfulness (`metrics.vehicle_region`, already an open decision)
     → tests proposal 2
4. **After run 1**, revisit proposal 2 only if the vehicle crops are bad, and only with a tight mask.

---

## 7. What was implemented

Items 2 and 3 of section 6, plus the telemetry needed to read the result. The full reasoning and
every config key is in `docs/training_budget_and_changes.md` §1a, §1b and §3; this is the index.

| Change | Where | Why |
|---|---|---|
| `logging.val_every` 2,500 → **500** | `configs/flow.yaml` | 40 validation points over the run instead of 8 |
| Held-out **training loss** (`validate_loss`) | `dawn/flow/trainer.py` | The actual overfitting signal. Measured on *live* weights, so it is comparable with the train curve; `validate`'s image scores stay on EMA |
| `flow_best.pt` selects on **CIEDE2000** (`logging.select_metric`) | `scripts/train_flow.py` | L\* PSNR is mostly the brightness grid fitted on thermal, and blind to colour going wrong. CIEDE2000 is brightness *and* colour. `psnr_L` stays selectable |
| **Patch retrieval** `top1` / `within_1` / `chance` | `trainer.patch_retrieval` | Tests proposal 1's premise directly. `within_1` separates residual misalignment from features that do not localize |
| **Edge-restricted** colour ceiling (`ab_err_*_edge`) | `dawn/eval/metrics.py: edge_quantile_mask`, `dawn/flow/oracle.py` | Tests 3.B — and proposal 3's goal with it. The whole-frame mean cannot see bleeding at boundaries |
| `ciede2000/gap_to_oracle`, `colorfulness/ratio_*` | `trainer.validate` | A colour twin of the PSNR gap; colourfulness is unreadable except against the ground truth's |
| `contrib/*`, `grad/*` per module | `train_step(detailed=True)` | Makes loss balancing and the detach gate legible — the exact risk the LUPI note raised about *adding* a loss applies to the four that are already there |
| **EMA decay warmup** (`optim.ema_warmup`) | `trainer.EMA` | A fixed 0.999 leaves the shadow 61% initialization at step 500. Without this, the extra validation points would have measured noise |
| **Conditioning ramp** (`flow.warmup_ramp_steps`) | `trainer.conditioning_scale` | The detach gate was a one-step flip from frozen features to moving ones. Gradient-only scaling, so a ramp of 0 is the old behaviour exactly |
| `optim.cosine.min_factor` | `build_schedule` | Exposed, default 0.0 — a knob, not a recommendation |
| W&B mirrored to **JSONL + CSV + `run_config.json`** | `scripts/train_flow.py: RunLog` | Plots rebuildable without W&B. One call site, so the records cannot drift (they already had) |
| `configs/gpu_5090.yaml` | new overlay | 32 GB spent on `timesteps_per_image: 8`, not on the batch, with `render_subset: 64` so L_render keeps its cost |

**What it cost.** 47 new tests (892 → 939, still ~70 s offline on CPU), `ruff check` clean. The
two schedule changes are strict generalizations: with `ema_warmup: false` and
`warmup_ramp_steps: 0` the run is bit-for-bit the old one, which is what keeps a baseline
comparable. Validation is now order 4–8% of wall-clock at `val_every: 500`; the probe gives the
real number, and `val_retrieval: false` is the first dial if it is too much.

**Still not done.** Proposal 2's vehicle-crop metrics (`metrics.vehicle_region`) remain the open
decision in `docs/inference_eval_phase.md`, and the plan's stage-2 refiner
(`dawn/models/refiner.py`, `scripts/train_refiner.py`, `configs/refiner.yaml`) does not exist —
it is first in the plan's own cut order.
