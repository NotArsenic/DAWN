# Training budget, the one-cycle schedule, and what changed on 2026-10-07

Operational notes, not conventions. `docs/conventions.md` is still the only authoritative file for
layouts and units; where this note touches a convention it points at the section rather than
restating it. The spec (`docs/Dawn Night-Vision to Day Colorization for Drone Imagery.md`) predates
everything here.

Written at the point where the data and flow phases were done and nothing had been trained yet.

---

## 1. Choosing the training budget

**Count epochs, not steps.** `flow.num_train_steps` is the knob, but the quantity that decides
whether the model underfits or memorizes is how many times it has seen the training set:

```
steps_per_epoch = len(train_pairs) / optim.batch_size = 3662 / batch_size
num_train_steps = epochs * steps_per_epoch
```

| `optim.batch_size` | steps / epoch | 80 epochs | 100 epochs | |
|---|---|---|---|---|
| 8 | 458 | 36,600 | 45,800 | |
| 16 (`flow.yaml` default) | 229 | 18,300 | 22,900 | |
| 32 | 114 | 9,200 | 11,400 | |
| 64 | 57 | 4,600 | 5,700 | the schedule constants need rescaling |
| 128 | **29** | 2,300 | 2,900 | **broken, see below** |

**Recommendation: 80-100 epochs for the first full run.** At batch 16 that is the 20,000 steps
already in `flow.yaml` (87 epochs).

### Why a big batch is the wrong way to spend an A100

The temptation on a 40/80 GB card is `batch_size: 128`. It does not work here, and the reason is
the dataset, not the card: 3,662 pairs over a batch of 128 is **29 steps per epoch**, so a
100-epoch run is 2,900 optimizer steps in total.

- 2,900 steps is **fewer than `flow.warmup_steps` (3,000)**, so the conditioning detach would
  never lift and the velocity net would train on frozen features for the entire run. That is a
  silent wrong result, not a crash.
- `logging.save_every: 5000` would never fire; `val_every: 2500` would fire once.
- EMA at 0.999 has a horizon of ~1,000 steps, which at batch 128 is 35 epochs — over a third of
  training, so the average lags the weights badly.
- More fundamentally, a 7M-parameter transformer plus LoRA wants **optimizer steps**, not just
  samples seen. Below a few thousand steps you are underfitting however many epochs that is.

**Spend the headroom on `flow.timesteps_per_image` instead.** It is the one knob that buys compute
without costing epoch budget: one encoder pass already feeds `timesteps_per_image` velocity passes,
so raising it from 4 to 8 doubles the velocity samples per step and leaves steps-per-epoch
untouched. `batch_size: 32` × `timesteps_per_image: 8` is 256 velocity samples per step at 114
steps per epoch — the arithmetic intensity you wanted from batch 128, with 4× the optimizer steps.

| what you want | do this | not this |
|---|---|---|
| use the A100's memory | `timesteps_per_image: 8`, `batch_size: 16-32` | `batch_size: 128` |
| finish sooner | fewer epochs, same batch | a bigger batch at the same epoch count |

If you do go past 32, rescale every constant in the table below **and** check that
`num_train_steps` stays comfortably above `warmup_steps`; a run whose budget is near its own
warmup is not the experiment you meant to run.

Why not more: there are 3,662 train pairs but only about **1,049 independent runs of consecutive
frames** in them (median run 1 frame, longest 127), so the count of genuinely distinct scenes is
nearer a thousand. Augmentation is flips and a monotone thermal contrast curve; `augment.rot90` is
off, because a 90° rotation turns 640×512 into 512×640 and breaks both batching and the velocity
net's learned 40×32 position embedding. `flow.yaml` already judges ~175 epochs "well into
overfitting" and that looks right.

Why not fewer: `flow.warmup_steps` (3,000, about 13 epochs at batch 16) is spent with the
encoder's patch tokens detached, so LoRA is learning from L_align and L_color alone for the first
sixth of the run. A short budget spends most of itself in that regime.

Extend only if **box-aligned** validation is still improving (`pairs.eval_subset: box_aligned`);
the fallback subset is a robustness number, not a selection signal.

### The rescaling trap

Every knob below is denominated in **steps**. Change `optim.batch_size` without changing them and
you have quietly run a different experiment: doubling the batch to 32 and halving the step count
takes the detached phase from 13 epochs to 26.

| key | what it means in epochs |
|---|---|
| `flow.num_train_steps` | the budget itself |
| `flow.warmup_steps` | how long the conditioning stays detached |
| `optim.lr_warmup_steps` | the LR ramp (ignored under `one_cycle`, see §2) |
| `logging.val_every` | how many validation points the run produces |
| `logging.save_every` | checkpoint spacing |
| `logging.log_every` | curve resolution |

Scale the **peak LR** with the batch too: about `2.8e-4` at batch 32 by the square-root rule, `4e-4`
by the linear rule, from the `2e-4` baseline at 16. One cycle is less forgiving of too high a peak
than the plain cosine is, because it sits near the peak for much longer.

### Hardware

Training is on an **A100**, not the 16 GB RTX 5060 Ti the spec and `README.md` assume. That retires
the VRAM concern those documents are built around: at batch 16 × 4 timesteps the velocity net's
attention needs roughly 5 GB (the DINO attention bias is an additive float mask, so SDPA cannot use
a flash kernel and has to materialize `64 × 4 × 1280 × 1280` scores per block and keep them for
backward). On 16 GB that was tight enough to expect the 8 × 4 fallback; on an A100 it is not a
constraint, and a larger batch is available if the step count is re-derived from the table above.

`configs/gpu_probe.yaml` is still worth 50 steps before an overnight run, for seconds/step.

---

## 2. The one-cycle schedule

`optim.schedule` selects it; `cosine` remains the default.

```yaml
optim:
  schedule: one_cycle
  one_cycle:
    pct_start: 0.2          # fraction of the budget spent ramping up
    div_factor: 25.0        # the ramp starts at lr / 25
    final_div_factor: 1.0e4 # the anneal ends at lr / 10,000
    cycle_momentum: true
    max_beta1: 0.95
    min_beta1: 0.85
```

### What it changes

The existing `cosine_warmup_schedule` is already a ramp followed by a cosine decay to zero, so the
shape of the tail is not the difference. The difference is:

| | `cosine` | `one_cycle` |
|---|---|---|
| ramp | `optim.lr_warmup_steps` = 1,000 (5% of the run), linear | `pct_start` = 20%, cosine |
| ramp starts at | ~0 | `lr / div_factor` |
| a group's `lr` is | the plateau it holds after warmup | the **peak**, touched once |
| ends at | 0 | `lr / final_div_factor` |
| β₁ | fixed at `optim.betas[0]` | cycled 0.95 → 0.85 → 0.95, opposite the LR |

Per-group peaks keep their ratio: `encoder.lora_lr` (1e-4) stays exactly half of `optim.lr` (2e-4)
at every step, and neither is ever exceeded.

### Three things to know before using it

1. **It is a fixed budget.** The anneal lands on `flow.num_train_steps` and most of the gain is in
   the final low-LR stretch. A one-cycle run stopped at 60% is worse than a one-cycle run that was
   *planned* for 60%. Decide the budget first (§1); do not early-stop a cycle.
2. **It interacts with the detach gate.** `pct_start: 0.2` of 20,000 steps peaks at step 4,000,
   which is 1,000 steps after `flow.warmup_steps` hands LoRA its velocity gradients. That ordering
   is deliberate: the features settle, LoRA joins, and only then is the LR hottest. If either number
   moves, keep the peak **after** the gate.
3. **Do not stack it with another change.** With a handful of runs available, changing the schedule
   and (say) `brightness_correction.enabled` in the same run makes the result unattributable. Run
   `cosine` first for the baseline, `one_cycle` second.

### Why not `torch.optim.lr_scheduler.OneCycleLR`

It raises `ValueError` if stepped past `total_steps`, which makes a resume that oversteps by one
fatal. `one_cycle_schedule` is a `LambdaLR` whose factor clamps instead, matching
`cosine_warmup_schedule`'s existing behaviour. Momentum lives in a separate `MomentumCycler` so the
plain cosine path can keep `optim.betas` untouched; its step count is in the checkpoint and
round-trips on resume.

Both are in `dawn/flow/trainer.py`; `build_schedule(optimizer, cfg)` is the one place that reads
`optim.schedule`. `train_step` logs `beta1` alongside the per-group learning rates whenever cycling
is on.

---

## 3. Weights & Biases: weight and gradient histograms

W&B itself was already wired (`WandbRun`, `logging.wandb_project`, `logging.wandb_mode`, and the
per-step loss / LR / attention-gate logging). What is new is weight and gradient histograms:

```yaml
logging:
  watch_scope: velocity   # none | velocity | all
  watch_freq: 500
```

`WandbRun.watch` calls `wandb.watch(..., log="all")`, which installs hooks that log weight and
gradient histograms.

**The scope matters more than the frequency.** The hooks fire on *every* backward pass whatever
`watch_freq` says; only the logging is throttled. And `wandb.watch` takes whole **modules**, not
parameters, so the scope is the only unit of choice available:

| scope | hooks |
|---|---|
| `velocity` (default) | the 7M-parameter net that learns the field |
| `all` | plus the encoder and the colour head. The encoder is 29M parameters of frozen DINO backbone wrapped around ~0.5M of LoRA, and watching it hooks all of them — for debugging LoRA, not for a real run |
| `none` | off |

A bad `watch_scope` raises at the start of the run rather than silently logging nothing for 20K
steps, and it raises even when `wandb_mode` is `disabled`, so a typo cannot hide in a debug run.

wandb is an optional dependency: if it is not installed, or `wandb_mode: disabled`, the run logs to
the console and `train_log.jsonl` and nothing here fires. `tests/test_wandb_logging.py` covers all
of that with a fake `wandb` module, since CI has no real one.

---

## 4. Progress reporting, and why not tqdm everywhere

`tqdm` was a declared dependency that **nothing imported**. It is now used, but deliberately not
everywhere, because a progress bar and a log file want opposite things: tqdm redraws with a
carriage return, so an overnight run under `nohup` or `tmux` fills the file that the person
debugging it later has to read.

`dawn/progress.py` makes the choice per stream rather than per caller:

| stderr | what you get |
|---|---|
| a TTY | a tqdm bar |
| redirected | one log line every 5%, with elapsed, rate and ETA |

| pass | before | now |
|---|---|---|
| `prepare_dronevehicle.py` stats / align / write | `align: 4200/8534` every 5% | the same line, plus rate and ETA. Always lines: these passes are the record of what was done |
| token stats (2,000 pairs, before step 1 of every training run) | **silent for minutes** | `token stats (2000 pairs)` with ETA |
| `check_oracle.py` over the val pairs | **silent** | `oracle` with ETA |
| the training loop | `step 1200 \| total 0.41 \| ... \| 0.42s` | `step 1200/20000 (6%) \| ... \| 0.42s/step \| ETA 2h04m` |
| `infer.py` | one line per frame | unchanged; frames are few and each line carries its spread |

The training loop keeps **lines, not a bar**, on purpose. It is the one thing always watched
through `tail -f` on a remote box, its output is the run's record, and W&B already has the curves.
What it was missing was the answer to "is this finishing tonight?", which is now on every line.

## 5. Dependencies actually installed

`pip install -e ".[dev]"` has not fully run in `.venv`. Three declared packages are missing:

| package | consequence |
|---|---|
| `wandb>=0.16` | **`WandbRun` logs a warning and falls back to console-only.** A 20K-step A100 run would train to completion with no W&B tracking at all and only one `WARNING` line to say so. Install it before the real run |
| `lpips>=0.1.4` | the plan's LPIPS metric cannot be computed (eval phase, not written yet) |
| `scikit-image>=0.21` | not imported anywhere yet; presumably intended for SSIM |

`black>=23.0` and `ipython>=8.0` are also missing from the dev extra and do not matter — `black`
is configured in `pyproject.toml` but `ruff format` is what is actually used, so the `[tool.black]`
section is dead configuration.

Everything else is installed and the suite runs offline on CPU.

## 6. What changed in the code

The data and flow phases were complete and tested before this; what follows is the first slice of
the inference/eval phase, plus the two training knobs above.

### New

| file | what it is |
|---|---|
| `dawn/uncertainty/spread.py` | sample-spread uncertainty: K rendered members → one image and two maps. `docs/conventions.md` §11 |
| `dawn/flow/inference.py` | `load_checkpoint` → encoder (+LoRA), velocity net, layout, stats, render settings, sizes |
| `tests/test_spread.py`, `test_inference_load.py`, `test_infer_script.py`, `test_schedule.py` | 47 tests |

### Rewritten

- **`scripts/infer.py`** was an old-regime stub about "night images" and "analytic propagation". It
  now runs the plan's inference path: read thermal → DINO once → K samples at 2 NFE → render each →
  reduce → write.
- **`dawn/flow/trainer.py`** gained `one_cycle_schedule`, `MomentumCycler` and `build_schedule`.
- **`scripts/train_flow.py`** gained `WandbRun.watch`.
- **`configs/flow.yaml`** gained `optim.schedule`, `optim.one_cycle.*`, `logging.watch_scope`,
  `logging.watch_freq`.
- **`docs/conventions.md`** gained §11 (uncertainty), an at-a-glance row and a §7 shapes row.

### Decisions worth knowing

- **Uncertainty is reported in CIELAB units.** `l_std` is the standard deviation of L\* across
  members, in L\* units; `ab_std` is `sqrt(var(a*) + var(b*))`, the RMS distance from the mean
  colour in the a\*b\* plane. The second is a chroma *distance*, on the same scale as a CIEDE2000,
  which is what makes the plan's "rank the spread against the realized error" evaluation
  well-posed. A per-channel a\*b\* spread would not be.
- **The spread is computed on rendered L\*a\*b\*, never on tokens.** A token's variance says nothing
  about how much of it survives the grid slice and the colour upsampling. Members are rendered one
  at a time and reduced by Welford, so peak memory is two renders whatever K is.
- **A spread is a disagreement, not an error bar.** A region the model is confidently wrong about
  has a small spread. `dawn/eval/calibration.py` is where that gets measured; until it exists the
  maps are "generated, not measured", which is what `README.md` promises.
- **Inference stats come from the checkpoint, never `grid.stats_cache`** (§5.1). A run that read the
  cache would unstandardize with whatever the cache holds today — plausible-looking wrongness rather
  than a crash. `tests/test_inference_load.py` pins it.
- **`infer.py` refuses to upsample thermal.** A frame smaller than the checkpoint's input is an
  error; upsampling would invent the texture the brightness grid is guided by. Raw 840×712 frames
  have their 100-px border cropped; anything larger is area-averaged down (§4a).
- **Uncertainty maps use a fixed `[0, scale]` ramp** (`--l-scale`, `--ab-scale`, both 10 by
  default), not per-image normalization: two frames' maps are only comparable on a shared scale, and
  normalizing per image makes the most confident frame look as uncertain as the worst.
- **EMA weights are the default at inference** (`--no-ema` to override). That is what validation
  used and what should be reported.

### Still missing from the inference/eval phase

- `scripts/evaluate.py`, `configs/eval.yaml` — still old-regime (`data_dir: data/visdrone`,
  "synthetic test set", a YOLO hallucination audit that no longer applies).
- `dawn/eval/metrics.py` — masked L\* PSNR, CIEDE2000 and colorfulness are implemented;
  `compute_ssim` and `evaluate_batch` are stubs, there is no LPIPS and no vehicle-crop path.
- `dawn/eval/calibration.py`, `dawn/eval/visualize.py` — every function raises `NotImplementedError`.
- `scripts/benchmark_latency.py` — old regime, and needs the GPU anyway.
- The spec's **local refiner** (`models/refiner.py`, `train_refiner.py`, `configs/refiner.yaml`)
  does not exist. `dawn/models/denoiser.py` is the old stage-2 denoiser; its
  `DepthwiseSeparableConv(k=5)` is a usable starting point, but `BoundedDenoiser`'s
  `(enhanced, log_variance)` signature belongs to the retired noise model, and
  `dawn/models/noise_head.py`, `scripts/train_denoiser.py` and `configs/denoiser.yaml` are dead.
