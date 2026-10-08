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

### Why a big batch is the wrong way to spend a 32 GB card

The temptation on a 32 GB card is `batch_size: 128`. It does not work here, and the reason is
the dataset, not the card: 3,662 pairs over a batch of 128 is **29 steps per epoch**, so a
100-epoch run is 2,900 optimizer steps in total.

- 2,900 steps is **fewer than `flow.warmup_steps` (3,000)**, so the conditioning detach would
  never lift and the velocity net would train on frozen features for the entire run. That is a
  silent wrong result, not a crash. The start-of-run `Budget:` line (`budget_summary`) now prints
  this in epochs, so the trap is visible rather than inferred.
- `logging.save_every: 5000` would never fire; `val_every: 500` would give 5 points, not 40.
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
| use the 5090's memory | `configs/gpu_5090.yaml` (`timesteps_per_image: 8`) | `batch_size: 128` |
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

Training is on an **RTX 5090 (32 GB)**, not the 16 GB RTX 5060 Ti the spec and `README.md` assume.
Same Blackwell architecture, so the setup requirement carries over unchanged: PyTorch built for
CUDA 12.8 or newer with `sm_120` in `torch.cuda.get_arch_list()`.

What 32 GB retires is the VRAM concern those documents are built around. At batch 16 × 4 timesteps
the velocity net's attention needs roughly 5 GB: the DINO attention bias is an additive float mask,
so SDPA cannot use a flash kernel and has to materialize `64 × 4 × 1280 × 1280` scores per block
and keep them for backward. On 16 GB that was tight enough to plan the 8 × 4 fallback; on 32 GB it
is not a constraint and the fallback can be ignored.

**`configs/gpu_5090.yaml`** is the overlay that spends the headroom, and it spends it on
`flow.timesteps_per_image: 8` rather than on the batch, for the reason in the section above: one
encoder pass already feeds `timesteps_per_image` velocity passes, so 4 → 8 doubles the velocity
samples per optimizer step and leaves steps-per-epoch untouched. It also sets
`flow.render_subset: 64`, because L_render would otherwise double to 128 rendered 256×320 images
per step; a random 64 of the 128 is an unbiased estimate of the same mean, so the render path
keeps its current cost while L_flow gets twice the samples.

Expect roughly **2× the step time** from that change. Run `configs/gpu_probe.yaml` *with* the 5090
overlay for 50 steps before committing a night to it — the probe is the only thing that knows the
real seconds/step and peak VRAM:

```bash
python scripts/train_flow.py --config configs/flow.yaml \
    --override configs/gpu_probe.yaml configs/gpu_5090.yaml \
    --device cuda --max-steps 50 --out-dir outputs/gpu_probe
```

If the probe shows comfortable headroom, the next-biggest wall-clock win is `encoder.
grad_checkpointing: false` — it trades a few GB for not recomputing DINO's 12 blocks on every
backward. It is left on in the overlay because an out-of-memory crash costs a whole run and the
recompute does not.

---

## 1a. Validation resolution

`logging.val_every` was 2,500, which over a 20,000-step run produces **eight validation points**.
That is not the resolution needed to see a divergence begin, and on 3,662 training pairs over ~87
epochs a divergence is the headline risk. It is now **500**, giving 40.

Three things were added with it, because more points are only useful if they measure the right
things:

| key | what it adds |
|---|---|
| `logging.val_loss_batches: 8` | `FlowTrainer.validate_loss`: the **training objective itself** on held-out pairs. `validate` scores sampled image quality, which is what the model is for; this is the loss it is minimizing, and a train curve falling while this one turns up is the memorization signal. Nothing in the run reported it before |
| `logging.select_metric: ciede2000` | which number picks `flow_best.pt`. It was L\* PSNR alone — and L\* comes mostly from the brightness grid fitted on thermal, the half of the model least able to memorize, so selection was blind to colour going wrong. CIEDE2000 is the full Lab difference: brightness **and** colour. `psnr_L` remains selectable |
| `logging.val_retrieval: true` | patch retrieval: does each adapted thermal token's nearest frozen-RGB token turn out to be its own? Reported with `within_1`, which separates residual misalignment (the neighbour wins) from features that do not localize at all. Chance is 1/1280 ≈ 0.08% |

The held-out loss is measured on the **live** weights, not the EMA ones, so it is comparable with
the train curve it is read against; `validate`'s image scores stay on the EMA weights, which is
what is reported and selected on.

**Cost.** A validation is now roughly 96 encoder passes of batch 4 (two subsets × 64 images, plus
the frozen pass for retrieval, plus the held-out loss batches) against a training step's 32
image-passes — order 4–8% of wall-clock at `val_every: 500`. The probe gives the real number. If
it is too much, the dials in order are `val_retrieval: false`, a smaller `val_loss_batches`, and
only then a larger `val_every`.

---

## 1b. Two smoothings of the schedule

Both are strict generalizations — with their knobs at the old values they reproduce the previous
behaviour exactly — and both are on by default.

**`optim.ema_warmup: true`.** A fixed decay of 0.999 leaves the EMA shadow `0.999 ** n`
initialization after `n` updates: **61% random weights at step 500** and still 5% at step 3,000.
Every validation inside that window was measuring the initialization as much as the model, which
did not matter at eight validation points starting at 2,500 and matters a great deal at forty
starting at 500. The warmup is the standard `min(decay, (1 + n) / (10 + n))`: it drives the
initialization's share to about 1e-5 within ten updates and only ever *lowers* the decay, so the
late-training average is exactly the one `ema_decay` asks for.

**`flow.warmup_ramp_steps: 500`.** `flow.warmup_steps` is a gate, not a ramp. At step 2,999 the
velocity net is fitting a frozen feature distribution that LoRA has been moving under it for 3,000
steps on L_align and L_color; at step 3,000 it would suddenly be fitting a moving one, and LoRA
would suddenly be taking a loss it has never seen. `conditioning_scale` now scales that gradient
linearly from 0 to 1 over 500 steps instead. The **forward pass is identical** either way — the
scaling is on the gradient only — so `warmup_ramp_steps: 0` is the old hard gate exactly, and
`null` derives `max(1, warmup_steps / 6)` so the short overlays scale with their own warmup.

`optim.cosine.min_factor` is also exposed now (floor of the cosine decay, as a fraction of each
group's LR). It defaults to **0.0**, the plan's schedule and the previous behaviour: a knob, not a
recommendation.

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
the console and the local files below and nothing here fires. `tests/test_wandb_logging.py` covers
all of that with a fake `wandb` module, since CI has no real one.

### Every metric is also on disk

W&B owns the live curves, but a plot has to be rebuildable months later by someone without the
run. `RunLog` is now the single place a metric is recorded, and it writes three ways from one
call, so they cannot drift:

| file (under `--out-dir`) | what it holds |
|---|---|
| `train_log.jsonl` | one JSON object per line: `step`, `kind`, and every key W&B got |
| `metrics/<kind>.csv` | one CSV per kind — `train`, `val_box_aligned`, `val_fallback`, `valloss_*` — with a stable header, rewritten whole on each append so a killed run still leaves complete files and a late column (`beta1`, `peak_vram_gb`) lands in the header retroactively |
| `run_config.json` | the fully merged config, so a plot can be traced to its settings |

Set `logging.metrics_csv: false` to skip the CSVs; the JSONL mirror is not optional. Before this
existed the JSONL held the raw loss dict while W&B got the prefixed one *plus* the attention
gates, so the two quietly disagreed about what a run had recorded.

### What else is logged now

| key | why |
|---|---|
| `contrib/{flow,render,align,color}` | each term's share of `total` (`weight × value / total`, summing to 1). Four hand-set weights over terms on different scales can leave one doing nothing, or everything, without any curve looking wrong on its own |
| `grad/{velocity,encoder,color_head}` | per-module gradient norm, read before clipping. One global `grad_norm` cannot say whether LoRA is receiving anything, and the step where that changes is the most load-bearing moment in the schedule |
| `loss/cond_scale`, `loss/ema_decay` | the two ramps above, so their effect is attributable |
| `progress/epoch` | the axis the budget is actually reasoned about in |
| `val_*/ciede2000/gap_to_oracle` | the colour twin of `psnr_L/gap_to_oracle`: distance from the measured ceiling, positive when the model is behind |
| `val_*/colorfulness/ratio_*` | colourfulness over the ground truth's. 1.0 is "as vivid as the real photo"; below 1 is the washed-out brown that averaging colourisers fall into. The raw number alone is not readable |
| `val_*/retrieval/{top1,within_1,chance}` | §1a |
| `valloss_*/loss/*` | §1a |

`grad/*` and `contrib/*` cost one extra pass over the gradients, so they are computed only on the
steps that are logged (`train_step(detailed=True)`).

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

**Resolved, 8 October 2026.** The three packages this section used to list as missing are now
present in `.venv`: `wandb 0.30.0`, `lpips 0.1.4`, `scikit-image 0.26.0`. In particular the one
that mattered — a 20K-step run completing with no W&B tracking at all and a single `WARNING` line
to say so — cannot happen now. Re-check on the training machine anyway, since `.venv` is not the
box the run happens on:

```bash
python -c "import wandb, lpips, skimage; print('ok')"
```

`black>=23.0` and `ipython>=8.0` are still absent from the dev extra and do not matter — `black`
is configured in `pyproject.toml` but `ruff format` is what is nominally used, so the
`[tool.black]` section is dead configuration. (`ruff check` is the gate that actually runs and
passes; the tree is hand-wrapped and has never been `ruff format`-clean.)

Everything else is installed and the suite runs offline on CPU: **939 tests in about 70 s**.

## 6. What changed in the code

The data and flow phases were complete and tested before this; what follows is the first slice of
the inference/eval phase, plus the two training knobs above.

### New

| file | what it is |
|---|---|
| `dawn/progress.py` | progress reporting: a tqdm bar on a TTY, lines with rate and ETA in a log |
| `dawn/uncertainty/spread.py` | sample-spread uncertainty: K rendered members → one image and two maps. `docs/conventions.md` §11 |
| `dawn/flow/inference.py` | `load_checkpoint` → encoder (+LoRA), velocity net, layout, stats, render settings, sizes |
| `tests/test_spread.py`, `test_inference_load.py`, `test_infer_script.py`, `test_schedule.py`, `test_wandb_logging.py`, `test_progress.py` | 79 tests |

### Rewritten

- **`scripts/infer.py`** was an old-regime stub about "night images" and "analytic propagation". It
  now runs the plan's inference path: read thermal → DINO once → K samples at 2 NFE → render each →
  reduce → write.
- **`dawn/flow/trainer.py`** gained `one_cycle_schedule`, `MomentumCycler` and `build_schedule`.
- **`scripts/train_flow.py`** gained `WandbRun.watch` and an ETA on every logged step.
- **`dawn/flow/trainer.py`** warns when `num_train_steps` is within 2x of `warmup_steps` (§1).
- **`dawn/data/sheets.py`** `run_parallel` gained a rate and an ETA; `dawn/flow/loaders.py` and
  `dawn/flow/oracle.py` report progress on their previously silent passes.
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
