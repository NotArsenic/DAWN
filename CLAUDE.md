# Dawn

Nighttime drone image **enhancement** (not night-to-day translation). DINOv3 ViT-S+ features (with LoRA) condition a flow-matched velocity transformer that predicts a 16×16×8 bilateral grid of 3×4 affine color transforms. The grid is sliced at full resolution, so the output is always `A·x + b` of the input pixels and the model cannot add or remove objects. Uncertainty is propagated analytically (`A·Σ·Aᵀ`). Overview, pipeline diagram and file layout: `README.md`.

## Conventions (authoritative)

@docs/conventions.md

If code and that file disagree, the file wins. To change a convention, edit the doc in the same change and update the affected docstrings and tests. Do not weaken a test in `tests/test_layout.py`, `test_guidance.py`, `test_normalize.py`, `test_propagate.py` or `test_grid_rules.py` to make code pass: they encode these rules.

## Status

This is the `scaffold` branch. Most modules are stubs that raise `NotImplementedError` with their final signature and docstring. Implement them in place and keep the documented shapes. Tests for stubs skip until the stub is implemented (`skip_until_implemented` in `tests/helpers.py`), so implementing one should make its tests run and pass.

## Commands

```bash
pip install -e ".[dev]"     # add ".[transformers]" for the transformers DINO backend
pytest                      # CPU only, offline
ruff check dawn scripts tests
```

Every script in `scripts/` takes `--help`. A tiny end-to-end debug run is `python scripts/train_flow.py --config configs/flow.yaml --override configs/debug.yaml --device cpu`; the debug overlay sets `loader.synthetic_images` and `encoder.pretrained: false`, so it needs neither VisDrone on disk nor an HF token. Add `--overfit` to validate on the training set (the sanity check to run before any full training). Ruff still reports some unused imports in scaffold files nobody has touched yet; leave those alone rather than churning unrelated files.

## Working rules

- All files are UTF-8 with LF line endings. The first scaffold commit was accidentally UTF-16; do not reintroduce it.
- Do not commit or push unless asked. The maintainer reviews and commits.
- LoRA comes from peft only; do not add a hand-rolled LoRA layer. `DINOEncoder.forward_frozen` uses `disable_adapter()` to produce the L_align target.
- DINOv3 weights are gated (HF token and license acceptance). Tests must not download weights; they build a randomly initialized, two-block backbone (`tests/flow_helpers.py: tiny_encoder`). Use `scripts/smoke_dino.py` for real-backend checks.
- `flow.warmup_steps` gates the conditioning detach: before it, the encoder's patch tokens and CLS are detached before the velocity net and the noise head, so LoRA trains on L_align alone. `optim.lr_warmup_steps` is the separate, unrelated LR warmup.
- Target hardware is a 16 GB GPU (RTX 5060 Ti): batch 32 images with 4 timesteps each, 30-50K steps.

## Open decisions (ask, do not guess)

Nothing open. Delete an item here once it is decided; add one rather than guessing.

Decided and implemented: the 6,497 images of the 49 `99999xx` still collections are auto-accepted as day by cutoffs on median luminance and highlight ratio, fitted to make zero errors on the hand-labelled frames (`dawn/data/splitting.py`, docs/conventions.md section 10.1). `prepare_visdrone.py split` writes a 200-image contact sheet of what they accepted, which is the intended spot check.

## Stale: `plans/night2day_bilateral_grid.md`

The plan predates the decisions above. Where it disagrees, the code, configs and conventions win. Known differences: `n2d/` is now `dawn/`; the encoder is ViT-S+, not ViT-S; batch 64 and 100K steps became batch 32 with 4 timesteps per image and 30-50K steps; the velocity transformer uses 256 tokens of 96 values; the project is enhancement, not night-to-day.
