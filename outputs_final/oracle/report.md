# Brightness and colour oracle

- config: `configs/flow.yaml` (+ no overrides)
- resolution (H, W): [512, 640]; grid ridge 0.001, refine 1; BGU {'bins': 8, 'grid': [64, 80], 'ridge_alpha': 0.1, 'refine_steps': 2}
- split: `val_pairs`, 345 pairs (194 box-aligned, 151 fallback)
- masks: `valid & ~moving` (box pairs), `valid & ~vehicles` (fallback pairs)

## L* PSNR, colour ceiling and assembled oracle

| metric | all mean (n=345) | all p10 / p50 / p90 | box-aligned mean (n=194) | box-aligned p10 / p50 / p90 | fallback mean (n=151) | fallback p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| grid only, refine 0 (L* PSNR dB) | 24.60 | 21.87 / 24.38 / 27.76 | 24.61 | 21.97 / 24.57 / 27.48 | 24.58 | 21.78 / 24.33 / 28.15 |
| grid only, refine 1 (L* PSNR dB) | 25.08 | 22.35 / 24.84 / 28.30 | 25.13 | 22.50 / 25.06 / 27.82 | 25.02 | 22.29 / 24.81 / 28.52 |
| grid + correction (bilinear), refine 0 (dB) | 26.68 | 23.80 / 26.49 / 29.90 | 26.86 | 24.26 / 26.89 / 29.53 | 26.45 | 23.36 / 26.37 / 30.26 |
| grid + correction (bilinear), refine 1 (dB) | 26.85 | 23.96 / 26.67 / 30.02 | 27.04 | 24.43 / 27.09 / 29.68 | 26.60 | 23.48 / 26.55 / 30.38 |
| grid + correction (BGU), refine 0 (dB) | 25.74 | 22.94 / 25.56 / 28.99 | 25.85 | 23.33 / 25.73 / 28.50 | 25.60 | 22.74 / 25.44 / 29.56 |
| grid + correction (BGU), refine 1 (dB) | 25.95 | 23.14 / 25.79 / 29.15 | 26.07 | 23.52 / 25.98 / 28.73 | 25.78 | 22.92 / 25.62 / 29.66 |
| colour ceiling, bilinear (mean abs delta a*b*) | 0.90 | 0.71 / 0.87 / 1.12 | 0.86 | 0.72 / 0.84 / 1.04 | 0.94 | 0.70 / 0.93 / 1.17 |
| colour ceiling, BGU (mean abs delta a*b*) | 1.28 | 0.95 / 1.25 / 1.60 | 1.23 | 0.95 / 1.22 / 1.51 | 1.34 | 0.97 / 1.37 / 1.70 |
| colour ceiling AT EDGES, bilinear (mean abs delta a*b*) | 1.45 | 1.07 / 1.41 / 1.83 | 1.39 | 1.07 / 1.35 / 1.74 | 1.51 | 1.09 / 1.50 / 1.94 |
| colour ceiling AT EDGES, BGU (mean abs delta a*b*) | 2.37 | 1.53 / 2.41 / 3.04 | 2.30 | 1.53 / 2.36 / 2.89 | 2.45 | 1.55 / 2.43 / 3.24 |
| assembled oracle, grid only, bilinear colour (CIEDE2000) | 3.33 | 2.41 / 3.22 / 4.25 | 3.25 | 2.42 / 3.12 / 4.20 | 3.43 | 2.32 / 3.32 / 4.37 |
| assembled oracle, grid only, BGU colour (CIEDE2000) | 3.50 | 2.57 / 3.40 / 4.41 | 3.42 | 2.59 / 3.27 / 4.39 | 3.60 | 2.50 / 3.53 / 4.57 |
| assembled oracle, with correction, bilinear (CIEDE2000) | 2.77 | 2.00 / 2.68 / 3.49 | 2.66 | 2.03 / 2.56 / 3.38 | 2.91 | 1.97 / 2.77 / 3.76 |
| assembled oracle, with correction, BGU (CIEDE2000) | 3.18 | 2.34 / 3.09 / 3.97 | 3.08 | 2.38 / 2.99 / 3.91 | 3.31 | 2.31 / 3.23 / 4.16 |

## The handoff's probe subset (120 pairs, seed 0)

| metric | all mean (n=120) | all p10 / p50 / p90 | box-aligned mean (n=57) | box-aligned p10 / p50 / p90 | fallback mean (n=63) | fallback p10 / p50 / p90 |
|---|---|---|---|---|---|---|
| grid only, refine 0 (L* PSNR dB) | 24.66 | 21.81 / 24.48 / 27.62 | 24.79 | 21.84 / 25.01 / 27.26 | 24.54 | 21.82 / 24.14 / 27.87 |
| grid only, refine 1 (L* PSNR dB) | 25.14 | 22.30 / 24.96 / 27.91 | 25.31 | 22.33 / 25.55 / 27.59 | 24.98 | 22.31 / 24.68 / 28.23 |
| grid + correction (bilinear), refine 0 (dB) | 26.70 | 23.57 / 26.41 / 29.55 | 27.05 | 24.20 / 27.24 / 29.45 | 26.37 | 23.44 / 26.13 / 30.09 |
| grid + correction (bilinear), refine 1 (dB) | 26.86 | 23.73 / 26.62 / 29.67 | 27.24 | 24.35 / 27.45 / 29.52 | 26.53 | 23.63 / 26.27 / 30.22 |
| grid + correction (BGU), refine 0 (dB) | 25.77 | 22.81 / 25.50 / 28.57 | 26.02 | 23.14 / 26.09 / 28.39 | 25.54 | 22.75 / 25.29 / 28.87 |
| grid + correction (BGU), refine 1 (dB) | 25.98 | 23.07 / 25.74 / 28.68 | 26.25 | 23.33 / 26.34 / 28.52 | 25.73 | 22.98 / 25.58 / 29.08 |
| colour ceiling, bilinear (mean abs delta a*b*) | 0.90 | 0.68 / 0.88 / 1.14 | 0.86 | 0.72 / 0.82 / 1.04 | 0.94 | 0.68 / 0.93 / 1.16 |
| colour ceiling, BGU (mean abs delta a*b*) | 1.28 | 0.93 / 1.26 / 1.58 | 1.22 | 0.96 / 1.24 / 1.46 | 1.33 | 0.91 / 1.38 / 1.62 |
| colour ceiling AT EDGES, bilinear (mean abs delta a*b*) | 1.46 | 1.09 / 1.40 / 1.85 | 1.37 | 1.07 / 1.34 / 1.71 | 1.54 | 1.13 / 1.55 / 1.93 |
| colour ceiling AT EDGES, BGU (mean abs delta a*b*) | 2.39 | 1.50 / 2.42 / 3.03 | 2.26 | 1.49 / 2.31 / 2.78 | 2.50 | 1.76 / 2.53 / 3.24 |
| assembled oracle, grid only, bilinear colour (CIEDE2000) | 3.31 | 2.32 / 3.19 / 4.23 | 3.23 | 2.42 / 3.02 / 4.21 | 3.38 | 2.32 / 3.39 / 4.23 |
| assembled oracle, grid only, BGU colour (CIEDE2000) | 3.49 | 2.49 / 3.34 / 4.38 | 3.41 | 2.60 / 3.19 / 4.38 | 3.56 | 2.47 / 3.59 / 4.37 |
| assembled oracle, with correction, bilinear (CIEDE2000) | 2.76 | 2.00 / 2.71 / 3.55 | 2.64 | 2.04 / 2.54 / 3.36 | 2.87 | 2.00 / 2.85 / 3.57 |
| assembled oracle, with correction, BGU (CIEDE2000) | 3.17 | 2.33 / 3.09 / 3.96 | 3.07 | 2.38 / 2.95 / 3.91 | 3.27 | 2.31 / 3.31 / 4.01 |

## BGU grid sweep (colour ceiling, lower is better)

| bins | grid (rows x cols) | ridge | refine | mean abs delta a*b* | pairs |
|---|---|---|---|---|---|
| bilinear (baseline) | - | - | - | 1.017 | 40 |
| 8 | 64 x 80 | 0.1 | 2 | 1.509 | 40 |
| 16 | 64 x 80 | 0.1 | 2 | 1.582 | 40 |
| 8 | 64 x 80 | 0.001 | 2 | 1.591 | 40 |
| 16 | 64 x 80 | 0.001 | 2 | 1.625 | 40 |
| 8 | 64 x 80 | 0.001 | 0 | 1.722 | 40 |
| 16 | 64 x 80 | 0.001 | 0 | 1.732 | 40 |
| 16 | 64 x 80 | 0.1 | 0 | 1.751 | 40 |
| 8 | 64 x 80 | 0.1 | 0 | 1.768 | 40 |
| 8 | 32 x 40 | 0.1 | 2 | 1.980 | 40 |
| 16 | 32 x 40 | 0.1 | 2 | 2.019 | 40 |
| 8 | 32 x 40 | 0.001 | 2 | 2.022 | 40 |
| 16 | 32 x 40 | 0.001 | 2 | 2.056 | 40 |
| 8 | 32 x 40 | 0.001 | 0 | 2.200 | 40 |
| 16 | 32 x 40 | 0.001 | 0 | 2.207 | 40 |
| 16 | 32 x 40 | 0.1 | 0 | 2.225 | 40 |
| 8 | 32 x 40 | 0.1 | 0 | 2.287 | 40 |
| 8 | 16 x 20 | 0.1 | 2 | 2.591 | 40 |
| 16 | 16 x 20 | 0.1 | 2 | 2.592 | 40 |
| 8 | 16 x 20 | 0.001 | 2 | 2.599 | 40 |
| 16 | 16 x 20 | 0.001 | 2 | 2.619 | 40 |
| 16 | 16 x 20 | 0.001 | 0 | 2.817 | 40 |
| 8 | 16 x 20 | 0.001 | 0 | 2.824 | 40 |
| 16 | 16 x 20 | 0.1 | 0 | 2.847 | 40 |
| 8 | 16 x 20 | 0.1 | 0 | 2.955 | 40 |

Best: bins, rows, cols, ridge, refine = (8, 64, 80, 0.1, 2)

## Worst 10% of grid-only scenes

- `outputs/oracle/worst10_grid_only_01.jpg`
- `outputs/oracle/worst10_grid_only_02.jpg`
- `outputs/oracle/worst10_grid_only_03.jpg`
- `outputs/oracle/worst10_grid_only_04.jpg`

## Checks against the handoff

- grid-only refine 0 on the probe subset: 24.66 dB vs the handoff's 24.72 dB (tolerance 0.5): **OK**
- colour ceiling over all pairs: bilinear 0.896 vs BGU 1.280 (configured {'bins': 8, 'grid': [64, 80], 'ridge_alpha': 0.1, 'refine_steps': 2}) mean abs delta a*b*; `color.upsample` is **bilinear** (the maintainer chose bilinear at Checkpoint A)
- colour ceiling AT EDGES (top 10% of gradient magnitude): bilinear 1.445 vs BGU 2.368 mean abs delta a*b* -- **bilinear ahead**
- assembled oracle CIEDE2000 (grid only): bilinear colour 3.33 vs BGU colour 3.50
- best swept BGU (8, 64, 80, 0.1, 2): 1.509 vs bilinear 1.017 on the sweep pairs (bilinear ahead)
