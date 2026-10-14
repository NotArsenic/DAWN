# Evaluation

- checkpoint: `/workspace/dawn/runs/baseline/flow_best.pt` (step 18000, EMA True)
- resolution (H, W): [512, 640]; colour upsampling `bilinear`; correction off
- split: `test_pairs`, NFE 2, K=8
- vehicle region: `mask` (the `vehicles` mask stands in for the plan's RGB boxes; see configs/eval.yaml)
- masks: `valid & ~moving` (box-aligned), `valid & ~vehicles` (fallback)
- structure on L\* with the peak at 100; LPIPS is whole-frame and unmasked

**The headline is box-aligned pairs.** Fallback pairs were aligned by edge correlation alone, so they are a robustness number, not the result. CIEDE2000 punishes a plausible colour that is not the true one, so read it beside colorfulness and the examples.

## box_aligned / euler (781 pairs)

| metric | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| psnr | 20.054 | 16.986 | 19.755 | 23.891 |
| ssim | 0.556 | 0.429 | 0.561 | 0.685 |
| ciede2000 | 8.310 | 5.066 | 8.204 | 11.565 |
| colorfulness | 11.988 | 4.076 | 10.629 | 20.516 |
| colorfulness_target | 18.633 | 8.844 | 17.146 | 28.769 |
| lpips | 0.535 | 0.436 | 0.533 | 0.637 |
| psnr_vehicles | 16.437 | 13.861 | 15.694 | 20.869 |
| ssim_vehicles | 0.374 | 0.279 | 0.359 | 0.479 |
| ciede2000_vehicles | 12.544 | 6.871 | 12.992 | 16.390 |
| mean_psnr | 20.285 | 17.181 | 19.925 | 24.203 |
| mean_ciede2000 | 8.035 | 4.863 | 7.986 | 11.197 |
| mean_colorfulness | 11.524 | 3.641 | 10.121 | 20.041 |
| l_std | 2.182 | 1.457 | 2.112 | 3.009 |
| ab_std | 1.096 | 0.578 | 1.022 | 1.640 |

## box_aligned / midpoint (781 pairs)

| metric | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| psnr | 19.380 | 16.498 | 19.042 | 22.985 |
| ssim | 0.543 | 0.414 | 0.548 | 0.673 |
| ciede2000 | 8.978 | 5.558 | 8.899 | 12.316 |
| colorfulness | 12.480 | 4.543 | 11.129 | 20.923 |
| colorfulness_target | 18.633 | 8.844 | 17.146 | 28.769 |
| lpips | 0.543 | 0.443 | 0.544 | 0.644 |
| psnr_vehicles | 15.868 | 13.472 | 15.125 | 20.183 |
| ssim_vehicles | 0.347 | 0.260 | 0.331 | 0.446 |
| ciede2000_vehicles | 13.361 | 7.550 | 13.967 | 17.122 |
| mean_psnr | 19.874 | 16.937 | 19.523 | 23.781 |
| mean_ciede2000 | 8.430 | 5.183 | 8.415 | 11.517 |
| mean_colorfulness | 11.623 | 3.684 | 10.216 | 20.300 |
| l_std | 3.403 | 2.421 | 3.413 | 4.357 |
| ab_std | 1.631 | 0.962 | 1.567 | 2.289 |

## fallback / euler (597 pairs)

| metric | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| psnr | 20.112 | 16.788 | 19.670 | 24.301 |
| ssim | 0.552 | 0.411 | 0.557 | 0.680 |
| ciede2000 | 8.374 | 4.841 | 8.187 | 11.659 |
| colorfulness | 11.481 | 5.402 | 9.756 | 20.418 |
| colorfulness_target | 18.729 | 9.920 | 17.537 | 29.009 |
| lpips | 0.532 | 0.417 | 0.535 | 0.643 |
| psnr_vehicles | 140.000 | 140.000 | 140.000 | 140.000 |
| ssim_vehicles | 0.000 | 0.000 | 0.000 | 0.000 |
| ciede2000_vehicles | 0.000 | 0.000 | 0.000 | 0.000 |
| mean_psnr | 20.307 | 17.019 | 19.787 | 24.610 |
| mean_ciede2000 | 8.156 | 4.565 | 7.966 | 11.157 |
| mean_colorfulness | 11.031 | 4.832 | 9.269 | 20.133 |
| l_std | 2.276 | 1.511 | 2.283 | 3.021 |
| ab_std | 1.078 | 0.626 | 1.010 | 1.535 |

## fallback / midpoint (597 pairs)

| metric | mean | p10 | p50 | p90 |
|---|---|---|---|---|
| psnr | 19.472 | 16.294 | 19.068 | 23.478 |
| ssim | 0.538 | 0.397 | 0.540 | 0.672 |
| ciede2000 | 8.992 | 5.329 | 8.881 | 12.293 |
| colorfulness | 11.997 | 6.000 | 10.311 | 20.754 |
| colorfulness_target | 18.729 | 9.920 | 17.537 | 29.009 |
| lpips | 0.542 | 0.437 | 0.542 | 0.654 |
| psnr_vehicles | 140.000 | 140.000 | 140.000 | 140.000 |
| ssim_vehicles | 0.000 | 0.000 | 0.000 | 0.000 |
| ciede2000_vehicles | 0.000 | 0.000 | 0.000 | 0.000 |
| mean_psnr | 19.914 | 16.639 | 19.382 | 24.320 |
| mean_ciede2000 | 8.516 | 4.871 | 8.408 | 11.556 |
| mean_colorfulness | 11.155 | 4.966 | 9.418 | 20.063 |
| l_std | 3.521 | 2.510 | 3.529 | 4.416 |
| ab_std | 1.617 | 1.026 | 1.569 | 2.187 |
