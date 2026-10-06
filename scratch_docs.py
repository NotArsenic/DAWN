import re

with open("docs/conventions.md", "r") as f:
    text = f.read()

# Update section 6
isp_old = """**The ISP.** On the native path only, the denoise blur and the unsharp mask are linear and each multiplies a white-noise variance by the sum of its squared weights, so `a`, `b` and `quant_var` are all scaled by that one factor (`CameraResult.isp_var_scale`). This is the same relation `scripts/calibrate_noise.py` runs backwards to turn a measured effective `(a, b)` into the sensor constants in `configs/camera_calib.yaml`."""
isp_new = """**The ISP.** On the native path only, the ISP applies a colour-specific denoise blur (on C1, C2) and a composite denoise and unsharp mask. Because these are linear filters that mix channels, the variance of the output `y` involves a mixing matrix `S` and a DC colour matrix `M`:

```
x = M⁻¹ y
Var[y] = S (a · x + b + quant_var) / n
```

where `S[i, j] = Σ_tap W[i, j, tap]²` and `M[i, j] = Σ_tap W[i, j, tap]` for the composite 3×3 channel kernel `W`. `CameraResult` reports `S` and `M` directly, and `DegradationResult.variance()` evaluates this exact formula. Off the native path, `S` and `M` are the identity."""

text = text.replace(isp_old, isp_new)

# Update section 6 formula
budget_old = """The full budget at the returned resolution is

```
Var[y_c] = (a_c · y_c + b_c + quant_var) / n
```

which is `DegradationResult.variance()`. JPEG and the final 8-bit step are nonlinear and are **not** in it."""
budget_new = """The full budget at the returned resolution is computed by `DegradationResult.variance()`, which uses the ISP matrices. Off the native path (where S and M are identity), it simplifies to:

```
Var[y_c] = (a_c · y_c + b_c + quant_var) / n
```

JPEG and the final 8-bit step are nonlinear and are **not** in the budget."""
text = text.replace(budget_old, budget_new)


# Update section 10.2
level_old = """Training takes a fresh random draw on every call, which is what makes the degradation an augmentation. Validation and test use fixed seeded sets at three darkness levels, `easy` / `medium` / `hard`, which are quantile bands of the `n_white` range: `hard` is darker *and* noisier because the auto-exposure gain caps out and the frame stays dark."""
level_new = """Training takes a fresh random draw on every call, which is what makes the degradation an augmentation. Validation and test use fixed seeded sets at three darkness levels, `easy` / `medium` / `hard`, which are quantile bands of the target brightness distribution: `hard` is darker because the auto-exposure target is sampled from the darkest 33% of real night frames. The photon count is drawn to hit the target; if it hits the bounds of its range, the frame stays dark and the gain caps out."""

text = text.replace(level_old, level_new)

with open("docs/conventions.md", "w") as f:
    f.write(text)
