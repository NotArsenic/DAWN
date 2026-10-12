#!/usr/bin/env python3
"""Per-frame comparison sheets plus a 'best cases' sheet, from outputs_final/infer.

Only frames that have saved inference outputs can be drawn (no trained checkpoint is needed).
The best-case sheet ranks frames by L* correlation minus mean Lab error / 10.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data/dronevehicle_aligned/test"
INFER = ROOT / "outputs_final/infer"
OUT = ROOT / "docs/images/sheets"
COLS = ["Thermal input", "Dawn RGB", "Brightness uncertainty", "Color uncertainty", "True RGB"]
W, H, PAD, HEAD = 320, 256, 6, 34


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def paths(i):
    s = f"test_{i}"
    return [ALIGNED / f"{s}_thermal.png", INFER / f"{s}_thermal_rgb.png",
            INFER / f"{s}_thermal_l_std.png", INFER / f"{s}_thermal_ab_std.png",
            ALIGNED / f"{s}_rgb.png"]


def lab(a):
    a = a / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    x = a @ m.T / np.array([.9505, 1, 1.089])
    f = np.where(x > .008856, np.cbrt(x), 7.787 * x + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def score(i):
    """Higher is better: L* structure correlation minus mean Lab error / 10 (perceptual, not PSNR)."""
    p = paths(i)
    load = lambda q: lab(np.asarray(Image.open(q).convert("RGB").resize((W, H)), float))
    a, b = load(p[1]), load(p[4])
    de = np.sqrt(((a - b) ** 2).sum(-1)).mean()
    return np.corrcoef(a[..., 0].ravel(), b[..., 0].ravel())[0, 1] - de / 10


def sheet(ids, out):
    c = Image.new("RGB", (len(COLS) * (W + PAD) + PAD, HEAD + len(ids) * (H + PAD) + PAD), "white")
    d, f = ImageDraw.Draw(c), font(17)
    for k, n in enumerate(COLS):
        d.text((PAD + k * (W + PAD) + W // 2, HEAD // 2), n, fill="black", font=f, anchor="mm")
    for r, i in enumerate(ids):
        for k, p in enumerate(paths(i)):
            im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
            c.paste(im, (PAD + k * (W + PAD), HEAD + PAD + r * (H + PAD)))
        d.text((12, HEAD + PAD + r * (H + PAD) + 8), f"test_{i}", fill="white", font=font(14))
    c.save(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name[5:10] for p in INFER.glob("test_*_thermal_rgb.png"))
    sheet(ids, OUT / "all_test_frames.png")
    scores = {i: score(i) for i in ids}
    best = sorted(ids, key=scores.get, reverse=True)[:3]
    sheet(best, OUT / "best_cases.png")
    for i in sorted(ids, key=scores.get, reverse=True):
        print(i, f"{scores[i]:.2f}")


if __name__ == "__main__":
    main()
