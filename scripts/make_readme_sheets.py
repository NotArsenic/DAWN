#!/usr/bin/env python3
"""Generate per-frame comparison sheets and best-case summaries from inference outputs."""
import argparse
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
    """Load a preferred TrueType font or fall back to the default font."""
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def paths(i, aligned=ALIGNED, infer=INFER):
    """Return image file paths for a given test frame stem."""
    s = f"test_{i}"
    return [aligned / f"{s}_thermal.png", infer / f"{s}_thermal_rgb.png",
            infer / f"{s}_thermal_l_std.png", infer / f"{s}_thermal_ab_std.png",
            aligned / f"{s}_rgb.png"]


def lab(a):
    """Convert sRGB image array to CIELAB colour space."""
    a = a / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    x = a @ m.T / np.array([.9505, 1, 1.089])
    f = np.where(x > .008856, np.cbrt(x), 7.787 * x + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def score(i, aligned=ALIGNED, infer=INFER):
    """Score pair based on structure correlation and colour error."""
    p = paths(i, aligned, infer)
    load = lambda q: lab(np.asarray(Image.open(q).convert("RGB").resize((W, H)), float))
    a, b = load(p[1]), load(p[4])
    de = np.sqrt(((a - b) ** 2).sum(-1)).mean()
    return np.corrcoef(a[..., 0].ravel(), b[..., 0].ravel())[0, 1] - de / 10


def sheet(ids, out, aligned=ALIGNED, infer=INFER):
    """Composite and save a grid image sheet for a list of frame IDs."""
    c = Image.new("RGB", (len(COLS) * (W + PAD) + PAD, HEAD + len(ids) * (H + PAD) + PAD), "white")
    d, f = ImageDraw.Draw(c), font(17)
    for k, n in enumerate(COLS):
        d.text((PAD + k * (W + PAD) + W // 2, HEAD // 2), n, fill="black", font=f, anchor="mm")
    for r, i in enumerate(ids):
        for k, p in enumerate(paths(i, aligned, infer)):
            im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
            c.paste(im, (PAD + k * (W + PAD), HEAD + PAD + r * (H + PAD)))
        d.text((12, HEAD + PAD + r * (H + PAD) + 8), f"test_{i}", fill="white", font=font(14))
    c.save(out)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=ALIGNED, help="Aligned test pairs")
    parser.add_argument("--infer", type=Path, default=INFER, help="Saved inference outputs")
    parser.add_argument("--out", type=Path, default=OUT, help="Where to write the sheets")
    parser.add_argument("--best", type=int, default=3, help="Frames on the best-cases sheet")
    return parser.parse_args(argv)


def main():
    """Generate and save comparison sheets."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.name[5:10] for p in args.infer.glob("test_*_thermal_rgb.png"))
    sheet(ids, args.out / "all_test_frames.png", args.aligned, args.infer)
    scores = {i: score(i, args.aligned, args.infer) for i in ids}
    ranked = sorted(ids, key=scores.get, reverse=True)
    sheet(ranked[: args.best], args.out / "best_cases.png", args.aligned, args.infer)
    for i in ranked:
        print(i, f"{scores[i]:.2f}")


if __name__ == "__main__":
    main()
