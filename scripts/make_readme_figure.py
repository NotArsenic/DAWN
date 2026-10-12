#!/usr/bin/env python3
"""Build the README comparison figure: thermal in, Dawn RGB, two uncertainty maps, true RGB."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data/dronevehicle_aligned/test"
INFER = ROOT / "outputs_final/infer"
IDS = ["00107", "00112", "00114", "00118"]
COLS = ["Thermal input", "Dawn RGB", "Brightness uncertainty", "Color uncertainty", "True RGB"]
W, H, PAD, HEAD = 320, 256, 6, 34


def font(size):
    for p in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    canvas = Image.new("RGB", (len(COLS) * (W + PAD) + PAD, HEAD + len(IDS) * (H + PAD) + PAD), "white")
    d, f = ImageDraw.Draw(canvas), font(17)
    for c, name in enumerate(COLS):
        x = PAD + c * (W + PAD)
        d.text((x + W // 2, HEAD // 2), name, fill="black", font=f, anchor="mm")
    for r, i in enumerate(IDS):
        s = f"test_{i}"
        paths = [ALIGNED / f"{s}_thermal.png", INFER / f"{s}_thermal_rgb.png",
                 INFER / f"{s}_thermal_l_std.png", INFER / f"{s}_thermal_ab_std.png",
                 ALIGNED / f"{s}_rgb.png"]
        for c, p in enumerate(paths):
            im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
            canvas.paste(im, (PAD + c * (W + PAD), HEAD + PAD + r * (H + PAD)))
    out = ROOT / "docs/images/results_comparison.png"
    canvas.save(out)
    print(out, canvas.size)


if __name__ == "__main__":
    main()
