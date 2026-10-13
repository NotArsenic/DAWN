#!/usr/bin/env python3
"""Build comparison figure showing thermal input, model RGB, uncertainty maps, and ground truth."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data/dronevehicle_aligned/test"
INFER = ROOT / "outputs_final/infer"
IDS = ["00107", "00112", "00114", "00118"]
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


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned", type=Path, default=ALIGNED, help="Aligned test pairs")
    parser.add_argument("--infer", type=Path, default=INFER, help="Saved inference outputs")
    parser.add_argument("--ids", nargs="+", default=IDS, help="Frame stems, without the split")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "docs/images/results_comparison.png",
        help="Where to write",
    )
    return parser.parse_args(argv)


def main():
    """Generate and save the comparison figure."""
    args = parse_args()
    rows = args.ids
    size = (len(COLS) * (W + PAD) + PAD, HEAD + len(rows) * (H + PAD) + PAD)
    canvas = Image.new("RGB", size, "white")
    d, f = ImageDraw.Draw(canvas), font(17)
    for c, name in enumerate(COLS):
        x = PAD + c * (W + PAD)
        d.text((x + W // 2, HEAD // 2), name, fill="black", font=f, anchor="mm")
    for r, i in enumerate(rows):
        s = f"test_{i}"
        paths = [args.aligned / f"{s}_thermal.png", args.infer / f"{s}_thermal_rgb.png",
                 args.infer / f"{s}_thermal_l_std.png", args.infer / f"{s}_thermal_ab_std.png",
                 args.aligned / f"{s}_rgb.png"]
        for c, p in enumerate(paths):
            im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
            canvas.paste(im, (PAD + c * (W + PAD), HEAD + PAD + r * (H + PAD)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out, canvas.size)


if __name__ == "__main__":
    main()
