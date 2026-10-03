#!/usr/bin/env python3
"""Hand-label VisDrone in the terminal: one keypress per sequence (or image).

Goes through every row of the hand-label CSV (`visdrone.labels_csv` in configs/data.yaml) whose
label is blank, in file order, and shows its frames in the terminal: the darkest, median and
brightest frame of a sequence, or the image itself for an image row. Each keypress is saved to
the CSV at once, so quit whenever you like and rerun to carry on.

    d  day        n  night        u  dusk        x  exclude
    s  skip       b  back         o  open every frame in the image viewer        q  quit

Images show inline in Ghostty, kitty and WezTerm (kitty graphics protocol) and in iTerm2 and
VS Code / Cursor terminals (iTerm2 protocol; VS Code needs terminal.integrated.enableImages).
Anywhere else the preview is written to a file for you to keep open. For a spot check of single
images, add their IDs to the CSV as rows with a blank label and run this again.

Don't edit the CSV in another program while this runs: every keypress rewrites the file.

Usage:
    python scripts/label_visdrone.py
    python scripts/label_visdrone.py --relabel          # go through every row, not just blank ones
    python scripts/label_visdrone.py --from 0000287     # start at this key
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KEYS = {"d": "day", "n": "night", "u": "dusk", "x": "exclude"}
PREVIEW_BOX = (300, 400)  # each frame fits (height, width) pixels
HELP = "d day   n night   u dusk   x exclude   |   s skip   b back   o open frames   q quit"


def detect_display() -> str:
    env = os.environ
    if (
        env.get("KITTY_WINDOW_ID")
        or env.get("TERM") in ("xterm-kitty", "xterm-ghostty")
        or env.get("TERM_PROGRAM") in ("ghostty", "WezTerm")
    ):
        return "kitty"
    if env.get("TERM_PROGRAM") in ("iTerm.app", "vscode") or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    return "file"


def clear(display: str) -> None:
    if display == "kitty":
        sys.stdout.write("\033_Ga=d,d=A,q=2\033\\")  # delete every image kitty holds
    sys.stdout.write("\033[H\033[2J\033[3J")
    sys.stdout.flush()


def show(image, display: str, columns: int, fallback: Path) -> None:
    """Draw a PIL image `columns` cells wide."""
    if display == "file":
        image.save(fallback)
        print(f"(no inline images in this terminal; preview: {fallback})")
        return
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = base64.standard_b64encode(buffer.getvalue())
    out = sys.stdout.buffer
    if display == "kitty":
        chunks = [data[i : i + 4096] for i in range(0, len(data), 4096)]
        for index, chunk in enumerate(chunks):
            more = int(index < len(chunks) - 1)
            head = f"a=T,f=100,q=2,c={columns},m={more}" if index == 0 else f"m={more}"
            out.write(b"\033_G" + head.encode() + b";" + chunk + b"\033\\")
    else:
        out.write(f"\033]1337;File=inline=1;width={columns};preserveAspectRatio=1:".encode())
        out.write(data + b"\a")
    out.write(b"\n")
    out.flush()


def read_key() -> str:
    """One keypress, lower-cased, without waiting for Enter."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return os.read(fd, 1).decode(errors="ignore").lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def open_frames(layout, frames) -> None:
    paths = [str(layout.raw_image(r.split, r.image_id)) for r in frames[:40]]
    if sys.platform == "darwin":
        subprocess.run(["open", *paths], check=False)
    else:
        print("\n".join(paths))


def main():
    parser = argparse.ArgumentParser(description="Hand-label VisDrone sequences in the terminal")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Data config")
    parser.add_argument("--relabel", action="store_true", help="Every row, not just blank ones")
    parser.add_argument("--from", dest="start", type=str, default=None, help="Start at this key")
    parser.add_argument("--display", choices=["auto", "kitty", "iterm", "file"], default="auto",
                        help="How to show images (default: detect the terminal)")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        raise SystemExit("label_visdrone.py needs an interactive terminal")

    from omegaconf import OmegaConf

    from dawn.data.prepare import frames_for_key, pick_three, preview_image, summarize_frames
    from dawn.data.visdrone import (
        VisDroneLayout,
        is_still_collection,
        read_labels,
        read_stats_csv,
        sequence_id,
        set_labels,
    )

    cfg = OmegaConf.to_container(OmegaConf.load(args.data_config), resolve=True)["visdrone"]
    layout = VisDroneLayout.from_config(cfg)
    labels_csv = Path(cfg["labels_csv"])
    if not labels_csv.exists():
        raise SystemExit(f"No {labels_csv}: run `python scripts/prepare_visdrone.py label-sheets`")
    stats = list(read_stats_csv(layout.stats_csv).values())
    queue = [key for key, label in read_labels(labels_csv).items() if args.relabel or not label]
    if args.start:
        if args.start not in queue:
            raise SystemExit(f"{args.start} is not among the rows to label")
        queue = queue[queue.index(args.start) :]
    if not queue:
        print("Every row has a label. Use --relabel to go through them again.")
        return
    display = detect_display() if args.display == "auto" else args.display
    fallback = layout.stats_csv.parent / "labelling" / "label_preview.png"  # not committed
    fallback.parent.mkdir(parents=True, exist_ok=True)

    def load(key):
        frames = frames_for_key(stats, key)
        return frames, preview_image(layout, pick_three(frames), PREVIEW_BOX)

    pool = ThreadPoolExecutor(max_workers=2)
    pending = {}

    def fetch(i):
        if queue[i] not in pending:
            pending[queue[i]] = pool.submit(load, queue[i])
        return pending[queue[i]]

    i = 0
    try:
        while 0 <= i < len(queue):
            key = queue[i]
            frames, image = fetch(i).result()
            if i + 1 < len(queue):
                fetch(i + 1)  # decode the next one while you look at this one
            labels = read_labels(labels_csv)
            done = sum(1 for label in labels.values() if label)
            summary = summarize_frames(frames)
            kind = "still" if is_still_collection(sequence_id(key)) else "video"
            columns = max(40, min(shutil.get_terminal_size().columns - 1, 220))
            clear(display)
            print(f"[{i + 1}/{len(queue)}]  {key}  ({kind})   now: {labels.get(key) or '-'}"
                  f"      labelled {done}/{len(labels)}")
            shown = "darkest, median, brightest" if len(frames) > 2 else "every frame"
            print(f"{summary['split']} · {summary['frames']} frames"
                  f" · log10 L {summary['log10_lum']} · noise {summary['noise']}"
                  f" · highlights {summary['highlight_ratio']}x   (shown: {shown})")
            show(image, display, columns, fallback)
            print(HELP)
            key_pressed = read_key()
            if key_pressed in KEYS:
                set_labels(labels_csv, {key: KEYS[key_pressed]})
                i += 1
            elif key_pressed in ("s", " "):
                i += 1
            elif key_pressed == "b":
                i = max(0, i - 1)
            elif key_pressed == "o":
                open_frames(layout, frames)
            elif key_pressed in ("q", "\x04"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        clear(display)
    labels = read_labels(labels_csv)
    counts = {name: sum(1 for v in labels.values() if v == name) for name in KEYS.values()}
    blank = sum(1 for v in labels.values() if not v)
    print(f"Saved to {labels_csv}: " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f", unlabelled {blank}")


if __name__ == "__main__":
    main()
