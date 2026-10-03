"""A tiny synthetic VisDrone-DET tree for tests: real JPEGs and annotation files, no download."""

from pathlib import Path

import numpy as np
from PIL import Image

from dawn.data.visdrone import VisDroneLayout

# split, image_id, (width, height), mean sRGB code. Sequence 0000001 is a two-frame video,
# 9999901 a still collection (labelled per image), 0000004 a video in val.
FAKE_IMAGES = [
    ("train", "0000001_00001_d_0000001", (640, 480), 200),
    ("train", "0000001_00002_d_0000002", (640, 480), 190),
    ("train", "0000002_00005_d_0000003", (800, 450), 20),
    ("train", "9999901_00000_d_0000001", (480, 640), 90),  # portrait
    ("val", "0000004_00004_d_0000004", (600, 400), 180),
]
FAKE_LABELS = {
    "0000001": "day",  # sequence key: both frames
    "0000002": "night",
    "9999901_00000_d_0000001": "dusk",  # image key
    "0000004": "day",
}


def annotation_rows(width: int, height: int) -> list[list[int]]:
    """One row per box rule: left, top, width, height, score, category, truncation, occlusion."""
    return [
        [100, 50, 60, 40, 1, 4, 0, 0],  # kept: a car
        [10, 10, 30, 30, 1, 0, 0, 0],  # category 0, ignored region: dropped
        [200, 200, 40, 40, 1, 11, 0, 0],  # category 11, "others": dropped
        [300, 100, 50, 50, 0, 1, 0, 0],  # score 0: dropped
        [5, 60, 10, 40, 1, 1, 0, 0],  # 10 px wide: dropped by min_size (label rules keep it)
        [width - 30, height - 30, 80, 80, 1, 2, 0, 0],  # past the corner: clipped to 30x30, kept
    ]


NUM_LABEL_KEPT = 3  # rows the label rules keep: the car, the 10-px box and the corner box
NUM_EVAL_BOXES = 2  # at native resolution: the car and the clipped corner box


def make_fake_visdrone(root: Path) -> VisDroneLayout:
    """Write the raw tree under `root` and return a layout whose derived paths are under it too."""
    layout = VisDroneLayout(
        root=root / "raw",
        stats_csv=root / "data" / "stats" / "visdrone_stats.csv",
        manifest_dir=root / "data" / "manifests",
        prepared_dir=root / "data" / "visdrone_512",
    )
    rng = np.random.default_rng(0)
    for split, image_id, (width, height), level in FAKE_IMAGES:
        image_path = layout.raw_image(split, image_id)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.clip(level + rng.integers(-15, 16, (height, width, 3)), 0, 255)
        Image.fromarray(pixels.astype(np.uint8)).save(image_path, quality=95)
        ann_path = layout.raw_annotation(split, image_id)
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        rows = annotation_rows(width, height)
        ann_path.write_text("".join(",".join(map(str, row)) + "\n" for row in rows))
    return layout
