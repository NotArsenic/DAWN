"""A tiny raw DroneVehicle tree for tests: 840x712 frames with a white 100-px border, JPEG images
and VOC-like XML in all three geometry forms (polygon, bndbox, point), the dataset's class typos,
and a few pairs built to fail each filter. Tests never read data/DroneVehicle.

Every pair is made from a known scene. The RGB content is the scene; the thermal content is the
scene's luminance sampled at `T(x_thermal)`, so the ground-truth similarity `T` maps thermal pixel
coordinates onto RGB ones (pixel-index coordinates, x = column, y = row, the convention of
dawn/data/align.py). The thermal XML holds the RGB boxes pulled back through `T^-1`.

`FAKE_PAIRS` lists what each stem is for, and `EXPECTED_FINAL` what the whole pipeline should
decide about it.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image
from scipy import ndimage

FRAME_H, FRAME_W, BORDER = 712, 840, 100
H, W = FRAME_H - 2 * BORDER, FRAME_W - 2 * BORDER  # the cropped frame, 512 x 640
DAY_CUTOFF = 0.05  # the fake config's day_filter.min_lum_median
CLASSES = ("car", "truck", "bus", "van", "feright car")  # the last one is a dataset typo
_GRID = [(x, y) for y in (110, 250, 400) for x in (80, 200, 320, 440, 560)]


def similarity(scale: float, rot_deg: float, tx: float, ty: float) -> np.ndarray:
    """3x3 similarity (rotation about the origin, then translation) in pixel-index coordinates."""
    a, b = scale * math.cos(math.radians(rot_deg)), scale * math.sin(math.radians(rot_deg))
    return np.array([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class PairSpec:
    """One fake stem. See the module docstring of the test that uses it for what it must do."""

    split: str
    stem: str
    note: str
    warp: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # scale, rot deg, tx, ty
    vehicles: int = 10
    seed: int = 0
    brightness: float = 1.0  # < 1 darkens the RGB scene: a night frame
    thermal_mode: str = "L"  # "L", or "RGB" (3 identical channels)
    has_thermal: bool = True
    copy_boxes: int = 0  # this many thermal boxes are exact copies of RGB boxes
    all_copied: bool = False  # the thermal label file equals the RGB one
    perspective: float = 0.0  # thermal boxes also bent by a perspective term: no similarity fits
    movers: tuple[int, ...] = ()  # vehicles whose thermal box is displaced 14 px (moved)
    content_shift_px: int = 0  # thermal content is displaced this much from what the labels say
    bad_border: bool = False  # RGB border has a dark stripe
    extras: bool = False  # `*` class, point-only, bndbox and typo objects
    duplicate_of: tuple[str, str] | None = None  # copy these four files from (split, stem)

    @property
    def image_id(self) -> str:
        return f"{self.split}_{self.stem}"


FAKE_PAIRS = [
    PairSpec(
        "train",
        "00001",
        "good; movers, copied box, odd forms",
        (1.01, 0.6, 4, -3),
        vehicles=10,
        seed=1,
        movers=(9,),
        copy_boxes=1,
        extras=True,
    ),
    PairSpec("train", "00002", "labels copied", vehicles=10, seed=2, all_copied=True),
    PairSpec("train", "00003", "few vehicles", (1.0, 0.3, 2, 1), vehicles=3, seed=3),
    PairSpec(
        "train", "00004", "night frame", (1.0, 0.0, 1, 1), vehicles=10, seed=4, brightness=0.12
    ),
    PairSpec("train", "00005", "no thermal image", vehicles=10, seed=5, has_thermal=False),
    PairSpec("train", "00006", "scale off by 8%", (1.08, 0.0, 0, 0), vehicles=10, seed=6),
    PairSpec(
        "train",
        "00007",
        "boxes bent: residual too high",
        (1.0, 0.0, 2, 2),
        vehicles=8,
        seed=7,
        perspective=8e-5,
    ),
    PairSpec(
        "train",
        "00008",
        "content shifted 6 px from the labels",
        (1.0, 0.2, 3, 2),
        vehicles=10,
        seed=8,
        content_shift_px=6,
    ),
    PairSpec("train", "00009", "bad border", vehicles=10, seed=9, bad_border=True),
    PairSpec(
        "train",
        "00010",
        "good, 3-channel thermal",
        (0.99, -0.4, -5, 3),
        vehicles=9,
        seed=10,
        thermal_mode="RGB",
    ),
    PairSpec("val", "00001", "duplicate of train 00001", duplicate_of=("train", "00001")),
    PairSpec("val", "00002", "good, unique", (1.005, 0.2, 3, 4), vehicles=9, seed=12),
    PairSpec("test", "00001", "good, unique", (0.995, -0.3, -2, 5), vehicles=9, seed=21),
]

# What `stats -> day -> align -> dedup -> write` should decide, by image ID.
EXPECTED_FINAL = {
    "train_00001": "ok",
    "train_00002": "labels_copied",
    "train_00003": "few_matches",
    "train_00004": "not_day",
    "train_00005": "missing_thermal",
    "train_00006": "scale",
    "train_00007": "residual",
    "train_00008": "edge_check",
    "train_00009": "bad_border",
    "train_00010": "ok",
    "val_00001": "near_duplicate",
    "val_00002": "ok",
    "test_00001": "ok",
}
EXPECTED_MANIFESTS = {
    "train": ["train_00001", "train_00010"],
    "val": ["val_00002"],
    "test": ["test_00001"],
}


@dataclass
class FakeTree:
    root: Path
    specs: list[PairSpec]
    truth: dict[str, np.ndarray] = field(default_factory=dict)  # image_id -> 3x3 thermal -> RGB


# --- scene ---------------------------------------------------------------------------------------


def _vehicles(spec: PairSpec) -> list[dict]:
    """Deterministic vehicle list in cropped RGB coordinates: centre, size, class."""
    rng = np.random.default_rng(spec.seed)
    out = []
    for i in range(spec.vehicles):
        gx, gy = _GRID[i]
        w, h = (28, 15) if CLASSES[i % 5] in ("truck", "bus") else (22, 12)
        cx, cy = gx + int(rng.integers(-12, 13)), gy + int(rng.integers(-12, 13))
        out.append({"cx": cx, "cy": cy, "w": w, "h": h, "cls": CLASSES[i % 5]})
    return out


def _box(v: dict) -> list[tuple[float, float]]:
    """Integer polygon (TL, TR, BR, BL) of an axis-aligned vehicle; pixel-index coordinates."""
    x0, x1 = v["cx"] - v["w"] // 2, v["cx"] + v["w"] // 2
    y0, y1 = v["cy"] - v["h"] // 2, v["cy"] + v["h"] // 2
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _scene(spec: PairSpec) -> np.ndarray:
    """`(H, W, 3)` sRGB-coded float scene: smooth background, buildings, vehicles."""
    rng = np.random.default_rng(1000 + spec.seed)
    base = ndimage.gaussian_filter(rng.random((H, W)), 10)
    base = 0.35 + 0.25 * (base - base.min()) / (np.ptp(base) + 1e-9)
    img = np.stack([base * 1.0, base * 0.95, base * 0.9], axis=-1)
    for _ in range(5):  # large buildings: strong, long edges
        x0, y0 = int(rng.integers(0, W - 120)), int(rng.integers(0, H - 100))
        w, h = int(rng.integers(50, 120)), int(rng.integers(40, 100))
        img[y0 : y0 + h, x0 : x0 + w] = rng.choice([0.15, 0.85]) * np.array([1.0, 0.9, 0.8])
    palette = [(0.9, 0.1, 0.1), (0.1, 0.2, 0.9), (0.95, 0.95, 0.9), (0.05, 0.05, 0.05)]
    for i, v in enumerate(_vehicles(spec)):
        (x0, y0), _, (x1, y1), _ = _box(v)
        img[y0 : y1 + 1, x0 : x1 + 1] = palette[i % 4]
    return np.clip(img, 0, 1) * spec.brightness


def _thermal(scene: np.ndarray, spec: PairSpec, warp: np.ndarray, rng) -> np.ndarray:
    """`(H, W)` float thermal: the scene's luminance sampled at T(x_thermal), a monotone gain
    curve and noise. `content_shift_px` displaces the content away from what the labels say."""
    luma = scene @ np.array([0.299, 0.587, 0.114])
    rows, cols = np.mgrid[0:H, 0:W].astype(np.float64)
    src = warp @ np.stack([cols.ravel(), rows.ravel(), np.ones(H * W)])
    sample = ndimage.map_coordinates(
        luma,
        [src[1] - spec.content_shift_px, src[0] - spec.content_shift_px],
        order=1,
        mode="nearest",
    ).reshape(H, W)
    thermal = (np.clip(sample, 0, 1) ** 0.8) * 0.9 + 0.05
    return np.clip(thermal + rng.normal(0, 0.01, thermal.shape), 0, 1)


# --- files ---------------------------------------------------------------------------------------


def _frame(content: np.ndarray, mode: str) -> Image.Image:
    """White 840x712 frame with `content` (uint8, (H, W) or (H, W, 3)) in the middle."""
    shape = (FRAME_H, FRAME_W) + content.shape[2:]
    frame = np.full(shape, 255, np.uint8)
    frame[BORDER : BORDER + H, BORDER : BORDER + W] = content
    return Image.fromarray(frame, mode="L" if frame.ndim == 2 else "RGB")


def _object_xml(name: str, geometry: str) -> str:
    return (
        f"  <object>\n    <name>{escape(name)}</name>\n    <pose>Unspecified</pose>\n"
        f"    <truncated>0</truncated>\n    <difficult>0</difficult>\n{geometry}  </object>\n"
    )


def _polygon_xml(poly) -> str:
    inner = "".join(
        f"      <x{i}>{round(x) + BORDER}</x{i}>\n      <y{i}>{round(y) + BORDER}</y{i}>\n"
        for i, (x, y) in enumerate(poly, 1)
    )
    return f"    <polygon>\n{inner}    </polygon>\n"


def _bndbox_xml(poly) -> str:
    xs, ys = [round(x) + BORDER for x, _ in poly], [round(y) + BORDER for _, y in poly]
    return (
        f"    <bndbox>\n      <xmin>{min(xs)}</xmin>\n      <ymin>{min(ys)}</ymin>\n"
        f"      <xmax>{max(xs)}</xmax>\n      <ymax>{max(ys)}</ymax>\n    </bndbox>\n"
    )


def _write_xml(path: Path, folder: str, filename: str, objects: list[str]) -> None:
    path.write_text(
        "<annotation>\n"
        f"  <folder>{escape(folder)}</folder>\n  <filename>{escape(filename)}</filename>\n"
        "  <path>C:\\fake\\x.jpg</path>\n  <source>\n    <database>Unknown</database>\n"
        f"  </source>\n  <size>\n    <width>{FRAME_W}</width>\n    <height>{FRAME_H}</height>\n"
        "    <depth>1</depth>\n  </size>\n  <segmented>0</segmented>\n"
        + "".join(objects)
        + "</annotation>\n",
        encoding="utf-8",
    )


def _labels(spec: PairSpec, warp: np.ndarray) -> tuple[list[str], list[str]]:
    """XML object lists `(rgb, thermal)` for a spec."""
    inverse = np.linalg.inv(warp)
    rgb, thermal = [], []
    for i, v in enumerate(_vehicles(spec)):
        poly = _box(v)
        form = _bndbox_xml if spec.extras and i == 1 else _polygon_xml  # one RGB bndbox
        rgb.append(_object_xml(v["cls"], form(poly)))
        if spec.all_copied or i < spec.copy_boxes:
            thermal.append(_object_xml(v["cls"], form(poly)))  # an exact copy of the RGB box
            continue
        bend = np.array([[1.0, 0, 0], [0, 1.0, 0], [spec.perspective, 0, 1.0]])
        pulled = []
        for x, y in poly:
            p = np.linalg.inv(bend) @ inverse @ np.array([x, y, 1.0])
            pulled.append((p[0] / p[2], p[1] / p[2]))
        offset = np.zeros(2)
        if i in spec.movers:
            offset = offset + np.array([14.0, 0.0])
        pulled = [(x + offset[0], y + offset[1]) for x, y in pulled]
        name = "truvk" if spec.extras and v["cls"] == "truck" else v["cls"]
        thermal.append(_object_xml(name, _polygon_xml(pulled)))
    if spec.extras:  # objects the reader must skip or drop
        rgb.append(_object_xml("*", _polygon_xml([(10, 10), (20, 10), (20, 20), (10, 20)])))
        rgb.append(
            _object_xml("car", "    <point>\n      <x>300</x>\n      <y>300</y>\n    </point>\n")
        )
    return rgb, thermal


def make_fake_dronevehicle(root: Path, specs: list[PairSpec] | None = None) -> FakeTree:
    """Write the raw tree under `root` (`<root>/<split>/<split>img/...`) and return its truth."""
    specs = FAKE_PAIRS if specs is None else specs
    tree = FakeTree(root=root, specs=list(specs))
    for split in ("train", "val", "test"):
        for kind in ("img", "imgr", "label", "labelr"):
            (root / split / f"{split}{kind}").mkdir(parents=True, exist_ok=True)
    for spec in specs:
        paths = {
            kind: root / spec.split / f"{spec.split}{kind}" / f"{spec.stem}.{ext}"
            for kind, ext in (("img", "jpg"), ("imgr", "jpg"), ("label", "xml"), ("labelr", "xml"))
        }
        if spec.duplicate_of:
            src_split, src_stem = spec.duplicate_of
            for kind, path in paths.items():
                src = root / src_split / f"{src_split}{kind}" / f"{src_stem}{path.suffix}"
                shutil.copyfile(src, path)
            tree.truth[spec.image_id] = tree.truth[f"{src_split}_{src_stem}"]
            continue
        warp = similarity(*spec.warp)
        tree.truth[spec.image_id] = warp
        rng = np.random.default_rng(3000 + spec.seed)
        scene = _scene(spec)
        rgb8 = np.round(scene * 255).astype(np.uint8)
        if spec.bad_border:
            rgb_frame = _frame(rgb8, "RGB")
            frame = np.asarray(rgb_frame).copy()
            frame[40:60, :] = 60  # a dark stripe inside the top border
            rgb_frame = Image.fromarray(frame, mode="RGB")
        else:
            rgb_frame = _frame(rgb8, "RGB")
        rgb_frame.save(paths["img"], quality=95)
        if spec.has_thermal:
            t8 = np.round(_thermal(scene, spec, warp, rng) * 255).astype(np.uint8)
            thermal_frame = _frame(t8, "L")
            if spec.thermal_mode == "RGB":
                thermal_frame = thermal_frame.convert("RGB")
            thermal_frame.save(paths["imgr"], quality=95)
        rgb_objects, thermal_objects = _labels(spec, warp)
        _write_xml(paths["label"], f"flight_{spec.stem}", f"DJI_{spec.stem}", rgb_objects)
        _write_xml(paths["labelr"], f"flight_{spec.stem}", f"DJI_{spec.stem}_R", thermal_objects)
    return tree


def write_framed_jpeg(path: Path, content: np.ndarray, mode: str = "L") -> None:
    """A bare JPEG for reader tests: uint8 `content` in the white frame, saved as `mode`."""
    _frame(content, "L" if content.ndim == 2 else "RGB").convert(mode).save(path, quality=95)


def fake_config(base: Path, raw_root: Path) -> dict:
    """The real configs/data.yaml with every path moved under `base` and the day cutoff set for
    the fake scenes. Returns a plain dict (what the scripts get from OmegaConf)."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "data.yaml")
    )
    dv = cfg["dronevehicle"]
    dv["root"] = str(raw_root)
    dv["aligned_dir"] = str(base / "aligned")
    dv["stats_csv"] = str(base / "stats" / "dronevehicle_stats.csv")
    dv["manifest_dir"] = str(base / "manifests")
    dv["outputs_dir"] = str(base / "outputs")
    cfg["day_filter"]["min_lum_median"] = DAY_CUTOFF
    cfg["day_filter"]["sheet_count"] = 8
    cfg["dedup"]["sheet_count"] = 4
    return cfg
