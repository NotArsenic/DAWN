"""scripts/infer.py: reading thermal frames, the fixed uncertainty ramp, and the files written."""

import argparse
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
pytest.importorskip("matplotlib")
pytest.importorskip("omegaconf")
pytest.importorskip("peft")
pytest.importorskip("timm")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dawn.flow.inference import load_checkpoint  # noqa: E402
from dawn.flow.targets import FitSettings, compute_token_stats, prepare_batch  # noqa: E402
from dawn.flow.tokens import TokenLayout  # noqa: E402
from dawn.flow.trainer import FlowTrainer  # noqa: E402
from scripts.infer import (  # noqa: E402
    colorize,
    find_inputs,
    read_thermal,
    run_one,
    write_outputs,
)
from tests.flow_helpers import TINY_INPUT, synthetic_batch, tiny_cfg, wake_up  # noqa: E402


def _args(**updates):
    defaults = dict(
        samples=3,
        nfe=2,
        method="euler",
        seed_from_name=True,
        save_mean=False,
        save_raw=False,
        l_scale=10.0,
        ab_scale=10.0,
    )
    return argparse.Namespace(**{**defaults, **updates})


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.manual_seed(0)
    cfg = tiny_cfg()
    batch = synthetic_batch(2, seed=0, fallback=[False, True])
    stats = compute_token_stats(
        [prepare_batch(batch, TINY_INPUT, cfg["pairs"])],
        TokenLayout.from_config(cfg),
        FitSettings.from_config(cfg),
    )
    trainer = FlowTrainer.from_config(cfg, stats, "cpu")
    trainer.train_step(batch)
    path = trainer.save_checkpoint(tmp_path_factory.mktemp("ckpt") / "flow.pt")
    out = load_checkpoint(path, "cpu", use_ema=True)
    wake_up(out.velocity, std=0.1, seed=3)
    return out


def _write_thermal(path: Path, size, value=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full(size, value, dtype=np.uint8)
    array[: size[0] // 2] = 200  # something for the grid to key on
    Image.fromarray(array).save(path)
    return path


def test_find_inputs_takes_a_file_a_directory_and_a_limit(tmp_path):
    one = _write_thermal(tmp_path / "a.png", TINY_INPUT)
    _write_thermal(tmp_path / "b.png", TINY_INPUT)
    _write_thermal(tmp_path / "nested" / "c.png", TINY_INPUT)
    (tmp_path / "notes.txt").write_text("ignored")

    assert find_inputs(one, None) == [one]
    assert len(find_inputs(tmp_path, None)) == 3, "recursive, images only"
    assert len(find_inputs(tmp_path, 2)) == 2
    assert find_inputs(tmp_path, None) == sorted(find_inputs(tmp_path, None)), "sorted"
    with pytest.raises(FileNotFoundError, match="no such file"):
        find_inputs(tmp_path / "missing", None)


def test_a_raw_dronevehicle_frame_loses_its_border(tmp_path):
    """840x712 in, the 640x512 content window out, then down to the model's input."""
    path = _write_thermal(tmp_path / "raw.png", (712, 840))
    thermal = read_thermal(path, TINY_INPUT)
    assert thermal.shape == (1, 1, *TINY_INPUT)
    assert 0.0 <= float(thermal.min()) and float(thermal.max()) <= 1.0


def test_a_frame_at_the_model_size_is_used_as_is(tmp_path):
    path = _write_thermal(tmp_path / "exact.png", TINY_INPUT, value=128)
    thermal = read_thermal(path, TINY_INPUT)
    assert thermal.shape == (1, 1, *TINY_INPUT)
    assert float(thermal.flatten()[-1]) == pytest.approx(128 / 255)


def test_a_frame_smaller_than_the_model_is_refused(tmp_path):
    """Upsampling thermal would invent the texture the grid is guided by."""
    path = _write_thermal(tmp_path / "small.png", (TINY_INPUT[0] // 2, TINY_INPUT[1] // 2))
    with pytest.raises(ValueError, match="does not upsample thermal"):
        read_thermal(path, TINY_INPUT)


def test_the_uncertainty_ramp_is_fixed_not_per_image():
    """The same value must give the same colour in two different frames."""
    quiet = torch.full((1, 1, 4, 4), 1.0)
    loud = torch.full((1, 1, 4, 4), 1.0)
    loud[0, 0, 0, 0] = 9.0
    assert np.array_equal(colorize(quiet, 10.0)[1, 1], colorize(loud, 10.0)[1, 1])
    assert not np.array_equal(colorize(loud, 10.0)[0, 0], colorize(loud, 10.0)[1, 1])
    # above the scale it saturates rather than rescaling the image
    assert np.array_equal(colorize(loud, 10.0 / 9)[0, 0], colorize(loud * 100, 10.0 / 9)[0, 0])


def test_end_to_end_writes_an_image_and_two_maps(tmp_path, bundle):
    path = _write_thermal(tmp_path / "frame.png", bundle.input_size)
    out = run_one(bundle, path, torch.device("cpu"), _args())
    write_outputs(out, path.stem, tmp_path / "out", _args(save_mean=True, save_raw=True))

    written = {p.name for p in (tmp_path / "out").iterdir()}
    assert written == {
        "frame_rgb.png",
        "frame_mean.png",
        "frame_l_std.png",
        "frame_ab_std.png",
        "frame_uncertainty.npz",
    }
    rgb = np.asarray(Image.open(tmp_path / "out" / "frame_rgb.png"))
    assert rgb.shape == (*bundle.input_size, 3) and rgb.dtype == np.uint8
    maps = np.load(tmp_path / "out" / "frame_uncertainty.npz")
    assert maps["l_std"].shape == bundle.input_size
    assert np.all(maps["l_std"] >= 0) and np.all(np.isfinite(maps["ab_std"]))


def test_one_sample_writes_the_image_but_no_uncertainty(tmp_path, bundle):
    path = _write_thermal(tmp_path / "single.png", bundle.input_size)
    args = _args(samples=1)
    out = run_one(bundle, path, torch.device("cpu"), args)
    write_outputs(out, path.stem, tmp_path / "one", args)
    assert {p.name for p in (tmp_path / "one").iterdir()} == {"single_rgb.png"}


def test_seeding_from_the_name_makes_a_frame_reproducible(tmp_path, bundle):
    path = _write_thermal(tmp_path / "stable.png", bundle.input_size)
    first = run_one(bundle, path, torch.device("cpu"), _args(seed_from_name=True))
    second = run_one(bundle, path, torch.device("cpu"), _args(seed_from_name=True))
    assert torch.equal(first.sample_rgb, second.sample_rgb)
    assert torch.equal(first.l_std, second.l_std)
