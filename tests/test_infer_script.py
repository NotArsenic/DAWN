"""scripts/infer.py: reading thermal frames, the fixed uncertainty ramp, and the files written.

Step 6, the optional Stage 2 refiner, is pinned from both sides: without `--refiner` the saved
bytes must be the Stage 1 render exactly (not merely close), and with one the two images change
while the two uncertainty PNGs do not -- the maps are the spread of the UNREFINED ensemble,
which is the one judgement call in this script (see its module docstring).
"""

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
from dawn.models.refiner import LocalRefiner  # noqa: E402
from dawn.uncertainty.spread import sample_with_spread  # noqa: E402
from scripts.infer import (  # noqa: E402
    colorize,
    find_inputs,
    main,
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
def flow_checkpoint(tmp_path_factory):
    """A tiny Stage 1 checkpoint on disk, so `main` can be driven the way a user drives it."""
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
    return trainer.save_checkpoint(tmp_path_factory.mktemp("ckpt") / "flow.pt")


@pytest.fixture(scope="module")
def bundle(flow_checkpoint):
    out = load_checkpoint(flow_checkpoint, "cpu", use_ema=True)
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

    assert find_inputs([one], None) == [one]
    assert len(find_inputs([tmp_path], None)) == 3, "recursive, images only"
    assert len(find_inputs([tmp_path], 2)) == 2
    assert find_inputs([tmp_path], None) == sorted(find_inputs([tmp_path], None)), "sorted"
    with pytest.raises(FileNotFoundError, match="no such file"):
        find_inputs([tmp_path / "missing"], None)


def test_find_inputs_takes_several_paths_and_deduplicates(tmp_path):
    """`--input` is `nargs="+"`, so a shell glob arrives as many paths, repeats and all."""
    one = _write_thermal(tmp_path / "a.png", TINY_INPUT)
    two = _write_thermal(tmp_path / "b.png", TINY_INPUT)

    assert find_inputs([one, two], None) == [one, two]
    assert find_inputs([two, one], None) == [one, two], "sorted, not argument order"
    assert find_inputs([one, one], None) == [one], "the same file twice is one frame"
    assert find_inputs([one, tmp_path], None) == sorted([one, two]), "a file and its directory"


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
    out, _, _ = run_one(bundle, path, torch.device("cpu"), _args())
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
    out, _, _ = run_one(bundle, path, torch.device("cpu"), args)
    write_outputs(out, path.stem, tmp_path / "one", args)
    assert {p.name for p in (tmp_path / "one").iterdir()} == {"single_rgb.png"}


def test_seeding_from_the_name_makes_a_frame_reproducible(tmp_path, bundle):
    path = _write_thermal(tmp_path / "stable.png", bundle.input_size)
    first, _, _ = run_one(bundle, path, torch.device("cpu"), _args(seed_from_name=True))
    second, _, _ = run_one(bundle, path, torch.device("cpu"), _args(seed_from_name=True))
    assert torch.equal(first.sample_rgb, second.sample_rgb)
    assert torch.equal(first.l_std, second.l_std)


# --- step 6: the optional local refiner --------------------------------------------------------


def _stage_1(bundle, path, args, device=torch.device("cpu")):
    """What `run_one` did before step 6 existed: read, one DINO pass, K members, render.

    Kept as a separate, duplicated reference rather than calling `run_one(..., refiner=None)`,
    because the thing being pinned is that the refiner-free path did not change at all.
    """
    thermal = read_thermal(path, bundle.input_size).to(device)
    encoded = bundle.encoder.encode(thermal, return_attn=True)
    return sample_with_spread(
        bundle.velocity,
        encoded,
        bundle.layout,
        bundle.mean,
        bundle.std,
        thermal,
        bundle.input_size,
        bundle.settings,
        num_samples=args.samples,
        method=args.method,
        nfe=args.nfe,
        image_ids=[path.stem] if args.seed_from_name else None,
    )


def _png_bytes(directory: Path) -> dict:
    """Every PNG in a directory, as raw bytes. PIL writes no timestamp, so these compare."""
    return {p.name: p.read_bytes() for p in sorted(directory.glob("*.png"))}


def test_without_a_refiner_the_output_is_bit_identical_to_stage_1(tmp_path, bundle):
    """`apply_refiner(..., None)` is an exact identity, so the default path must not move."""
    path = _write_thermal(tmp_path / "plain.png", bundle.input_size)
    args = _args(save_mean=True, save_raw=True)
    out, refined, residual = run_one(bundle, path, torch.device("cpu"), args)
    before = _stage_1(bundle, path, args)

    assert torch.equal(residual, torch.zeros_like(residual))
    for name in ("sample_rgb", "sample_lab", "rgb", "lab", "l_std", "ab_std"):
        assert torch.equal(getattr(out, name), getattr(before, name)), name
        # with no refiner the two returns are the same render, so neither branch can drift
        assert torch.equal(getattr(refined, name), getattr(before, name)), name

    write_outputs(before, "f", tmp_path / "stage1", args)
    write_outputs(out, "f", tmp_path / "now", args)
    assert _png_bytes(tmp_path / "stage1") == _png_bytes(tmp_path / "now")
    old = np.load(tmp_path / "stage1" / "f_uncertainty.npz")
    new = np.load(tmp_path / "now" / "f_uncertainty.npz")
    for name in ("l_std", "ab_std"):  # the .npz itself carries zip timestamps, its arrays do not
        assert np.array_equal(old[name], new[name]), name


def test_with_a_refiner_the_images_change_and_the_maps_do_not(tmp_path, bundle):
    """The spread stays the spread of the unrefined members (docs/conventions.md section 11)."""
    path = _write_thermal(tmp_path / "refined.png", bundle.input_size)
    args = _args(save_mean=True)
    refiner = wake_up(LocalRefiner(hidden_channels=4, num_blocks=1, kernel_size=3), std=0.5, seed=7)
    refiner.eval().requires_grad_(False)

    plain, _, _ = run_one(bundle, path, torch.device("cpu"), args)
    unrefined, refined, residual = run_one(bundle, path, torch.device("cpu"), args, refiner)

    for name in ("sample_rgb", "lab", "l_std", "ab_std"):
        assert torch.equal(getattr(unrefined, name), getattr(plain, name)), name

    assert 100 * float(residual.abs().mean()) > 0.5, "the refiner has to actually move L*"
    assert not torch.equal(refined.sample_rgb, plain.sample_rgb)
    assert not torch.equal(refined.rgb, plain.rgb)
    assert torch.equal(refined.sample_lab[:, 1:], plain.sample_lab[:, 1:]), "a*b* untouched"
    assert torch.equal(refined.l_std, plain.l_std)
    assert torch.equal(refined.ab_std, plain.ab_std)
    assert refined.num_samples == plain.num_samples

    write_outputs(plain, "f", tmp_path / "stage1", args)
    write_outputs(refined, "f", tmp_path / "stage2", args)
    changed = {"f_rgb.png", "f_mean.png"}
    stage_1, stage_2 = _png_bytes(tmp_path / "stage1"), _png_bytes(tmp_path / "stage2")
    assert set(stage_1) == set(stage_2) == changed | {"f_l_std.png", "f_ab_std.png"}
    for name, data in stage_1.items():
        assert (data == stage_2[name]) is (name not in changed), name


def _save_refiner(path: Path, refiner: LocalRefiner, cfg: dict) -> Path:
    """The pinned Stage 2 checkpoint, synthesized here: scripts/train_refiner.py is a separate
    piece of work, and this file tests the script that *reads* one."""
    torch.save(
        {
            "step": 5,
            "config": {"refiner": dict(cfg)},
            "refiner": refiner.state_dict(),
            "refiner_config": dict(cfg),
            "ema": {
                "decay": 0.999,
                "warmup": True,
                "num_updates": 7,
                "shadow": {
                    f"refiner.{name}": param.detach().clone().float()
                    for name, param in refiner.named_parameters()
                },
            },
            "optimizer": {},
            "scheduler": {},
        },
        path,
    )
    return path


def test_main_writes_the_same_maps_with_and_without_the_refiner(
    tmp_path, flow_checkpoint, monkeypatch
):
    """The whole script, as a user runs it.

    `main` writes the unrefined render to `--output` either way and puts the refined images in
    an `refined/` subfolder, so switching step 6 on must leave the top level byte-identical and
    add a folder whose image differs while its two uncertainty maps do not.
    """
    frame = _write_thermal(tmp_path / "in" / "frame.png", TINY_INPUT)
    cfg = {"hidden_channels": 4, "num_blocks": 1, "kernel_size": 3}
    refiner = wake_up(LocalRefiner(**cfg), std=0.5, seed=7)
    checkpoint = _save_refiner(tmp_path / "refiner.pt", refiner, cfg)

    base = [
        "infer.py",
        "--checkpoint",
        str(flow_checkpoint),
        "--input",
        str(frame),
        "--samples",
        "3",
        "--seed-from-name",
        "--output",
    ]
    # The tiny config's DINO backbone is randomly initialized (`pretrained: false`), so each
    # `main` draws its own; reseeding gives the two runs the same encoder, which is what a real
    # run gets for free from the pretrained weights.
    torch.manual_seed(0)
    monkeypatch.setattr(sys, "argv", [*base, str(tmp_path / "plain")])
    main()
    torch.manual_seed(0)
    monkeypatch.setattr(
        sys, "argv", [*base, str(tmp_path / "refined"), "--refiner", str(checkpoint)]
    )
    main()

    written = {"frame_rgb.png", "frame_l_std.png", "frame_ab_std.png"}
    plain = _png_bytes(tmp_path / "plain")
    top = _png_bytes(tmp_path / "refined")
    step_6 = _png_bytes(tmp_path / "refined" / "refined")

    assert set(plain) == set(top) == set(step_6) == written
    assert not (tmp_path / "plain" / "refined").exists(), "no subfolder without --refiner"
    assert top == plain, "the unrefined output is untouched by step 6"
    assert step_6["frame_rgb.png"] != plain["frame_rgb.png"]
    assert step_6["frame_l_std.png"] == plain["frame_l_std.png"]
    assert step_6["frame_ab_std.png"] == plain["frame_ab_std.png"]


def test_help_works_and_offers_the_refiner_flag(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["infer.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0
    assert "--refiner" in capsys.readouterr().out
