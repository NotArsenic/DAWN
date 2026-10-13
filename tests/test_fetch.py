"""Tests for dataset archive discovery, safe extraction, and folder arrangement."""

import zipfile

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from dawn.data.fetch import (  # noqa: E402
    SPLITS,
    arrange_split,
    discover_split_dirs,
    folder_names,
    safe_extract,
    safe_members,
    verify_split,
)

STEMS = [f"{i:05d}" for i in range(120)]  # over MIN_FILES_PER_FOLDER


def _populate(directory, split, stems=STEMS, size=(840, 712), skip=()):
    """Populate directory with mock image and annotation files for split testing."""
    for name in folder_names(split):
        if name in skip:
            continue
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            if name.endswith("label") or name.endswith("labelr"):
                (target / f"{stem}.xml").write_text("<annotation/>", encoding="utf-8")
            else:
                Image.new("RGB", size, (255, 255, 255)).save(target / f"{stem}.jpg")
    return directory


# --- discovery -----------------------------------------------------------------------------------


def test_the_four_folders_are_named_after_the_split():
    assert folder_names("train") == ("trainimg", "trainimgr", "trainlabel", "trainlabelr")
    assert folder_names("val")[1] == "valimgr"


@pytest.mark.parametrize(
    "nesting",
    [
        (),  # the archive unpacks the folders at its root
        ("train",),  # wrapped in the split name
        ("DroneVehicle", "train"),  # wrapped twice
    ],
    ids=["flat", "split_dir", "nested_twice"],
)
def test_the_folders_are_found_however_deep_the_archive_buried_them(tmp_path, nesting):
    """The reason this searches rather than hard-codes a path: mirrors differ, and the Hub
    README documents none of this."""
    inner = tmp_path.joinpath(*nesting) if nesting else tmp_path
    _populate(inner, "train")
    found = discover_split_dirs(tmp_path, "train")
    assert set(found) == set(folder_names("train"))
    assert found["trainimg"] == inner / "trainimg"


def test_folder_names_are_matched_case_insensitively(tmp_path):
    for name in ("TrainImg", "TRAINIMGR", "trainlabel", "trainLabelR"):
        (tmp_path / name).mkdir()
    assert set(discover_split_dirs(tmp_path, "train")) == set(folder_names("train"))


def test_a_macos_shadow_tree_does_not_win(tmp_path):
    """Archives zipped on a Mac carry a __MACOSX mirror; linking that would find no images."""
    _populate(tmp_path / "__MACOSX" / "train", "train", stems=STEMS[:1])
    _populate(tmp_path / "train", "train", stems=STEMS[:1])
    assert discover_split_dirs(tmp_path, "train")["trainimg"] == tmp_path / "train" / "trainimg"


def test_the_shallowest_copy_wins_when_an_archive_nests_a_duplicate(tmp_path):
    (tmp_path / "trainimg").mkdir()
    (tmp_path / "extra" / "deeper" / "trainimg").mkdir(parents=True)
    assert discover_split_dirs(tmp_path, "train")["trainimg"] == tmp_path / "trainimg"


def test_a_split_that_is_not_there_is_reported_as_nothing_found(tmp_path):
    _populate(tmp_path, "train", stems=STEMS[:1])
    assert discover_split_dirs(tmp_path, "val") == {}


# --- extraction safety ---------------------------------------------------------------------------


def test_a_member_that_escapes_its_directory_is_refused(tmp_path):
    """Downloaded data: a zip naming `../escaped` must not be written outside the destination."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("fine.txt", "ok")
        handle.writestr("../escaped.txt", "not ok")
    with zipfile.ZipFile(archive) as handle:
        with pytest.raises(ValueError, match="escapes"):
            safe_members(handle, tmp_path / "dest")
    with pytest.raises(ValueError, match="escapes"):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_an_ordinary_archive_extracts_and_is_not_unpacked_twice(tmp_path):
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("train/trainimg/00001.jpg", "x")
    out = safe_extract(archive, tmp_path / "work")
    assert (out / "train" / "trainimg" / "00001.jpg").exists()
    assert (out / ".extracted").exists()

    # The marker is what makes a 14 GB re-run cheap: a second call must not rewrite the tree.
    (out / "train" / "trainimg" / "00001.jpg").write_text("edited", encoding="utf-8")
    safe_extract(archive, tmp_path / "work")
    assert (out / "train" / "trainimg" / "00001.jpg").read_text() == "edited"
    safe_extract(archive, tmp_path / "work", force=True)
    assert (out / "train" / "trainimg" / "00001.jpg").read_text() == "x"


# --- arranging -----------------------------------------------------------------------------------


def test_arranging_symlinks_the_folders_into_the_layout_data_yaml_expects(tmp_path):
    source = _populate(tmp_path / "extracted" / "train", "train", stems=STEMS[:3])
    root = tmp_path / "data" / "DroneVehicle"
    arrange_split(discover_split_dirs(tmp_path / "extracted", "train"), root, "train")
    for name in folder_names("train"):
        link = root / "train" / name
        assert link.is_symlink() and link.resolve() == (source / name).resolve()
    assert (root / "train" / "trainimg" / "00000.jpg").exists()


def test_arranging_twice_is_idempotent(tmp_path):
    _populate(tmp_path / "extracted", "train", stems=STEMS[:2])
    found = discover_split_dirs(tmp_path / "extracted", "train")
    root = tmp_path / "root"
    arrange_split(found, root, "train")
    actions = arrange_split(found, root, "train")
    assert all("already linked" in a for a in actions), actions


def test_a_real_directory_already_in_place_is_left_alone(tmp_path):
    """Somebody who unpacked by hand must not have their tree replaced by a link."""
    _populate(tmp_path / "extracted", "train", stems=STEMS[:2])
    root = tmp_path / "root"
    (root / "train" / "trainimg").mkdir(parents=True)
    (root / "train" / "trainimg" / "mine.jpg").write_text("keep me", encoding="utf-8")
    actions = arrange_split(discover_split_dirs(tmp_path / "extracted", "train"), root, "train")
    assert (root / "train" / "trainimg" / "mine.jpg").read_text() == "keep me"
    assert any("left alone" in a for a in actions)


@pytest.mark.parametrize("mode", ["copy", "move"])
def test_copy_and_move_put_real_directories_there(tmp_path, mode):
    _populate(tmp_path / "extracted", "train", stems=STEMS[:2])
    root = tmp_path / "root"
    arrange_split(discover_split_dirs(tmp_path / "extracted", "train"), root, "train", mode)
    target = root / "train" / "trainimg"
    assert target.is_dir() and not target.is_symlink()
    assert (target / "00000.jpg").exists()


def test_arranging_a_split_missing_a_folder_says_which_one(tmp_path):
    _populate(tmp_path / "extracted", "train", stems=STEMS[:2], skip=("trainlabelr",))
    with pytest.raises(ValueError, match="trainlabelr"):
        arrange_split(discover_split_dirs(tmp_path / "extracted", "train"), tmp_path / "r", "train")


def test_an_unknown_arrange_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        arrange_split({}, tmp_path, "train", "hardlink")


# --- verification --------------------------------------------------------------------------------


def test_a_correctly_arranged_split_has_no_problems(tmp_path):
    _populate(tmp_path / "train", "train")
    assert verify_split(tmp_path, "train") == []


def test_a_missing_folder_is_reported_without_raising(tmp_path):
    _populate(tmp_path / "train", "train", skip=("trainlabelr",))
    problems = verify_split(tmp_path, "train")
    assert any("trainlabelr is missing" in p for p in problems)


def test_stems_that_do_not_pair_across_the_four_folders_are_caught(tmp_path):
    """The thing that actually breaks a prepare run: RGB frames with no thermal counterpart.

    Reported once they are more than a rounding error -- a few are tolerated, see below.
    """
    _populate(tmp_path / "train", "train")
    for i in range(20):
        (tmp_path / "train" / "trainimg" / f"9{i:04d}.jpg").write_bytes(b"")
    problems = verify_split(tmp_path, "train")
    assert any("trainimg has 20 stems the other three do not" in p for p in problems)


def test_a_frame_of_the_wrong_size_is_caught_before_the_pipeline_crops_it(tmp_path):
    """Everything downstream assumes 840x712 with a 100-px border -> 640x512."""
    _populate(tmp_path / "train", "train", size=(640, 512))
    problems = verify_split(tmp_path, "train")
    assert any("640x512, expected 840x712" in p for p in problems)


def test_an_almost_empty_folder_is_caught(tmp_path):
    _populate(tmp_path / "train", "train", stems=STEMS[:3])
    assert any("holds only 3" in p for p in verify_split(tmp_path, "train"))


def test_an_unreadable_jpeg_is_reported_rather_than_raising(tmp_path):
    _populate(tmp_path / "train", "train")
    (tmp_path / "train" / "trainimg" / "00000.jpg").write_bytes(b"not a jpeg")
    assert any("cannot be read" in p for p in verify_split(tmp_path, "train"))


def test_the_splits_are_the_three_the_dataset_ships():
    assert SPLITS == ("train", "val", "test")


def test_one_corrupt_member_does_not_destroy_the_whole_extraction(tmp_path):
    """`extractall` aborts the archive on the first bad CRC: a 14 GB unpack thrown away over
    one unreadable JPEG out of 56,878. The pair it belongs to drops out later instead."""
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("train/trainimg/00001.jpg", "good")
        handle.writestr("train/trainimgr/04991.jpg", "AAAACORRUPTMEAAAA")
        handle.writestr("train/trainimg/00002.jpg", "also good")
    # Flip a byte of the stored data so its CRC no longer matches -- a bad download.
    raw = bytearray(archive.read_bytes())
    raw[raw.index(b"CORRUPTME")] = ord("X")
    archive.write_bytes(bytes(raw))

    out = safe_extract(archive, tmp_path / "work")
    assert (out / "train" / "trainimg" / "00001.jpg").exists(), "files before the bad one"
    assert (out / "train" / "trainimg" / "00002.jpg").exists(), "and files after it"
    assert not (out / "train" / "trainimgr" / "04991.jpg").exists()
    assert "skipped train/trainimgr/04991.jpg" in (out / ".extracted").read_text(encoding="utf-8")


def test_an_archive_that_is_corrupt_throughout_still_raises(tmp_path):
    """One bad file is tolerable; a bad download is not, and must not look like success."""
    archive = tmp_path / "train.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for i in range(6):
            handle.writestr(f"train/trainimg/{i:05d}.jpg", f"ROT{i}ROT")
    raw = bytearray(archive.read_bytes())
    for i in range(6):
        raw[raw.index(f"ROT{i}ROT".encode())] = ord("Z")
    archive.write_bytes(bytes(raw))
    with pytest.raises(zipfile.BadZipFile, match="archive-level corruption"):
        safe_extract(archive, tmp_path / "work", max_bad=2)


def test_a_handful_of_unpaired_stems_is_tolerated(tmp_path, caplog):
    """A bad download costs a few files; the day stage already drops pairs missing a modality
    (docs/conventions.md 10.1), so stopping the whole run over 0.1% is out of proportion."""
    _populate(tmp_path / "train", "train")  # 120 paired stems
    (tmp_path / "train" / "trainimg" / "99999.jpg").write_bytes(b"")  # 1 unpaired, 0.8%
    with caplog.at_level("WARNING"):
        assert verify_split(tmp_path, "train") == []
    assert "tolerated" in caplog.text


def test_many_unpaired_stems_still_fail(tmp_path):
    """Losing a tenth of a folder is a broken download, not a few bad files."""
    _populate(tmp_path / "train", "train")
    for i in range(20):  # 14% of the folder
        (tmp_path / "train" / "trainimg" / f"9{i:04d}.jpg").write_bytes(b"")
    assert any("stems the other three do not" in p for p in verify_split(tmp_path, "train"))
