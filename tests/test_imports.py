"""Tests verifying that all package modules import cleanly and all scripts provide help."""

import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import dawn  # noqa: E402

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "scripts").glob("*.py"))


@pytest.mark.parametrize(
    "name", [m.name for m in pkgutil.walk_packages(dawn.__path__, prefix="dawn.")]
)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_help(script, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"script_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as exit_info:
        module.main()
    assert exit_info.value.code == 0
