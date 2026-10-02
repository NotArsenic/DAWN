"""Test helpers."""

import functools
import traceback
from pathlib import Path

import pytest

import dawn

_PACKAGE_DIR = Path(dawn.__file__).resolve().parent


def skip_until_implemented(fn):
    """Skip a test while the scaffold code it exercises is still a `NotImplementedError` stub.

    A NotImplementedError raised from inside the `dawn` package becomes a skip. Anything else
    fails as usual, including a NotImplementedError raised by torch or by the test itself, so
    each test starts running as soon as the stub it depends on is implemented.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except NotImplementedError as exc:
            frames = traceback.extract_tb(exc.__traceback__)
            if frames and Path(frames[-1].filename).resolve().is_relative_to(_PACKAGE_DIR):
                pytest.skip(f"scaffold stub: {frames[-1].name}()")
            raise

    return wrapper
