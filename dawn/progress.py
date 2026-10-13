"""Terminal and log file progress reporting utilities."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator, Sized
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def format_duration(seconds: float) -> str:
    """Format elapsed duration in seconds into a concise string (e.g. 7s, 1m42s, 3h05m)."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def eta_line(desc: str, done: int, total: int | None, elapsed: float) -> str:
    """Format a progress status line with rate and estimated time of completion."""
    rate = done / elapsed if elapsed > 0 else 0.0
    head = f"{desc}: {done}" + (f"/{total}" if total else "")
    share = f" ({100 * done / total:.0f}%)" if total else ""
    tail = ""
    if total and rate > 0 and done < total:
        tail = f", ETA {format_duration((total - done) / rate)}"
    return f"{head}{share} {format_duration(elapsed)} elapsed, {rate:.1f}/s{tail}"


def use_bar(force_lines: bool = False) -> bool:
    """Check whether an interactive progress bar should be displayed."""
    if force_lines:
        return False
    try:
        import tqdm  # noqa: F401
    except ImportError:
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def progress(
    iterable: Iterable[T],
    desc: str,
    total: int | None = None,
    every: float = 0.05,
    force_lines: bool = False,
) -> Iterator[T]:
    """Yield items from iterable while logging progress or updating a progress bar.

    Args:
        iterable: Iterable sequence or generator to wrap.
        desc: Description prefix for status messages.
        total: Optional total item count if len(iterable) is unavailable.
        every: Interval fraction between progress log lines when not using a progress bar.
        force_lines: If True, log lines periodically instead of displaying a bar.

    Yields:
        Items yielded from the input iterable.
    """
    if total is None and isinstance(iterable, Sized):
        total = len(iterable)

    if use_bar(force_lines):
        from tqdm import tqdm

        yield from tqdm(iterable, desc=desc, total=total, unit="it", leave=False)
        return

    start = time.perf_counter()
    step = max(1, int(total * every)) if total else 200
    done = 0
    for item in iterable:
        yield item
        done += 1
        if done % step == 0:
            logger.info(eta_line(desc, done, total, time.perf_counter() - start))
    elapsed = time.perf_counter() - start
    if done:
        logger.info(f"{desc}: {done} done in {format_duration(elapsed)}")
