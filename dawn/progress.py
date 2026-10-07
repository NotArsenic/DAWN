"""Progress reporting that is readable both in a terminal and in a log file.

Several passes here take minutes with nothing to show for it: the 28,439-stem `prepare` passes,
the 2,000-pair token-stats fit at the start of every training run, the oracle over the val pairs.

A tqdm bar is the right answer **interactively** and the wrong one in a log. tqdm redraws with a
carriage return, so an overnight run under `nohup`, `tmux` or a CI job turns megabytes of `\\r`
into the file the person debugging it later has to read. So:

    stderr is a TTY   ->  a tqdm bar, if tqdm is installed
    otherwise         ->  one log line every `every` of the work, with rate and ETA

Callers do not choose; they call `progress(...)` and get whichever fits. `force_lines=True` opts
out of the bar for a caller that is always logged rather than watched.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable, Iterator, Sized
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def format_duration(seconds: float) -> str:
    """`7s`, `1m42s`, `3h05m`. Short enough to sit inside a log line."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def eta_line(desc: str, done: int, total: int | None, elapsed: float) -> str:
    """`align: 4200/8534 (49%) 2m10s elapsed, 18.2/s, ETA 3m58s`.

    With no `total` the share and the ETA are left out rather than guessed at.
    """
    rate = done / elapsed if elapsed > 0 else 0.0
    head = f"{desc}: {done}" + (f"/{total}" if total else "")
    share = f" ({100 * done / total:.0f}%)" if total else ""
    tail = ""
    if total and rate > 0 and done < total:
        tail = f", ETA {format_duration((total - done) / rate)}"
    return f"{head}{share} {format_duration(elapsed)} elapsed, {rate:.1f}/s{tail}"


def use_bar(force_lines: bool = False) -> bool:
    """Whether a tqdm bar would be seen by a person rather than written to a file."""
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
    """Yield from `iterable`, reporting progress the way the output stream wants.

    Args:
        iterable: what to iterate. Consumed lazily; a generator stays a generator.
        desc: the label, e.g. `"align"` or `"token stats"`.
        total: how many items, when `len()` cannot say (a generator, a stream).
        every: fraction of the work between log lines in the non-TTY path. 0.05 is 20 lines.
        force_lines: never use a bar, even on a TTY. For passes whose output is always kept.

    Yields:
        the items of `iterable`, unchanged.
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
