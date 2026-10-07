"""Progress reporting (dawn/progress.py).

The rule worth pinning: a tqdm bar is for a terminal, lines are for a log file. An overnight run
is redirected, and a bar there fills the file with carriage returns.
"""

import logging
import sys

import pytest

from dawn.progress import eta_line, format_duration, progress, use_bar


@pytest.mark.parametrize(
    "seconds, text",
    [(0, "0s"), (7.4, "7s"), (59.6, "60s"), (60, "1m00s"), (102, "1m42s"), (3600, "1h00m"),
     (11100, "3h05m")],
)
def test_durations_stay_short_enough_for_a_log_line(seconds, text):
    assert format_duration(seconds) == text


def test_eta_line_reports_share_rate_and_remaining():
    line = eta_line("align", 4000, 8000, 200.0)
    assert "align: 4000/8000" in line and "(50%)" in line
    assert "20.0/s" in line and "ETA 3m20s" in line


def test_eta_line_omits_what_it_cannot_know():
    """No total means no percentage and no ETA, rather than a guess."""
    line = eta_line("stats", 300, None, 10.0)
    assert "300" in line and "30.0/s" in line
    assert "%" not in line and "ETA" not in line


def test_eta_line_drops_the_eta_when_the_work_is_done():
    assert "ETA" not in eta_line("align", 8000, 8000, 100.0)


def test_no_bar_when_stderr_is_not_a_tty(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    assert not use_bar()


def test_no_bar_when_the_caller_forces_lines(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    assert use_bar() == (pytest.importorskip("tqdm") is not None)
    assert not use_bar(force_lines=True)


def test_no_bar_when_tqdm_is_missing(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
    monkeypatch.setitem(sys.modules, "tqdm", None)
    assert not use_bar()


def test_every_item_is_yielded_unchanged(monkeypatch):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    assert list(progress(range(10), "x")) == list(range(10))


def test_a_generator_is_not_materialized(monkeypatch):
    """Wrapping a stream must not pull it all into memory to count it first."""
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    pulled = []

    def source():
        for i in range(5):
            pulled.append(i)
            yield i

    wrapped = progress(source(), "x")
    assert pulled == [], "nothing consumed before the first next()"
    assert next(wrapped) == 0
    assert pulled == [0], "exactly one item consumed"


def test_lines_are_logged_at_the_requested_interval(monkeypatch, caplog):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    with caplog.at_level(logging.INFO, logger="dawn.progress"):
        list(progress(range(100), "job", every=0.25))
    lines = [r.message for r in caplog.records]
    assert sum("job:" in line and "elapsed" in line for line in lines) == 4, "25/50/75/100"
    assert lines[-1].startswith("job: 100 done in")


def test_an_empty_iterable_logs_nothing(monkeypatch, caplog):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    with caplog.at_level(logging.INFO, logger="dawn.progress"):
        assert list(progress([], "job")) == []
    assert caplog.records == []


def test_an_explicit_total_overrides_a_missing_len(monkeypatch, caplog):
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False, raising=False)
    with caplog.at_level(logging.INFO, logger="dawn.progress"):
        list(progress((i for i in range(40)), "job", total=40, every=0.5))
    assert any("(50%)" in r.message for r in caplog.records)


def test_run_parallel_logs_a_rate_and_an_eta(caplog):
    """The data-prep passes keep line logging whatever the terminal is: they are the record."""
    pytest.importorskip("torch")
    from dawn.data.sheets import run_parallel

    with caplog.at_level(logging.INFO):
        assert run_parallel(abs, [(-i,) for i in range(40)], workers=1, desc="job") == list(
            range(40)
        )
    messages = [r.message for r in caplog.records]
    assert any("/s" in m and "job:" in m for m in messages)
    assert messages[-1].startswith("job: 40 done in")


def test_run_parallel_stays_silent_without_a_desc(caplog):
    pytest.importorskip("torch")
    from dawn.data.sheets import run_parallel

    with caplog.at_level(logging.INFO):
        run_parallel(abs, [(-1,)], workers=1)
    assert caplog.records == []
