#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Session-wide safety net for the VibeVoice suite.

Two problems, one cause: the engine finishes its work on **daemon threads**, so
a worker can outlive the test that started it.

1. It can reach the real keyboard. `autosend()` pastes via the clipboard and
   `type_text()` posts synthetic keystrokes into whatever app is frontmost —
   which is exactly what happened on 2026-08-02, when the streaming-paste tests
   typed "il polpo ha" into the user's screen and overwrote their clipboard.
2. It can land in the NEXT test's recorder. A leaked worker running the real
   Whisper on a buffer of zeros (which hallucinates "Grazie a tutti.") called
   `engine.autosend` while a later test in another file had monkeypatched it,
   and that test failed with a paste it never made.

Both disappear if the real outbound functions are simply never installed during
a test session. A test that wants to observe a paste still monkeypatches them
per-test: function-scoped `monkeypatch` layers on top of these no-ops and
reverts to the no-ops, never to the real thing.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import engine


# Every module-level path the engine can write. Per-test fixtures redirect these
# to their own tmp_path; this list is the floor underneath them.
_WRITABLE_STATE = (
    "STATE_FILE", "LEVELS_FILE", "LEVELS_TMP", "RAW_FILE", "HISTORY_FILE",
    "METRICS_FILE", "CORRECTIONS_FILE", "DICT_FILE", "MUTED_FILE", "CORPUS_FILE",
    "PARTIAL_FILE", "PARTIAL_TMP", "AUTOSEND_PAUSE_FLAG", "PASTE_FILE",
    "FIXES_FILE", "LEARNED_FIXES_FILE",
)


@pytest.fixture(autouse=True, scope="session")
def _no_outbound_effects_during_tests(tmp_path_factory):
    """Cut every path out of the process, for the whole run.

    Two kinds of escape, one mechanism. The functions that reach the keyboard
    and clipboard become no-ops, and every writable state path is moved into a
    session-scoped temp dir.

    The second half is not redundant with the per-test fixtures. Those use
    function-scoped `monkeypatch`, which is undone at teardown — so a worker
    that outlives its test finds the module attribute restored to the user's
    real `~/.vibevoice/` and writes there. Measured 2026-08-02: 13 phantom rows
    in the live metrics.jsonl, each 0.1 s of audio transcribed as "Grazie a
    tutti.", which then skewed the KPI report enough to make a decode-latency
    problem look three times worse than it is. Telemetry you cannot trust is
    worse than none — it sends you optimising the wrong thing.
    """
    home = tmp_path_factory.mktemp("vibevoice_state")
    patch = pytest.MonkeyPatch()
    # Same escape hatch as autosend.py's below: a test asserting something
    # ABOUT the real paste (which keys it presses) calls this, and is safe only
    # with `_press_key_cg` and `subprocess.run` patched.
    engine._REAL_autosend = engine.autosend
    patch.setattr(engine, "autosend", lambda text: None)
    patch.setattr(engine, "type_text", lambda text: True)
    # run() must not load the real model: overlapping warm-ups crashed MLX on
    # CI. The warm-up itself is tested by calling Engine._warm_up directly.
    patch.setattr(engine, "WARM_UP", False)
    # autosend.py presses Return through osascript. It is a third process with
    # its own path to the keyboard, so it needs the same floor as the engine.
    try:
        import autosend as _autosend
        # Keep a handle on the real one. The no-op below is the default for the
        # whole session — nothing reaches the keyboard by accident — but a test
        # that wants to assert something ABOUT the real function (that it bounds
        # its osascript, say) would otherwise be asserting against the stub and
        # passing for free. Using it is deliberate, and safe only with
        # `subprocess.run` patched.
        _autosend._REAL_simulate_return = _autosend.simulate_return
        patch.setattr(_autosend, "simulate_return",
                      lambda *a, **k: type("R", (), {"returncode": 0, "stderr": ""})())
        patch.setattr(_autosend, "afplay_sound", lambda *a, **k: None)
        patch.setattr(_autosend, "PAUSE_FLAG", home / "autosend_pause_flag")
    except Exception:
        pass   # pynput absent: nothing of autosend can run anyway
    for attr in _WRITABLE_STATE:
        if hasattr(engine, attr):
            patch.setattr(engine, attr, home / attr.lower())
    # The pill's lifecycle notes (chip.log) resolve STATE_DIR at call time: a
    # lifecycle test that calls applicationWillTerminate_ must not write
    # «app quit» into the user's real ledger.
    try:
        import vibevoice as _pill
        patch.setattr(_pill, "STATE_DIR", home)
    except Exception:
        pass   # no PyObjC: the pill cannot be imported, nothing to redirect
    yield
    patch.undo()


def test_no_writable_state_points_at_the_real_home():
    """The floor itself. If this fails, the suite can corrupt the user's data."""
    real = Path.home() / ".vibevoice"
    for attr in _WRITABLE_STATE:
        value = getattr(engine, attr, None)
        if value is not None:
            assert real not in Path(value).parents, f"{attr} still points at {value}"


# How long to wait for a test's own threads before giving up and moving on.
# Generous: a real transcription on this machine is ~200 ms.
_THREAD_SETTLE_TIMEOUT = 8.0


@pytest.fixture(autouse=True)
def _let_each_test_take_its_threads_with_it():
    """No test may finish while the threads it started are still running.

    The session no-ops above are not enough on their own: while a test has
    monkeypatched `engine.autosend` with its own recorder, a worker leaked by an
    EARLIER test calls that recorder, not the no-op. That is how a transcription
    of silence ("Grazie a tutti.", Whisper's hallucination on a zero buffer)
    from `test_contract.py` turned up inside a paste-ordering assertion in
    `test_dictation_quality.py` — a failure with no relationship to the code
    under test, and reproducible only in a full-suite run.

    So the leak is closed at the source. Not asserted, only waited for: a hung
    worker should slow the suite down, not fail an unrelated test.
    """
    before = set(threading.enumerate())
    yield
    deadline = time.monotonic() + _THREAD_SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        if not [t for t in threading.enumerate() if t not in before and t.is_alive()]:
            return
        time.sleep(0.02)
