#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Formatting of one history entry (never touches live ~/.vibevoice).

`history.jsonl` is written by the engine on the transcription path — the one place
that must never raise and never block — so a torn or malformed line is a real
possibility, not a hypothetical. The settings window renders that file; one bad
line must not blank the whole list.

The timestamp has always been in the file and was always discarded, leaving the
list unable to answer "when did I say that?".
"""
from __future__ import annotations

from datetime import datetime

from vibevoice import format_history_line


def test_shows_the_time_and_the_text():
    ts = datetime(2026, 7, 26, 9, 41, 30).timestamp()
    assert format_history_line({"ts": ts, "text": "ciao come stai"}) == "09:41  ciao come stai"


def test_text_is_trimmed():
    ts = datetime(2026, 7, 26, 9, 41).timestamp()
    assert format_history_line({"ts": ts, "text": "  spazi  "}) == "09:41  spazi"


def test_missing_timestamp_degrades_to_the_text():
    """Losing the time is a nuisance; losing the transcription is data loss."""
    assert format_history_line({"text": "senza orario"}) == "senza orario"


def test_unusable_timestamp_degrades_to_the_text():
    for bad in (None, "boh", float("nan"), [], 10**20):
        assert format_history_line({"ts": bad, "text": "testo"}) == "testo", bad


def test_entries_without_usable_text_are_dropped():
    for record in ({"ts": 0}, {"text": ""}, {"text": "   "}, {"text": 42}, {}):
        assert format_history_line(record) is None, record


def test_a_non_dict_line_is_dropped_not_raised():
    """A torn JSONL line can decode to anything at all."""
    for junk in (None, "riga", 7, [], ["ts", "text"]):
        assert format_history_line(junk) is None, junk


def test_last_dictation_is_the_newest_usable_line(tmp_path):
    """«Copia l'ultima dettatura» in the menu: newest text wins, a torn last
    line falls back to the one before, a missing file gives nothing."""
    from vibevoice import last_dictation_text

    path = tmp_path / "history.jsonl"
    assert last_dictation_text(path) is None
    path.write_text('{"ts": 1, "text": "prima"}\n{"ts": 2, "text": "seconda"}\n{"ts": 3, "te')
    assert last_dictation_text(path) == "seconda"
