#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Tests for tools/vibevoice_correct.py — the corrections loop v0.

The tool pairs the last dictation (history.jsonl) with the user-corrected text,
records the pair in corrections.jsonl, and grows dictionary.txt with the terms
the engine got wrong — so the same mistake is not repeated (dictionary biases
Whisper; the pairs feed the cleanup prompt as few-shot examples).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "vibevoice_correct.py"
spec = importlib.util.spec_from_file_location("vibevoice_correct", _TOOL)
correct_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correct_tool)


# ── extract_new_terms ─────────────────────────────────────────────────────────

def test_extract_new_terms_finds_corrected_words_only():
    raw = "ho sentito con con line e con fatom"
    corrected = "Ho sentito Moonline e con Fathom"
    terms = correct_tool.extract_new_terms(raw, corrected)
    assert "Moonline" in terms
    assert "Fathom" in terms
    assert "sentito" not in terms  # unchanged words are not new terms


def test_extract_new_terms_ignores_short_and_case_only_changes():
    terms = correct_tool.extract_new_terms("ciao a tutti ok", "Ciao a tutti ok!")
    assert terms == []  # case/punctuation changes are not vocabulary


def test_extract_new_terms_dedups():
    terms = correct_tool.extract_new_terms(
        "gol gol gol", "GoHighLevel GoHighLevel GoHighLevel"
    )
    assert terms == ["GoHighLevel"]


# ── record_correction ─────────────────────────────────────────────────────────

def test_record_correction_appends_pair_and_grows_dictionary(tmp_path):
    history = tmp_path / "history.jsonl"
    corrections = tmp_path / "corrections.jsonl"
    dictionary = tmp_path / "dictionary.txt"
    history.write_text(json.dumps({"ts": 1.0, "text": "ciao mun line"}) + "\n")

    result = correct_tool.record_correction(
        "ciao Moonline",
        history_file=history,
        corrections_file=corrections,
        dict_file=dictionary,
    )

    pair = json.loads(corrections.read_text().splitlines()[-1])
    assert pair["raw"] == "ciao mun line"
    assert pair["corrected"] == "ciao Moonline"
    assert "Moonline" in dictionary.read_text().splitlines()
    assert "Moonline" in result["new_terms"]


def test_record_correction_does_not_duplicate_dictionary_terms(tmp_path):
    history = tmp_path / "history.jsonl"
    dictionary = tmp_path / "dictionary.txt"
    history.write_text(json.dumps({"ts": 1.0, "text": "ciao mun line"}) + "\n")
    dictionary.write_text("Moonline\n")

    correct_tool.record_correction(
        "ciao Moonline",
        history_file=history,
        corrections_file=tmp_path / "corrections.jsonl",
        dict_file=dictionary,
    )

    assert dictionary.read_text().splitlines().count("Moonline") == 1


def test_record_correction_without_history_raises(tmp_path):
    import pytest

    with pytest.raises(SystemExit):
        correct_tool.record_correction(
            "testo",
            history_file=tmp_path / "missing.jsonl",
            corrections_file=tmp_path / "corrections.jsonl",
            dict_file=tmp_path / "dictionary.txt",
        )
