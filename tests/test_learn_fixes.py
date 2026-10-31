#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The mishearing learner: learns «crowd code» → Claude Code, never «online» →
Moonline, never rewrites a word you put in your own glossary."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import vibevoice_learn_fixes as lf  # noqa: E402

GLOSSARY = ["Claude Code", "Moonline", "Second Brain", "branch", "notch", "Notion"]
UNKNOWN = {"crowd", "clod", "moonline", "crowdcode"}


def unknown(w):
    return w in UNKNOWN


def run(texts, fixed=lambda f: False, never=()):
    return [(f, t, n) for f, t, n, _ in lf.candidates(texts, GLOSSARY, unknown, fixed, never)]


def test_a_mishearing_seen_twice_is_learned():
    texts = ["apri crowd code adesso", "lancia crowd code nel terminale"]
    assert ("crowd code", "Claude Code", 2) in run(texts)


def test_once_is_not_enough():
    assert run(["apri crowd code adesso"]) == []


def test_an_ordinary_word_is_never_rewritten():
    assert run(["vado online", "sono online"]) == []          # «online» is Italian enough


def test_a_glossary_word_is_never_a_mishearing():
    assert run(["il brain di casa", "un brain nuovo", "brain brain"]) == []
    assert run(["il notch con la chip", "nel notch con calma"]) == []


def test_already_fixed_and_refused_forms_are_skipped():
    texts = ["apri crowd code", "apri crowd code"]
    assert run(texts, fixed=lambda f: "crowd" in f) == []
    assert run(texts, never={"crowd code"}) == []


def test_the_rule_matches_the_form_with_any_spacing():
    rx = re.compile(lf.rule_for("crowd code"), re.IGNORECASE)
    assert rx.sub("Claude Code", "apri Crowd  code ora") == "apri Claude Code ora"
    assert rx.sub("X", "crowdcoder") == "crowdcoder"


def test_the_engine_applies_learned_rules_next_to_the_house_ones(tmp_path, monkeypatch):
    house = tmp_path / "fixes.json"
    learned = tmp_path / "learned_fixes.json"
    house.write_text(json.dumps([[r"\bcloud\b", "Claude"]]))
    learned.write_text(json.dumps({"correzioni": [[lf.rule_for("crowd code"), "Claude Code"]]}))
    monkeypatch.setattr(engine, "FIXES_FILE", house)
    monkeypatch.setattr(engine, "LEARNED_FIXES_FILE", learned)
    assert engine.apply_fixes("chiedi a cloud di aprire crowd code") == \
        "chiedi a Claude di aprire Claude Code"


def test_a_phrase_of_real_words_close_enough_is_learned():
    texts = ["apri cloud code", "su cloud code"]
    assert ("cloud code", "Claude Code", 2) in run(texts)
