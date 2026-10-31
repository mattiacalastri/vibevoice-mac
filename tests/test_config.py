#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Contract tests for config.json — pill-owned settings (never touch live ~/.vibevoice)."""
from __future__ import annotations

import json

import config


def _redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "CONFIG_TMP", tmp_path / "config.json.tmp")


def test_load_missing_file_returns_defaults(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    assert config.load() == config.DEFAULTS


def test_save_then_load_roundtrip(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    cfg = dict(config.DEFAULTS, lang="en", dock=False)
    config.save(cfg)
    assert config.load() == cfg
    assert not (tmp_path / "config.json.tmp").exists()  # atomic: no staging left


def test_corrupt_file_returns_defaults(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text("{not json")
    assert config.load() == config.DEFAULTS


def test_existing_config_without_vp_gets_the_default(tmp_path, monkeypatch):
    """Adding a key must not require a migration: files written before it exist keep working."""
    _redirect(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps(
        {"lang": "en", "autosend": False, "autosend_return": False, "dock": False}))
    cfg = config.load()
    assert cfg["vp"] is True, "a pre-existing config must inherit the vp default"
    assert cfg["lang"] == "en", "the other values must survive untouched"


def test_vp_survives_a_roundtrip(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS, vp=False))
    assert config.load()["vp"] is False


def test_save_fills_omitted_keys_instead_of_raising(tmp_path, monkeypatch):
    """Regression: save() used to demand every key, so adding one armed a KeyError
    in any caller that built a partial dict (that is exactly what happened to
    settingsChanged_ when `vp` was introduced)."""
    _redirect(tmp_path, monkeypatch)
    config.save({"lang": "en"})                      # deliberately partial
    cfg = config.load()
    assert cfg["lang"] == "en"
    assert cfg["vp"] == config.DEFAULTS["vp"]
    assert cfg["dock"] == config.DEFAULTS["dock"]


def test_dock_alone_does_not_restart_the_engine():
    """THE defect: ticking "Dock icon" restarted the engine and killed the
    transcription in flight, for a preference the engine never receives."""
    old = dict(config.DEFAULTS, dock=True)
    new = dict(config.DEFAULTS, dock=False)
    assert config.engine_restart_needed(old, new) is False


def test_nothing_changed_does_not_restart_the_engine():
    assert config.engine_restart_needed(dict(config.DEFAULTS), dict(config.DEFAULTS)) is False


def test_every_engine_key_restarts_the_engine():
    flipped = {"lang": "en", "autosend": not config.DEFAULTS["autosend"],
               "autosend_return": not config.DEFAULTS["autosend_return"],
               "vp": not config.DEFAULTS["vp"], "ptt": not config.DEFAULTS["ptt"]}
    for key, value in flipped.items():
        old = dict(config.DEFAULTS)
        new = dict(config.DEFAULTS, **{key: value})
        assert config.engine_restart_needed(old, new) is True, f"{key} must force a restart"


def test_engine_keys_never_contain_pill_only_preferences():
    """A guard for the next key: anything the engine does not receive must stay out."""
    assert "dock" not in config.ENGINE_KEYS
    assert config.ENGINE_KEYS <= set(config.DEFAULTS), "ENGINE_KEYS must be a subset of the schema"


def test_restart_decision_tolerates_missing_keys():
    """Comparing a pre-existing config (no `vp`) against a fresh one must not raise."""
    old = {"lang": "it"}                       # legacy file, most keys absent
    new = dict(config.DEFAULTS)
    assert config.engine_restart_needed(old, new) is False


def test_unknown_keys_dropped_known_kept(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    (tmp_path / "config.json").write_text(json.dumps({"lang": "en", "evil": 1}))
    cfg = config.load()
    assert cfg["lang"] == "en"
    assert "evil" not in cfg


# ── settings i18n ─────────────────────────────────────────────────────────────

def test_every_offered_language_can_actually_dress_the_window():
    """A language in the picker with no strings behind it is a window with holes.

    The picker and the translation table are two lists that must agree, and
    nothing but this test makes them: adding a flag is one line, and the six
    dictionaries it needs to grow are somewhere else in the file.
    """
    import vibevoice

    reference = set(vibevoice._S_TEXT["en"])
    for code, flag, name in vibevoice._S_LANGS:
        assert code in vibevoice._S_TEXT, f"{name} is offered but has no strings"
        assert set(vibevoice._S_TEXT[code]) == reference, (
            f"{name} is missing {reference - set(vibevoice._S_TEXT[code])}"
        )
        assert flag and name, f"{code} has no flag or no name"


def test_an_unknown_language_still_gets_a_readable_window():
    """A hand-edited config can name a language the picker never offered. English
    labels beat missing ones — the window must never come up blank."""
    import vibevoice

    assert vibevoice.settings_text("zz") == vibevoice._S_TEXT["en"]


def test_toggle_send_return_flips_the_setting_the_engine_reads(tmp_path, monkeypatch):
    """The menu's «Invio automatico» showed the legacy autosend.py loop (off)
    while the engine's own Return after paste (config `autosend_return`) was on
    and sending every dictation — the menu said the opposite of what happened
    (Mattia, 2026-10-31). The menu now drives `autosend_return`."""
    import config

    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    config.save({"autosend_return": True})
    assert config.toggle_send_return() is False
    assert config.load()["autosend_return"] is False
    assert config.toggle_send_return() is True
    assert config.load()["autosend_return"] is True
