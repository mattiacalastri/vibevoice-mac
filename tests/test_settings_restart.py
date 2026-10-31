#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Does saving from the settings window restart the engine only when it must?
(never touches live ~/.vibevoice)

`config.engine_restart_needed` has its own unit tests, but they would stay green
even if the call site went back to restarting unconditionally — and the call site
is where the defect lived: ticking "Dock icon" killed the transcription in flight.

The callback itself (`settingsChanged_`) cannot be driven from a test: PyObjC
requires a real Objective-C `self` for anything exposed as a selector. That is why
the decision was moved out into `vibevoice.apply_settings`, which is what these
tests exercise. What remains inside the selector is two statements of wiring —
read the controls, obey the answer.
"""
from __future__ import annotations

import config
import vibevoice


class _NSAppStub:
    """Records the activation policy instead of talking to a real NSApplication."""

    def __init__(self):
        self.policies = []

    def setActivationPolicy_(self, policy):
        self.policies.append(policy)


def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(config, "CONFIG_TMP", tmp_path / "config.json.tmp")
    stub = _NSAppStub()
    monkeypatch.setattr(vibevoice, "NSApp", stub)
    return stub


def _from_window(**controls):
    """Exactly what the window's four controls would have produced."""
    return vibevoice.apply_settings(controls)


def test_dock_alone_does_not_restart_the_engine(tmp_path, monkeypatch):
    """THE regression: restarting for a preference the engine never receives
    interrupts whatever is being transcribed at that moment."""
    _isolate(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS, dock=True))

    restart = _from_window(lang="it", autosend=True, autosend_return=config.DEFAULTS["autosend_return"], dock=False)

    assert restart is False, "the engine would be restarted for a pill-only preference"
    assert config.load()["dock"] is False, "the change must still be persisted"


def test_language_change_does_restart_the_engine(tmp_path, monkeypatch):
    _isolate(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS, lang="it"))

    restart = _from_window(lang="en", autosend=True, autosend_return=config.DEFAULTS["autosend_return"], dock=True)

    assert restart is True
    assert config.load()["lang"] == "en"


def test_no_change_at_all_does_not_restart_the_engine(tmp_path, monkeypatch):
    """Re-saving identical settings — clicking a checkbox twice — must be inert."""
    _isolate(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS))

    restart = _from_window(lang="it", autosend=True, autosend_return=config.DEFAULTS["autosend_return"], dock=True)

    assert restart is False


def test_dock_policy_is_applied_even_without_a_restart(tmp_path, monkeypatch):
    """Skipping the restart must not skip the visible effect of the setting."""
    stub = _isolate(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS, dock=True))

    restart = _from_window(lang="it", autosend=True, autosend_return=config.DEFAULTS["autosend_return"], dock=False)

    assert stub.policies == [vibevoice.NSApplicationActivationPolicyAccessory]
    assert restart is False


def test_vp_is_preserved_by_a_save_from_the_window(tmp_path, monkeypatch):
    """The window has no control for `vp` yet: saving from it must not reset the key,
    and carrying it over must not look like a change."""
    _isolate(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS, vp=False))

    restart = _from_window(lang="it", autosend=True, autosend_return=config.DEFAULTS["autosend_return"], dock=False)

    assert config.load()["vp"] is False, "vp was clobbered by a save from the window"
    assert restart is False, "carrying vp over must not be mistaken for a change"


def test_a_vp_change_would_restart_the_engine(tmp_path, monkeypatch):
    """Forward guard: once the window grows a vp control, it must force a restart —
    the engine only reads VIBEVOICE_VP at spawn time."""
    _isolate(tmp_path, monkeypatch)
    config.save(dict(config.DEFAULTS, vp=True))

    restart = _from_window(lang="it", autosend=True, autosend_return=True,
                           dock=True, vp=False)

    assert restart is True
    assert config.load()["vp"] is False
