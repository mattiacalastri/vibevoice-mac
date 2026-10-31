#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pill-owned persistent settings (~/.vibevoice/config.json).

Leaf module: imported ONLY by vibevoice.py. The engine gets these values via
environment variables at spawn time (the pill exports them); autosend.py keeps
its own `autosend` state file. Writer and only reader: the pill.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

STATE_DIR = Path(os.path.expanduser("~")) / ".vibevoice"
CONFIG_FILE = STATE_DIR / "config.json"
CONFIG_TMP = STATE_DIR / "config.json.tmp"

# Adding a key here is backward compatible by construction: `load()` fills any key
# missing from an existing config.json with the default below, so no migration is
# needed. `vp` = macOS voice-processing capture (echo cancellation + noise
# suppression); it reaches the engine as VIBEVOICE_VP at spawn time. Default on,
# and a voice-processing failure already falls back to sounddevice (AGENTS.md rule 8).
DEFAULTS = {
    "lang": "it",
    "autosend": True,
    # Off for a fresh install: Return after every dictation sends half-thoughts
    # into chats. One click on «Invio automatico» in the menu turns it on.
    "autosend_return": False,
    "dock": True,
    "vp": True,
    # Push-to-talk: hold Control to dictate, let go to send (the chip in
    # the notch shows it). Off = the always-listening VAD engine of before.
    "ptt": True,
}


# Keys whose value the engine reads (it receives them as environment variables at
# spawn time, see vibevoice._start_engine). Changing one of these means the running
# engine is stale and must be restarted; changing anything else must NOT disturb it.
# "dock" is deliberately absent: it is a pill-only preference.
ENGINE_KEYS = frozenset({"lang", "autosend", "autosend_return", "vp", "ptt"})


def engine_restart_needed(old: dict, new: dict) -> bool:
    """True if a value the engine actually reads changed between `old` and `new`.

    The settings window used to restart the engine on *every* change, so ticking
    "Dock icon" — a preference the engine never even receives — killed whatever
    was being transcribed at that moment.
    """
    return any(
        old.get(k, DEFAULTS[k]) != new.get(k, DEFAULTS[k])
        for k in ENGINE_KEYS
    )


def load() -> dict:
    try:
        raw = json.loads(CONFIG_FILE.read_text())
    except (OSError, ValueError):
        return dict(DEFAULTS)
    return {k: raw.get(k, v) for k, v in DEFAULTS.items()}


def save(cfg: dict) -> None:
    """Persist `cfg` atomically, filling anything it omits from DEFAULTS.

    Symmetric with `load()` on purpose. The previous form — `cfg[k] for k in
    DEFAULTS` — demanded that every caller be exhaustive, so adding a key here
    silently armed a KeyError in every caller that built a partial dict. That is
    a trap for whoever adds the next key, not a useful strictness: the defaults
    are right here.
    """
    CONFIG_TMP.parent.mkdir(parents=True, exist_ok=True)
    merged = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    CONFIG_TMP.write_text(json.dumps(merged, indent=2))
    os.replace(CONFIG_TMP, CONFIG_FILE)


def toggle_send_return() -> bool:
    """Flip `autosend_return` (the engine presses Return after each paste) and
    return the new value. The menu's «Invio automatico» drives this one: it is
    what actually sends the message. The caller restarts the engine, which
    reads the setting from its environment at spawn."""
    cfg = load()
    cfg["autosend_return"] = not cfg["autosend_return"]
    save(cfg)
    return cfg["autosend_return"]
