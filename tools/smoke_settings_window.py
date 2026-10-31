#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Smoke test for the settings window: it actually BUILDS the window.

`pytest` deliberately never opens a GUI, so nothing there proves the AppKit calls
in `openSettings_` are valid — `NSColor.secondaryLabelColor()`,
`buttonWithTitle_target_action_`, `setAutoresizingMask_` and friends only fail at
runtime. Compiling is not working: this script constructs the window, counts its
subviews, drives the controls, and closes it without ever entering the run loop.

It earned its keep on the first run by catching a live defect the unit tests could
not see: the new "Voice processing" checkbox existed but `settingsChanged_` never
read it, so toggling it did nothing.

Not a pytest: creating a window needs a GUI session, and CI runs headless.

    python3 tools/smoke_settings_window.py

State is redirected to a throwaway HOME — it never touches the real ~/.vibevoice.
"""
import json
import os
import pathlib
import sys
import time

FAKE = pathlib.Path("/private/tmp/vv_smoke_home")
(FAKE / ".vibevoice").mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(FAKE)

sys.path.insert(0, "/Users/mattiacalastri/projects/💼-prodotti/vibevoice")

# A history with one good line, one malformed, one empty: the good lines
# must survive the others.
hist = FAKE / ".vibevoice/history.jsonl"
hist.write_text(
    json.dumps({"ts": time.time() - 120, "text": "first good line"}) + "\n"
    + "{ this line is torn\n"
    + json.dumps({"ts": time.time() - 60, "text": ""}) + "\n"
    + json.dumps({"ts": time.time(), "text": "second good line"}) + "\n"
)

import config  # noqa: E402
import vibevoice  # noqa: E402
from AppKit import NSApplication  # noqa: E402

config.STATE_DIR = FAKE / ".vibevoice"
config.CONFIG_FILE = FAKE / ".vibevoice/config.json"
config.CONFIG_TMP = FAKE / ".vibevoice/config.json.tmp"
vibevoice.STATE_DIR = FAKE / ".vibevoice"

NSApplication.sharedApplication()
ctrl = vibevoice.Controller.alloc().initWithDemo_place_(False, False)

fails = 0


def check(label, ok, extra=""):
    global fails
    print(f"  {'✓' if ok else '✗'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        fails += 1


ctrl.openSettings_(None)
win = ctrl._settings_win
check("the window builds without raising", win is not None)

if win is not None:
    # The window speaks the language you dictate in, so the title is not a
    # constant: it is whatever that language calls it.
    T = vibevoice.settings_text(config.load()["lang"])
    check("title follows the dictation language",
          str(win.title()) == T["title"], str(win.title()))
    style = int(win.styleMask())
    check("resizable", bool(style & 8), f"styleMask={style}")
    frame = win.frame()
    check("height is sane", frame.size.height >= 300, f"h={frame.size.height:.0f}")
    # Counted through the tree, not across the top: rows live inside their cards
    # now, so a flat count of the content view sees 9 children and calls a fuller
    # window emptier than the one it replaced.
    def _deep_count(view):
        subs = view.subviews() or []
        return len(subs) + sum(_deep_count(s) for s in subs)

    n = _deep_count(win.contentView())
    check("views present (4 section labels + 3 cards + rows + Clear + history)",
          n >= 20, f"n={n}")
    cards = [s for s in win.contentView().subviews()
             if s.layer() is not None and float(s.layer().cornerRadius()) > 0
             and len(s.subviews() or []) >= 2]
    check("settings are grouped into cards", len(cards) >= 3, f"cards={len(cards)}")

    for name in ("_set_lang", "_set_vp", "_set_as", "_set_ar", "_set_dk", "_set_hist"):
        check(f"control {name}", getattr(ctrl, name, None) is not None)

    body = str(ctrl._set_hist.string())
    check("good lines survive the torn one",
          "first good line" in body and "second good line" in body)
    check("a torn line did not blank the history", body.strip() != T["empty"])
    check("timestamp shown (HH:MM)", ":" in body.split()[0], repr(body.split("\n")[0][:30]))
    check("history timer armed", getattr(ctrl, "_hist_timer", None) is not None)

    # clearing
    ctrl.clearHistory_(None)
    check("Clear empties the pane",
          str(ctrl._set_hist.string()).strip() == T["empty"], T["empty"])
    check("the file was truncated", hist.read_text() == "")

    # the vp checkbox must have an effect: turn it off and check it persists
    ctrl._set_vp.setState_(0)
    ctrl.settingsChanged_(None)
    check("turning Voice processing off persists", config.load()["vp"] is False,
          f"vp={config.load()['vp']}")
    ctrl._set_vp.setState_(1)
    ctrl.settingsChanged_(None)
    check("turning it back on persists", config.load()["vp"] is True)

    # "Dock icon" must NOT require an engine restart (the original defect)
    before = config.load()
    ctrl._set_dk.setState_(0)
    check("Dock icon needs no engine restart",
          vibevoice.config.engine_restart_needed(before, dict(before, dock=False)) is False)

    win.close()
    ctrl.historyTick_(None)
    check("the timer self-invalidates once closed",
          getattr(ctrl, "_hist_timer", None) is None)

print()
print("SMOKE PASSED" if fails == 0 else f"{fails} CHECKS FAILED")
sys.exit(1 if fails else 0)
