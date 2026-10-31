#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# VibeVoice — MIT
#
# Copyright (c) 2026 VibeVoice contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
vibevoice.py — VibeVoice "Dynamic Island" STT pill + robot command center

A minimal floating UI for a speech-to-text engine. A borderless, floating,
non-activating NSPanel docked under the notch. Decoupled from the engine: it
READS the state files written by the engine and draws. It never touches the
audio pipeline directly.

  idle          → invisible (alpha 0)
  recording     → fade-in, live waveform
  transcribing  → keeps drawing the transcript that streams in
  silence       → fade-out after ~1.5s (or ~2.5s while text is shown)

A floating ROBOT widget doubles as the command center: click it for the menu,
drag it to reposition (persisted). It exists because on recent macOS a status
item created before launch-finish can be parked off-screen behind the notch —
the robot is our own deterministic, always-visible control. The menu-bar status
item is still created (and self-heals if parked); both share the same menu.

STATE-FILE CONTRACT (shared pill <-> engine, under ~/.vibevoice/):
  ~/.vibevoice/state       text file, one of: idle | recording | transcribing
  ~/.vibevoice/levels.bin  60 float32 little-endian (RMS 0..1), atomic write
  ~/.vibevoice/raw.txt     last transcription, plain text (just the sentence)
  ~/.vibevoice/partial.txt live draft while you are still speaking (absent = none)
  ~/.vibevoice/history.jsonl  last 20 transcriptions, JSONL {"ts","text"}, newest last

The engine WRITES these files; the pill READS them.

CONTROL FILES (this pill WRITES them; the engine / this pill honor them — the
same external-control pattern as autosend's pause flag, NOT engine-owned state):
  ~/.vibevoice/muted       presence = mic paused (engine ignores audio, stays alive)
  ~/.vibevoice/locked      presence = pill stays visible (no auto-hide)
  ~/.vibevoice/return_held presence = the engine pastes but never presses Return
                         (the widget's play/stop key; the engine reads it live)
                         (`muted` is also the widget's rightmost pause/play key)
~/.vibevoice/autosend    "on" | "off" — armed state of the auto-Return daemon
                           (autosend.py owns this file; the pill toggles it and
                           spawns the daemon when it isn't already running)

OPTIONAL TTS-REACTIVITY (self-contained hook; ANY external text-to-speech may
write these, the pill only READS them — no engine/autosend change needed):
  ~/.vibevoice/tts         presence = the feature is enabled
  ~/.vibevoice/tts.txt     line 1: "<start_epoch> <duration_s>"; line 2+: spoken text
  ~/.vibevoice/tts_levels.bin  60 float32 LE (RMS 0..1) of the TTS audio
When tts.txt is present and fresh, the pill turns RED and types out the spoken
sentence in sync with the audio (a mirror of the green dictation waveform).

  ~/.vibevoice/robot_pos   "x,y" — saved position of the floating robot (drag)
  ~/.vibevoice/widget      presence = hardware-look voice widget shown (menu toggle)
  ~/.vibevoice/widget_pos  "x,y" — saved position of the hardware widget (drag)

Menu / robot acts as the master switch:
  - toggle the engine (launches/kills engine.py via subprocess)
  - Mute mic (pause without killing the engine)
  - Lock pill (pin it so it never auto-hides)
  - Auto-send loop (arm autosend.py: every dictation → automatic Return)
  - Restart / Quit

Run:
  python3 vibevoice.py            # live (reads the engine state files)
  python3 vibevoice.py --demo     # animated demo (to preview the design)
  python3 vibevoice.py --place    # placement mode (stays visible)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import config
import objc
from notch_chip import NotchChip

from AppKit import (
    NSApplication, NSApp, NSPanel, NSView, NSColor, NSBezierPath, NSAnimationContext,
    NSScreen, NSTimer, NSFont, NSForegroundColorAttributeName,
    NSFontAttributeName, NSMakeRect, NSMakePoint, NSImage, NSCursor, NSEvent,
    NSTrackingArea, NSMenu,
    NSTrackingMouseEnteredAndExited, NSTrackingMouseMoved,
    NSTrackingActiveAlways, NSTrackingCursorUpdate,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
    NSWindowStyleMaskTitled, NSWindowStyleMaskClosable, NSWindowStyleMaskResizable,
    NSViewWidthSizable, NSViewHeightSizable,
    NSBackingStoreBuffered, NSStatusWindowLevel,
    NSApplicationActivationPolicyAccessory, NSApplicationActivationPolicyRegular,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorStationary,
    NSString, NSPasteboard, NSPasteboardTypeString,
    NSStatusBar, NSMenuItem, NSVariableStatusItemLength,
    NSWindow, NSTextField, NSButton, NSPopUpButton, NSTextView, NSScrollView,
    NSVisualEffectView, NSWindowStyleMaskFullSizeContentView,
    NSVisualEffectBlendingModeBehindWindow, NSVisualEffectMaterialMenu,
    NSVisualEffectStateFollowsWindowActiveState,
    NSAppearance, NSAppearanceNameDarkAqua,
    NSSwitch, NSKernAttributeName, NSViewMinXMargin, NSViewMinYMargin,
)
from Foundation import NSObject, NSMakeSize, NSBundle, NSAttributedString

# ── state-file contract (under $HOME/.vibevoice) ──────────────────────────────
STATE_DIR  = Path(os.path.expanduser("~/.vibevoice"))
STATE_FILE = STATE_DIR / "state"        # idle | recording | transcribing
LEVELS_BIN = STATE_DIR / "levels.bin"   # 60 float32 LE (RMS 0..1)
RAW_TXT    = STATE_DIR / "raw.txt"       # last transcription (plain text)
PARTIAL_TXT = STATE_DIR / "partial.txt"  # live draft while speaking (absent = nothing in flight)
# control files (the pill writes these; not engine-owned state):
RETURN_HELD_FILE = STATE_DIR / "return_held"  # presence = engine pastes but never presses Return (widget play/stop key)
MUTED_FILE  = STATE_DIR / "muted"        # presence = mic paused (engine reads, stays alive)
LOCKED_FILE = STATE_DIR / "locked"       # presence = pill stays visible (pill-only, no auto-hide)
AUTOSEND_FILE = STATE_DIR / "autosend"   # "on" | "off" — autosend.py owns this; the pill toggles it
# optional TTS-reactivity hook (any external TTS writes these; the pill reads):
TTS_FLAG   = STATE_DIR / "tts"            # presence = feature enabled
TTS_TEXT   = STATE_DIR / "tts.txt"        # line1 "<start> <dur>", line2+ spoken text
TTS_LEVELS = STATE_DIR / "tts_levels.bin" # 60 float32 LE (RMS of the TTS audio)
ROBOT_POS  = STATE_DIR / "robot_pos"      # "x,y" saved position of the floating robot
WIDGET_FLAG = STATE_DIR / "widget"        # presence = hardware-look widget shown (pill writes via menu)
WIDGET_POS  = STATE_DIR / "widget_pos"    # "x,y" saved position of the hardware widget (drag)

SOUNDS_DIR = Path(os.path.abspath(__file__)).parent / "assets" / "sounds"

# engine.py / autosend.py live next to this file
ENGINE_PATH   = Path(os.path.abspath(__file__)).parent / "engine.py"
AUTOSEND_PATH = Path(os.path.abspath(__file__)).parent / "autosend.py"

# Menu-bar-only mode: the floating pill + robot widget are the live
# waveform/TTS visualiser, but the user wants a single surface — the menu-bar item.
# False → the pill panel is never shown and the robot widget is never built; the
# NSStatusBar item stays the sole UI. Flip to True to bring the floating pill back.
SHOW_PILL = False




def clamp_to_visible(x, y, w, h, screens):
    """Pull a saved window origin back onto a screen that actually exists.

    A position is restored verbatim, so dragging the widget onto an external
    monitor and unplugging it leaves the panel born off-screen — and toggling it
    off and on only re-orders the existing panel, never repositions it. With
    SHOW_PILL False that is the whole floating surface, and the only cure was
    deleting `widget_pos` by hand.

    `screens` is a list of (origin_x, origin_y, width, height). An origin that
    sits on any of them is left exactly where the user put it: a second monitor
    is a legitimate home. With no screens known, change nothing — never invent a
    position from ignorance.
    """
    try:
        if not screens:
            return (x, y)
        for sx, sy, sw, sh in screens:
            if sx <= x <= sx + sw - min(w, sw) and sy <= y <= sy + sh - min(h, sh):
                return (x, y)
        sx, sy, sw, sh = screens[0]
        return (min(max(x, sx), sx + max(sw - w, 0)),
                min(max(y, sy), sy + max(sh - h, 0)))
    except Exception:
        return (x, y)


def first_run_defaults(state_dir) -> list:
    """Turn on what a brand-new install must show, exactly once.

    `SHOW_PILL` is False, so the notch pill never appears: the visible surface
    is the floating hardware widget, and its flag lives in the state dir. A
    packaged app installed fresh therefore opened with nothing on screen but a
    menu-bar icon and looked dead — which is what the first bundle did on
    2026-08-02, while it was in fact transcribing.

    The marker makes it a DEFAULT, not a policy: turn the widget off and it
    stays off, here and across upgrades. Returns the paths created, so the
    caller can log them; never raises.
    """
    from pathlib import Path
    created = []
    try:
        state_dir = Path(state_dir)
        marker = state_dir / "configured"
        if marker.exists():
            return created
        state_dir.mkdir(parents=True, exist_ok=True)
        widget = state_dir / "widget"
        if not widget.exists():
            widget.touch()
            created.append(widget)
        marker.touch()
    except Exception:
        pass
    return created


def _flag_on(path) -> bool:
    """True if a control flag file exists (defensive: never raises)."""
    try:
        return path.exists()
    except Exception:
        return False


# ── hardware widget: pure logic (locked by tests/test_widget_logic.py) ────────
WIDGET_BARS = 24        # VU columns on the widget's recessed display


def widget_bar_levels(levels, n=WIDGET_BARS) -> list:
    """Downsample the RMS history to n VU bars, peak per bucket, clamped 0..1.

    Peak (not mean): a single spike must survive downsampling or the meter
    reads dead during short plosives — VU behavior, not smoothing.
    """
    if not levels:
        return [0.0] * n
    total = len(levels)
    bars = []
    for i in range(n):
        a = int(i * total / n)
        b = max(a + 1, int((i + 1) * total / n))
        bars.append(max(0.0, min(1.0, max(levels[a:b]))))
    return bars


def widget_bar_color(lvl: float, engine_on: bool) -> tuple:
    """(r, g, b, a) of one VU bar. Red means CLIPPING, not a loud voice:
    Apple's AGC pushes normal speech well past 0.85, so the hot threshold sits
    at the very top (feedback: too much red)."""
    if not engine_on:
        return (0.20, 0.30, 0.24, 0.8)
    if lvl > 0.96:
        return (1.0, 0.25, 0.18, 0.95)
    return (0.10 + 0.5 * lvl, 0.92, 0.30, 0.95)


def widget_led(state: str, muted: bool, engine_on: bool) -> tuple:
    """(r, g, b) tint of the widget's status LED.

    Priority mirrors what the hardware would do: power off → grey, muted →
    amber (a warning, whatever the engine thinks), REC → red, listening → green.
    """
    if not engine_on:
        return (0.42, 0.42, 0.42)
    if muted:
        return (1.0, 0.62, 0.08)
    if state == "recording":
        return (1.0, 0.16, 0.14)
    if state == "transcribing":
        return (1.0, 0.78, 0.12)
    return (0.12, 0.95, 0.35)


def _toggle_flag(path) -> bool:
    """Create the flag if absent, remove it if present. Returns the new state."""
    try:
        if path.exists():
            path.unlink()
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return True
    except Exception:
        return _flag_on(path)


# ── the widget's play/stop key (locked by tests/test_widget_logic.py) ─────────
SEND_KEY_R = 9.5        # cap radius of the round push key, right of the display


def widget_send_key_center(W, H) -> tuple:
    """Center of the play/stop key: right of the VU display, left of the mic key."""
    return (W - 50.0, H / 2.0)


def widget_mic_key_center(W, H) -> tuple:
    """Center of the mic pause/play key: the rightmost key, vertically centered."""
    return (W - 23.0, H / 2.0)


def widget_mic_key_hit(x, y, W, H) -> bool:
    """True if a click at (x, y) lands on the mic key (same bezel slack)."""
    cx, cy = widget_mic_key_center(W, H)
    return (x - cx) ** 2 + (y - cy) ** 2 <= (SEND_KEY_R + 2.5) ** 2


def widget_mic_key_look(muted: bool, pressing: bool) -> dict:
    """The mic key, same latching push button as the play/stop key.

    Listening = cap proud, pause glyph (press to pause). Paused = the cap
    latches IN with a play glyph (press to resume): depth tells the state first.
    """
    if pressing:
        depth = 1.0
    elif muted:
        depth = 0.6
    else:
        depth = 0.0
    return {"glyph": "play" if muted else "pause", "depth": depth}


def widget_send_key_hit(x, y, W, H) -> bool:
    """True if a click at (x, y) lands on the key (a little slack for the bezel)."""
    cx, cy = widget_send_key_center(W, H)
    return (x - cx) ** 2 + (y - cy) ** 2 <= (SEND_KEY_R + 2.5) ** 2


def widget_send_key_look(held: bool, pressing: bool) -> dict:
    """How the key sits in the metal, like a latching push button.

    Play (Return armed) = cap proud of the bezel. Stop (Return held) = the cap
    latches IN, so the state is readable from the depth alone, before the
    glyph. Under the finger it bottoms out. `depth` 0 = proud, 1 = bottomed.
    """
    if pressing:
        depth = 1.0
    elif held:
        depth = 0.6
    else:
        depth = 0.0
    return {"glyph": "stop" if held else "play", "depth": depth}


def _toggle_return_held() -> bool:
    """Flip the play/stop key. Returns True when Return is now held (stop).

    Stop means nothing sends by itself: the one-shot autosend daemon is
    disarmed too, so a Return cannot sneak in from the other path."""
    held = _toggle_flag(RETURN_HELD_FILE)
    if held:
        try:
            if AUTOSEND_FILE.read_text().strip() == "on":
                AUTOSEND_FILE.write_text("off")
        except Exception:
            pass
    try:
        subprocess.Popen(["afplay", "-v", "0.35", "/System/Library/Sounds/Pop.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    return held


def _toggle_mic_paused() -> bool:
    """Flip the mic key: pause/resume the dictation without killing the engine
    (the same `muted` master switch as the menu). Returns True when paused."""
    muted = _toggle_flag(MUTED_FILE)
    try:
        subprocess.Popen(["afplay", "-v", "0.35", "/System/Library/Sounds/Pop.aiff"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    return muted


def _autosend_armed() -> bool:
    """True if the auto-send loop is armed ('on' in the shared autosend file)."""
    try:
        return AUTOSEND_FILE.read_text().strip() == "on"
    except Exception:
        return False


# ── design ────────────────────────────────────────────────────────────────────
N_BARS       = 32
PILL_W       = 460.0
PILL_H       = 140.0    # taller: the waveform never grazes the notch
PILL_RADIUS  = 22.0
GAIN         = 12.0     # waveform sensitivity: higher = more reactive
VOICE_THRESH = 0.018    # raw RMS onset threshold (floor ~0.005 < thresh < speech ~0.05)

IDLE_HIDE_S = 1.5            # seconds of silence before fade-out
TICK        = 1.0 / 60.0     # 60 fps — see IDLE_SKIP for the idle-time saving
# 24 fps was the original cadence and it read as visibly stepped on the VU bars
# The bars are a continuous horizontal motion, the display runs at
# 120 Hz, and every frame therefore held for five refreshes. Judder on smooth
# translation is exactly what that produces. 60 fps halves the hold to two.
IDLE_SKIP   = 8              # while HIDDEN, run 1 tick in 8 (~7.5 fps).
# Kept as a DURATION, not a count: at 24 fps the old `% 3` meant ~125 ms of
# worst-case onset latency, and 8/60 = 133 ms preserves it. Raising the frame
# rate without raising this divisor would have quietly tripled idle CPU.

MATRIX = (0.12, 1.00, 0.32)  # Matrix-terminal flúor green (#1fff52) — dictation
AMBER  = (1.00, 0.66, 0.18)  # 🔒 locked / 🔁 auto-send armed
RED    = (0.95, 0.27, 0.27)  # 🔇 muted / TTS speaking

_CTRL  = None   # strong refs (avoid GC of controller/timer)
_TIMER = None


def _ensure_state_dir():
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # A fresh install must show its one visible surface, or it looks dead while
    # it is in fact transcribing.
    for path in first_run_defaults(STATE_DIR):
        sys.stderr.write(f"VibeVoice: first run — enabled {path.name}\n")


def _pill_path(w, h, r):
    """Path with a SQUARE top (flush with the notch) and a ROUNDED bottom — it
    looks like the black notch rectangle extending downward."""
    r = min(r, h / 2.0, w / 2.0)
    p = NSBezierPath.bezierPath()
    p.moveToPoint_(NSMakePoint(0.0, h))                 # top-left square
    p.lineToPoint_(NSMakePoint(w, h))                   # top-right square
    p.lineToPoint_(NSMakePoint(w, r))                   # down the right edge
    p.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
        NSMakePoint(w - r, r), r, 0.0, -90.0, True)     # bottom-right corner
    p.lineToPoint_(NSMakePoint(r, 0.0))                 # along the bottom
    p.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
        NSMakePoint(r, r), r, -90.0, 180.0, True)       # bottom-left corner
    p.closePath()
    return p


class PillView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(PillView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.levels = [0.04] * N_BARS
        self.text = ""
        self.active = False
        self.tint = MATRIX          # current colour: green (dictation) or red (TTS speaking)
        self.phase = 0.0
        self.copied_flash = 0.0
        self.hover_x = False
        self.hover_c = False
        self.copy_rect = None
        # control panel — re-read the flags from disk on launch, otherwise a pill
        # restart (e.g. launchd kickstart) would show mic-ON even if muted/locked
        # already exist, and the UI would lie about the real state.
        self.muted = _flag_on(MUTED_FILE)
        self.locked = _flag_on(LOCKED_FILE)
        self.muta_rect = None
        self.blocca_rect = None
        self.hover_m = False
        self.hover_l = False
        self.autoloop = _autosend_armed()
        self.loop_rect = None
        self.hover_o = False
        self._loop_chk = 0.0
        return self

    # NSTrackingArea: pointing-hand cursor over clickable icons, even on a
    # non-key panel. The per-frame .set() gets reset by the WindowServer;
    # cursorUpdate_ is the correct path.
    def updateTrackingAreas(self):
        for ta in list(self.trackingAreas()):
            self.removeTrackingArea_(ta)
        opts = (NSTrackingMouseEnteredAndExited | NSTrackingMouseMoved
                | NSTrackingActiveAlways | NSTrackingCursorUpdate)
        ta = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
            self.bounds(), opts, self, None)
        self.addTrackingArea_(ta)
        objc.super(PillView, self).updateTrackingAreas()

    def _over_icon(self, loc):
        """True if loc (window coords) is over a clickable icon (✕/copy/mute/lock/loop)."""
        b = self.bounds()
        w, h = b.size.width, b.size.height
        if loc.x >= w - 40.0 and loc.y >= h - 34.0:        # ✕
            return True
        for r in (self.copy_rect, self.muta_rect, self.blocca_rect, self.loop_rect):
            if r and r[0] <= loc.x <= r[0] + r[2] and r[1] <= loc.y <= r[1] + r[3]:
                return True
        return False

    def cursorUpdate_(self, event):
        if self._over_icon(event.locationInWindow()):
            NSCursor.pointingHandCursor().set()
        else:
            NSCursor.arrowCursor().set()

    # mouseMoved_ is the most reliable path on a non-key NSPanel: the .set()
    # during the real event is honoured by the WindowServer (the per-frame
    # version in the tick was being reset). Delivered via the tracking area.
    def mouseMoved_(self, event):
        if self._over_icon(event.locationInWindow()):
            NSCursor.pointingHandCursor().set()
        else:
            NSCursor.arrowCursor().set()

    def mouseDown_(self, event):
        b = self.bounds()
        loc = event.locationInWindow()   # the view fills the window → coords match
        w, h = b.size.width, b.size.height
        # ✕ STOP (top-right) → kills the engine, pill stays as a service
        if loc.x >= w - 40.0 and loc.y >= h - 34.0:
            _stop_engine()
            return
        # 🔇 MUTE (below the ✕) → pause the mic without killing the engine
        mr = self.muta_rect
        if mr and mr[0] <= loc.x <= mr[0] + mr[2] and mr[1] <= loc.y <= mr[1] + mr[3]:
            self.muted = _toggle_flag(MUTED_FILE)
            self.setNeedsDisplay_(True)
            return
        # 🔒 LOCK (padlock) → keep the pill visible (no auto-hide)
        lr = self.blocca_rect
        if lr and lr[0] <= loc.x <= lr[0] + lr[2] and lr[1] <= loc.y <= lr[1] + lr[3]:
            self.locked = _toggle_flag(LOCKED_FILE)
            self.setNeedsDisplay_(True)
            return
        # 🔁 AUTO-SEND LOOP (below the padlock) → arm/disarm autosend.py
        orr = self.loop_rect
        if orr and orr[0] <= loc.x <= orr[0] + orr[2] and orr[1] <= loc.y <= orr[1] + orr[3]:
            _toggle_autosend()
            self.autoloop = _autosend_armed()
            self.setNeedsDisplay_(True)
            return
        # ⧉ COPY inline (at the end of the text) → re-copy last sentence
        cr = self.copy_rect
        if cr and self.text and cr[0] <= loc.x <= cr[0] + cr[2] and cr[1] <= loc.y <= cr[1] + cr[3]:
            try:
                pb = NSPasteboard.generalPasteboard()
                pb.clearContents()
                pb.setString_forType_(self.text, NSPasteboardTypeString)
                self.copied_flash = time.time()
            except Exception:
                pass

    def setLevels_text_active_(self, levels, text, active):
        self.levels = levels
        self.text = text
        was = self.active
        self.active = active
        # the loop can change from OUTSIDE the pill (hotkey, daemon auto-disarm):
        # re-read the armed state at most once/second and redraw if it changed.
        now = time.time()
        if now - self._loop_chk > 1.0:
            self._loop_chk = now
            cur = _autosend_armed()
            if cur != self.autoloop:
                self.autoloop = cur
                self.setNeedsDisplay_(True)
        # redraw ONLY while visible (or on the transition frame to hidden, for a
        # clean fade). From idle/hidden it does NOT redraw: the fade is the native
        # animator on alpha. Fixes the constant idle CPU and the "freeze" when the
        # engine dies (an orphan pill spinning the waveform on stale levels).
        if active or was:
            self.setNeedsDisplay_(True)

    def drawRect_(self, rect):
        b = self.bounds()
        hf = b.size.height
        # ── pure-BLACK background (extension of the notch) ──
        w, h = b.size.width, b.size.height
        bg = _pill_path(w, h, PILL_RADIUS)   # square top, rounded bottom
        NSColor.blackColor().set()           # PURE BLACK — no border, stroke, or shadow
        bg.fill()

        # ── collapsed state (inside the notch): a single bar in the current tint ──
        if hf < 34.0:
            NSColor.colorWithCalibratedRed_green_blue_alpha_(self.tint[0], self.tint[1], self.tint[2], 0.85).set()
            NSBezierPath.bezierPathWithRect_(NSMakeRect(b.size.width / 2.0 - 16, hf / 2.0 - 1.5, 32, 3)).fill()
            return

        # ── PIXEL Matrix-terminal waveform: columns of square LEDs, FULL WIDTH ──
        cell, vgap, col_w, cgap = 4.0, 2.0, 5.0, 3.0
        wf_x0 = 24.0                      # left padding
        wf_x1 = b.size.width - 54.0       # space on the right for ✕ / ⧉ icons
        wf_w = wf_x1 - wf_x0
        text_band = 40.0                  # padding below the text
        base_y = text_band + 6.0          # =46 — gap between waveform and text
        top_y = hf - 44.0                 # clearance from the notch (~38px) + margin
        rows = max(3, int((top_y - base_y) / (cell + vgap)))
        pitch = col_w + cgap
        ncols = max(1, int(wf_w / pitch))    # fill the FULL width (no cap)
        nlv = len(self.levels)
        x = wf_x0
        for i in range(ncols):
            j = min(nlv - 1, int(i * nlv / ncols))
            lv = max(0.0, min(1.0, self.levels[j]))
            lit = int(round(lv * rows))
            for r in range(rows):
                if r < lit:
                    a = (0.55 + 0.45 * (r / max(1, rows))) if self.active else 0.32
                else:
                    a = 0.04
                NSColor.colorWithCalibratedRed_green_blue_alpha_(self.tint[0], self.tint[1], self.tint[2], a).set()
                NSBezierPath.bezierPathWithRect_(NSMakeRect(x, base_y + r * (cell + vgap), cell, cell)).fill()
            x += pitch

        # ── blinking caret █ at the bottom, BEFORE the transcribed sentence ──
        cb = 0.5 + 0.5 * math.sin(self.phase * 4.0)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(self.tint[0], self.tint[1], self.tint[2], 0.30 + 0.70 * cb).set()
        NSBezierPath.bezierPathWithRect_(NSMakeRect(24.0, 18.0, 5.0, 14.0)).fill()
        self.copy_rect = None
        if self.text:
            font = NSFont.fontWithName_size_("Menlo", 11.0)
            if font is None:
                font = NSFont.systemFontOfSize_(11.0)
            attrs = {
                NSFontAttributeName: font,
                NSForegroundColorAttributeName:    # dynamic tint: green dictation / red TTS
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(self.tint[0], self.tint[1], self.tint[2], 0.95),
            }
            s = self.text
            if len(s) > 52:
                s = "…" + s[-51:]
            ns = NSString.stringWithString_(s)
            tw = ns.sizeWithAttributes_(attrs).width
            ns.drawAtPoint_withAttributes_(NSMakePoint(35.0, 18.0), attrs)
            # ── ⧉ inline COPY, at the end of the transcribed text ──
            cix = min(35.0 + tw + 10.0, w - 28.0)
            ciy = 18.0
            recent = (time.time() - self.copied_flash) < 1.0
            ic = self.tint if (self.hover_c or recent) else (1.0, 1.0, 1.0)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(ic[0], ic[1], ic[2], 0.92).set()
            if recent:
                chk = NSBezierPath.bezierPath()
                chk.moveToPoint_(NSMakePoint(cix, ciy + 4))
                chk.lineToPoint_(NSMakePoint(cix + 4, ciy))
                chk.lineToPoint_(NSMakePoint(cix + 11, ciy + 11))
                chk.setLineWidth_(2.0)
                chk.stroke()
            else:
                bk = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(cix + 4, ciy, 9, 10), 2, 2)
                bk.setLineWidth_(1.4)
                bk.stroke()
                fr = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(cix, ciy + 3, 9, 10), 2, 2)
                fr.setLineWidth_(1.4)
                fr.stroke()
            self.copy_rect = (cix - 3, ciy - 3, 22, 20)

        # ── ✕ STOP top-right (white, tinted on hover) — copy is inline with text ──
        WHITE = (1.0, 1.0, 1.0)
        cxc = w - 26.0
        s = 11.0
        xc = self.tint if self.hover_x else WHITE
        NSColor.colorWithCalibratedRed_green_blue_alpha_(xc[0], xc[1], xc[2], 0.92).set()
        qy = h - 26.0
        xq = NSBezierPath.bezierPath()
        xq.moveToPoint_(NSMakePoint(cxc - s / 2, qy))
        xq.lineToPoint_(NSMakePoint(cxc + s / 2, qy + s))
        xq.moveToPoint_(NSMakePoint(cxc + s / 2, qy))
        xq.lineToPoint_(NSMakePoint(cxc - s / 2, qy + s))
        xq.setLineWidth_(1.6)
        xq.stroke()

        # ── 🔇 MUTE (below the ✕) — speaker + waves / slash when muted ──
        mx, my = cxc, h - 52.0
        self.muta_rect = (mx - 11, my - 11, 22, 22)
        mc = RED if self.muted else (self.tint if self.hover_m else WHITE)
        ma = 0.95 if (self.muted or self.hover_m) else 0.62
        NSColor.colorWithCalibratedRed_green_blue_alpha_(mc[0], mc[1], mc[2], ma).set()
        sp = NSBezierPath.bezierPath()
        sp.moveToPoint_(NSMakePoint(mx - 6, my - 2))
        sp.lineToPoint_(NSMakePoint(mx - 3, my - 2))
        sp.lineToPoint_(NSMakePoint(mx + 1, my - 5))
        sp.lineToPoint_(NSMakePoint(mx + 1, my + 5))
        sp.lineToPoint_(NSMakePoint(mx - 3, my + 2))
        sp.lineToPoint_(NSMakePoint(mx - 6, my + 2))
        sp.closePath()
        sp.fill()
        if self.muted:
            sl = NSBezierPath.bezierPath()
            sl.moveToPoint_(NSMakePoint(mx - 7, my + 7))
            sl.lineToPoint_(NSMakePoint(mx + 7, my - 7))
            sl.setLineWidth_(1.7)
            sl.stroke()
        else:
            for rr in (3.2, 5.4):
                wv = NSBezierPath.bezierPath()
                wv.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_(
                    NSMakePoint(mx + 1, my), rr, -42.0, 42.0)
                wv.setLineWidth_(1.2)
                wv.stroke()

        # ── 🔒 LOCK (padlock) — filled when locked (no auto-hide), outline when unlocked ──
        lx, ly = cxc, h - 78.0
        self.blocca_rect = (lx - 11, ly - 11, 22, 22)
        lc = AMBER if self.locked else (self.tint if self.hover_l else WHITE)
        la = 0.95 if (self.locked or self.hover_l) else 0.62
        NSColor.colorWithCalibratedRed_green_blue_alpha_(lc[0], lc[1], lc[2], la).set()
        body = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(lx - 5, ly - 6, 10, 8), 1.6, 1.6)
        shackle = NSBezierPath.bezierPath()
        shackle.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_(
            NSMakePoint(lx, ly + 2), 3.2, 0.0, 180.0)
        shackle.setLineWidth_(1.4)
        if self.locked:
            body.fill()
            shackle.stroke()
        else:
            body.setLineWidth_(1.4)
            body.stroke()
            shackle.stroke()

        # ── 🔁 AUTO-SEND LOOP (below the padlock) — two looping arrows
        #    armed = AMBER filled · off = white/tint-hover outline
        ox, oy = cxc, h - 104.0
        self.loop_rect = (ox - 11, oy - 11, 22, 22)
        oc = AMBER if self.autoloop else (self.tint if self.hover_o else WHITE)
        oa = 0.95 if (self.autoloop or self.hover_o) else 0.62
        NSColor.colorWithCalibratedRed_green_blue_alpha_(oc[0], oc[1], oc[2], oa).set()
        R_LOOP = 5.6
        # top arc (left→right) + bottom arc (right→left)
        for a0, a1 in ((150.0, 30.0), (330.0, 210.0)):
            arc = NSBezierPath.bezierPath()
            arc.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
                NSMakePoint(ox, oy), R_LOOP, a0, a1, True)
            arc.setLineWidth_(1.5)
            arc.stroke()
        # arrow heads at the arc tips (at 30° and 210°)
        for ang, drot in ((30.0, -90.0), (210.0, -90.0)):
            rad = math.radians(ang)
            tipx = ox + R_LOOP * math.cos(rad)
            tipy = oy + R_LOOP * math.sin(rad)
            t = math.radians(ang + drot)   # clockwise tangent direction
            ah = NSBezierPath.bezierPath()
            ah.moveToPoint_(NSMakePoint(tipx + 3.4 * math.cos(t + 2.6), tipy + 3.4 * math.sin(t + 2.6)))
            ah.lineToPoint_(NSMakePoint(tipx, tipy))
            ah.lineToPoint_(NSMakePoint(tipx + 3.4 * math.cos(t - 2.6), tipy + 3.4 * math.sin(t - 2.6)))
            ah.setLineWidth_(1.5)
            ah.stroke()


class RobotView(NSView):
    """Floating ROBOT widget — the always-visible command center (the menu bar
    can park new status items off-screen behind the notch). Click = control
    center (the same menu as the status item) · drag = reposition (persisted)."""

    def initWithFrame_(self, frame):
        self = objc.super(RobotView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.on = False
        self.loop = False
        self.owner = None        # Controller (for the menu + the icon)
        self._drag0 = None       # (mouse_screen, win_origin) at drag start
        self._moved = False
        return self

    def drawRect_(self, rect):
        b = self.bounds()
        from AppKit import NSGradient
        # ── disc with a body: radial gradient + an accent ring tinted by state +
        #    a hairline. Glass-orb style.
        inset = NSMakeRect(4.0, 4.0, b.size.width - 8.0, b.size.height - 8.0)
        disc = NSBezierPath.bezierPathWithOvalInRect_(inset)
        # state tint: amber (loop) · green (listening) · grey (off)
        if self.loop and self.on:
            tc = (1.0, 0.72, 0.05)
        elif self.on:
            tc = (0.12, 1.0, 0.32)
        else:
            tc = (0.48, 0.51, 0.58)
        # soft outer halo (state glow)
        for gr, ga in ((2.6, 0.10), (1.4, 0.16)):
            NSColor.colorWithCalibratedRed_green_blue_alpha_(tc[0], tc[1], tc[2], ga).set()
            halo = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(inset.origin.x - gr, inset.origin.y - gr,
                           inset.size.width + 2 * gr, inset.size.height + 2 * gr))
            halo.setLineWidth_(1.6)
            halo.stroke()
        # body: radial gradient panel→near-black (depth, light from top-left)
        g = NSGradient.alloc().initWithStartingColor_endingColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.14, 0.16, 0.22, 0.97),
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.03, 0.035, 0.055, 0.97))
        g.drawInBezierPath_relativeCenterPosition_(disc, NSMakePoint(-0.25, 0.35))
        # accent ring (thin, full character)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(tc[0], tc[1], tc[2], 0.55).set()
        ring = NSBezierPath.bezierPathWithOvalInRect_(inset)
        ring.setLineWidth_(1.3)
        ring.stroke()
        # inner white hairline (glass sheen)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.10).set()
        hl = NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(inset.origin.x + 1.2, inset.origin.y + 1.2,
                       inset.size.width - 2.4, inset.size.height - 2.4))
        hl.setLineWidth_(0.8)
        hl.stroke()
        if self.owner is not None:
            img = self.owner._make_robot_icon(self.on, self.loop, S=20.0)
            img.drawInRect_fromRect_operation_fraction_(
                NSMakeRect((b.size.width - 20.0) / 2.0, (b.size.height - 20.0) / 2.0, 20.0, 20.0),
                NSMakeRect(0, 0, 0, 0), 2, 1.0)   # NSCompositeSourceOver

    def mouseDown_(self, event):
        w = self.window()
        self._drag0 = (NSEvent.mouseLocation(), w.frame().origin)
        self._moved = False

    def mouseDragged_(self, event):
        if not self._drag0:
            return
        m0, o0 = self._drag0
        m1 = NSEvent.mouseLocation()
        dx, dy = m1.x - m0.x, m1.y - m0.y
        if abs(dx) + abs(dy) > 3:
            self._moved = True
        self.window().setFrameOrigin_(NSMakePoint(o0.x + dx, o0.y + dy))

    def mouseUp_(self, event):
        if self._moved:
            try:
                o = self.window().frame().origin
                ROBOT_POS.write_text(f"{o.x},{o.y}")
            except Exception:
                pass
        else:
            # a plain click → control center (the same menu as the status item)
            if self.owner is not None and getattr(self.owner, "mb_menu", None):
                NSMenu.popUpContextMenu_withEvent_forView_(self.owner.mb_menu, event, self)
        self._drag0 = None


WIDGET_W, WIDGET_H = 231.0, 50.0   # +59 px on the right for the play/stop and mic keys


class HardwareView(NSView):
    """Hardware-look floating voice widget (Wispr-style): an anodized-aluminum
    capsule with corner screws, a status LED and a recessed VU display — pure
    UI pretending to be a device. Click = control center (same menu as the
    status item) · drag = reposition (persisted in WIDGET_POS)."""

    def initWithFrame_(self, frame):
        self = objc.super(HardwareView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.levels = []          # RMS history (the engine's levels.bin, via tick)
        self.state = ""           # idle | recording | transcribing ("" = engine off)
        self.muted = False
        self.engine_on = False
        self.phase = 0.0          # animation clock (LED pulse)
        self.owner = None         # Controller (for the popup menu)
        self.return_held = _flag_on(RETURN_HELD_FILE)  # play/stop key latched in
        self._key_down = False    # finger on the key (cap bottomed out)
        self._mic_down = False    # finger on the mic key
        self._drag0 = None
        self._moved = False
        return self

    def drawRect_(self, rect):
        import math as _math
        from AppKit import NSGradient
        b = self.bounds()
        W, H = b.size.width, b.size.height

        # ── body: anodized capsule, light from the top ────────────────────────
        body_r = NSMakeRect(3.0, 3.0, W - 6.0, H - 6.0)
        body = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(body_r, 10.0, 10.0)
        g = NSGradient.alloc().initWithStartingColor_endingColor_(
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.31, 0.33, 0.37, 0.98),
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.12, 0.13, 0.16, 0.98))
        g.drawInBezierPath_angle_(body, -90.0)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.55).set()
        body.setLineWidth_(1.0)
        body.stroke()
        # machined edge: inner top highlight + inner bottom shade
        hi = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(4.2, 4.2, W - 8.4, H - 8.4), 8.8, 8.8)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.14).set()
        hi.setLineWidth_(0.8)
        hi.stroke()

        # ── corner screws (the tell that it "is" hardware) ────────────────────
        for sx, sy, ang in ((10.0, 10.0, 0.6), (W - 10.0, 10.0, 2.2),
                            (10.0, H - 10.0, 1.5), (W - 10.0, H - 10.0, 0.2)):
            sr = 2.2
            screw = NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(sx - sr, sy - sr, sr * 2, sr * 2))
            sg = NSGradient.alloc().initWithStartingColor_endingColor_(
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.55, 0.57, 0.60, 1.0),
                NSColor.colorWithCalibratedRed_green_blue_alpha_(0.20, 0.21, 0.24, 1.0))
            sg.drawInBezierPath_angle_(screw, -90.0)
            # slot, each screw at its own angle (real assembly, not a texture)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(0.06, 0.06, 0.08, 0.9).set()
            slot = NSBezierPath.bezierPath()
            dx, dy = sr * 0.85 * _math.cos(ang), sr * 0.85 * _math.sin(ang)
            slot.moveToPoint_(NSMakePoint(sx - dx, sy - dy))
            slot.lineToPoint_(NSMakePoint(sx + dx, sy + dy))
            slot.setLineWidth_(0.8)
            slot.stroke()

        # ── status LED (left), glow + pulse while recording ───────────────────
        tc = widget_led(self.state, self.muted, self.engine_on)
        cx, cy, lr = 21.0, H / 2.0 + 4.0, 3.6
        pulse = 1.0
        if self.state == "recording":
            pulse = 0.70 + 0.30 * _math.sin(self.phase * 6.0)
        for gr, ga in ((3.2, 0.10), (1.8, 0.18)):
            NSColor.colorWithCalibratedRed_green_blue_alpha_(tc[0], tc[1], tc[2], ga * pulse).set()
            NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(cx - lr - gr, cy - lr - gr, (lr + gr) * 2, (lr + gr) * 2)).fill()
        # bezel (the LED sits IN the metal)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.04, 0.04, 0.06, 1.0).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - lr - 1.2, cy - lr - 1.2, (lr + 1.2) * 2, (lr + 1.2) * 2)).fill()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(
            tc[0] * pulse, tc[1] * pulse, tc[2] * pulse, 1.0).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - lr, cy - lr, lr * 2, lr * 2)).fill()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.35 * pulse).set()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - lr * 0.45, cy + lr * 0.05, lr * 0.8, lr * 0.8)).fill()

        # ── recessed VU display ───────────────────────────────────────────────
        dx0, dy0 = 36.0, 11.0
        dw, dh = W - dx0 - 72.0, H - 22.0
        disp_r = NSMakeRect(dx0, dy0, dw, dh)
        disp = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(disp_r, 5.0, 5.0)
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.015, 0.02, 0.03, 1.0).set()
        disp.fill()
        # recess: dark rim + faint reflected light on the lower lip
        NSColor.colorWithCalibratedRed_green_blue_alpha_(0.0, 0.0, 0.0, 0.8).set()
        disp.setLineWidth_(1.2)
        disp.stroke()
        NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.10).set()
        lip = NSBezierPath.bezierPath()
        lip.moveToPoint_(NSMakePoint(dx0 + 5.0, dy0 - 0.8))
        lip.lineToPoint_(NSMakePoint(dx0 + dw - 5.0, dy0 - 0.8))
        lip.setLineWidth_(0.8)
        lip.stroke()
        # VU bars (green, red-tipped when hot; dim baseline when the engine is off)
        bars = widget_bar_levels(self.levels if self.engine_on else [])
        gap, pad = 2.0, 5.0
        bw = (dw - 2 * pad - gap * (WIDGET_BARS - 1)) / WIDGET_BARS
        for i, lvl in enumerate(bars):
            x = dx0 + pad + i * (bw + gap)
            bh = max(1.6, lvl * (dh - 8.0))
            y = dy0 + (dh - bh) / 2.0
            cr, cg, cb, ca = widget_bar_color(lvl, self.engine_on)
            NSColor.colorWithCalibratedRed_green_blue_alpha_(cr, cg, cb, ca).set()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x, y, bw, bh), 1.0, 1.0).fill()

        self._draw_key(widget_send_key_center(W, H),
                       widget_send_key_look(self.return_held, self._key_down),
                       {"play": (0.12, 0.95, 0.35), "stop": (1.0, 0.62, 0.08)})
        self._draw_key(widget_mic_key_center(W, H),
                       widget_mic_key_look(self.muted, self._mic_down),
                       {"pause": (0.12, 0.95, 0.35), "play": (1.0, 0.62, 0.08)})

        # ── engraved serial (the wink) ────────────────────────────────────────
        label = NSString.stringWithString_("VV·01")
        label.drawAtPoint_withAttributes_(
            NSMakePoint(12.0, 5.0),
            {NSFontAttributeName: NSFont.systemFontOfSize_(5.0),
             NSForegroundColorAttributeName:
                 NSColor.colorWithCalibratedRed_green_blue_alpha_(1.0, 1.0, 1.0, 0.28)})

    def _draw_key(self, center, look, tints):
        """Round latching push key: a dark well cut in the metal, a machined
        cap with a cast shadow that shrinks as it goes in, an engraved glyph
        tinted per glyph (green = running, amber = held/paused)."""
        from AppKit import NSGradient
        d = look["depth"]
        cx, cy = center
        r = SEND_KEY_R

        def oval(x, y, rad):
            return NSBezierPath.bezierPathWithOvalInRect_(
                NSMakeRect(x - rad, y - rad, rad * 2, rad * 2))

        def rgba(r_, g_, b_, a_):
            return NSColor.colorWithCalibratedRed_green_blue_alpha_(r_, g_, b_, a_)

        # well: the hole the key sits in, its lower lip catching the light
        rgba(1.0, 1.0, 1.0, 0.10).set()
        oval(cx, cy - 0.7, r + 2.6).fill()
        rgba(0.02, 0.02, 0.03, 1.0).set()
        oval(cx, cy, r + 2.2).fill()

        # cast shadow: long and soft when proud, almost gone when pressed in
        lift = 1.0 - d
        for k, a in ((2.0, 0.18), (1.0, 0.30)):
            rgba(0.0, 0.0, 0.0, a * (0.25 + 0.75 * lift)).set()
            oval(cx, cy - 0.4 - 1.6 * lift, r + 0.2 * k * lift).fill()

        # cap: sinks a hair and turns concave in the light as it goes in
        ccy = cy - 0.8 * d
        cr = r - 0.7 * d
        cap = oval(cx, ccy, cr)
        top = 0.46 - 0.20 * d
        bot = 0.15 + 0.10 * d
        NSGradient.alloc().initWithStartingColor_endingColor_(
            rgba(top, top + 0.02, top + 0.05, 1.0),
            rgba(bot, bot + 0.01, bot + 0.03, 1.0)).drawInBezierPath_angle_(cap, -90.0)
        rgba(0.0, 0.0, 0.0, 0.65).set()
        cap.setLineWidth_(0.8)
        cap.stroke()
        # machined rim highlight on the upper edge (fades as the cap sinks)
        rim = NSBezierPath.bezierPath()
        rim.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_(
            NSMakePoint(cx, ccy), cr - 0.9, 25.0, 155.0)
        rgba(1.0, 1.0, 1.0, 0.08 + 0.22 * lift).set()
        rim.setLineWidth_(0.8)
        rim.stroke()

        # engraved glyph, tinted like the LED it replaces
        tint = tints[look["glyph"]]
        if look["glyph"] == "play":
            g = NSBezierPath.bezierPath()
            g.moveToPoint_(NSMakePoint(cx - 2.4, ccy + 3.4))
            g.lineToPoint_(NSMakePoint(cx + 3.6, ccy))
            g.lineToPoint_(NSMakePoint(cx - 2.4, ccy - 3.4))
            g.closePath()
        elif look["glyph"] == "pause":
            g = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(cx - 2.9, ccy - 3.2, 2.0, 6.4), 0.5, 0.5)
            g.appendBezierPath_(NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(cx + 0.9, ccy - 3.2, 2.0, 6.4), 0.5, 0.5))
        else:
            g = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(cx - 2.6, ccy - 2.6, 5.2, 5.2), 0.8, 0.8)
        if not self.engine_on:
            tint = (0.42, 0.42, 0.42)
        # the engraving's lower edge catches light, the fill sits inside it
        from AppKit import NSAffineTransform
        t = NSAffineTransform.transform()
        t.translateXBy_yBy_(0.0, -0.6)
        lip = g.copy()
        lip.transformUsingAffineTransform_(t)
        rgba(1.0, 1.0, 1.0, 0.12).set()
        lip.fill()
        rgba(tint[0], tint[1], tint[2], 0.92).set()
        g.fill()

    # ── drag anywhere (persisted) · plain click = control center ─────────────
    # · the play/stop and mic keys are buttons, not drag handles
    def mouseDown_(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)
        b = self.bounds()
        if widget_send_key_hit(p.x, p.y, b.size.width, b.size.height):
            self._key_down = True
            self._drag0 = None
            self.setNeedsDisplay_(True)
            return
        if widget_mic_key_hit(p.x, p.y, b.size.width, b.size.height):
            self._mic_down = True
            self._drag0 = None
            self.setNeedsDisplay_(True)
            return
        w = self.window()
        self._drag0 = (NSEvent.mouseLocation(), w.frame().origin)
        self._moved = False

    def mouseDragged_(self, event):
        if self._key_down or self._mic_down:
            return
        if not self._drag0:
            return
        m0, o0 = self._drag0
        m1 = NSEvent.mouseLocation()
        dx, dy = m1.x - m0.x, m1.y - m0.y
        if abs(dx) + abs(dy) > 3:
            self._moved = True
        self.window().setFrameOrigin_(NSMakePoint(o0.x + dx, o0.y + dy))

    def mouseUp_(self, event):
        if self._key_down:
            self._key_down = False
            p = self.convertPoint_fromView_(event.locationInWindow(), None)
            b = self.bounds()
            if widget_send_key_hit(p.x, p.y, b.size.width, b.size.height):
                self.return_held = _toggle_return_held()
            self.setNeedsDisplay_(True)
            return
        if self._mic_down:
            self._mic_down = False
            p = self.convertPoint_fromView_(event.locationInWindow(), None)
            b = self.bounds()
            if widget_mic_key_hit(p.x, p.y, b.size.width, b.size.height):
                self.muted = _toggle_mic_paused()
            self.setNeedsDisplay_(True)
            return
        if self._moved:
            try:
                o = self.window().frame().origin
                WIDGET_POS.write_text(f"{o.x},{o.y}")
            except Exception:
                pass
        else:
            if self.owner is not None and getattr(self.owner, "mb_menu", None):
                NSMenu.popUpContextMenu_withEvent_forView_(self.owner.mb_menu, event, self)
        self._drag0 = None


def _child_python():
    """Interpreter for spawning the engine.py/autosend.py siblings.

    Inside the py2app bundle sys.executable is the app LAUNCHER: it ignores
    argv and always boots the pill main again, so spawning children through it
    forks a second pill instead of the engine. py2app ships a real embedded
    interpreter at Contents/MacOS/python — use that when frozen. Keeping the
    script path in argv preserves invariant #8 (pgrep/pkill -f engine.py)."""
    if getattr(sys, "frozen", None) == "macosx_app":
        cand = Path(sys.executable).parent / "python"
        if cand.exists():
            return str(cand)
    return sys.executable or "python3"


def _argv_pattern(path) -> str:
    """`path` as a pgrep/pkill pattern that matches itself and nothing else.

    Both take an *extended regular expression*, not a literal — and a repo path
    is user-chosen. Clone into `~/projects/vibe(voice)` and the raw path becomes
    a regex with a capture group that matches nothing: pgrep returns 1, the pill
    concludes its engine is dead, and every toggle spawns another one. Verified
    against `(`, `+`, `[`, `*`, `^$`, a space and this repo's own emoji — all
    seven fail raw and all seven pass escaped.
    """
    return re.escape(str(path))


def _engine_running():
    """True when OUR engine is alive — matched by absolute path, not by name.

    `pgrep -f engine.py` matches any process with that string anywhere in argv:
    an editor open on the file, a `tail -f`, an unrelated project's engine.py
    (verified — a sleeping `python3 /tmp/.../engine.py` is a hit). Two failures
    followed from it: the pill saw a stranger and refused to start its own
    engine, and `_stop_engine`'s pkill killed that stranger. ENGINE_PATH is the
    exact argv the pill spawns 40 lines down, so the match is now the process
    we own — and it still ends in engine.py, keeping invariant #8 intact.
    """
    try:
        r = subprocess.run(["pgrep", "-f", _argv_pattern(ENGINE_PATH)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


_ENGINE_SEEN = {"t": -1e9, "on": False}


def _engine_running_cached(ttl=1.0, now=None):
    """`_engine_running()` at most once per `ttl` seconds.

    The pill's tick asked it on EVERY frame, before the idle skip: a `pgrep`
    subprocess spawned 60 times a second on the main thread — the same thread
    that reads the Control key and plays the cues. Measured 30/10/2026: the
    chip's «sent» arrived 2.1 s after release with the text ready in 0.5 s.
    Whether the engine is alive changes on a human timescale; one look a
    second is plenty."""
    now = time.monotonic() if now is None else now
    if now - _ENGINE_SEEN["t"] >= ttl:
        _ENGINE_SEEN["on"] = _engine_running()
        _ENGINE_SEEN["t"] = now
    return _ENGINE_SEEN["on"]


def _clean_child_env(base=None):
    """Environment for spawning the embedded interpreter (engine/autosend).

    Inside the py2app bundle os.environ carries PYTHONHOME / PYTHONPATH /
    PYTHONEXECUTABLE / RESOURCEPATH pointing INTO the app. Passing them to
    Contents/MacOS/python makes it miss its own stdlib and die on the first
    import (`ModuleNotFoundError: wave`, engine.py) within ~1s — so the menu-bar
    toggle looked dead. Strip them so the child interpreter resolves its own
    home. (Scar: standalone spawn works, app-spawn crashed — the diff
    was the inherited env, not the signature.)"""
    env = dict(os.environ if base is None else base)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "RESOURCEPATH",
              "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
              "PYVENV_LAUNCHER", "__PYVENV_LAUNCHER__"):
        env.pop(k, None)
    # Stripping is not enough on its own: without a home, the embedded
    # interpreter falls back to the prefix compiled into it — the BUILD
    # machine's Homebrew Python. On the builder that "worked" (and was the
    # fix above); on any other Mac there is no such folder, and under the
    # hardened runtime even the builder refuses those foreign .so files. The
    # missing `wave` of the old scar was a module absent from the bundle,
    # not a wrong home. Point it at the bundle itself.
    # Only a bundle that carries its own stdlib gets the override: the dev
    # (alias) bundle is "frozen" too but its python is Homebrew's, and pointing
    # that one at the bundle left it without `encodings` (engine dead at start,
    # 2026-10-31).
    res = Path(sys.executable).parent.parent / "Resources"
    if getattr(sys, "frozen", None) == "macosx_app" and (res / "lib" / "python310.zip").is_file():
        env["PYTHONHOME"] = str(res)
        env["PYTHONNOUSERSITE"] = "1"
        # A .pyc written inside a signed app breaks its seal.
        env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def rotate_log(path, max_bytes=1_000_000):
    """Keep one previous generation of an append-only log, so engine.err (one
    model-download progress bar per start) cannot grow without bound."""
    try:
        if path.stat().st_size > max_bytes:
            os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


def _start_engine():
    try:
        cfg = config.load()
        env = _clean_child_env()
        env.setdefault("VIBEVOICE_LANG", cfg["lang"])
        env.setdefault("VIBEVOICE_AUTOSEND", "1" if cfg["autosend"] else "0")
        env.setdefault("VIBEVOICE_AUTOSEND_RETURN", "1" if cfg["autosend_return"] else "0")
        env.setdefault("VIBEVOICE_VP", "1" if cfg["vp"] else "0")
        env.setdefault("VIBEVOICE_PTT", "1" if cfg["ptt"] else "0")
        # Engine stderr → file, NOT devnull: the engine prints its capture
        # backend, VAD choice and every transcription failure there — with
        # devnull a deaf engine is indistinguishable from a healthy one
        # (scar: "not transcribing" with zero evidence anywhere).
        try:
            rotate_log(STATE_DIR / "engine.err")
            err = open(STATE_DIR / "engine.err", "ab")
        except Exception:
            err = subprocess.DEVNULL
        subprocess.Popen([_child_python(), str(ENGINE_PATH)],
                         stdout=subprocess.DEVNULL, stderr=err,
                         start_new_session=True, env=env)
        _note_life("engine spawned")
    except Exception as e:
        # Never silent: a spawn that fails here is a dictation that never starts.
        _note_life(f"engine spawn failed: {type(e).__name__}: {e}")


def _note_life(line):
    """App and engine lifecycle in chip.log, next to the key events: when the
    pill vanishes, the last line says whether it was a quit (this hook ran) or
    a crash (it did not)."""
    try:
        log = STATE_DIR / "chip.log"
        lines = log.read_text().splitlines()[-199:] if log.exists() else []
        lines.append(time.strftime("%H:%M:%S ") + line)
        log.write_text("\n".join(lines) + "\n")
    except Exception:
        pass


def format_history_line(record) -> str | None:
    """One `history.jsonl` entry as "HH:MM  text", or None if there is nothing to show.

    The timestamp is already in the file (`{"ts": float, "text": str}`) and used to be
    thrown away, which left the list unable to answer "when did I say that?".

    Deliberately tolerant: the engine appends to this file on the transcription path,
    so a torn or malformed line is possible — and one bad line must not blank the
    whole list. An unusable timestamp degrades to the text alone rather than hiding it.
    """
    if not isinstance(record, dict):
        return None
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    try:
        return f"{datetime.fromtimestamp(float(record['ts'])):%H:%M}  {text}"
    except (KeyError, TypeError, ValueError, OSError, OverflowError):
        return text


def _set_menu_symbol(item, symbol: str) -> None:
    """Put an SF Symbol on a menu item; text-only if the symbol is unavailable."""
    try:
        img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(symbol, None)
        if img is not None:
            img.setTemplate_(True)
            item.setImage_(img)
    except Exception:
        pass


def last_dictation_text(path) -> str | None:
    """The newest usable text in `history.jsonl`, for «Copia l'ultima dettatura».
    Same tolerance as format_history_line: a torn line is skipped, not fatal."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return None
    for line in reversed(lines):
        try:
            text = json.loads(line).get("text")
        except Exception:
            continue
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


# ── settings i18n ─────────────────────────────────────────────────────────────
# The six languages the picker offers. The CODE is what reaches Whisper and the
# engine; the flag is decoration and the name is what carries the meaning, which
# is why they always travel together — a flag alone would claim that a language
# belongs to a country, and "English" is not a British possession. Emoji flags
# are safe here and nowhere else: this app is macOS-only, so Apple Color Emoji
# is guaranteed to be present and to render them identically on every machine.
_S_LANGS = [
    ("it", "🇮🇹", "Italiano"),
    ("en", "🇬🇧", "English"),
    ("es", "🇪🇸", "Español"),
    ("fr", "🇫🇷", "Français"),
    ("de", "🇩🇪", "Deutsch"),
    ("pt", "🇵🇹", "Português"),
]

# Every string the settings window shows, in each of those languages. The window
# speaks whatever language you dictate in: someone who talks to the app in
# Spanish should not have to read English to change a switch. Subtitles are
# written in plain words on purpose — they say what the option DOES to your
# machine, not what it is called in the code.
_S_TEXT = {
    "it": {
        "title": "VibeVoice — Impostazioni",
        "voice": "Voce", "behaviour": "Comportamento",
        "appearance": "Aspetto", "history": "Cronologia",
        "clear": "Svuota", "empty": "(nessuna trascrizione)",
        "lang": ("Lingua", "La lingua in cui parli quando detti"),
        "vp": ("Elaborazione voce", "Toglie l'eco e i rumori intorno a te"),
        "as": ("Incolla automatico",
               "Scrive il testo dove hai il cursore, senza copiare e incollare"),
        "ar": ("Invio automatico",
               "Preme Invio da solo appena il testo è arrivato"),
        "dk": ("Icona nel Dock",
               "Mostra VibeVoice in fondo allo schermo, tra le app aperte"),
    },
    "en": {
        "title": "VibeVoice — Settings",
        "voice": "Voice", "behaviour": "Behaviour",
        "appearance": "Appearance", "history": "History",
        "clear": "Clear", "empty": "(no transcriptions yet)",
        "lang": ("Language", "The language you speak when you dictate"),
        "vp": ("Voice processing", "Removes echo and the noise around you"),
        "as": ("Auto-paste",
               "Types the text where your cursor is, no copy and paste"),
        "ar": ("Auto-Return",
               "Presses Return for you as soon as the text has landed"),
        "dk": ("Dock icon",
               "Shows VibeVoice at the bottom of the screen, with the open apps"),
    },
    "es": {
        "title": "VibeVoice — Ajustes",
        "voice": "Voz", "behaviour": "Comportamiento",
        "appearance": "Apariencia", "history": "Historial",
        "clear": "Vaciar", "empty": "(todavía no hay transcripciones)",
        "lang": ("Idioma", "El idioma en el que hablas al dictar"),
        "vp": ("Procesamiento de voz", "Quita el eco y el ruido a tu alrededor"),
        "as": ("Pegado automático",
               "Escribe el texto donde está el cursor, sin copiar y pegar"),
        "ar": ("Intro automático",
               "Pulsa Intro por ti en cuanto llega el texto"),
        "dk": ("Icono en el Dock",
               "Muestra VibeVoice abajo, entre las apps abiertas"),
    },
    "fr": {
        "title": "VibeVoice — Réglages",
        "voice": "Voix", "behaviour": "Comportement",
        "appearance": "Apparence", "history": "Historique",
        "clear": "Effacer", "empty": "(aucune transcription)",
        "lang": ("Langue", "La langue que vous parlez quand vous dictez"),
        "vp": ("Traitement de la voix", "Supprime l'écho et les bruits autour de vous"),
        "as": ("Collage automatique",
               "Écrit le texte là où se trouve le curseur, sans copier-coller"),
        "ar": ("Retour automatique",
               "Appuie sur Entrée dès que le texte est arrivé"),
        "dk": ("Icône du Dock",
               "Affiche VibeVoice en bas de l'écran, parmi les apps ouvertes"),
    },
    "de": {
        "title": "VibeVoice — Einstellungen",
        "voice": "Stimme", "behaviour": "Verhalten",
        "appearance": "Darstellung", "history": "Verlauf",
        "clear": "Leeren", "empty": "(noch keine Transkriptionen)",
        "lang": ("Sprache", "Die Sprache, in der du diktierst"),
        "vp": ("Sprachverarbeitung", "Entfernt Echo und Geräusche um dich herum"),
        "as": ("Automatisch einfügen",
               "Schreibt den Text dorthin, wo dein Cursor steht"),
        "ar": ("Automatische Eingabe",
               "Drückt für dich die Eingabetaste, sobald der Text da ist"),
        "dk": ("Dock-Symbol",
               "Zeigt VibeVoice unten am Bildschirm bei den offenen Apps"),
    },
    "pt": {
        "title": "VibeVoice — Definições",
        "voice": "Voz", "behaviour": "Comportamento",
        "appearance": "Aparência", "history": "Histórico",
        "clear": "Limpar", "empty": "(ainda sem transcrições)",
        "lang": ("Idioma", "O idioma em que falas quando ditas"),
        "vp": ("Processamento de voz", "Remove o eco e o ruído à tua volta"),
        "as": ("Colar automático",
               "Escreve o texto onde está o cursor, sem copiar e colar"),
        "ar": ("Enter automático",
               "Carrega em Enter por ti assim que o texto chega"),
        "dk": ("Ícone na Dock",
               "Mostra o VibeVoice em baixo, entre as apps abertas"),
    },
}


def settings_text(lang: str) -> dict:
    """The strings for `lang`, falling back to English for anything unknown.
    A language the picker does not offer can still reach here from a hand-edited
    config, and a window with missing labels is worse than one in English."""
    return _S_TEXT.get(lang, _S_TEXT["en"])


def apply_settings(cfg: dict) -> bool:
    """Persist `cfg`, apply the Dock policy, and say whether the engine must restart.

    Module-level on purpose. The settings window's callback is an Objective-C
    selector and cannot be driven from a test, so the part worth testing lives
    here instead: the window used to restart the engine on *every* change, which
    meant ticking "Dock icon" — a preference the engine never even receives —
    killed whatever was being transcribed at that moment.

    Keys the window has no control for (currently `vp`) are carried over from the
    stored config rather than silently reset to their default.
    """
    previous = config.load()
    merged = {k: cfg.get(k, previous[k]) for k in config.DEFAULTS}
    config.save(merged)
    NSApp.setActivationPolicy_(
        NSApplicationActivationPolicyRegular if merged["dock"]
        else NSApplicationActivationPolicyAccessory)
    return config.engine_restart_needed(previous, merged)


def _stop_engine():
    # Absolute path, for the reason spelled out in _engine_running: a bare
    # `pkill -f engine.py` is a loaded gun pointed at every other process on the
    # machine that happens to carry that string in argv.
    try:
        subprocess.Popen(["pkill", "-f", _argv_pattern(ENGINE_PATH)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _autosend_running():
    # Same anchoring as _engine_running: a name is not an identity.
    try:
        r = subprocess.run(["pgrep", "-f", _argv_pattern(AUTOSEND_PATH)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False


def _toggle_autosend():
    """Arm/disarm the auto-send loop. Writes "on"/"off" to the shared autosend
    file (autosend.py reads it) and spawns the daemon when arming if it isn't
    already running — a switch with no wire is useless."""
    try:
        cur = ""
        try:
            cur = AUTOSEND_FILE.read_text().strip()
        except Exception:
            pass
        if cur == "on":
            AUTOSEND_FILE.parent.mkdir(parents=True, exist_ok=True)
            AUTOSEND_FILE.write_text("off")
            subprocess.Popen(["afplay", "/System/Library/Sounds/Submarine.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            AUTOSEND_FILE.parent.mkdir(parents=True, exist_ok=True)
            AUTOSEND_FILE.write_text("on")
            subprocess.Popen(["afplay", "/System/Library/Sounds/Tink.aiff"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not _autosend_running():
                # Keep the daemon's stderr, for the same reason _start_engine
                # does 90 lines up (scar): with devnull a deaf daemon
                # is indistinguishable from a healthy one. Everything it says —
                # "skip — paused by flag", "window changed", osascript errors,
                # a missing pynput — was being thrown away, which is precisely
                # what made a swallowed Return impossible to diagnose.
                try:
                    err = open(STATE_DIR / "autosend.err", "ab")
                except Exception:
                    err = subprocess.DEVNULL
                subprocess.Popen([_child_python(), str(AUTOSEND_PATH)],
                                 stdout=subprocess.DEVNULL, stderr=err,
                                 start_new_session=True, env=_clean_child_env())
    except Exception:
        pass


class Controller(NSObject):
    def initWithDemo_place_(self, demo, place):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        self.demo = bool(demo)
        self.place = bool(place)
        self.alpha = 0.0
        self.last_active = False
        self.last_voice = 0.0
        self.demo_full = "open the dashboard and show me the real margin"
        self.demo_i = 0
        self.t0 = time.time()
        self._restart_timer = None
        self._restart_ticks = 0
        self._build_window()
        self._build_menubar()
        if SHOW_PILL:
            self._build_robot()
        # Hardware-look widget: independent of SHOW_PILL (its own control flag),
        # so it can be the sole floating surface in menu-bar-only mode.
        self.widget_panel = None
        self.widget_view = None
        if _flag_on(WIDGET_FLAG):
            self._build_widget()
        # Push-to-talk chip in the notch: hold Control to dictate. Built
        # only when the engine runs in push-to-talk, so the chord stays free
        # for the user's own shortcuts otherwise.
        self.chip = None
        if not self.demo and not self.place and config.load().get("ptt", True):
            try:
                self.chip = NotchChip.alloc().initWithStateDir_soundDir_engineOn_(
                    str(STATE_DIR), str(SOUNDS_DIR), _engine_running)
            except Exception as err:
                sys.stderr.write(f"VibeVoice: notch chip unavailable: {err}\n")
        # Optional: come up already dictating when launchd-managed. Gated by env so
        # the default (manual toggle) is unchanged. The engine is spawned here in
        # the pill's GUI/TCC context, where the mic permission resolves correctly.
        if not self.demo and not self.place:
            _note_life("app start, autostart=%s, engine running=%s" % (
                os.environ.get("VIBEVOICE_ENGINE_AUTOSTART"), _engine_running()))
        if (not self.demo and not self.place
                and os.environ.get("VIBEVOICE_ENGINE_AUTOSTART") == "1"
                and not _engine_running()):
            _start_engine()
        return self

    def _build_robot(self):
        """Always-visible floating robot (functional stand-in for a status item
        that recent macOS can park off-screen behind the notch)."""
        scr = None
        for s in NSScreen.screens():
            try:
                if s.safeAreaInsets().top > 0:
                    scr = s
                    break
            except Exception:
                pass
        if scr is None:
            scr = NSScreen.mainScreen()
        sf = scr.frame()
        SZ = 34.0   # room for the glow ring + gradient body
        # position: saved, else top-right just below the menu bar
        x = sf.origin.x + sf.size.width - SZ - 14.0
        y = sf.origin.y + sf.size.height - SZ - 30.0
        try:
            sx, sy = ROBOT_POS.read_text().strip().split(",")
            x, y = float(sx), float(sy)
        except Exception:
            pass
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        rp = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, SZ, SZ), style, NSBackingStoreBuffered, False)
        rp.setOpaque_(False)
        rp.setBackgroundColor_(NSColor.clearColor())
        rp.setLevel_(NSStatusWindowLevel)
        rp.setHasShadow_(False)
        rp.setIgnoresMouseEvents_(False)
        rp.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces |
            NSWindowCollectionBehaviorStationary)
        rv = RobotView.alloc().initWithFrame_(NSMakeRect(0, 0, SZ, SZ))
        rv.owner = self
        rv.on = self._mic_is_on()
        rv.loop = _autosend_armed()
        rp.setContentView_(rv)
        rp.setAlphaValue_(1.0)
        rp.orderFrontRegardless()
        self.robot_panel = rp
        self.robot_view = rv

    def _build_widget(self):
        """Hardware-look floating voice widget (HardwareView). Same panel recipe
        as the robot; default berth bottom-center, Wispr-style."""
        scr = NSScreen.mainScreen()
        sf = scr.frame()
        x = sf.origin.x + (sf.size.width - WIDGET_W) / 2.0
        y = sf.origin.y + 84.0
        try:
            sx, sy = WIDGET_POS.read_text().strip().split(",")
            x, y = float(sx), float(sy)
            # The saved berth may belong to a monitor that is no longer attached.
            # Restoring it verbatim buries the widget off-screen, and the menu
            # toggle only re-orders the panel — it never moves it — so the only
            # cure was deleting widget_pos by hand.
            screens = []
            for s in (NSScreen.screens() or []):
                f = s.frame()
                screens.append((f.origin.x, f.origin.y, f.size.width, f.size.height))
            x, y = clamp_to_visible(x, y, WIDGET_W, WIDGET_H, screens)
        except Exception:
            pass
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        wp = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(x, y, WIDGET_W, WIDGET_H), style, NSBackingStoreBuffered, False)
        wp.setOpaque_(False)
        wp.setBackgroundColor_(NSColor.clearColor())
        wp.setLevel_(NSStatusWindowLevel)
        wp.setHasShadow_(True)   # a real drop shadow sells the "device on the desk"
        wp.setIgnoresMouseEvents_(False)
        wp.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces |
            NSWindowCollectionBehaviorStationary)
        wv = HardwareView.alloc().initWithFrame_(NSMakeRect(0, 0, WIDGET_W, WIDGET_H))
        wv.owner = self
        wv.engine_on = self._mic_is_on()
        wp.setContentView_(wv)
        wp.setAlphaValue_(1.0)
        wp.orderFrontRegardless()
        self.widget_panel = wp
        self.widget_view = wv

    def _build_window(self):
        # find the screen with the NOTCH (built-in), NOT mainScreen
        scr = None
        for s in NSScreen.screens():
            try:
                if s.safeAreaInsets().top > 0:
                    scr = s
                    break
            except Exception:
                pass
        if scr is None:
            scr = NSScreen.mainScreen()
        screen = scr.frame()
        try:
            notch = scr.safeAreaInsets().top
        except Exception:
            notch = 0.0
        if notch <= 0:
            notch = 38.0
        cx = screen.origin.x + screen.size.width / 2.0
        top = screen.origin.y + screen.size.height           # absolute top edge (flush with notch)
        # real notch width (menu-bar aux areas on the sides) for the collapsed footprint
        notch_w = 210.0
        try:
            la = scr.auxiliaryTopLeftArea()
            ra = scr.auxiliaryTopRightArea()
            nw = screen.size.width - la.size.width - ra.size.width
            if 120.0 < nw < 420.0:
                notch_w = nw
        except Exception:
            pass
        # expanded: full pill, top edge FLUSH with the screen edge → looks like the notch extended
        self.exp = (cx - PILL_W / 2.0, top - PILL_H, PILL_W, PILL_H)
        # collapsed: EXACT notch footprint (real width+height) → it "is" the notch
        self.col = (cx - notch_w / 2.0, top - notch, notch_w, notch)
        rect = NSMakeRect(self.col[0], self.col[1], self.col[2], self.col[3])
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, NSBackingStoreBuffered, False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setLevel_(NSStatusWindowLevel)
        panel.setHasShadow_(False)            # no shadow (pure black)
        panel.setIgnoresMouseEvents_(False)   # click on the copy icon (no drag)
        panel.setAcceptsMouseMovedEvents_(True)   # mouseMoved_/hover on a non-key panel
        panel.disableCursorRects()   # stop the window's auto cursor-rect reset (it would
                                     # reset to arrow on every mouseMoved right after my set)
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces |
            NSWindowCollectionBehaviorStationary)
        panel.setAlphaValue_(0.0)
        view = PillView.alloc().initWithFrame_(NSMakeRect(0, 0, PILL_W, PILL_H))
        view.setAutoresizingMask_(18)   # NSViewWidthSizable|HeightSizable — grows with the panel
        panel.setContentView_(view)
        if SHOW_PILL:
            panel.orderFrontRegardless()
        else:
            panel.orderOut_(None)    # menu-bar-only: keep the pill panel off-screen
        view.updateTrackingAreas()   # install the tracking area now (AppKit won't on a hand-built panel)
        self.panel = panel
        self.view = view

    def _make_robot_icon(self, on, loop, S=18.0):
        """Robot head with an antenna, two eyes and a mouth. States:
          mic OFF          → soft white outline, hollow eyes (sleeping)
          mic ON           → green Matrix eyes lit (listening)
          mic ON + loop 🔁 → amber eyes (listening AND auto-sending)
        Used both as the menu-bar icon and (scaled) by the floating widget."""
        img = NSImage.alloc().initWithSize_(NSMakeSize(S, S))
        img.lockFocus()
        if S != 18.0:
            # the drawing is calibrated at 18px → scale the context for other sizes
            from AppKit import NSAffineTransform
            t = NSAffineTransform.transform()
            t.scaleBy_(S / 18.0)
            t.concat()
        S = 18.0
        cx = S / 2.0
        WHITE = (1.0, 1.0, 1.0)
        GREEN = (0.12, 1.0, 0.32)
        AMBER_I = (1.0, 0.72, 0.05)
        body_a = 0.92 if on else 0.55

        NSColor.colorWithCalibratedRed_green_blue_alpha_(WHITE[0], WHITE[1], WHITE[2], body_a).set()
        # antenna: stalk + ball
        st = NSBezierPath.bezierPath()
        st.moveToPoint_(NSMakePoint(cx, 13.0))
        st.lineToPoint_(NSMakePoint(cx, 15.0))
        st.setLineWidth_(1.4)
        st.stroke()
        ball = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(cx - 1.4, 15.0, 2.8, 2.8))
        ball.fill()
        # head: rounded rect 14×10
        head = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(cx - 7.0, 3.0, 14.0, 10.0), 3.2, 3.2)
        head.setLineWidth_(1.5)
        head.stroke()
        # side ears
        for ex in (cx - 8.6, cx + 7.6):
            ear = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(ex, 6.2, 1.0, 3.6), 0.5, 0.5)
            ear.fill()
        # eyes: lit (green/amber) when mic on, hollow when off
        if on:
            ec = AMBER_I if loop else GREEN
            NSColor.colorWithCalibratedRed_green_blue_alpha_(ec[0], ec[1], ec[2], 1.0).set()
            for ox_ in (cx - 3.6, cx + 1.2):
                eye = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(ox_, 7.4, 2.4, 2.4))
                eye.fill()
            NSColor.colorWithCalibratedRed_green_blue_alpha_(WHITE[0], WHITE[1], WHITE[2], body_a).set()
        else:
            for ox_ in (cx - 3.6, cx + 1.2):
                eye = NSBezierPath.bezierPathWithOvalInRect_(NSMakeRect(ox_, 7.4, 2.4, 2.4))
                eye.setLineWidth_(1.0)
                eye.stroke()
        # mouth
        mo = NSBezierPath.bezierPath()
        mo.moveToPoint_(NSMakePoint(cx - 2.6, 5.2))
        mo.lineToPoint_(NSMakePoint(cx + 2.6, 5.2))
        mo.setLineWidth_(1.2)
        mo.stroke()
        img.unlockFocus()
        img.setTemplate_(False)   # NOT a template → keep the colour (template = mono white)
        return img

    def _build_menubar(self):
        # icon ALWAYS present in the menu bar → command center (even when mic is off)
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.status_item.button().setTitle_("")
        self.status_item.button().setImage_(self._make_robot_icon(self._mic_is_on(), _autosend_armed()))
        self.status_item.button().setToolTip_("VibeVoice — click for the menu")
        menu = NSMenu.alloc().init()
        # status header (disabled = label only)
        self.mb_status = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("VibeVoice", "", "")
        self.mb_status.setEnabled_(False)
        menu.addItem_(self.mb_status)
        menu.addItem_(NSMenuItem.separatorItem())
        # Push-to-talk menu (2026-10-31): the always-listening controls (voice
        # on/off, pill lock, hardware widget, restart pill) are gone; one pause
        # switch, auto-send as a checkmark, the last dictation one click away.
        self.mb_mute = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Pausa dettatura", "toggleMute:", "")
        self.mb_mute.setTarget_(self)
        menu.addItem_(self.mb_mute)
        self.mb_loop = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Invio automatico", "toggleSendReturn:", "")
        self.mb_loop.setTarget_(self)
        menu.addItem_(self.mb_loop)
        cp = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Copia l'ultima dettatura", "copyLast:", "")
        cp.setTarget_(self)
        menu.addItem_(cp)
        st = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Impostazioni…", "openSettings:", ",")
        st.setTarget_(self)
        menu.addItem_(st)
        menu.addItem_(NSMenuItem.separatorItem())
        ri = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Riavvia VibeVoice", "restartAll:", "")
        ri.setTarget_(self)
        menu.addItem_(ri)
        qi = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Esci", "quitAll:", "")
        qi.setTarget_(self)
        menu.addItem_(qi)
        # SF Symbols, template-rendered: they follow light/dark and the
        # highlight colour like the system's own menus (2026-10-31).
        # A missing symbol leaves the item text-only.
        for item, symbol in ((self.mb_status, "waveform"),
                             (self.mb_mute, "pause.circle"),
                             (self.mb_loop, "paperplane"),
                             (cp, "doc.on.doc"),
                             (st, "gearshape"),
                             (ri, "arrow.clockwise"),
                             (qi, "power")):
            _set_menu_symbol(item, symbol)
        # The checkmark is read from config.json each time the menu opens:
        # Settings can change the same switch while the menu is closed.
        menu.setDelegate_(self)
        self.menuNeedsUpdate_(menu)
        self.status_item.setMenu_(menu)
        self.mb_menu = menu   # reused by the floating robot (popup)
        self._mb_last = None

    def _mic_is_on(self):
        return _engine_running()

    def toggleMute_(self, sender):
        # 🔇 pause/resume the mic without killing the engine (the engine reads
        # the `muted` control file and ignores audio while it exists).
        _toggle_flag(MUTED_FILE)

    def menuNeedsUpdate_(self, menu):
        try:
            self.mb_loop.setState_(1 if config.load()["autosend_return"] else 0)
        except Exception:
            pass

    def toggleSendReturn_(self, sender):
        # «Invio automatico»: Return after each paste, owned by the engine.
        on = config.toggle_send_return()
        self.mb_loop.setState_(1 if on else 0)
        subprocess.Popen(["afplay", "/System/Library/Sounds/%s.aiff" % ("Tink" if on else "Submarine")],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.restartEngine_(sender)

    def copyLast_(self, sender):
        text = last_dictation_text(STATE_DIR / "history.jsonl")
        if not text:
            return
        try:
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(text, NSPasteboardTypeString)
        except Exception:
            pass

    def restartAll_(self, sender):
        # Engine running → kill and respawn it; engine down → just start it.
        if _engine_running():
            self.restartEngine_(sender)
        else:
            _start_engine()

    # Poll cadence/timeout for the post-kill respawn below: _stop_engine() fires
    # `pkill -f engine.py` async (no wait), so we can't just fire-once after a
    # fixed delay — if the old process is still exiting when the timer fires,
    # skipping the start would leave the engine down for good (no lang/env
    # update, silently). Instead poll until it's actually dead, bounded so a
    # wedged old process can't spawn a second engine on top of it.
    _RESTART_POLL_INTERVAL = 0.25
    _RESTART_MAX_TICKS = 20  # ~5s

    def restartEngine_(self, _sender=None):
        # Kill + respawn engine.py so it picks up the new env (lang/autosend/
        # autosend_return from config.json). No-op if the engine isn't running
        # right now — Settings shouldn't autostart the mic.
        if not _engine_running():
            return
        # Rapid successive settings changes must not stack timers.
        if getattr(self, "_restart_timer", None) is not None:
            self._restart_timer.invalidate()
            self._restart_timer = None
        _stop_engine()
        self._restart_ticks = 0
        self._restart_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self._RESTART_POLL_INTERVAL, self, "_pollRestartEngine:", None, True)

    def _pollRestartEngine_(self, timer):
        if not _engine_running():
            timer.invalidate()
            self._restart_timer = None
            _start_engine()
            return
        self._restart_ticks += 1
        if self._restart_ticks >= self._RESTART_MAX_TICKS:
            timer.invalidate()
            self._restart_timer = None
            print(
                "[vibevoice] restartEngine_: old engine.py still exiting after "
                "%.1fs, giving up (not starting a second engine)"
                % (self._RESTART_MAX_TICKS * self._RESTART_POLL_INTERVAL),
                file=sys.stderr,
            )

    _SET_W = 460            # settings window width
    _SET_HISTORY_TICK = 2.0  # history refresh cadence, only while on screen

    # ── settings window: layout constants (8pt rhythm) ───────────────────────
    _S_PAD      = 22        # window margin
    _S_TOP      = 62        # first section clears the transparent titlebar
    _S_ROW      = 36        # row with a title only
    _S_ROW_SUB  = 50        # row that also carries a subtitle
    _S_HEAD     = 28        # section header + its gap to the card
    _S_GAP      = 20        # card to the next section header
    _S_HIST     = 168       # history card
    _S_RADIUS   = 10.0

    def _s_section(self, parent, title, y):
        """Small-caps section label. Tracked, quiet, tertiary — a section label
        is signage, not a headline: the eye should find it and move on."""
        lbl = NSTextField.labelWithString_("")
        lbl.setAttributedStringValue_(NSAttributedString.alloc()
            .initWithString_attributes_(title.upper(), {
                NSFontAttributeName: NSFont.systemFontOfSize_weight_(10.0, 0.3),
                NSKernAttributeName: 1.1,
                NSForegroundColorAttributeName:
                    NSColor.colorWithCalibratedRed_green_blue_alpha_(0.12, 1.0, 0.32, 0.72),
            }))
        lbl.setFrame_(NSMakeRect(self._S_PAD + 2, y, self._SET_W - 2 * self._S_PAD, 14))
        lbl.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
        parent.addSubview_(lbl)

    def _s_card(self, parent, y, h):
        """A translucent card, never an opaque one: an opaque panel would cover
        the glass the window is made of and put us back where we started."""
        card = NSView.alloc().initWithFrame_(
            NSMakeRect(self._S_PAD, y, self._SET_W - 2 * self._S_PAD, h))
        card.setWantsLayer_(True)
        card.layer().setCornerRadius_(self._S_RADIUS)
        card.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.055).CGColor())
        card.layer().setBorderWidth_(1.0)
        card.layer().setBorderColor_(
            NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.085).CGColor())
        card.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
        parent.addSubview_(card)
        return card

    def _s_divider(self, card, y):
        """Hairline inset from both edges, the way grouped rows separate on this
        platform — a full-bleed line reads as a cut, an inset one as a seam."""
        d = NSView.alloc().initWithFrame_(
            NSMakeRect(14, y, card.frame().size.width - 28, 1))
        d.setWantsLayer_(True)
        d.layer().setBackgroundColor_(
            NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.07).CGColor())
        d.setAutoresizingMask_(NSViewWidthSizable)
        card.addSubview_(d)

    def _s_row(self, card, title, subtitle, y, h):
        """One setting per row: title, and the explanation UNDER it in muted type.
        The old form said everything twice — "Auto-paste" beside "paste the
        transcription" — because the caption was the checkbox's own label. The
        control carries no words now, so the sentence is free to be useful."""
        w = card.frame().size.width
        t = NSTextField.labelWithString_(title)
        t.setFont_(NSFont.systemFontOfSize_(13.0))
        t.setTextColor_(NSColor.labelColor())
        t.setFrame_(NSMakeRect(14, y + h - (24 if subtitle else 26), w - 100, 18))
        t.setAutoresizingMask_(NSViewWidthSizable)
        card.addSubview_(t)
        if subtitle:
            s = NSTextField.labelWithString_(subtitle)
            s.setFont_(NSFont.systemFontOfSize_(11.0))
            s.setTextColor_(NSColor.secondaryLabelColor())
            s.setFrame_(NSMakeRect(14, y + h - 42, w - 100, 16))
            s.setAutoresizingMask_(NSViewWidthSizable)
            card.addSubview_(s)

    def _s_switch(self, card, on, y, h):
        """NSSwitch, not a checkbox. Same state API the tests already drive, but
        a checkbox reads as a form to fill in and a switch reads as a setting
        that is on — which is what every one of these actually is."""
        sw = NSSwitch.alloc().initWithFrame_(
            NSMakeRect(card.frame().size.width - 52, y + (h - 22) / 2.0, 38, 22))
        sw.setState_(1 if on else 0)
        sw.setTarget_(self)
        sw.setAction_("settingsChanged:")
        sw.setAutoresizingMask_(NSViewMinXMargin)
        card.addSubview_(sw)
        return sw

    def openSettings_(self, _sender):
        if getattr(self, "_settings_win", None):
            self._settings_win.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            self._start_history_timer()
            return

        cfg = config.load()
        W = self._SET_W
        # The layout is declared, then measured, then drawn. Height used to be a
        # sum of magic numbers; now the same list that draws the window is the one
        # that sizes it, so a new row cannot silently overflow the frame.
        T = settings_text(cfg["lang"])
        self._s_empty = T["empty"]
        self._s_lang_code = cfg["lang"]
        SECTIONS = [
            (T["voice"], [T["lang"] + ("lang",), T["vp"] + ("vp",)]),
            (T["behaviour"], [T["as"] + ("as",), T["ar"] + ("ar",)]),
            (T["appearance"], [T["dk"] + ("dk",)]),
        ]

        def card_h(rows):
            return sum(self._S_ROW_SUB if sub else self._S_ROW for _, sub, _ in rows) \
                + (len(rows) - 1)          # hairlines

        # Measure with the SAME step the drawing loop walks. They used to be two
        # arithmetics for one layout — the window came out 52pt taller than its
        # content and the surplus showed as dead space under the history.
        step = self._S_HEAD + self._S_GAP - 8      # card bottom -> next card top
        H = self._S_TOP + self._S_HIST + self._S_PAD
        for _, rows in SECTIONS:
            H += card_h(rows) + step

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H),
            NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable
            | NSWindowStyleMaskFullSizeContentView,
            NSBackingStoreBuffered, False)
        win.setTitle_(T["title"])
        win.setMinSize_(NSMakeSize(W, H))   # sotto il contenuto non si scende
        win.center()
        win.setReleasedWhenClosed_(False)
        # Liquid glass: the window IS the material. Behind-window vibrancy under a
        # transparent titlebar; the cards above it are translucent too, so the
        # material reads through the whole surface instead of being covered by it.
        # The "Reduce transparency" fallback is the system's own: NSVisualEffectView
        # goes opaque by itself. Dark is pinned on purpose — this is the Matrix
        # pill's chrome, one committed look, no light-mode contrast roulette.
        win.setTitlebarAppearsTransparent_(True)
        win.setAppearance_(NSAppearance.appearanceNamed_(NSAppearanceNameDarkAqua))
        glass = NSVisualEffectView.alloc().initWithFrame_(NSMakeRect(0, 0, W, H))
        glass.setBlendingMode_(NSVisualEffectBlendingModeBehindWindow)
        # Menu, not HUDWindow. Measured on this desktop over a live dashboard —
        # colour variance inside an empty strip of the window, which is exactly
        # "how much of what is behind survives the glass":
        #   menu 37.2 · popover 17.8 · sidebar 14.5 · HUDWindow 13.8
        # HUDWindow, the obvious pick by name, is the most OPAQUE of the five: it
        # renders as a flat dark panel and the effect reads as no effect at all.
        # Menu is the dark smoked glass that actually refracts, and dark suits the
        # Matrix chrome. Re-measure before changing this, do not pick by name.
        glass.setMaterial_(NSVisualEffectMaterialMenu)
        glass.setState_(NSVisualEffectStateFollowsWindowActiveState)
        glass.setEmphasized_(True)
        win.setContentView_(glass)
        v = glass

        made = {}
        y = H - self._S_TOP
        for title, rows in SECTIONS:
            self._s_section(v, title, y + 8)
            ch = card_h(rows)
            y -= ch
            card = self._s_card(v, y, ch)
            ry = ch                          # walk down inside the card
            for i, (label, sub, key) in enumerate(rows):
                h = self._S_ROW_SUB if sub else self._S_ROW
                ry -= h
                self._s_row(card, label, sub, ry, h)
                if key == "lang":
                    pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
                        NSMakeRect(card.frame().size.width - 170, ry + (h - 24) / 2.0,
                                   156, 24), False)
                    pop.addItemsWithTitles_(
                        [f"{flag}  {name}" for _, flag, name in _S_LANGS])
                    codes = [c for c, _, _ in _S_LANGS]
                    pop.selectItemAtIndex_(
                        codes.index(cfg["lang"]) if cfg["lang"] in codes else 0)
                    pop.setTarget_(self)
                    pop.setAction_("settingsChanged:")
                    pop.setAutoresizingMask_(NSViewMinXMargin)
                    card.addSubview_(pop)
                    made[key] = pop
                else:
                    made[key] = self._s_switch(card, cfg[{"vp":"vp","as":"autosend","ar":"autosend_return","dk":"dock"}[key]], ry, h)
                if i < len(rows) - 1:
                    ry -= 1
                    self._s_divider(card, ry)
            y -= step

        # History: the one surface that is content, not chrome. Its own darker
        # berth and full-contrast type, because dense text must be read, not
        # admired through glass.
        self._s_section(v, T["history"], y + 8)
        clear = NSButton.buttonWithTitle_target_action_(T["clear"], self, "clearHistory:")
        clear.setFont_(NSFont.systemFontOfSize_(11.0))
        clear.setBezelStyle_(1)              # NSBezelStyleRounded
        clear.setFrame_(NSMakeRect(W - self._S_PAD - 62, y + 2, 62, 20))
        clear.setAutoresizingMask_(NSViewMinXMargin | NSViewMinYMargin)
        v.addSubview_(clear)

        y -= self._S_HIST
        sc = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(self._S_PAD, y, W - 2 * self._S_PAD, self._S_HIST))
        tv = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, W - 2 * self._S_PAD, self._S_HIST))
        tv.setEditable_(False)
        sc.setBorderType_(0)                 # NSNoBorder
        sc.setDrawsBackground_(False)
        sc.setWantsLayer_(True)
        sc.layer().setCornerRadius_(self._S_RADIUS)
        sc.layer().setMasksToBounds_(True)
        tv.setDrawsBackground_(True)
        tv.setBackgroundColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.30))
        tv.setTextColor_(NSColor.labelColor())
        tv.setFont_(NSFont.monospacedDigitSystemFontOfSize_weight_(11.0, 0.0))
        tv.setTextContainerInset_(NSMakeSize(10, 10))
        sc.setDocumentView_(tv)
        sc.setHasVerticalScroller_(True)
        sc.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
        v.addSubview_(sc)

        self._set_lang = made["lang"]
        self._set_vp = made["vp"]
        self._set_as = made["as"]
        self._set_ar = made["ar"]
        self._set_dk = made["dk"]
        self._set_hist = tv
        self._settings_win = win
        self._reload_history()
        win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        self._start_history_timer()

    # ── history pane ──────────────────────────────────────────────────────────

    def _start_history_timer(self):
        """Refresh the list while the window is up. Dictating with Settings open used
        to leave a list frozen at whatever it said when the window was created."""
        if getattr(self, "_hist_timer", None) is not None:
            return
        self._hist_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            self._SET_HISTORY_TICK, self, "historyTick:", None, True)

    def _stop_history_timer(self):
        timer = getattr(self, "_hist_timer", None)
        if timer is not None:
            timer.invalidate()
            self._hist_timer = None

    def historyTick_(self, _timer):
        # Self-invalidating instead of relying on a window delegate: a closed
        # Settings window must not cost a wakeup every two seconds forever.
        win = getattr(self, "_settings_win", None)
        if win is None or not win.isVisible():
            self._stop_history_timer()
            return
        self._reload_history()

    def clearHistory_(self, _sender):
        """Truncate history.jsonl. Allowed: invariant #1 names `state`, `levels.bin`
        and `raw.txt` — not this file. The engine appends, so the next utterance
        simply starts a fresh list."""
        try:
            (STATE_DIR / "history.jsonl").write_text("")
        except OSError:
            pass
        self._reload_history()

    def _reload_history(self):
        import json

        rows = []
        try:
            lines = (STATE_DIR / "history.jsonl").read_text().splitlines()
        except OSError:
            lines = []
        for ln in reversed(lines):
            if not ln.strip():
                continue
            # Per line, not around the loop: the previous form wrapped the whole
            # read in one try/except and blanked `rows`, so a single torn line
            # erased every good line above it too.
            try:
                record = json.loads(ln)
            except ValueError:
                continue
            shown = format_history_line(record)
            if shown:
                rows.append(shown)
        self._set_hist.setString_("\n".join(rows) if rows else
                                  getattr(self, "_s_empty", "(no transcriptions yet)"))

    def settingsChanged_(self, _sender):
        # Pure wiring: read the controls, hand them to apply_settings, obey the answer.
        # The decision deliberately does NOT live here — a selector cannot be driven
        # from a test (PyObjC demands a real Objective-C `self`), and this is precisely
        # where the "restart the engine for every change" defect hid unnoticed.
        lang = _S_LANGS[max(0, int(self._set_lang.indexOfSelectedItem()))][0]
        restart = apply_settings({
            "lang": lang,
            "autosend": bool(self._set_as.state()),
            "autosend_return": bool(self._set_ar.state()),
            "dock": bool(self._set_dk.state()),
            "vp": bool(self._set_vp.state()),
        })
        if lang != getattr(self, "_s_lang_code", lang):
            # The window is written in the language you dictate in, so changing
            # that language has to rewrite the window. Rebuilt rather than
            # relabelled: every row would otherwise need a handle kept alive
            # purely to be renamed once in a while.
            self._rebuild_settings()
        if restart:
            self.restartEngine_(None)

    def _rebuild_settings(self):
        win = getattr(self, "_settings_win", None)
        if win is None:
            return
        self._stop_history_timer()
        frame = win.frame()
        win.orderOut_(None)
        self._settings_win = None
        self.openSettings_(None)
        new_win = getattr(self, "_settings_win", None)
        if new_win is not None:
            # Keep the corner where the user left it; the height is the layout's
            # to decide, and it can differ between languages.
            new_win.setFrameTopLeftPoint_(
                NSMakePoint(frame.origin.x, frame.origin.y + frame.size.height))

    def quitAll_(self, sender):
        _stop_engine()
        NSApp.terminate_(self)

    # NSApplicationDelegate. The menu item above is only ONE way out. Cmd+Q,
    # the Dock's Quit and a logout all go straight to NSApp.terminate_ without
    # passing through it, and the engine is spawned with start_new_session=True
    # — so it survived as an orphan (PPID 1), still holding the microphone and
    # still pasting what it heard. Measured 2026-09-27: the app was closed, the
    # menu-bar mic indicator still read "VibeVoice", and speech still landed on
    # screen. A quit the user can see must be a quit the microphone can feel.
    def applicationWillTerminate_(self, notification):
        _note_life("app quit (terminate)")
        if getattr(self, "chip", None) is not None:
            self.chip.shutdown()
        _stop_engine()

    # ── live sources ──
    def _read_levels(self):
        try:
            data = LEVELS_BIN.read_bytes()
            if len(data) < 60 * 4:          # torn/partial read → skip the frame
                return None
            vals = struct.unpack("<60f", data[:60 * 4])
            out = []
            for i in range(N_BARS):
                j = int(i * 60 / N_BARS)
                v = abs(vals[j]) * GAIN                 # higher sensitivity
                out.append(min(1.0, v ** 0.6))          # perceptual curve: quiet voice = more visible
            return out
        except Exception:
            return None

    def _read_raw_energy(self):
        # raw RMS (no GAIN) for immediate voice onset
        try:
            data = LEVELS_BIN.read_bytes()
            if len(data) < 60 * 4:
                return 0.0
            vals = struct.unpack("<60f", data[:60 * 4])
            return max(abs(v) for v in vals)
        except Exception:
            return 0.0

    def _read_state(self):
        try:
            return STATE_FILE.read_text().strip()
        except Exception:
            return ""

    def _read_text(self):
        try:
            lines = RAW_TXT.read_text().strip().splitlines()
            return lines[-1].strip() if lines else ""
        except Exception:
            return ""

    # A draft older than this was left behind by an engine that died without
    # running its handler (SIGKILL, crash, OOM). Comfortably longer than
    # MAX_DUR + SILENCE_SEC, so a slow real utterance is never mistaken for one.
    _PARTIAL_STALE_S = 25.0

    def _read_partial(self):
        """The engine's live draft, or None when no utterance is in flight.

        The distinction matters: absent = fall back to the last finished
        sentence; present-but-empty = you ARE speaking and nothing is stable
        yet, so show nothing rather than the previous dictation.

        A draft that has stopped changing is treated as absent. An engine killed
        mid-utterance leaves both `state = recording` and `partial.txt` behind,
        and the pill then refreshes `last_voice` on every tick, so it never
        fades: a red LED, frozen VU bars and a dead draft at 24 fps with no
        engine alive, curable only by killing the pill. Age is the honest
        signal — `_engine_running()` is `pgrep -f engine.py`, which matches any
        long-lived process with that string in argv, including an editor open on
        the file (review 2026-08-02).
        """
        try:
            if time.time() - PARTIAL_TXT.stat().st_mtime > self._PARTIAL_STALE_S:
                return None
            return PARTIAL_TXT.read_text().strip()
        except OSError:
            return None
        except Exception:
            return None

    def _tts_speaking(self):
        """Return the slice of the sentence already spoken by an external TTS (a
        typewriter synced to the audio), or '' when not speaking / feature off.
        TTS_TEXT format (any TTS writes it during playback, clears it at the end):
          line 1: '<start_epoch> <duration_s>'
          line 2+: the full clean sentence
        Reveals text[:progress*len] where progress = (now-start)/duration. The
        presence of TTS_FLAG enables the feature. Back-compat: a legacy single
        line returns the whole text."""
        if not TTS_FLAG.exists():
            return ""
        try:
            raw = TTS_TEXT.read_text()
        except Exception:
            return ""
        if not raw.strip():
            return ""
        lines = raw.split("\n", 1)
        try:
            parts = lines[0].split()
            start = float(parts[0])
            dur = float(parts[1])
            full = lines[1].rstrip("\n") if len(lines) > 1 else ""
            if not full:
                return ""
            if dur <= 0.01:
                return full
            prog = max(0.0, min(1.0, (time.time() - start) / dur))
            n = max(1, int(round(prog * len(full))))
            return full[:n]
        except (ValueError, IndexError):
            return raw.strip()   # legacy single line → all of it

    def _read_tts_levels(self):
        """Waveform of the real TTS audio (rolling RMS, 60 float32 LE written by
        the external TTS). Mapped to N_BARS like the live waveform. Returns None
        when absent/torn (→ the caller uses a synthetic fallback)."""
        try:
            data = TTS_LEVELS.read_bytes()
            if len(data) < 60 * 4:
                return None
            vals = struct.unpack("<60f", data[:60 * 4])
            return [max(0.04, min(1.0, abs(vals[int(i * 60 / N_BARS)]))) for i in range(N_BARS)]
        except Exception:
            return None

    def _animate_(self, show):
        # Dynamic Island: EXPANDS from the notch / RE-COLLAPSES into it.
        # the collapsed snap stays OUTSIDE the grouping, otherwise it jumps a frame.
        if not SHOW_PILL:
            return                    # menu-bar-only: the pill panel never surfaces
        if show:
            self.panel.setFrame_display_(NSMakeRect(*self.col), True)
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.34)
        if show:
            self.panel.animator().setAlphaValue_(1.0)
            self.panel.animator().setFrame_display_(NSMakeRect(*self.exp), True)
        else:
            self.panel.animator().setAlphaValue_(0.0)
            self.panel.animator().setFrame_display_(NSMakeRect(*self.col), True)
        NSAnimationContext.endGrouping()

    def tick_(self, timer):
        self.view.phase = time.time() - self.t0
        # control panel state (cheap stat every tick)
        self.view.muted = _flag_on(MUTED_FILE)
        self.view.locked = _flag_on(LOCKED_FILE)
        engine_on = _engine_running_cached()
        # adaptive polling: while the pill is HIDDEN, work 1 tick in IDLE_SKIP
        # (~7.5 fps) instead of 60. Cuts idle CPU. Visible → full 60 fps. Onset
        # latency ~133ms max. Compute the TTS-speaking slice once; if it speaks,
        # don't idle-skip (smooth typewriter).
        tspk = "" if (self.demo or self.place) else self._tts_speaking()
        if (not self.demo and not self.place and not self.last_active
                and not tspk and not self.view.locked):
            self._idle_skip = (getattr(self, "_idle_skip", 0) + 1) % IDLE_SKIP
            if self._idle_skip:
                return
        if self.demo or self.place:
            active = True
            ph = self.view.phase
            levels = []
            for i in range(N_BARS):
                base = 0.5 + 0.45 * math.sin(ph * 5.0 + i * 0.5)
                env = 0.5 + 0.5 * math.sin(ph * 1.3)
                levels.append(max(0.04, base * env * (0.6 + 0.4 * random.random())))
            # typewriter
            if int(ph * 12) > self.demo_i and self.demo_i < len(self.demo_full):
                self.demo_i += 1
            if self.demo_i >= len(self.demo_full) and ph % 6 < 0.1:
                self.demo_i = 0
            text = self.demo_full[: self.demo_i]
            if self.place:
                text = "↔ drag me anywhere · position is saved"
        elif tspk:
            # TTS is speaking → pill turns RED. Waveform from the REAL TTS audio
            # (the TTS writes the rolling RMS); the text types out in sync. The
            # mic is HW-muted (anti-echo), so the waves come from the TTS audio,
            # not the mic: same rhythm as the speech, zero loop risk.
            self.view.tint = RED
            lv = self._read_tts_levels()
            if lv is not None:
                self._tts_last = lv
                levels = lv
            elif getattr(self, "_tts_last", None) is not None:
                levels = self._tts_last       # HOLD the last real frame → no mid-speech flicker
            else:
                # only the first ~100ms (audio not decoded yet) → soft synthetic wave
                ph = self.view.phase
                levels = [max(0.04, (0.5 + 0.45 * math.sin(ph * 5.0 + i * 0.5))
                                    * (0.5 + 0.5 * math.sin(ph * 1.3))
                                    * (0.6 + 0.4 * random.random())) for i in range(N_BARS)]
            text = tspk
            active = True
            self.last_voice = time.time()   # keep the pill alive while it speaks
        else:
            self.view.tint = MATRIX   # dictation input → green
            self._tts_last = None      # reset the TTS waveform hold (next turn starts clean)
            # if the mic is OFF (engine dead) don't trust levels.bin: it stays
            # frozen on the last frame, whose residual energy can read > threshold
            # → the pill would believe itself eternally active, redraw at 24 fps,
            # and stay on screen. Mic off = no voice → goes idle → hides → CPU ~0.
            # 🔇 MUTE = full pause: treat as mic-off (flat waves, no false-active).
            mic_off = (not engine_on) or self.view.muted
            state = "" if mic_off else self._read_state()
            levels = self._read_levels()
            if levels is None or mic_off:
                levels = [0.04] * N_BARS
            raw = 0.0 if mic_off else self._read_raw_energy()
            # IMMEDIATE onset: appears as soon as voice clears the noise floor (raw
            # RMS), or when the engine is recording/transcribing.
            if raw > VOICE_THRESH or state in ("recording", "transcribing"):
                self.last_voice = time.time()
            # While the utterance is still open the LIVE draft wins: words appear
            # as they are spoken instead of after the trailing silence. Showing
            # the previous sentence during a new dictation would be a lie.
            live = self._read_partial() if state == "recording" else None
            text_now = live if live is not None else self._read_text()
            hold = 2.5 if text_now else IDLE_HIDE_S   # with text it stays visible to click copy
            active = (time.time() - self.last_voice) <= hold
            text = text_now if active else ""

        # 🔒 lock pins the pill visible regardless of silence.
        if self.view.locked:
            active = True
        # show/hide transition → native AppKit animation (fade + slide from the
        # notch), triggered ONCE on state change. The waveform updates every tick.
        if active != self.last_active:
            self._animate_(active)
            self.last_active = active
        # icon hover (poll the mouse without an event stream) — before the redraw
        try:
            mloc = self.panel.mouseLocationOutsideOfEventStream()
            vb = self.view.bounds()
            ww, hh = vb.size.width, vb.size.height
            self.view.hover_x = bool(mloc.x >= ww - 40.0 and mloc.y >= hh - 34.0)
            cr = self.view.copy_rect
            self.view.hover_c = bool(cr and cr[0] <= mloc.x <= cr[0] + cr[2]
                                     and cr[1] <= mloc.y <= cr[1] + cr[3])
            mr = self.view.muta_rect
            self.view.hover_m = bool(mr and mr[0] <= mloc.x <= mr[0] + mr[2]
                                     and mr[1] <= mloc.y <= mr[1] + mr[3])
            lr = self.view.blocca_rect
            self.view.hover_l = bool(lr and lr[0] <= mloc.x <= lr[0] + lr[2]
                                     and lr[1] <= mloc.y <= lr[1] + lr[3])
            orr = self.view.loop_rect
            self.view.hover_o = bool(orr and orr[0] <= mloc.x <= orr[0] + orr[2]
                                     and orr[1] <= mloc.y <= orr[1] + orr[3])
            # the pointing-hand cursor is handled by PillView.cursorUpdate_
            # (NSTrackingArea), not here.
        except Exception:
            pass
        self.view.setLevels_text_active_(levels, text, active)
        # hardware widget: mirror the same tick data (its own redraw, cheap)
        if getattr(self, "widget_view", None) is not None and _flag_on(WIDGET_FLAG):
            try:
                wv = self.widget_view
                wv.levels = list(levels)
                wv.engine_on = engine_on or self.demo or self.place
                wv.muted = self.view.muted
                wv.return_held = _flag_on(RETURN_HELD_FILE)
                wv.state = ("recording" if (self.demo or self.place)
                            else (self._read_state() if engine_on else ""))
                wv.phase = self.view.phase
                wv.setNeedsDisplay_(True)
            except Exception:
                pass
        # SELF-HEAL the status item: on recent macOS an item created before
        # launch-finish can be parked off-screen (y<0, screen=None). Detect it →
        # destroy and recreate the item at runtime (max 3 attempts, every ~3s).
        # The floating robot covers the gap meanwhile.
        if time.time() - self.t0 > 3.0 and getattr(self, "_mb_heal", 0) < 3:
            try:
                win = self.status_item.button().window()
                fr = win.frame()
                parked = (win.screen() is None) or (fr.origin.y < 0)
                if parked:
                    self._mb_heal = getattr(self, "_mb_heal", 0) + 1
                    NSStatusBar.systemStatusBar().removeStatusItem_(self.status_item)
                    self._build_menubar()
                    self.t0 = time.time()   # give the new item 3s of grace
                else:
                    self._mb_heal = 3   # healthy: stop checking
            except Exception:
                self._mb_heal = getattr(self, "_mb_heal", 0) + 1
        # menu bar: icon + control center update only on a state change
        on = engine_on
        state_now = (on, self.view.muted, self.view.locked, self.view.autoloop)
        if state_now != self._mb_last:
            self._mb_last = state_now
            self.status_item.button().setImage_(self._make_robot_icon(on, self.view.autoloop))
            # floating robot: same state, same icon
            if getattr(self, "robot_view", None) is not None:
                self.robot_view.on = on
                self.robot_view.loop = self.view.autoloop
                self.robot_view.setNeedsDisplay_(True)
            if self.view.muted:
                st = "in pausa"
            elif on:
                st = "pronto · tieni premuto Control"
            else:
                st = "motore spento · Riavvia"
            self.mb_status.setTitle_("VibeVoice — %s" % st)
            # Paused: the item says what a click does, with the play symbol.
            self.mb_mute.setTitle_("Riprendi dettatura" if self.view.muted else "Pausa dettatura")
            _set_menu_symbol(self.mb_mute, "play.circle" if self.view.muted else "pause.circle")


def _apply_app_identity(app) -> None:
    """Give the running process VibeVoice's icon AND name in the Dock.

    `CFBundleIconFile` / `CFBundleName` in the .app only apply when the *bundle*
    is launched. The LaunchAgent runs `python3 vibevoice.py`, so the process
    inherits the identity of Homebrew's `Python.app` — a rocket labelled
    "Python". Both halves need fixing and they need different fixes:

    * the icon is a property of the running NSApplication → set the image;
    * the name is read from the main bundle's info dictionary → patch that
      dictionary. It is mutable in place, and this is the same approach the
      menu-bar libraries use. It must happen before AppKit reads the name for
      the Dock tile, so this runs early in `main()`.

    A no-op for the real bundle, which already declares both. Cosmetic by
    definition: any failure here must leave dictation untouched.
    """
    try:
        icon = Path(__file__).resolve().parent / "assets" / "icon" / "VibeVoice_LED.icns"
        if icon.exists():
            image = NSImage.alloc().initWithContentsOfFile_(str(icon))
            if image is not None:
                app.setApplicationIconImage_(image)
    except Exception:
        pass
    try:
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        if info is not None:
            info["CFBundleName"] = "VibeVoice"
            info["CFBundleDisplayName"] = "VibeVoice"
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="VibeVoice — Dynamic Island STT pill (MIT)")
    ap.add_argument("--demo", action="store_true", help="animated demo to preview the design")
    ap.add_argument("--place", action="store_true", help="placement mode: stays visible")
    args = ap.parse_args()

    _ensure_state_dir()

    global _CTRL, _TIMER
    app = NSApplication.sharedApplication()
    cfg = config.load()
    app.setActivationPolicy_(
        NSApplicationActivationPolicyRegular if cfg.get("dock", True)
        else NSApplicationActivationPolicyAccessory)
    _apply_app_identity(app)
    _CTRL = Controller.alloc().initWithDemo_place_(args.demo, args.place)
    # Without a delegate, applicationWillTerminate_ is never delivered and the
    # engine outlives every quit that is not our own menu item.
    app.setDelegate_(_CTRL)
    _TIMER = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        TICK, _CTRL, "tick:", None, True)
    # Zero tolerance: the run loop is otherwise free to coalesce this timer with
    # other work, and a frame delivered late-then-early is judder even when the
    # average rate is right. Cadence alone does not buy smoothness.
    _TIMER.setTolerance_(0.0)
    NSApp.run()


if __name__ == "__main__":
    main()
