#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The push-to-talk chip in the notch (2026-10-30).

Hold Control (alone, since 30/10): the notch grows two black wings — a red dot on the left,
the live voice on the right — and a soft rising cue plays. Let go: a falling
cue, the dot turns amber while the engine transcribes, then a bright tick and a
green dot when the text is in the terminal (or a low cue, grey, when nothing was
heard). Then the wings fold back into the notch.

Lives in the pill process (imported by vibevoice.py only). It talks to the
engine through the same file contract as everything else: it WRITES the `ptt`
control file while the chord is held and READS `state`, `levels.bin` and
`raw.txt`. The chord is read from `NSEvent.modifierFlags()` — the modifier state,
no grant needed. Control alone is also the first half of every terminal
shortcut (Ctrl+C, Ctrl+R…), so a global key/click monitor watches for any other
key while Control is held and cancels the talk before it starts; it only tells
"a key went down", never which one, and without the Accessibility grant it
simply stays silent and the hold threshold alone protects the shortcuts.

The top half of this file is pure logic, tested headless in
tests/test_notch_chip.py; the bottom half is the AppKit surface.
"""
from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

# ── pure logic ────────────────────────────────────────────────────────────────

# NSEvent modifier masks (stable AppKit values, spelled out so the pure half
# needs no AppKit import).
FLAG_SHIFT = 1 << 17
FLAG_CONTROL = 1 << 18
FLAG_OPTION = 1 << 19
FLAG_COMMAND = 1 << 20
_CHORD_MASK = FLAG_SHIFT | FLAG_CONTROL | FLAG_OPTION | FLAG_COMMAND

HOLD_S = 0.25            # Control held shorter than this is a shortcut, not a talk
RESULT_TIMEOUT = 20.0    # an engine that never answers does not keep the chip up
EMPTY_GRACE_S = 1.0      # idle right after release is the tail, not the verdict
WING = 74.0              # width of each wing beside the notch
BIG_WING = 150.0         # the missed-paste card: wider…
BIG_EXTRA_H = 64.0       # …and hanging this far below the notch
LIVE_EXTRA_H = 34.0      # the live-text strip under the notch while you speak
REPORT_WAIT_S = 1.5      # after raw.txt changes, how long the paste report may take


def is_chord(flags: int) -> bool:
    """Control and nothing else — with Option, Shift or Command it is someone's
    shortcut (Control+Option belongs to another app)."""
    return (int(flags) & _CHORD_MASK) == FLAG_CONTROL


class PttGate:
    """Turns the polled chord state into one `press` and one `release`.

    `press` fires once the chord has been held for `hold` seconds; a release
    before that is silent. `cancel()` — another key or a click while Control is
    down — blocks this hold for good: Ctrl+C held a little long is still Ctrl+C.
    After `press` a cancel changes nothing: the talk is already on.
    """

    def __init__(self, hold: float = HOLD_S) -> None:
        self.hold = hold
        self.since: float | None = None
        self.active = False
        self.blocked = False

    def cancel(self) -> None:
        if not self.active and self.since is not None:
            self.blocked = True

    def update(self, down: bool, now: float) -> str | None:
        if down:
            if self.since is None:
                self.since = now
            if self.blocked:
                return None
            if not self.active and now - self.since >= self.hold:
                self.active = True
                return "press"
            return None
        self.since = None
        self.blocked = False
        if self.active:
            self.active = False
            return "release"
        return None


def chip_rect(sx: float, sy: float, sw: float, sh: float,
              notch_w: float, notch_h: float, wing: float) -> tuple:
    """(x, y, w, h) of the chip: centred on the notch, flush with the top edge,
    `wing` points wider on each side (0 = folded into the notch)."""
    w = notch_w + 2.0 * wing
    return (sx + (sw - w) / 2.0, sy + sh - notch_h, w, notch_h)


def chip_bars(levels, n: int = 5, gain: float = 12.0) -> list:
    """The last `n` RMS values as bar heights 0..1, newest on the right."""
    if not levels:
        return [0.0] * n
    tail = list(levels)[-n:]
    return [min(1.0, (abs(v) * gain) ** 0.6) for v in tail]


def release_outcome(raw_mtime_before: float, raw_mtime_now: float, state: str,
                    elapsed: float) -> str | None:
    """After release: "sent" once a new sentence reached raw.txt, "empty" when the
    engine is idle again with nothing new (or never answers), None while working."""
    if raw_mtime_now and raw_mtime_now != raw_mtime_before:
        return "sent"
    if state == "idle" and elapsed >= EMPTY_GRACE_S:
        return "empty"
    if elapsed >= RESULT_TIMEOUT:
        return "empty"
    return None


MISSED_REASONS = {
    "paste": "la tastiera finta non è partita",
    "secure": "un campo protetto blocca la tastiera",
    "vibevoice": "era in primo piano VibeVoice",
    "nofocus": "nessun campo di testo in primo piano",
}
MISSED_LINGER_S = 15.0   # the offer stays this long unless used or closed


def reason_label(reason: str) -> str:
    if reason.startswith("role:"):
        reason = "nofocus"
    return MISSED_REASONS.get(reason, "non sono sicuro che sia arrivato")


def read_report(path, since: float):
    """The engine's last paste report if written after `since`, else None."""
    import json
    try:
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        return rep if isinstance(rep, dict) and float(rep.get("ts", 0)) >= since else None
    except Exception:
        return None


def read_missed(path, since: float):
    """(text, reason label) when the last paste report, written after `since`,
    says the text did not land; None otherwise (landed, older, absent, torn)."""
    rep = read_report(path, since)
    if rep is None or rep.get("ok") or not rep.get("text"):
        return None
    return str(rep["text"]), reason_label(str(rep.get("reason", "")))


def preview(text: str, limit: int) -> str:
    """The END of a long dictation (the part just spoken), after an ellipsis."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    tail = text[-(limit - 1):]
    if " " in tail:
        tail = tail[tail.index(" ") + 1:]
    return "…" + tail


# ── AppKit surface ────────────────────────────────────────────────────────────

import objc  # noqa: E402
from AppKit import (  # noqa: E402
    NSAnimationContext, NSBackingStoreBuffered, NSBezierPath, NSColor, NSEvent,
    NSMakePoint, NSMakeRect, NSPanel, NSScreen, NSSound, NSStatusWindowLevel,
    NSTimer, NSView, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
    NSWindowStyleMaskNonactivatingPanel, NSFont, NSFontAttributeName,
    NSForegroundColorAttributeName, NSPasteboard, NSPasteboardTypeString,
)
from Foundation import NSAttributedString  # noqa: E402
from Foundation import NSObject  # noqa: E402

RED = (1.00, 0.23, 0.19)
AMBER = (1.00, 0.66, 0.18)
GREEN = (0.12, 1.00, 0.32)
GREY = (0.55, 0.55, 0.58)
TICK_S = 1.0 / 60.0
LINGER_S = 0.55          # the verdict stays visible this long before folding


def _notch_screen():
    for s in NSScreen.screens() or []:
        try:
            if s.safeAreaInsets().top > 0:
                return s
        except Exception:
            pass
    return NSScreen.mainScreen()


def _notch_geometry(scr) -> tuple:
    """(screen frame tuple, notch width, notch height) — a plausible island on
    a screen without a notch."""
    f = scr.frame()
    frame = (f.origin.x, f.origin.y, f.size.width, f.size.height)
    try:
        h = float(scr.safeAreaInsets().top)
    except Exception:
        h = 0.0
    notch_w = 0.0
    try:
        la, ra = scr.auxiliaryTopLeftArea(), scr.auxiliaryTopRightArea()
        notch_w = f.size.width - la.size.width - ra.size.width
    except Exception:
        pass
    if h <= 0:
        h = 32.0
    if not 120.0 < notch_w < 420.0:
        notch_w = 180.0
    return frame, notch_w, h


class ChipView(NSView):
    def initWithFrame_(self, frame):
        self = objc.super(ChipView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.mode = "rec"      # rec | busy | sent | empty | missed
        self.bars = [0.0] * 5
        self.phase = 0.0
        self.notch_w = 180.0
        self.band_h = 32.0     # the notch-high band with the dot and the bars
        self.text = ""
        self.label = ""
        self.copied = False
        self.owner = None
        self.copy_rect = None
        self.close_rect = None
        return self

    def acceptsFirstMouse_(self, event):
        return True

    def mouseDown_(self, event):
        p = self.convertPoint_fromView_(event.locationInWindow(), None)

        def inside(r):
            return r is not None and r[0] <= p.x <= r[0] + r[2] and r[1] <= p.y <= r[1] + r[3]
        if self.owner is None:
            return
        if inside(self.copy_rect):
            self.owner.copyMissed()
        elif inside(self.close_rect):
            self.owner.closeMissed()

    def drawRect_(self, rect):
        b = self.bounds()
        w, h = b.size.width, b.size.height
        r = min(self.band_h * 0.42 if h <= self.band_h + 1 else 18.0, w / 2.0)
        p = NSBezierPath.bezierPath()
        p.moveToPoint_(NSMakePoint(0.0, h))
        p.lineToPoint_(NSMakePoint(w, h))
        p.lineToPoint_(NSMakePoint(w, r))
        p.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
            NSMakePoint(w - r, r), r, 0.0, -90.0, True)
        p.lineToPoint_(NSMakePoint(r, 0.0))
        p.appendBezierPathWithArcWithCenter_radius_startAngle_endAngle_clockwise_(
            NSMakePoint(r, r), r, -90.0, 180.0, True)
        p.closePath()
        NSColor.blackColor().setFill()
        p.fill()

        wing = (w - self.notch_w) / 2.0
        if wing < 18.0:
            return                      # folding: wings too narrow to draw in
        cy = h - self.band_h / 2.0
        if self.mode == "missed":
            self._draw_missed(w, h)
        elif self.mode in ("rec", "busy", "sent") and self.text and h > self.band_h + 8:
            font = NSFont.systemFontOfSize_(13.0)
            white = 0.96 if self.mode == "rec" else 0.6
            attrs = {NSFontAttributeName: font,
                     NSForegroundColorAttributeName: NSColor.colorWithCalibratedWhite_alpha_(white, 1.0)}
            # Fill the strip to its right edge: the tail is cut by measured
            # width, not by a character count that left ~10% of it empty.
            words, avail = " ".join(self.text.split()).split(" "), w - 36.0
            line = " ".join(words)
            while len(words) > 1:
                s = NSAttributedString.alloc().initWithString_attributes_(line, attrs)
                if s.size().width <= avail:
                    break
                words = words[1:]
                line = "…" + " ".join(words)
            NSAttributedString.alloc().initWithString_attributes_(
                line, attrs).drawAtPoint_(NSMakePoint(18.0, 9.0))

        # left wing: the status dot
        color = {"rec": RED, "busy": AMBER, "sent": GREEN, "missed": AMBER}.get(self.mode, GREY)
        if self.mode == "rec":
            alpha = 0.88 + 0.12 * math.sin(self.phase * 6.0)
        elif self.mode == "busy":
            alpha = 0.55 + 0.45 * math.sin(self.phase * 12.0)
        else:
            alpha = 1.0
        rad = 5.0
        cx = min(wing * 0.55, 34.0)
        NSColor.colorWithSRGBRed_green_blue_alpha_(*color, alpha * 0.18).setFill()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - rad * 2, cy - rad * 2, rad * 4, rad * 4)).fill()
        NSColor.colorWithSRGBRed_green_blue_alpha_(*color, alpha).setFill()
        NSBezierPath.bezierPathWithOvalInRect_(
            NSMakeRect(cx - rad, cy - rad, rad * 2, rad * 2)).fill()

        # right wing: the voice
        n = len(self.bars)
        bw, gap = 3.0, 3.0
        total = n * bw + (n - 1) * gap
        x0 = w - min(wing * 0.55, 34.0) - total / 2.0
        max_h = self.band_h * 0.56
        for i, v in enumerate(self.bars):
            if self.mode == "rec":
                bh = 3.0 + v * (max_h - 3.0)
                NSColor.colorWithCalibratedWhite_alpha_(0.94, 0.95).setFill()
            else:
                bh = 3.0 + 0.18 * max_h * (0.5 + 0.5 * math.sin(self.phase * 9.0 + i * 0.9)) \
                    if self.mode == "busy" else 3.0
                NSColor.colorWithSRGBRed_green_blue_alpha_(*color, 0.8).setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(x0 + i * (bw + gap), cy - bh / 2.0, bw, bh), 1.5, 1.5).fill()


    @objc.python_method
    def _draw_missed(self, w, h):
        pad = 18.0
        def text_at(s, x, y, size, white, bold=False):
            font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
            attrs = {NSFontAttributeName: font,
                     NSForegroundColorAttributeName: NSColor.colorWithCalibratedWhite_alpha_(*white)}
            NSAttributedString.alloc().initWithString_attributes_(s, attrs).drawAtPoint_(NSMakePoint(x, y))
        top = h - self.band_h
        text_at("Non incollato · " + self.label, pad, top - 22.0, 11.0, (0.62, 1.0))
        text_at(preview(self.text, 50), pad, 14.0, 13.0, (0.96, 1.0))
        # Copy key, bottom right
        bw, bh = 74.0, 26.0
        bx, by = w - pad - bw, 10.0
        self.copy_rect = (bx, by, bw, bh)
        fill = GREEN if self.copied else (1.0, 1.0, 1.0)
        NSColor.colorWithSRGBRed_green_blue_alpha_(*fill, 0.95 if self.copied else 0.14).setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(bx, by, bw, bh), 13.0, 13.0).fill()
        label = "Copiato" if self.copied else "Copia"
        text_at(label, bx + (18.0 if self.copied else 22.0), by + 5.0, 12.0,
                (0.05 if self.copied else 1.0, 1.0), bold=True)
        # close, top right of the card, on the label's line
        cx, cy = w - pad - 12.0, top - 24.0
        self.close_rect = (cx - 8.0, cy - 6.0, 28.0, 26.0)
        text_at("✕", cx, cy, 12.0, (0.55, 1.0))


class NotchChip(NSObject):
    """Owns the chord gate, the `ptt` flag, the chip panel and the cues."""

    def initWithStateDir_soundDir_engineOn_(self, state_dir, sound_dir, engine_on):
        self = objc.super(NotchChip, self).init()
        if self is None:
            return None
        self.state_dir = Path(state_dir)
        self.ptt_file = self.state_dir / "ptt"
        self.engine_on = engine_on
        self.gate = PttGate()
        self.mode = None            # None = folded; rec | busy | sent | empty
        self.t_mode = 0.0
        self.raw_before = 0.0
        self.t_release = 0.0
        self.t_raw = 0.0
        self.t0 = time.time()
        self.sounds = {}
        for name in ("press", "release", "sent", "empty"):
            try:
                snd = NSSound.alloc().initWithContentsOfFile_byReference_(
                    str(Path(sound_dir) / f"{name}.wav"), True)
                if snd is not None:
                    snd.setVolume_(1.0)    # thuds carry less than notes
                    self.sounds[name] = snd
            except Exception:
                pass
        self.clear_flag()           # a flag left by a crashed pill would hold the mic open
        self._build_panel()
        # The screen edge lights up while Control is held (style A, 30/10/2026).
        self.glow, self.last_levels = None, None
        try:
            from edge_glow import EdgeGlow
            self.glow = EdgeGlow.alloc().init()
        except Exception as err:
            self._note(f"edge glow unavailable: {err}")
        # Any key or click while Control is down = a shortcut, not a talk.
        self.monitor = None
        try:
            mask = (1 << 10) | (1 << 1) | (1 << 3) | (1 << 25)   # keyDown, left/right/other mouseDown
            self.monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask, lambda _ev: self.gate.cancel())
        except Exception as err:
            self._note(f"key monitor unavailable: {err}")
        self.timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            TICK_S, self, "tick:", None, True)
        self.timer.setTolerance_(0.0)
        return self

    # -- the flag ------------------------------------------------------------

    @objc.python_method
    def clear_flag(self):
        try:
            self.ptt_file.unlink()
        except Exception:
            pass

    @objc.python_method
    def _raw_mtime(self) -> float:
        try:
            return (self.state_dir / "raw.txt").stat().st_mtime
        except Exception:
            return 0.0

    @objc.python_method
    def _state(self) -> str:
        try:
            return (self.state_dir / "state").read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    @objc.python_method
    def _levels(self):
        try:
            data = (self.state_dir / "levels.bin").read_bytes()
            if len(data) < 60 * 4:
                return None
            return struct.unpack("<60f", data[:60 * 4])
        except Exception:
            return None

    @objc.python_method
    def _draft(self) -> str:
        """The engine's live draft of the sentence being spoken ('' if none)."""
        try:
            path = self.state_dir / "partial.txt"
            if time.time() - path.stat().st_mtime > 25.0:
                return ""
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    @objc.python_method
    def _play(self, name):
        snd = self.sounds.get(name)
        if snd is None:
            return
        try:
            snd.stop()
            snd.play()
        except Exception:
            pass

    # -- the panel -----------------------------------------------------------

    @objc.python_method
    def _build_panel(self):
        style = NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
        panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 200, 32), style, NSBackingStoreBuffered, False)
        panel.setOpaque_(False)
        panel.setBackgroundColor_(NSColor.clearColor())
        panel.setLevel_(NSStatusWindowLevel + 1)
        panel.setHasShadow_(False)
        panel.setIgnoresMouseEvents_(True)      # never steals a click
        panel.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary)
        view = ChipView.alloc().initWithFrame_(NSMakeRect(0, 0, 200, 32))
        view.setAutoresizingMask_(18)
        view.owner = self
        panel.setContentView_(view)
        panel.setAlphaValue_(0.0)
        self.panel = panel
        self.view = view

    @objc.python_method
    def _show(self, big=False, extra=BIG_EXTRA_H, clickable=None):
        """Wings beside the notch; `big` = a card hanging `extra` points below
        it — the live text while you speak, or the missed-paste card, which is
        the only state that takes clicks."""
        clickable = big if clickable is None else clickable
        frame, notch_w, notch_h = _notch_geometry(_notch_screen())
        self.view.notch_w = notch_w
        self.view.band_h = notch_h
        folded = NSMakeRect(*chip_rect(*frame, notch_w, notch_h, 0.0))
        if big:
            x, y, w, h = chip_rect(*frame, notch_w, notch_h, BIG_WING)
            open_ = NSMakeRect(x, y - extra, w, h + extra)
        else:
            open_ = NSMakeRect(*chip_rect(*frame, notch_w, notch_h, WING))
        self.panel.setIgnoresMouseEvents_(not clickable)
        # The shape the chip is heading to — the edge glow bends around it.
        self.cutout = (open_.size.width, open_.size.height,
                       18.0 if big else min(notch_h * 0.42, open_.size.width / 2))
        if not self.panel.isVisible() or self.panel.alphaValue() < 0.05:
            self.panel.setFrame_display_(folded, True)
        self.panel.orderFrontRegardless()
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.18)
        self.panel.animator().setAlphaValue_(1.0)
        self.panel.animator().setFrame_display_(open_, True)
        NSAnimationContext.endGrouping()

    @objc.python_method
    def _hide(self):
        frame, notch_w, notch_h = _notch_geometry(_notch_screen())
        folded = NSMakeRect(*chip_rect(*frame, notch_w, notch_h, 0.0))
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.22)
        self.panel.setIgnoresMouseEvents_(True)
        self.panel.animator().setAlphaValue_(0.0)
        self.panel.animator().setFrame_display_(folded, True)
        NSAnimationContext.endGrouping()

    @objc.python_method
    def _set_mode(self, mode):
        self.mode = mode
        self.t_mode = time.time()
        self.view.mode = mode or "rec"

    # -- the loop ------------------------------------------------------------

    def tick_(self, _timer):
        # An exception escaping an NSTimer callback becomes an NSException in
        # the run loop and can take the whole pill down with the menu-bar
        # switch. The chip is a bonus: it never gets to crash dictation.
        try:
            self._tick()
        except Exception as err:
            self._note(f"error {type(err).__name__}: {err}")

    @objc.python_method
    def _note(self, line):
        """One line per event in chip.log (capped): the evidence when the user
        says it did not start and nothing else on disk saw the keys."""
        try:
            log = self.state_dir / "chip.log"
            lines = log.read_text(encoding="utf-8").splitlines()[-199:] if log.exists() else []
            lines.append(time.strftime("%H:%M:%S") + ".%03d " % (time.time() % 1 * 1000) + line)
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    @objc.python_method
    def _tick(self):
        now = time.time()
        try:
            flags = int(NSEvent.modifierFlags())
        except Exception:
            flags = 0
        event = self.gate.update(is_chord(flags), time.monotonic())

        if event == "press":
            if not self.engine_on():
                # Nothing would hear it: say so instead of pretending.
                self._note("press, engine off")
                self._play("empty")
                self._set_mode("empty")
                self.view.bars = [0.0] * 5
                self._show()
            else:
                try:
                    self.ptt_file.touch()
                except Exception:
                    pass
                self._play("press")
                self._note("press")
                self._set_mode("rec")
                self.view.copied = False
                self.view.text = ""
                self.live_open = False
                self._show()
        elif event == "release":
            self.clear_flag()
            if self.mode == "rec":
                self.raw_before = self._raw_mtime()
                self.t_release = now
                self.t_raw = 0.0
                self.noted_wait = False
                self._play("release")
                self._note("release")
                self._set_mode("busy")

        # Files at 20 Hz (the rate the engine writes levels.bin), the keys and
        # the drawing at 60: three times fewer reads for no visible change.
        self.n_ticks = getattr(self, "n_ticks", 0) + 1
        poll = event is not None or self.n_ticks % 3 == 0
        if self.mode == "rec" and poll:
            self.last_levels = self._levels()
            self.view.bars = chip_bars(self.last_levels)
            draft = self._draft()
            if draft:
                self.view.text = draft
                if not getattr(self, "live_open", False):
                    self.live_open = True
                    self._show(big=True, extra=LIVE_EXTRA_H, clickable=False)
        elif self.mode == "busy" and poll:
            verdict = release_outcome(self.raw_before, self._raw_mtime(),
                                      self._state(), now - self.t_mode)
            if verdict == "sent":
                # The text exists; did it LAND? The engine writes paste.json
                # right after the paste — give it a moment before trusting it.
                if not self.t_raw:
                    self._note("text ready")
                self.t_raw = self.t_raw or now
                report = read_report(self.state_dir / "paste.json", self.t_release - 0.2)
                if report is None and now - self.t_raw < REPORT_WAIT_S:
                    if not getattr(self, "noted_wait", False):
                        self.noted_wait = True
                        try:
                            ts = json.loads((self.state_dir / "paste.json").read_text(encoding="utf-8")).get("ts")
                        except Exception as err:
                            ts = f"unreadable {err}"
                        self._note(f"waiting report: since={self.t_release - 0.2:.3f} file ts={ts}")
                    verdict = None
                elif report is not None and not report.get("ok", True):
                    verdict = "missed"
                    self.view.text = str(report.get("text", ""))
                    self.view.label = reason_label(str(report.get("reason", "")))
                    self.view.copied = False
            if verdict == "sent":
                # Close on what was SENT, not on the last draft: the draft
                # trails the voice by a pass, so it ended a few words early.
                try:
                    final = (self.state_dir / "raw.txt").read_text(encoding="utf-8").strip()
                    if final:
                        self.view.text = final
                except Exception:
                    pass
            if verdict:
                self._note(f"{verdict} after {now - self.t_mode:.2f}s")
                self._play("empty" if verdict == "missed" else verdict)
                self._set_mode(verdict)
                if verdict == "missed":
                    self._show(big=True)
        elif self.mode in ("sent", "empty") and not self.gate.active:
            if now - self.t_mode >= LINGER_S:
                self.mode = None
                self._hide()
        elif self.mode == "missed" and now - self.t_mode >= MISSED_LINGER_S:
            self.mode = None
            self._hide()

        if self.mode is not None:
            self.view.phase = now - self.t0
            self.view.setNeedsDisplay_(True)
        if self.glow is not None:
            try:
                self.glow.update(self.mode == "rec", self.last_levels, now,
                                 getattr(self, "cutout", None))
            except Exception as err:
                self._note(f"edge glow error: {err}")
                self.glow = None    # one failure turns it off; dictation goes on

    def copyMissed(self):
        try:
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(self.view.text, NSPasteboardTypeString)
            self.view.copied = True
            self.t_mode = time.time() - (MISSED_LINGER_S - 0.9)   # fold 0.9 s from now
            self._note("missed: copied")
            self._play("sent")
        except Exception as err:
            self._note(f"copy failed: {err}")

    def closeMissed(self):
        self.mode = None
        self._hide()

    @objc.python_method
    def shutdown(self):
        self.clear_flag()
        try:
            if self.monitor is not None:
                NSEvent.removeMonitor_(self.monitor)
        except Exception:
            pass
        try:
            self.timer.invalidate()
        except Exception:
            pass
