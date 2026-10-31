#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PROTOTYPE — the screen-edge glow (Siri / Apple Intelligence style) for the Mac.

Not wired into VibeVoice: a standalone demo to look at before deciding.

    python3 tools/edge_glow_demo.py                 # the parade: 5 styles, 6 s each, labelled A–E
    python3 tools/edge_glow_demo.py --stile C       # one style, simulated voice, 8 s
    python3 tools/edge_glow_demo.py --stile C --live  # one style, live: lights while you hold
                                                    # Control and dictate (Ctrl+C to quit)

How it is built (the parts that matter for the real thing):
  * one borderless, transparent, click-through window per screen, above apps,
    on every Space and over full-screen apps — it never takes a click;
  * the ring follows the display: rounded corners, and on a MacBook it dips
    around the notch, so the glow and the notch chip read as one object;
  * four stacked rings, same path, from a wide soft halo to a thin bright core
    (widths/alphas/blurs below) — the bloom comes from the stack, not one blur;
  * each ring is a CONIC gradient (blue → violet → coral → amber → blue, the
    palette sampled from Apple's animation) masked by the stroke, its angle
    advanced by a 60 Hz timer (one turn ≈ 1.8 s, clockwise) — a timer, not a
    CABasicAnimation, which froze or jumped after the window was hidden;
  * the voice (RMS) widens and brightens the halo, so it breathes with you;
  * in 0.25 s, out 0.4 s.
"""
from __future__ import annotations

import math
import struct
import sys
import time
from pathlib import Path

import objc
from AppKit import (
    NSApp, NSApplication, NSBackingStoreBuffered, NSColor, NSFont, NSMakeRect, NSScreen,
    NSStatusWindowLevel, NSTextField,
    NSTimer, NSView, NSWindow, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
)
from Foundation import NSObject
from Quartz import (
    CAGradientLayer, CALayer, CAShapeLayer, CATransaction, CGPathAddArcToPoint,
    CGPathAddLineToPoint, CGPathCloseSubpath, CGPathCreateMutable, CGPathMoveToPoint, CIFilter,
    kCAGradientLayerConic,
)

APPLE = ["#0894FF", "#C959DD", "#FF2E54", "#FF9004", "#0894FF"]   # sampled from Siri
POLPO = ["#FF2A2A", "#FF6A3D", "#FFB020", "#E0185E", "#FF2A2A"]   # the Polpo is red
# (stroke width pt, alpha, blur radius) — wide soft halo → thin bright core
SOFT = [(50.0, 0.45, 22.0), (20.0, 0.7, 8.0), (6.0, 0.95, 2.0), (2.0, 1.0, 0.0)]
MID = [(90.0, 0.7, 30.0), (40.0, 0.9, 12.0), (16.0, 1.0, 3.0), (4.0, 1.0, 0.0)]
BOLD = [(140.0, 0.95, 42.0), (60.0, 1.0, 16.0), (24.0, 1.0, 4.0), (6.0, 1.0, 0.0)]
STYLES = {
    "A": ("Apple · discreto", APPLE, SOFT),
    "B": ("Apple · medio", APPLE, MID),
    "C": ("Apple · acceso", APPLE, BOLD),
    "D": ("Polpo · medio", POLPO, MID),
    "E": ("Polpo · acceso", POLPO, BOLD),
}
PARADE_S = 6.0
TURN_S = 1.8
FADE_IN, FADE_OUT = 0.25, 0.4
STATE = Path.home() / ".vibevoice"


def cg(hexstr, alpha=1.0):
    h = hexstr.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha).CGColor()


def notch_size(scr):
    try:
        top = float(scr.safeAreaInsets().top)
        f = scr.frame()
        la, ra = scr.auxiliaryTopLeftArea(), scr.auxiliaryTopRightArea()
        w = f.size.width - la.size.width - ra.size.width
        if top > 0 and 120 < w < 420:
            return w, top
    except Exception:
        pass
    return 0.0, 0.0


def ring_path(w, h, inset, corner, notch_w, notch_h):
    """Closed path along the screen edge, inset by `inset`, with rounded
    corners and — when there is one — a dip around the notch."""
    p = CGPathCreateMutable()
    L, R, B, T = inset, w - inset, inset, h - inset
    c = corner
    CGPathMoveToPoint(p, None, L + c, T)
    if notch_w:
        nl, nr = (w - notch_w) / 2 - inset, (w + notch_w) / 2 + inset
        nb = h - notch_h - inset
        r = min(8.0 + inset, notch_h / 2)
        CGPathAddLineToPoint(p, None, nl - r, T)
        CGPathAddArcToPoint(p, None, nl, T, nl, T - r, r)
        CGPathAddArcToPoint(p, None, nl, nb, nl + r, nb, r)
        CGPathAddArcToPoint(p, None, nr, nb, nr, nb + r, r)
        CGPathAddArcToPoint(p, None, nr, T, nr + r, T, r)
    CGPathAddArcToPoint(p, None, R, T, R, T - c, c)
    CGPathAddArcToPoint(p, None, R, B, R - c, B, c)
    CGPathAddArcToPoint(p, None, L, B, L, B + c, c)
    CGPathAddArcToPoint(p, None, L, T, L + c, T, c)
    CGPathCloseSubpath(p)
    return p


class GlowView(NSView):
    def initWithFrame_screen_style_(self, frame, scr, style):
        self = objc.super(GlowView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.setWantsLayer_(True)
        self.setLayerUsesCoreImageFilters_(True)
        w, h = frame.size.width, frame.size.height
        nw, nh = notch_size(scr)
        self.grads, self.halos = [], []
        root = self.layer()
        _name, palette, rings = STYLES[style]
        for width, alpha, blur in rings:
            holder = CALayer.layer()
            holder.setFrame_(((0, 0), (w, h)))
            if blur:
                f = CIFilter.filterWithName_("CIGaussianBlur")
                f.setDefaults()
                f.setValue_forKey_(blur, "inputRadius")
                holder.setFilters_([f])
            grad = CAGradientLayer.layer()
            grad.setType_(kCAGradientLayerConic)
            grad.setFrame_(((0, 0), (w, h)))
            grad.setColors_([cg(c) for c in palette])
            grad.setStartPoint_((0.5, 0.5))
            grad.setEndPoint_((0.5, 1.0))
            mask = CAShapeLayer.layer()
            mask.setFrame_(((0, 0), (w, h)))
            mask.setPath_(ring_path(w, h, 1.5, 12.0, nw, nh))
            mask.setLineWidth_(width)
            mask.setFillColor_(None)
            mask.setStrokeColor_(NSColor.whiteColor().CGColor())
            grad.setMask_(mask)
            holder.addSublayer_(grad)
            holder.setOpacity_(alpha)
            root.addSublayer_(holder)
            self.grads.append(grad)
            self.halos.append((holder, mask, width, alpha))
        return self

    @objc.python_method
    def frame_at(self, angle, level, fade):
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        ex, ey = 0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)
        for g in self.grads:
            g.setEndPoint_((ex, ey))
        for i, (holder, mask, width, alpha) in enumerate(self.halos):
            boost = (1.0 + 1.4 * level) if i < 2 else (1.0 + 0.3 * level)
            mask.setLineWidth_(width * boost)
            holder.setOpacity_(min(1.0, alpha * (0.75 + 0.6 * level)) * fade)
        CATransaction.commit()


class Demo(NSObject):
    def initWithStyles_live_(self, styles, live):
        self = objc.super(Demo, self).init()
        self.styles, self.live, self.t0 = styles, live, time.time()
        self.shown_at, self.hidden_at, self.level, self.current = None, None, 0.0, None
        self.windows = []
        for scr in NSScreen.screens():
            f = scr.frame()
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                f, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
            win.setOpaque_(False)
            win.setBackgroundColor_(NSColor.clearColor())
            win.setHasShadow_(False)
            win.setIgnoresMouseEvents_(True)
            win.setLevel_(NSStatusWindowLevel)
            win.setCollectionBehavior_(
                NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary
                | NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorIgnoresCycle)
            win.orderFrontRegardless()
            self.windows.append([win, scr, None])
        self.label = self._label_window()
        return self

    @objc.python_method
    def _label_window(self):
        scr = NSScreen.mainScreen().frame()
        w, h = 360.0, 64.0
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(scr.origin.x + (scr.size.width - w) / 2, scr.origin.y + 120, w, h),
            NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
        win.setOpaque_(False)
        win.setBackgroundColor_(NSColor.clearColor())
        win.setIgnoresMouseEvents_(True)
        win.setLevel_(NSStatusWindowLevel + 1)
        box = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
        box.setWantsLayer_(True)
        box.layer().setBackgroundColor_(NSColor.colorWithCalibratedWhite_alpha_(0.0, 0.82).CGColor())
        box.layer().setCornerRadius_(h / 2)
        text = NSTextField.labelWithString_("")
        text.setFont_(NSFont.boldSystemFontOfSize_(24.0))
        text.setTextColor_(NSColor.whiteColor())
        text.setAlignment_(1)
        text.setFrame_(NSMakeRect(0, 15, w, 32))
        box.addSubview_(text)
        win.setContentView_(box)
        win.orderFrontRegardless()
        self.label_text = text
        return win

    @objc.python_method
    def _use(self, style):
        if style == self.current:
            return
        self.current = style
        for entry in self.windows:
            win, scr, _old = entry
            f = scr.frame()
            view = GlowView.alloc().initWithFrame_screen_style_(
                NSMakeRect(0, 0, f.size.width, f.size.height), scr, style)
            win.setContentView_(view)
            entry[2] = view
        self.label_text.setStringValue_(f"{style}  ·  {STYLES[style][0]}")

    def tick_(self, _):
        t = time.time() - self.t0
        if len(self.styles) > 1:
            k = int(t // PARADE_S)
            if k >= len(self.styles):
                NSApp.terminate_(None)
                return
            self._use(self.styles[k])
            local = t - k * PARADE_S
            on = 0.5 < local < PARADE_S - 0.6
        else:
            self._use(self.styles[0])
            on = (STATE / "ptt").exists() if self.live else 0.6 < t < 7.0
            if not self.live and t > 7.8:
                NSApp.terminate_(None)
                return
        if self.live:
            try:
                vals = struct.unpack("<60f", (STATE / "levels.bin").read_bytes()[:240])
                raw = min(1.0, (max(abs(v) for v in vals[-6:]) * 12) ** 0.6)
            except Exception:
                raw = 0.0
        else:
            # a fake voice: syllables on a slower phrase envelope
            raw = max(0.0, math.sin(t * 13.0) * 0.5 + 0.5) * (0.55 + 0.45 * math.sin(t * 2.1))
        self.level += (raw - self.level) * 0.25            # smooth like a meter
        now = time.time()
        if on and self.shown_at is None:
            self.shown_at, self.hidden_at = now, None
        if not on and self.shown_at is not None and self.hidden_at is None:
            self.hidden_at = now
        if self.shown_at is None:
            fade = 0.0
        elif self.hidden_at is None:
            fade = min(1.0, (now - self.shown_at) / FADE_IN)
        else:
            fade = max(0.0, 1.0 - (now - self.hidden_at) / FADE_OUT)
            if fade == 0.0:
                self.shown_at = None
        angle = -2 * math.pi * t / TURN_S                   # clockwise
        for _win, _scr, view in self.windows:
            if view is not None:
                view.frame_at(angle, self.level, fade)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)       # Ctrl+C quits the live mode
    if "--stile" in sys.argv:
        styles = [sys.argv[sys.argv.index("--stile") + 1].upper()]
    else:
        styles = list(STYLES)
    demo = Demo.alloc().initWithStyles_live_(styles, "--live" in sys.argv)
    timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
        1 / 60, demo, "tick:", None, True)
    timer.setTolerance_(0.0)
    app.run()


if __name__ == "__main__":
    main()
