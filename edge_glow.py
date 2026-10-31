#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The screen-edge glow while Control is held (Siri style, «A · discreto»).

Style A of five was picked on 30/10/2026 («tutti gli altri sono troppo
invasivi»): Apple's sampled palette, a thin core and a soft halo. The prototype
with all five lives in tools/edge_glow_demo.py.

Lives in the pill process (imported by notch_chip.py). One borderless,
transparent, click-through window per screen, above apps, on every Space and
over full-screen apps; the ring follows the display's corners and dips around
the notch so it meets the chip. Four stacked rings (halo → core), each a conic
gradient masked by the stroke, turned by the caller's 60 Hz tick — a timer, not
a CABasicAnimation, which froze or jumped after the window was hidden.
The windows leave the screen entirely while the glow is off: no compositing
cost when nobody is talking.
"""
from __future__ import annotations

import math

# ── pure logic ────────────────────────────────────────────────────────────────

PALETTE = ["#0894FF", "#C959DD", "#FF2E54", "#FF9004", "#0894FF"]   # sampled from Siri
RINGS = [(50.0, 0.45, 22.0), (20.0, 0.7, 8.0), (6.0, 0.95, 2.0), (2.0, 1.0, 0.0)]  # (width, alpha, blur)
TURN_S = 1.8             # one turn of the colours, clockwise
FADE_IN, FADE_OUT = 0.25, 0.4
REVEAL_IN = 0.3          # the light runs from the notch round the screen (30/10:
REVEAL_OUT = 0.3         # «parte dal notch e si espande»; short — dictation happens all day).
                         # At a CONSTANT speed: any ease-out lands slowly where the two
                         # runs meet under the Dock, the far side from the notch, and
                         # that slow join pulled the eye («mi distrae»). Windows are
                         # built ahead, so the first frames are no longer lost.
REVEAL_S = REVEAL_OUT


def glow_fade(shown_at, hidden_at, now) -> float:
    """0..1 opacity: ramps in after `shown_at`, out after `hidden_at`."""
    if shown_at is None:
        return 0.0
    if hidden_at is None:
        return min(1.0, (now - shown_at) / FADE_IN)
    return max(0.0, 1.0 - (now - hidden_at) / FADE_OUT)


def glow_reveal(shown_at, hidden_at, now) -> float:
    """0..1 share of the ring lit, grown from the notch at constant speed, and
    run back into the notch on the way out."""
    if shown_at is None:
        return 0.0
    if hidden_at is None:
        return min(1.0, (now - shown_at) / REVEAL_IN)
    x = min(1.0, (now - hidden_at) / REVEAL_OUT)
    return (1 - x) ** 3


def voice_level(levels, gain: float = 12.0) -> float:
    """The newest RMS values as a 0..1 loudness for the halo."""
    if not levels:
        return 0.0
    return min(1.0, (max(abs(v) for v in list(levels)[-6:]) * gain) ** 0.6)


# ── AppKit surface ────────────────────────────────────────────────────────────

import objc  # noqa: E402
from AppKit import (  # noqa: E402
    NSBackingStoreBuffered, NSColor, NSMakeRect, NSScreen, NSStatusWindowLevel, NSView,
    NSWindow, NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary, NSWindowCollectionBehaviorIgnoresCycle,
    NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
)
from Foundation import NSObject  # noqa: E402
from Quartz import (  # noqa: E402
    CAGradientLayer, CALayer, CAShapeLayer, CATransaction, CGPathAddArcToPoint,
    CGPathAddLineToPoint, CGPathCloseSubpath, CGPathCreateMutable, CGPathMoveToPoint, CIFilter,
    kCAGradientLayerConic,
)


def _cg(hexstr):
    h = hexstr.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 1.0).CGColor()


def _notch(scr):
    try:
        top = float(scr.safeAreaInsets().top)
        f = scr.frame()
        w = f.size.width - scr.auxiliaryTopLeftArea().size.width - scr.auxiliaryTopRightArea().size.width
        if top > 0 and 120 < w < 420:
            return w, top
    except Exception:
        pass
    return 0.0, 0.0


def _ring_path(w, h, inset, corner, notch_w, notch_h, notch_r=8.0):
    """The screen edge as ONE symmetric loop that starts and ends at the bottom
    centre, so its midpoint is the middle of the notch: revealing the stroke
    from 0.5 outwards makes the light leave the notch and run down both sides
    together, meeting at the bottom."""
    p = CGPathCreateMutable()
    L, R, B, T, c = inset, w - inset, inset, h - inset, corner
    CGPathMoveToPoint(p, None, w / 2, B)
    CGPathAddArcToPoint(p, None, L, B, L, B + c, c)
    CGPathAddArcToPoint(p, None, L, T, L + c, T, c)
    if notch_w:
        nl, nr = (w - notch_w) / 2 - inset, (w + notch_w) / 2 + inset
        nb = h - notch_h - inset
        r = min(notch_r + inset, notch_h / 2, notch_w / 2)
        CGPathAddLineToPoint(p, None, nl - r, T)
        CGPathAddArcToPoint(p, None, nl, T, nl, T - r, r)
        CGPathAddArcToPoint(p, None, nl, nb, nl + r, nb, r)
        CGPathAddArcToPoint(p, None, nr, nb, nr, nb + r, r)
        CGPathAddArcToPoint(p, None, nr, T, nr + r, T, r)
    CGPathAddArcToPoint(p, None, R, T, R, T - c, c)
    CGPathAddArcToPoint(p, None, R, B, R - c, B, c)
    CGPathAddLineToPoint(p, None, w / 2, B)
    CGPathCloseSubpath(p)
    return p


class _GlowView(NSView):
    def initWithFrame_screen_(self, frame, scr):
        self = objc.super(_GlowView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.setWantsLayer_(True)
        self.setLayerUsesCoreImageFilters_(True)
        w, h = frame.size.width, frame.size.height
        nw, nh = _notch(scr)
        self.size, self.notch, self.cutout = (w, h), (nw, nh), None
        self.masks = []
        self.grads, self.rings = [], []
        for width, alpha, blur in RINGS:
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
            grad.setColors_([_cg(c) for c in PALETTE])
            grad.setStartPoint_((0.5, 0.5))
            mask = CAShapeLayer.layer()
            mask.setFrame_(((0, 0), (w, h)))
            mask.setPath_(_ring_path(w, h, 1.5, 12.0, nw, nh))
            mask.setLineWidth_(width)
            mask.setFillColor_(None)
            mask.setStrokeColor_(NSColor.whiteColor().CGColor())
            grad.setMask_(mask)
            self.masks.append(mask)
            holder.addSublayer_(grad)
            self.layer().addSublayer_(holder)
            self.grads.append(grad)
            self.rings.append((holder, mask, width, alpha))
        return self

    @objc.python_method
    def follow(self, cutout):
        """Dip around the chip as it is drawn now — (width, height, radius) —
        instead of the bare notch: the chip grows wings and a text strip, and
        a ring bent around the small notch cut through them (30/10)."""
        if cutout == self.cutout or not self.notch[0]:
            return
        self.cutout = cutout
        w, h = self.size
        cw, ch, cr = cutout if cutout else (self.notch[0], self.notch[1], 8.0)
        path = _ring_path(w, h, 1.5, 12.0, cw, ch, cr)
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        for m in self.masks:
            m.setPath_(path)
        CATransaction.commit()

    @objc.python_method
    def paint(self, angle, level, fade, reveal=1.0):
        CATransaction.begin()
        CATransaction.setDisableActions_(True)
        end = (0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle))
        for g in self.grads:
            g.setEndPoint_(end)
        for m in self.masks:
            m.setStrokeStart_(0.5 - reveal / 2)
            m.setStrokeEnd_(0.5 + reveal / 2)
        for i, (holder, mask, width, alpha) in enumerate(self.rings):
            mask.setLineWidth_(width * ((1.0 + 1.4 * level) if i < 2 else (1.0 + 0.3 * level)))
            holder.setOpacity_(min(1.0, alpha * (0.75 + 0.6 * level)) * fade)
        CATransaction.commit()


class EdgeGlow(NSObject):
    """`update(on, levels)` from a 60 Hz tick; everything else is internal."""

    def init(self):
        self = objc.super(EdgeGlow, self).init()
        if self is None:
            return None
        self.windows = []
        self.shown_at = self.hidden_at = None
        self.level = 0.0
        self.t0 = None
        try:
            self._build()
            self.screens_sig = [(tuple(s.frame().origin), tuple(s.frame().size))
                                for s in NSScreen.screens() or []]
        except Exception:
            self.windows = []
        return self

    @objc.python_method
    def _build(self):
        for win, _view in self.windows:
            win.orderOut_(None)
        self.windows = []
        for scr in NSScreen.screens() or []:
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
            view = _GlowView.alloc().initWithFrame_screen_(
                NSMakeRect(0, 0, f.size.width, f.size.height), scr)
            win.setContentView_(view)
            self.windows.append((win, view))

    @objc.python_method
    def update(self, on, levels, now, cutout=None):
        if on and self.shown_at is None:
            # Built ahead, rebuilt only when the screens changed (lid,
            # monitors): building on the press ate the first frames of the run.
            sig = [(tuple(s.frame().origin), tuple(s.frame().size)) for s in NSScreen.screens() or []]
            if sig != getattr(self, "screens_sig", None) or not self.windows:
                self._build()
                self.screens_sig = sig
            for _win, view in self.windows:
                view.paint(0.0, 0.0, 1.0, 0.0)     # start from nothing lit
            self.shown_at, self.hidden_at, self.t0 = now, None, now
            for win, _v in self.windows:
                win.orderFrontRegardless()
        if not on and self.shown_at is not None and self.hidden_at is None:
            self.hidden_at = now
        if self.shown_at is None:
            return
        # In: the light runs out of the notch (reveal) at full opacity — a fade
        # on top would hide the run. Out: it runs back while it dims.
        fade = 1.0 if self.hidden_at is None else glow_fade(self.shown_at, self.hidden_at, now)
        self.level += (voice_level(levels) - self.level) * 0.25
        angle = -2 * math.pi * (now - self.t0) / TURN_S
        for _win, view in self.windows:
            view.follow(cutout)
            view.paint(angle, self.level, fade,
                       glow_reveal(self.shown_at, self.hidden_at, now))
        if self.hidden_at is not None and fade == 0.0:
            for win, _v in self.windows:
                win.orderOut_(None)
            self.shown_at = self.hidden_at = None
