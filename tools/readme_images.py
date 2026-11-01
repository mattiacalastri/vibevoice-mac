#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Render the README screenshots from the real drawing code — no screen capture.

    python3 tools/readme_images.py            # writes docs/hero.png, docs/notch_states.png

The chip is the app's own `notch_chip.ChipView`, drawn offscreen. The edge glow
uses the app's own ring path, palette and ring widths (`edge_glow`), composited
here because Core Animation filters do not render offscreen. The desktop behind
them is a neutral mock: nobody's real screen ends up in the repo.
"""
from __future__ import annotations

import math
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import edge_glow as eg  # noqa: E402
import notch_chip as nc  # noqa: E402
from AppKit import (  # noqa: E402
    NSBezierPath, NSBitmapImageRep, NSColor, NSDeviceRGBColorSpace, NSFont,
    NSFontAttributeName, NSForegroundColorAttributeName, NSGraphicsContext,
    NSMakePoint, NSMakeRect, NSPNGFileType,
)
from Foundation import NSAttributedString  # noqa: E402
from Quartz import (  # noqa: E402
    CGBitmapContextCreate, CGBitmapContextCreateImage, CGColorSpaceCreateDeviceGray,
    CGContextAddPath, CGContextScaleCTM, CGContextSetGrayStrokeColor,
    CGContextSetLineCap, CGContextSetLineJoin, CGContextSetLineWidth,
    CGContextStrokePath, CGImageGetBytesPerRow, CGDataProviderCopyData,
    CGImageGetDataProvider, kCGImageAlphaNone, kCGLineCapRound, kCGLineJoinRound,
)

SCALE = 2                       # render at Retina density
SCREEN_W, SCREEN_H = 1512, 982  # a 14" MacBook Pro, in points
NOTCH_W, NOTCH_H = 186.0, 32.0
LEVEL = 0.55                    # a voice mid-sentence
SAID = "add a loading state to the settings page and write a test for it"
TYPED = "add a loading state to the settings page and"   # confirmed words land first


# ── AppKit offscreen helpers ──────────────────────────────────────────────────

def _canvas(w: float, h: float):
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, int(w * SCALE), int(h * SCALE), 8, 4, True, False, NSDeviceRGBColorSpace, 0, 0)
    rep.setSize_((w, h))
    return rep


def _draw(rep, fn) -> Image.Image:
    ctx = NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(ctx)
    fn()
    ctx.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    data = rep.representationUsingType_properties_(NSPNGFileType, {})
    return Image.open(BytesIO(bytes(data))).convert("RGBA")


def _text(s, x, y, size, rgb, mono=False, alpha=1.0, bold=False):
    if mono:
        font = NSFont.monospacedSystemFontOfSize_weight_(size, 0.3 if bold else 0.0)
    else:
        font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
    attrs = {NSFontAttributeName: font,
             NSForegroundColorAttributeName: NSColor.colorWithSRGBRed_green_blue_alpha_(*rgb, alpha)}
    a = NSAttributedString.alloc().initWithString_attributes_(s, attrs)
    a.drawAtPoint_(NSMakePoint(x, y))
    return a.size().width


def _fill(rect, rgb, alpha=1.0, radius=0.0):
    NSColor.colorWithSRGBRed_green_blue_alpha_(*rgb, alpha).setFill()
    NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(NSMakeRect(*rect), radius, radius).fill()


# ── the chip, drawn by the app's own view ─────────────────────────────────────

def chip(mode: str, text: str = "", label: str = "", bars=(0.35, 0.8, 1.0, 0.6, 0.3),
         big: bool = False, extra: float = nc.LIVE_EXTRA_H) -> Image.Image:
    wing = nc.BIG_WING if big else nc.WING
    w, h = NOTCH_W + 2 * wing, NOTCH_H + (extra if big else 0.0)
    view = nc.ChipView.alloc().initWithFrame_(NSMakeRect(0, 0, w, h))
    view.mode, view.text, view.label = mode, text, label
    view.bars, view.notch_w, view.band_h = list(bars), NOTCH_W, NOTCH_H
    view.phase = math.pi / (24 if mode == "busy" else 12)   # the dot at the top of its pulse
    return _draw(_canvas(w, h), lambda: view.drawRect_(view.bounds()))


# ── the edge glow, from the app's own geometry and constants ──────────────────

def _ring_mask(width_pt: float, cutout) -> np.ndarray:
    W, H = SCREEN_W * SCALE, SCREEN_H * SCALE
    ctx = CGBitmapContextCreate(None, W, H, 8, W, CGColorSpaceCreateDeviceGray(), kCGImageAlphaNone)
    CGContextScaleCTM(ctx, SCALE, SCALE)
    cw, ch, cr = cutout
    CGContextAddPath(ctx, eg._ring_path(SCREEN_W, SCREEN_H, 1.5, 12.0, cw, ch, cr))
    CGContextSetGrayStrokeColor(ctx, 1.0, 1.0)
    CGContextSetLineWidth(ctx, width_pt)
    CGContextSetLineCap(ctx, kCGLineCapRound)
    CGContextSetLineJoin(ctx, kCGLineJoinRound)
    CGContextStrokePath(ctx)
    img = CGBitmapContextCreateImage(ctx)
    raw = CGDataProviderCopyData(CGImageGetDataProvider(img))
    row = CGImageGetBytesPerRow(img)
    a = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(H, row)[:, :W]
    return a.astype(np.float32) / 255.0


def _conic(angle: float) -> np.ndarray:
    W, H = SCREEN_W * SCALE, SCREEN_H * SCALE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = (np.arctan2(-(yy - H / 2), xx - W / 2) - angle) % (2 * np.pi)
    t = theta / (2 * np.pi) * (len(eg.PALETTE) - 1)
    stops = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in eg.PALETTE], np.float32)
    i0 = np.floor(t).astype(int).clip(0, len(stops) - 2)
    f = (t - i0)[..., None]
    return stops[i0] * (1 - f) + stops[i0 + 1] * f


def glow(cutout, level: float = LEVEL, angle: float = 0.9) -> Image.Image:
    colors = _conic(angle)
    out = np.zeros((SCREEN_H * SCALE, SCREEN_W * SCALE, 4), np.float32)
    for i, (width, alpha, blur) in enumerate(eg.RINGS):
        grow = (1.0 + 1.4 * level) if i < 2 else (1.0 + 0.3 * level)
        mask = _ring_mask(width * grow, cutout)
        if blur:
            m = Image.fromarray((mask * 255).astype(np.uint8))
            mask = np.asarray(m.filter(ImageFilter.GaussianBlur(blur * SCALE)), np.float32) / 255.0
        a = (mask * min(1.0, alpha * (0.75 + 0.6 * level)))[..., None]
        out[..., :3] = colors * a + out[..., :3] * (1 - a)    # premultiplied "over"
        out[..., 3:] = a + out[..., 3:] * (1 - a)
    rgb = np.where(out[..., 3:] > 0, out[..., :3] / np.maximum(out[..., 3:], 1e-6), 0)
    return Image.fromarray(np.dstack([rgb, out[..., 3:] * 255]).clip(0, 255).astype(np.uint8), "RGBA")


# ── a neutral desktop: wallpaper, menu bar, one terminal ──────────────────────

def desktop(typed: str) -> Image.Image:
    def paint():
        # wallpaper: a quiet dark gradient
        for k in range(48):
            t = k / 47
            _fill((0, SCREEN_H * (1 - (k + 1) / 48), SCREEN_W, SCREEN_H / 48 + 1),
                  (0.06 + 0.05 * t, 0.07 + 0.02 * t, 0.12 + 0.06 * t))
        # menu bar
        _fill((0, SCREEN_H - NOTCH_H, SCREEN_W, NOTCH_H), (0.0, 0.0, 0.0), 0.35)
        x = 22
        for word, bold in (("Terminal", True), ("Shell", False), ("Edit", False),
                           ("View", False), ("Window", False)):
            x += _text(word, x, SCREEN_H - 23, 13, (1, 1, 1), bold=bold, alpha=0.92) + 20
        _text("Sun 1 Nov  17:42", SCREEN_W - 132, SCREEN_H - 23, 13, (1, 1, 1), alpha=0.92)
        # terminal window
        wx, wy, ww, wh = 236, 150, 1040, 600
        _fill((wx - 1, wy - 1, ww + 2, wh + 2), (1, 1, 1), 0.10, 11)
        _fill((wx, wy, ww, wh), (0.075, 0.078, 0.09), 1.0, 10)
        _fill((wx, wy + wh - 36, ww, 36), (0.13, 0.135, 0.15), 1.0, 10)
        _fill((wx, wy + wh - 36, ww, 12), (0.13, 0.135, 0.15))
        for i, c in enumerate(((1.0, 0.37, 0.34), (1.0, 0.74, 0.18), (0.16, 0.79, 0.25))):
            _fill((wx + 18 + i * 20, wy + wh - 24, 12, 12), c, 1.0, 6)
        _text("my-app — zsh — 120×32", wx + ww / 2 - 78, wy + wh - 26, 13, (1, 1, 1), alpha=0.55)
        green, dim, white = (0.32, 0.86, 0.45), (0.55, 0.57, 0.62), (0.92, 0.93, 0.95)
        lines = [
            ([("~/my-app", (0.45, 0.68, 1.0)), (" on ", dim), ("main", (0.80, 0.55, 1.0))], 0),
            ([("$ ", green), ("npm test", white)], 1),
            ([("  ✓ ", green), ("settings page renders (41 ms)", dim)], 2),
            ([("  ✓ ", green), ("saves the form (63 ms)", dim)], 3),
            ([("  Tests: ", dim), ("2 passed", green), (", 2 total", dim)], 4),
            ([("~/my-app", (0.45, 0.68, 1.0)), (" on ", dim), ("main", (0.80, 0.55, 1.0))], 6),
            ([("$ ", green), (typed, white)], 7),
        ]
        for parts, row in lines:
            x, y = wx + 24, wy + wh - 76 - row * 26
            for s, rgb in parts:
                x += _text(s, x, y, 15, rgb, mono=True)
        _fill((x + 3, wy + wh - 76 - 7 * 26 - 2, 9, 20), white, 0.85)   # the cursor
    return _draw(_canvas(SCREEN_W, SCREEN_H), paint)


def hero() -> Image.Image:
    big = nc.BIG_WING
    cw, ch = NOTCH_W + 2 * big, NOTCH_H + nc.LIVE_EXTRA_H
    base = desktop(TYPED)
    base.alpha_composite(glow((cw, ch, 18.0)))
    c = chip("rec", text=SAID, big=True)
    base.alpha_composite(c, (int((SCREEN_W - cw) / 2 * SCALE), 0))
    return base


def states() -> Image.Image:
    tiles = [
        chip("rec", text=SAID, big=True),
        chip("busy", bars=(0.2,) * 5),
        chip("sent", bars=(0.0,) * 5),
        chip("missed", text="add a loading state to the settings page",
             label=nc.reason_label("nofocus"), big=True, extra=nc.BIG_EXTRA_H),
    ]
    captions = ["1 · Holding Control: live text", "2 · Released: transcribing",
                "3 · Delivered at the cursor", "4 · Couldn't type it: «Copia» keeps it"]
    cell_w, cell_h, cap_h, gap = 660, 108, 34, 18     # points

    def cell():
        # the top strip of a screen: wallpaper and a menu bar the chip hangs from
        for k in range(12):
            _fill((0, cell_h * (1 - (k + 1) / 12), cell_w, cell_h / 12 + 1),
                  (0.08 + 0.006 * k, 0.085 + 0.003 * k, 0.15 + 0.006 * k))
        _fill((0, cell_h - NOTCH_H, cell_w, NOTCH_H), (0.0, 0.0, 0.0), 0.35)
    sheet_w = 2 * cell_w + 3 * gap
    sheet_h = 2 * (cell_h + cap_h) + 3 * gap
    sheet = Image.new("RGBA", (sheet_w * SCALE, sheet_h * SCALE), (24, 25, 30, 255))
    for i, (tile, cap) in enumerate(zip(tiles, captions)):
        x = gap + (i % 2) * (cell_w + gap)
        y = gap + (i // 2) * (cell_h + cap_h + gap)
        bg = _draw(_canvas(cell_w, cell_h), cell)
        bg.alpha_composite(tile, ((bg.width - tile.width) // 2, 0))
        mask = Image.new("L", bg.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, bg.width - 1, bg.height - 1), 12 * SCALE, fill=255)
        sheet.paste(bg, (x * SCALE, y * SCALE), mask)
        label = _draw(_canvas(cell_w, cap_h), lambda cap=cap: _text(cap, 4, 8, 15, (0.86, 0.87, 0.9)))
        sheet.alpha_composite(label, (x * SCALE, (y + cell_h) * SCALE))
    return sheet


def main() -> None:
    docs = ROOT / "docs"
    h = hero()
    h.convert("RGB").resize((1600, round(1600 * h.height / h.width)), Image.LANCZOS) \
        .save(docs / "hero.png", optimize=True)
    states().convert("RGB").save(docs / "notch_states.png", optimize=True)
    print("wrote docs/hero.png, docs/notch_states.png")


if __name__ == "__main__":
    main()
