#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate VibeVoice icon candidates: teal squircle + waveform.

Scar sess.9161: the icon must be a squircle with TRANSPARENT corners, never an
opaque full square, or macOS wraps it in a grey rounded rect. A transparent
margin (~10%) is the safe default. But the margin itself reads as grey "padding"
in the dock next to full-bleed neighbours — sess.9203 Mattia asked for it gone,
so `--margin 0` renders a full-bleed squircle (edges touch the canvas, corners
still transparent → no grey-wrap). Keep corners transparent at any margin.

Usage: python3 make_icon.py --variant 1 --margin 0 --out candidate1.png
"""
from __future__ import annotations

import argparse

from PIL import Image, ImageDraw

S = 1024
CORNER_RATIO = 0.2237                # Apple macOS squircle corner radius ratio

PALETTES = {1: ((13, 148, 136), (4, 47, 46)),    # teal → deep teal (AI Accelerator family)
            2: ((45, 212, 191), (15, 118, 110)), # bright aqua → teal
            3: ((20, 184, 166), (2, 26, 25))}    # teal → near-black

BARS = {1: [.30, .55, .90, .65, 1.0, .70, .45, .25],
        2: [.20, .45, .75, 1.0, .75, .45, .20],
        3: [.35, .70, 1.0, .55, .85, .40]}


def build(variant: int, margin: float = 0.0) -> Image.Image:
    M = int(S * margin)                       # transparent margin (0 = full-bleed)
    R = int((S - 2 * M) * CORNER_RATIO)       # macOS squircle corner radius
    top, bot = PALETTES[variant]
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    grad = Image.new("RGBA", (S, S))
    for y in range(S):
        t = y / S
        grad.paste(tuple(int(a + (b - a) * t) for a, b in zip(top, bot)) + (255,), (0, y, S, y + 1))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).rounded_rectangle([M, M, S - M, S - M], radius=R, fill=255)
    img.paste(grad, (0, 0), mask)
    d = ImageDraw.Draw(img)
    bars = BARS[variant]
    bw = int((S - 2 * M) * 0.055)
    gap = int((S - 2 * M) * 0.045)
    total = len(bars) * bw + (len(bars) - 1) * gap
    x = (S - total) // 2
    max_h = (S - 2 * M) * 0.52
    for h in bars:
        bh = int(max_h * h)
        d.rounded_rectangle([x, (S - bh) // 2, x + bw, (S + bh) // 2], radius=bw // 2, fill=(255, 255, 255, 235))
        x += bw + gap
    return img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, default=1, choices=sorted(PALETTES))
    ap.add_argument("--margin", type=float, default=0.0,
                    help="transparent margin fraction (0 = full-bleed, 0.10 = legacy safe)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.variant, a.margin).save(a.out)
    print(f"wrote {a.out} (variant {a.variant}, margin {a.margin})")
