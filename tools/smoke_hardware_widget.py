#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
smoke_hardware_widget.py — render the hardware widget's states to PNGs.

The widget's look only fails at runtime (AppKit drawing), so pytest proves
nothing about it: this renders HardwareView offscreen in every state and
writes PNGs for eyeball verification. Not a pytest on purpose — drawing needs
a GUI session and CI is headless (same reasoning as smoke_settings_window.py).

Usage:  python3 tools/smoke_hardware_widget.py [outdir]   (default /tmp)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from AppKit import (  # noqa: E402
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered,
    NSBitmapImageFileTypePNG, NSMakeRect, NSWindow, NSWindowStyleMaskBorderless,
)

import vibevoice  # noqa: E402


def _levels(kind: str) -> list:
    if kind == "flat":
        return [0.04] * 60
    if kind == "hot":
        return [min(1.0, 0.55 + 0.45 * abs(math.sin(i * 0.6))) for i in range(60)]
    return [0.25 + 0.35 * abs(math.sin(i * 0.35)) for i in range(60)]


STATES = [
    # (filename, engine_on, muted, state, levels)
    ("widget_off.png", False, False, "", _levels("flat")),
    ("widget_listening.png", True, False, "idle", _levels("mid")),
    ("widget_recording.png", True, False, "recording", _levels("hot")),
    ("widget_muted.png", True, True, "idle", _levels("flat")),
]


def main() -> int:
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")
    outdir.mkdir(parents=True, exist_ok=True)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)

    frame = NSMakeRect(0, 0, vibevoice.WIDGET_W, vibevoice.WIDGET_H)
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
    win.setOpaque_(False)

    for name, engine_on, muted, state, levels in STATES:
        view = vibevoice.HardwareView.alloc().initWithFrame_(frame)
        view.engine_on = engine_on
        view.muted = muted
        view.state = state
        view.levels = levels
        view.phase = 1.3   # frozen animation clock (deterministic pulse)
        win.setContentView_(view)
        rep = view.bitmapImageRepForCachingDisplayInRect_(view.bounds())
        view.cacheDisplayInRect_toBitmapImageRep_(view.bounds(), rep)
        png = rep.representationUsingType_properties_(NSBitmapImageFileTypePNG, {})
        path = outdir / name
        png.writeToFile_atomically_(str(path), True)
        print(f"rendered {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
