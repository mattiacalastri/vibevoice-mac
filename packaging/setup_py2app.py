#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""py2app build: python3 packaging/setup_py2app.py py2app (run from repo root).

Fallback if py2app can't cope (document WHY in the commit):
  pyinstaller --windowed --name VibeVoice --icon assets/icon/VibeVoice_LED.icns vibevoice.py
  then: check libportaudio via `otool -L` and graft with install_name_tool.

This file is the ONLY place the shipped bundle's identity is declared — icon,
name, version, usage strings. `packaging/install_dev_app.sh` reads the same
OPTIONS through py2app's alias mode, so the daily driver and the release
bundle cannot drift apart in what they claim to be (sess.9767: the release
bundle was still shipping the retired teal waveform icon that BRAND.md forbids
by name, and calling itself "Copyright not specified").
"""
import sys

from setuptools import setup

# One string, two plist keys. CFBundleVersion is what LaunchServices compares
# when two copies of com.vibevoice.app exist on disk — keeping it in a single
# constant is what stops the "installed app is older than the repo" question
# from needing an archaeology session to answer.
VERSION = "1.1.0"

# modulegraph walks the AST of every scanned module recursively (no explicit
# stack) — numpy's __init__ has deeply nested conditional imports that blow
# past Python's default recursion limit (1000) during the py2app scan itself
# (not at runtime). Bump it for the duration of the build only.
sys.setrecursionlimit(10000)

APP = ["vibevoice.py"]
DATA_FILES = ["engine.py", "autosend.py", "config.py", "notch_chip.py", "edge_glow.py", "requirements.txt",
              ("assets/sounds", ["assets/sounds/press.wav", "assets/sounds/release.wav",
                                 "assets/sounds/sent.wav", "assets/sounds/empty.wav"])]
OPTIONS = {
    # The LED, not the waveform. `assets/icon/VibeVoice.icns` is the RETIRED
    # teal wave — BRAND.md forbids both the wave ("names the category, not the
    # product") and the teal (it is Astra Digital's colour, and this is not
    # agency work). It stayed wired here for a month after the mark changed,
    # so every bundle built in that window shipped the wrong logo.
    "iconfile": "assets/icon/VibeVoice_LED.icns",
    # mlx_whisper is deliberately NOT in "packages": that option forces py2app
    # to recursively bundle the ENTIRE package tree, which for mlx_whisper
    # drags in transformers' optional torch/jax/onnxruntime/numba backends
    # (all installed system-wide here) and blows up on Windows-only PyInstaller
    # hook stubs (`No module named PyInstaller.hooks.hook-PyQt5.QtX11Extras`).
    # engine.py already lazy-imports mlx_whisper inside a guarded try/except
    # (_ensure_mlx_whisper), so "includes" (normal reachability-based
    # discovery, following the actual import graph) is enough to ship it
    # without forcing in its unused optional heavyweight dependencies.
    "packages": ["numpy", "sounddevice"],
    # AVFoundation: the F1 voice-processing capture imports it lazily
    # (engine._ensure_avfoundation), so modulegraph never sees it — include it
    # explicitly like the other pyobjc frameworks.
    "includes": ["objc", "AppKit", "Foundation", "Quartz", "AVFoundation", "config", "notch_chip", "edge_glow", "mlx_whisper"],
    # This Mac's system site-packages is polluted with heavyweight ML backends
    # (torch, jax, onnxruntime, PyInstaller, ...) that mlx_whisper's transitive
    # graph can reach but never needs at runtime. modulegraph crashes scanning
    # PyInstaller's dashed hook filenames (`hook-PySide2.QtWebEngine` is not an
    # importable module name), so cut the whole cluster out of the scan.
    # onnxruntime IS needed at runtime by the F2 Silero decider, but it stays
    # in this exclude list deliberately (anti-crash): build_release.sh grafts
    # the real package post-build, same pattern as mlx/tiktoken.
    "excludes": [
        "PyInstaller",
        "torch",
        "torchvision",
        "torchaudio",
        "torchgen",
        "jax",
        "jaxlib",
        "onnxruntime",
        "matplotlib",
        "sympy",
        "pandas",
        "PIL",
        "black",
        "pytest",
        "test",
    ],
    "plist": {
        "CFBundleName": "VibeVoice",
        "CFBundleDisplayName": "VibeVoice",
        "CFBundleIdentifier": "com.vibevoice.app",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "CFBundleDevelopmentRegion": "en",
        # py2app's default is the literal string "Copyright not specified",
        # which is what the About box and the Finder inspector were showing on
        # an MIT-licensed product that has a LICENSE file at its root.
        "NSHumanReadableCopyright": "© 2026 Mattia Calastri — MIT licence",
        # Finder, Launchpad and the App Store category shelf all read this; with
        # it absent the app files under "Other".
        "LSApplicationCategoryType": "public.app-category.productivity",
        # Retina: without it AppKit renders the pill through the 1× path and
        # every hairline in the UI lands on a half pixel.
        "NSHighResolutionCapable": True,
        # Two pills fight over ~/.vibevoice/ and over the microphone. A second
        # double click should raise the running one, not start a rival.
        "LSMultipleInstancesProhibited": True,
        "LSMinimumSystemVersion": "12.0",
        "LSUIElement": False,  # Dock ON by default; runtime policy follows config.json
        # A double-clicked app must LISTEN when you open it. The engine autostart
        # lives in an environment variable, which the LaunchAgent supplies and a
        # double click does not — so the packaged app opened, drew its pill, and
        # transcribed nothing until you found the 🎙 toggle in the menu. That is
        # the difference between a project you configure and an app that works.
        "LSEnvironment": {"VIBEVOICE_ENGINE_AUTOSTART": "1"},
        "NSMicrophoneUsageDescription": "VibeVoice transcribes your voice into text, locally on your Mac.",
        "NSAppleEventsUsageDescription": "VibeVoice uses AppleScript to detect the frontmost app and press Return after dictation.",
    },
}

setup(app=APP, data_files=DATA_FILES, options={"py2app": OPTIONS}, setup_requires=["py2app"])
