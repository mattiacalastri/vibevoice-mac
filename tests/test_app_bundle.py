#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Bundle tests for VibeVoice.

These build a real VibeVoice.app via build_app.sh into a tmp dir and lock down
its shape: the bundle layout, the Info.plist identity + usage strings (without
which macOS silently denies the mic), and a launcher that is executable, valid
bash, and runs the pill from Resources. They run headless — the app is assembled
and inspected, never launched (launching would open the mic/GUI).

Run:  pytest -q
"""
from __future__ import annotations

import plistlib
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from setup_options import py2app_options

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "build_app.sh"


@pytest.fixture(scope="module")
def app(tmp_path_factory) -> Path:
    """Build VibeVoice.app once into a tmp dir; never touch the real ./dist."""
    out = tmp_path_factory.mktemp("vv_dist")
    subprocess.run(["bash", str(BUILD), str(out)], check=True,
                   capture_output=True, text=True)
    bundle = out / "VibeVoice.app"
    assert bundle.is_dir(), "build_app.sh did not produce VibeVoice.app"
    return bundle


# ── bundle layout ─────────────────────────────────────────────────────────────

def test_bundle_layout(app):
    """The .app has the canonical macOS skeleton + self-contained sources."""
    assert (app / "Contents" / "Info.plist").is_file()
    assert (app / "Contents" / "MacOS" / "VibeVoice").is_file()
    assert (app / "Contents" / "PkgInfo").is_file()
    res = app / "Contents" / "Resources"
    for name in ("vibevoice.py", "engine.py", "autosend.py", "requirements.txt"):
        assert (res / name).is_file(), f"missing bundled resource: {name}"


# ── Info.plist identity + usage strings ───────────────────────────────────────

def test_infoplist_is_valid_and_identified(app):
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == "com.vibevoice.app"
    assert info["CFBundleExecutable"] == "VibeVoice"
    assert info["CFBundlePackageType"] == "APPL"


def test_infoplist_is_accessory(app):
    """LSUIElement must match the app's NSApplicationActivationPolicyAccessory."""
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["LSUIElement"] is True


def test_infoplist_has_permission_strings(app):
    """No usage string => macOS denies the mic/AppleEvents prompt silently."""
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info.get("NSMicrophoneUsageDescription", "").strip()
    assert info.get("NSAppleEventsUsageDescription", "").strip()


@pytest.mark.skipif(shutil.which("plutil") is None, reason="plutil is macOS-only")
def test_infoplist_passes_plutil(app):
    r = subprocess.run(["plutil", "-lint", str(app / "Contents" / "Info.plist")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# ── launcher ──────────────────────────────────────────────────────────────────

def test_launcher_is_executable_bash(app):
    launcher = app / "Contents" / "MacOS" / "VibeVoice"
    assert launcher.read_text().startswith("#!/bin/bash")
    assert launcher.stat().st_mode & stat.S_IXUSR, "launcher is not executable"
    # Valid bash — catches a generated-heredoc syntax error before a user double-clicks.
    subprocess.run(["bash", "-n", str(launcher)], check=True,
                   capture_output=True, text=True)


def test_launcher_runs_the_pill_all_in_one(app):
    """The launcher execs the pill from Resources with autostart on."""
    body = (app / "Contents" / "MacOS" / "VibeVoice").read_text()
    assert 'exec "$PY" "$RES/vibevoice.py"' in body
    assert "VIBEVOICE_ENGINE_AUTOSTART" in body


# ── icon ──────────────────────────────────────────────────────────────────────

def test_bundle_has_icon(app):
    assert (app / "Contents/Resources/VibeVoice.icns").stat().st_size > 10_000
    with (app / "Contents/Info.plist").open("rb") as f:
        assert plistlib.load(f)["CFBundleIconFile"] == "VibeVoice"


def test_icon_respects_transparent_margin():
    """Scar sess.9161: corners must be transparent (squircle inside the canvas).

    Guards the LED master — the mark. `VibeVoice.icns` (no suffix) is the
    retired teal waveform, and this test used to guard THAT: a green assertion
    on a file no bundle was supposed to ship any more (sess.9767).
    """
    PIL = pytest.importorskip("PIL.Image")
    img = PIL.open(REPO / "assets/icon/VibeVoice_LED.icns")
    img = img.convert("RGBA")
    w, h = img.size
    for xy in [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]:
        assert img.getpixel(xy)[3] == 0, f"corner {xy} not transparent"


# ── shipped identity (what the Dock, Finder and About box read) ──────────────

def test_shipped_bundle_wears_the_led_mark():
    """The release bundle shipped the RETIRED teal waveform for a month.

    BRAND.md forbids it twice over — the wave "names the category, not the
    product", and the teal is Astra Digital's colour on an app that is not
    agency work. Nothing that builds a bundle may point back at it.
    """
    assert py2app_options()["iconfile"] == "assets/icon/VibeVoice_LED.icns"
    launcher_build = (REPO / "build_app.sh").read_text()
    assert "assets/icon/VibeVoice_LED.icns" in launcher_build
    assert 'cp "$SRC/assets/icon/VibeVoice.icns"' not in launcher_build


def test_shipped_plist_is_fully_identified():
    """Metadata a real app carries — and py2app does not fill in for you.

    Every one of these was missing or wrong on the installed bundle: the About
    box read the py2app default "Copyright not specified", Finder filed the app
    under Other for want of a category, and NSHighResolutionCapable was absent
    on a Retina-only product.
    """
    plist = py2app_options()["plist"]
    assert plist["CFBundleName"] == "VibeVoice"
    assert plist["CFBundleDisplayName"] == "VibeVoice"
    assert plist["CFBundleIdentifier"] == "com.vibevoice.app"
    assert plist["CFBundleShortVersionString"] == plist["CFBundleVersion"]
    assert "not specified" not in plist["NSHumanReadableCopyright"]
    assert "MIT" in plist["NSHumanReadableCopyright"]
    assert plist["LSApplicationCategoryType"].startswith("public.app-category.")
    assert plist["NSHighResolutionCapable"] is True
    assert plist["LSMultipleInstancesProhibited"] is True


def test_dev_installer_asserts_the_dock_name_by_effect():
    """The daily driver's whole point is the name macOS registers.

    A plist that *says* VibeVoice proves nothing: the old LaunchAgent ran the
    framework interpreter, which re-execs into Python.app, and the Dock said
    "Python" while every plist on disk said otherwise. So the installer has to
    read the name back out of LaunchServices, and that check must not quietly
    disappear from it.
    """
    body = (REPO / "packaging" / "install_dev_app.sh").read_text()
    assert "lsappinfo info -only name" in body
    assert 'NAME" = "VibeVoice"' in body or '[ "$NAME" = "VibeVoice" ]' in body
    # _child_python() needs the embedded interpreter, or the pill forks itself.
    assert "Contents/MacOS/python" in body


# ── child process spawning (py2app trap) ─────────────────────────────────────

def test_child_spawns_never_use_raw_sys_executable():
    """Scar sess.9191: in the py2app bundle sys.executable is the app LAUNCHER —
    it ignores argv and boots the pill main again, so Popen([sys.executable,
    engine.py]) forks a second pill instead of the engine. Every child spawn
    must go through _child_python(), which picks the embedded
    Contents/MacOS/python when frozen (and keeps the script path in argv so
    invariant #8's pgrep/pkill -f engine.py still matches)."""
    src = (REPO / "vibevoice.py").read_text()
    assert "Popen([sys.executable" not in src
    assert "Popen([_child_python(), str(ENGINE_PATH)]" in src
    assert "Popen([_child_python(), str(AUTOSEND_PATH)]" in src
