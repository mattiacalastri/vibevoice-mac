#!/bin/bash
# SPDX-License-Identifier: MIT
#
# build_app.sh — assemble a double-clickable, all-in-one VibeVoice.app.
#
# This is a *lightweight* bundle: it ships the source + a launcher + an Info.plist
# that gives VibeVoice its own TCC identity (so the Microphone / Accessibility
# prompts attach to "VibeVoice", not to your Terminal). It does NOT embed a Python
# interpreter — it runs the first python3 it finds that has the deps (see
# requirements.txt). That keeps the bundle small and reliable; for a fully
# self-contained, signed, notarized app use py2app/PyInstaller on top of this.
#
# ── what this bundle CANNOT do (sess.9767) ───────────────────────────────────
# Own the process name. The launcher below is a shell script that `exec`s the
# host python3, and on a framework build that path resolves into
# Python.framework/…/Resources/Python.app — the exec'd image, and therefore
# `NSBundle.mainBundle()`, is *Python.app*. TCC identity is a property of the
# bundle you launch; the Dock tile name is a property of the executable that
# ends up running. This bundle wins the first and loses the second, so the Dock
# says "Python" and the icon set here never gets a name to sit under.
# For an app that is called VibeVoice:
#   daily driver  →  bash packaging/install_dev_app.sh   (live source)
#   distribution  →  bash packaging/build_release.sh     (self-contained)
#
# Usage:
#   ./build_app.sh            # builds ./dist/VibeVoice.app
#   ./build_app.sh /tmp/out   # builds /tmp/out/VibeVoice.app
#
# Verify (no mic, no launch):
#   plutil -lint dist/VibeVoice.app/Contents/Info.plist
#   ls -R dist/VibeVoice.app
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$SRC/dist}"
APP="$OUT/VibeVoice.app"
VERSION="0.2.0"
BUNDLE_ID="com.vibevoice.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# ── source the launcher runs (self-contained under Resources/) ────────────────
cp "$SRC/vibevoice.py" "$SRC/engine.py" "$SRC/autosend.py" \
   "$SRC/requirements.txt" "$APP/Contents/Resources/"
# The LED master, landed under the plist's CFBundleIconFile name. The file
# literally called VibeVoice.icns is the RETIRED teal waveform that BRAND.md
# forbids by name — it stayed wired here after the mark changed.
cp "$SRC/assets/icon/VibeVoice_LED.icns" "$APP/Contents/Resources/VibeVoice.icns"

# ── Info.plist — the bundle's identity + usage strings ────────────────────────
# LSUIElement matches the app's NSApplicationActivationPolicyAccessory (notch
# pill, no Dock icon). The Usage strings are what macOS shows in the permission
# prompts; without them the prompts (and thus mic/keys) silently fail.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>VibeVoice</string>
    <key>CFBundleDisplayName</key>
    <string>VibeVoice</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>VibeVoice</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>VibeVoice transcribes your voice into text, locally on your Mac.</string>
    <key>NSAppleEventsUsageDescription</key>
    <string>VibeVoice uses AppleScript to detect the frontmost app and press Return after dictation.</string>
    <key>CFBundleIconFile</key>
    <string>VibeVoice</string>
</dict>
</plist>
PLIST

# ── launcher: find a python3 with the deps, then run the pill ─────────────────
# The pill autostarts the engine (VIBEVOICE_ENGINE_AUTOSTART=1), so this single
# executable brings up the whole capture -> transcribe -> paste stack.
cat > "$APP/Contents/MacOS/VibeVoice" <<'LAUNCH'
#!/bin/bash
# VibeVoice.app launcher (MIT). Runs the pill, which autostarts the engine.
set -u
RES="$(cd "$(dirname "$0")/../Resources" && pwd)"

# Prefer a python3 that actually has the deps. Homebrew first (where pip installs
# usually land), then PATH, then the system interpreter as a last resort.
PY=""
for cand in /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3 2>/dev/null)" /usr/bin/python3; do
    [ -n "$cand" ] && [ -x "$cand" ] || continue
    if "$cand" -c "import objc, AppKit" >/dev/null 2>&1; then PY="$cand"; break; fi
    [ -z "$PY" ] && PY="$cand"   # remember a fallback even if deps missing
done

# If nothing can import the GUI deps, tell the user how to fix it (no silent fail).
if [ -z "$PY" ] || ! "$PY" -c "import objc, AppKit" >/dev/null 2>&1; then
    /usr/bin/osascript -e 'display dialog "VibeVoice needs its Python dependencies.\n\nOpen Terminal and run:\n\n    pip3 install -r requirements.txt\n\n(pyobjc, mlx-whisper, sounddevice, numpy)\n\nThen reopen VibeVoice." with title "VibeVoice — missing dependencies" buttons {"OK"} default button "OK" with icon caution' >/dev/null 2>&1
    exit 1
fi

# All-in-one defaults — override by exporting before launch if you like.
export VIBEVOICE_ENGINE_AUTOSTART="${VIBEVOICE_ENGINE_AUTOSTART:-1}"
export VIBEVOICE_LANG="${VIBEVOICE_LANG:-en}"
export VIBEVOICE_AUTOSEND="${VIBEVOICE_AUTOSEND:-1}"
export VIBEVOICE_AUTOSEND_RETURN="${VIBEVOICE_AUTOSEND_RETURN:-1}"

exec "$PY" "$RES/vibevoice.py"
LAUNCH
chmod +x "$APP/Contents/MacOS/VibeVoice"

# ── classic PkgInfo (harmless, expected by some tools) ────────────────────────
printf 'APPL????' > "$APP/Contents/PkgInfo"

echo "Built: $APP"
echo "Open with:  open \"$APP\"   (first launch will prompt for Microphone + Accessibility)"
