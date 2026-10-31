#!/bin/bash
# SPDX-License-Identifier: MIT
# Install the DAILY-DRIVER VibeVoice.app and point the LaunchAgent at it.
# Run from anywhere:  bash packaging/install_dev_app.sh
#
# ── why this script exists ────────────────────────────────────────────────────
# The LaunchAgent used to run `/opt/homebrew/bin/python3 vibevoice.py`. That
# path resolves — through two symlinks — into
#   …/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python
# and a framework build of CPython re-executes itself there to become GUI
# capable. `_NSGetExecutablePath()` proves it: whatever you type on the command
# line, the running image lives inside *Python.app*. So `NSBundle.mainBundle()`
# is Python.app, and the Dock tile said **Python** — next to Obsidian, which
# says Obsidian.
#
# That cannot be fixed from inside the process. `_apply_app_identity()` in
# vibevoice.py patches the info dictionary and does win the icon, because the
# icon is a property of the live NSApplication; it never wins the NAME, because
# the name is read from the bundle LaunchServices registered at exec time. The
# only fix is to *be* a bundle: an executable that sits in Contents/MacOS and is
# not the framework interpreter.
#
# ── why ALIAS mode, and how this differs from build_release.sh ────────────────
# py2app's alias mode ships the same C launcher stub as a full build, but its
# Resources are SYMLINKS into this repo. So:
#   * the Dock says VibeVoice, with the LED icon and real metadata — the whole
#     point, identical to the release bundle
#   * `vibevoice.py` stays live: edit, restart the agent, you are running the
#     edit. A frozen release bundle as the daily driver means the code you read
#     and the code you run drift apart silently, which is the one failure this
#     repo's invariants exist to prevent
#   * it is NOT distributable — it needs this repo and the host's site-packages.
#     Distribution is `bash packaging/build_release.sh`, which stays the only
#     path allowed to produce a release (CLAUDE.md invariant #9). Alias mode is
#     not that target and does not touch packaging/dist.
# Both read their identity from the same OPTIONS dict in setup_py2app.py, so the
# daily driver and the shipped app cannot disagree about what they are.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
APP_DIR="$HOME/Applications"
APP="$APP_DIR/VibeVoice.app"
AGENT="$HOME/Library/LaunchAgents/com.vibevoice.pill.plist"
STAGE="$REPO/packaging/dist-dev"

echo "── build: alias bundle from $REPO ──"
rm -rf "$STAGE" "$REPO/packaging/build-dev"
python3 packaging/setup_py2app.py py2app -A \
    --dist-dir "$STAGE" --bdist-base "$REPO/packaging/build-dev" >/dev/null
test -x "$STAGE/VibeVoice.app/Contents/MacOS/VibeVoice"
# _child_python() spawns engine.py through Contents/MacOS/python: without it the
# pill would spawn itself through the launcher stub and fork a SECOND pill.
test -e "$STAGE/VibeVoice.app/Contents/MacOS/python"

# Ordering fix for py2app's site.py vs Homebrew's sitecustomize — see the
# docstring in packaging/sitecustomize_shim.py. Without it the bundle dies at
# init_import_site before it can draw a pixel. Alias-mode only: a release
# bundle carries its own stdlib and never sees Homebrew's file.
cp packaging/sitecustomize_shim.py "$STAGE/VibeVoice.app/Contents/Resources/sitecustomize.py"

echo "── install: $APP ──"
mkdir -p "$APP_DIR"
if [ -e "$APP" ]; then
    BAK="$APP.bak.$(date +%Y-%m-%d_%H%M%S)"
    mv "$APP" "$BAK"
    echo "   previous bundle kept at $BAK"
fi
ditto "$STAGE/VibeVoice.app" "$APP"

# LaunchServices caches name+icon per bundle path. Without this the Dock keeps
# showing whatever it learned from the copy it saw first (scar sess.9767: the
# new icon was on disk and the Dock still drew the old one).
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
    -f "$APP" >/dev/null 2>&1 || true

echo "── agent: point com.vibevoice.pill at the bundle ──"
python3 - "$AGENT" "$APP" <<'PYEOF'
import plistlib, sys
agent, app = sys.argv[1], sys.argv[2]
with open(agent, "rb") as fh:
    pl = plistlib.load(fh)
pl["ProgramArguments"] = [f"{app}/Contents/MacOS/VibeVoice"]
with open(agent, "wb") as fh:
    plistlib.dump(pl, fh)
print(f"   ProgramArguments -> {pl['ProgramArguments'][0]}")
PYEOF

DOMAIN="gui/$(id -u)"
launchctl bootout "$DOMAIN/com.vibevoice.pill" 2>/dev/null || true
# bootout returns before launchd has finished draining the service, and
# bootstrapping into that window fails with the unhelpful "Bootstrap failed: 5:
# Input/output error". Wait for the label to actually leave the domain.
for _ in $(seq 1 40); do
    launchctl print "$DOMAIN/com.vibevoice.pill" >/dev/null 2>&1 || break
    sleep 0.25
done
launchctl bootstrap "$DOMAIN" "$AGENT"

echo "── verify by effect ──"
for _ in $(seq 1 25); do
    PID="$(pgrep -f 'VibeVoice.app/Contents/MacOS/VibeVoice' | head -1 || true)"
    [ -n "$PID" ] && break
    sleep 0.4
done
test -n "${PID:-}" || { echo "FAIL: the pill did not come up"; exit 1; }
# lsappinfo reads the name LaunchServices registered — the same string the Dock
# tooltip draws. Asserting on it is the difference between "the plist says
# VibeVoice" and "the Dock says VibeVoice". Registration lands a beat AFTER the
# process appears, so poll: reading it once returned empty on a healthy app.
NAME=""
for _ in $(seq 1 25); do
    NAME="$(lsappinfo info -only name "$PID" | sed -n 's/.*"LSDisplayName"="\(.*\)"/\1/p')"
    [ -n "$NAME" ] && break
    sleep 0.4
done
echo "   pid $PID · LaunchServices name: ${NAME:-<none>}"
[ "$NAME" = "VibeVoice" ] || { echo "FAIL: the Dock would still say '${NAME}'"; exit 1; }
echo "OK — VibeVoice.app is the daily driver, and it is called VibeVoice."
