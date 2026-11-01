#!/bin/bash
# SPDX-License-Identifier: MIT
# Sign VibeVoice.app (Developer ID, hardened runtime), package a DMG, and
# notarize if a notarytool keychain profile exists. Run from repo root AFTER
# packaging/build_release.sh:  bash packaging/sign_and_dmg.sh
set -euo pipefail
cd "$(dirname "$0")/.."

APP="packaging/dist/VibeVoice.app"
DMG="packaging/dist/VibeVoice.dmg"
IDENTITY="Developer ID Application: MATTIA CALASTRI (54582KN4KW)"
NOTARY_PROFILE="${NOTARY_PROFILE:-vibevoice-notary}"
test -d "$APP" || { echo "missing $APP — run packaging/build_release.sh first"; exit 1; }

ENTITLEMENTS="$(mktemp -t vibevoice-entitlements).plist"
# Hardened runtime blocks exactly what a py2app+MLX app needs, so re-allow:
#   audio-input                       — mic capture (TCC prompt still applies)
#   allow-unsigned-executable-memory  — MLX Metal JIT + ctypes trampolines
#   disable-library-validation        — python .so files are signed by us, not Apple
cat > "$ENTITLEMENTS" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.device.audio-input</key><true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
</dict></plist>
EOF

echo "── sign: nested Mach-O first, then the bundle ──"
find "$APP" \( -name "*.so" -o -name "*.dylib" -o -name "*.metallib" \) -print0 |
  xargs -0 -n 50 codesign --force --timestamp --options runtime --sign "$IDENTITY" 2>/dev/null
# Frameworks + embedded executables (python binary inside Resources, if any)
find "$APP/Contents" -type f -perm +111 ! -name "*.py*" ! -name "*.so" ! -name "*.dylib" -print0 |
  xargs -0 -n 20 codesign --force --timestamp --options runtime --sign "$IDENTITY" 2>/dev/null || true
# Python.framework's main dylib is named "Python" (no extension) and shipped
# mode 644 — both finds above skip it, and the notary rejects the DMG for that
# one file (submission a3e90245). Sign the versioned framework explicitly.
for fw in "$APP"/Contents/Frameworks/*.framework; do
  [ -d "$fw" ] || continue
  ver="$(ls "$fw/Versions" 2>/dev/null | grep -v Current | head -1)"
  [ -n "$ver" ] && codesign --force --timestamp --options runtime --sign "$IDENTITY" "$fw/Versions/$ver"
done
# The engine runs under Contents/MacOS/python, a SECOND executable: the find
# above signed it without entitlements, so the hardened runtime killed it the
# moment numba/llvmlite asked for executable memory (SIGKILL, Code Signature
# Invalid, in LLVMPY_TryAllocateExecutableMemory — the 1.1.0 open test). It
# needs the same entitlements as the app, mic included.
if [ -f "$APP/Contents/MacOS/python" ]; then
  codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP/Contents/MacOS/python"
fi
codesign --force --timestamp --options runtime \
  --entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"

echo "── verify ──"
codesign --verify --strict --deep "$APP" && echo "codesign: OK"
spctl -a -vv "$APP" 2>&1 | tail -2 || echo "(spctl rejection is EXPECTED before notarization)"

# Built twice on purpose. The first DMG is the one that goes to Apple; the app
# inside it cannot carry a ticket yet, because the ticket does not exist until
# Apple has looked at it. Shipping THAT one leaves every user with an unstapled
# app: it opens while they have a network (Gatekeeper fetches the ticket) and
# fails the day they do not. So after stapling, the DMG is rebuilt around the
# now-stapled app and stapled itself — the ticket is already issued, `stapler`
# just fetches it again.
make_dmg() {
  rm -f "$DMG"
  STAGE="$(mktemp -d -t vibevoice-dmg)"
  cp -R "$APP" "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname "VibeVoice" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
  rm -rf "$STAGE"
  codesign --force --timestamp --sign "$IDENTITY" "$DMG"
}

echo "── notarize (best-effort) ──"
if [ -z "${SKIP_NOTARY:-}" ] && xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1; then
  # TWO submissions, in this order, and the order is the whole point. A ticket is
  # bound to the exact bytes it was issued for, so a DMG rebuilt after stapling is
  # a file Apple has never seen: `stapler` answers "Record not found". The app must
  # therefore be notarised and stapled BEFORE the DMG that carries it is built.
  echo "── notarize the app itself ──"
  APPZIP="$(mktemp -d)/App.zip"
  ditto -c -k --keepParent "$APP" "$APPZIP"
  xcrun notarytool submit "$APPZIP" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  rm -f "$APPZIP"

  echo "── dmg around the stapled app ──"
  make_dmg
  du -sh "$DMG"

  echo "── notarize the dmg ──"
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
  echo "notarized + stapled (the app inside the dmg carries its own ticket)"
else
  cat <<EOF
notarytool profile '$NOTARY_PROFILE' not found — SKIPPING notarization.
One-time human step (needs the Apple ID app-specific password):
  xcrun notarytool store-credentials $NOTARY_PROFILE \\
    --apple-id <apple-id-email> --team-id 54582KN4KW
then re-run: bash packaging/sign_and_dmg.sh
EOF
fi
echo "Done: $DMG"
