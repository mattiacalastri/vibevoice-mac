#!/bin/bash
# SPDX-License-Identifier: MIT
# Local Developer ID sign + DMG for VibeVoice, run BY MATTIA in his own Terminal.
# Fixes the "keychain keeps asking for the password" spam: unlock once + disable
# the idle auto-lock so codesign can hit the private key 260× without re-prompting.
# The partition-list (codesign: ACL) must already be set once (scar sess.9187).
#
#   cd ~/projects/💼-prodotti/vibevoice && bash packaging/sign_local.sh
#
set -uo pipefail
cd "$(dirname "$0")/.."

APP="packaging/dist/VibeVoice.app"
DMG="packaging/dist/VibeVoice.dmg"
IDENTITY="Developer ID Application: MATTIA CALASTRI (54582KN4KW)"
KC="$HOME/Library/Keychains/login.keychain-db"
test -d "$APP" || { echo "missing $APP — run packaging/build_release.sh first"; exit 1; }

echo "── unlock keychain (one password prompt) + kill idle auto-lock ──"
security unlock-keychain "$KC"          # prompts ONCE, securely
security set-keychain-settings "$KC"    # no -t/-l → no timeout, no lock-on-sleep

ENT="$(mktemp -t vv-ent).plist"
cat > "$ENT" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.device.audio-input</key><true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
</dict></plist>
EOF

echo "── sign every Mach-O in the bundle (errors shown, don't stop on one) ──"
# Ask the FILE what it is, never its name or its permission bits. The old pass
# signed *.so/*.dylib/*.metallib plus whatever carried -perm +111; the embedded
# CPython — Contents/Frameworks/Python.framework/Versions/3.10/Python — is mode
# 644 and has no extension, so it matched neither test and shipped with py2app's
# ADHOC signature. `codesign --deep --verify` still said OK (adhoc IS a valid
# signature) and the script exited 0, so the defect was invisible until Apple
# rejected the DMG: "not signed with a valid Developer ID certificate" + "the
# signature does not include a secure timestamp" (submission 1702f77c,
# 27/09/2026). A green exit over a red artifact.
ok=0; fail=0
while IFS= read -r f; do
  if codesign --force --timestamp --options runtime --sign "$IDENTITY" "$f" 2>/tmp/vv_err.txt; then
    ok=$((ok + 1))
  else
    fail=$((fail + 1)); echo "  FAIL $f"; head -1 /tmp/vv_err.txt
  fi
done < <(find "$APP/Contents" -type f ! -name "*.py" ! -name "*.pyc" \
           -exec sh -c 'file -b "$1" | grep -q "Mach-O"' _ {} \; -print)
echo "  mach-o: ok=$ok fail=$fail"

echo "── sign the embedded frameworks as bundles ──"
# A framework is signed at its version directory: signing only the Mach-O inside
# leaves the framework bundle itself unsigned.
for fw in "$APP/Contents/Frameworks/"*.framework/Versions/*/; do
  [ -d "$fw" ] || continue
  case "$fw" in *"/Versions/Current/") continue ;; esac
  codesign --force --timestamp --options runtime --sign "$IDENTITY" "$fw" || echo "  FAIL $fw"
done

echo "── gate: no adhoc signature may survive ──"
# The whole defect was that adhoc passes --verify. Name it here, before Apple does.
adhoc=0
while IFS= read -r f; do
  if codesign -dv "$f" 2>&1 | grep -q "Signature=adhoc"; then
    adhoc=$((adhoc + 1)); echo "  ADHOC $f"
  fi
done < <(find "$APP/Contents" -type f ! -name "*.py" ! -name "*.pyc" \
           -exec sh -c 'file -b "$1" | grep -q "Mach-O"' _ {} \; -print)
if [ "$adhoc" -gt 0 ]; then
  echo "  $adhoc adhoc binaries left — the notary would reject this DMG. Stopping."
  exit 1
fi
echo "  adhoc: 0"

echo "── sign the bundle (entitlements + hardened runtime) ──"
codesign --force --timestamp --options runtime --entitlements "$ENT" --sign "$IDENTITY" "$APP"

echo "── verify ──"
codesign --verify --strict --deep "$APP" && echo "  codesign: OK"
codesign -dv --verbose=2 "$APP" 2>&1 | grep -E "Authority=Developer|flags"

echo "── dmg ──"
rm -f "$DMG"
STAGE="$(mktemp -d -t vv-dmg)"
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "VibeVoice" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
echo "  DMG signed:"; du -sh "$DMG"

echo "── notarize (needs the one-time app-specific password profile) ──"
if xcrun notarytool history --keychain-profile polpo-notary >/dev/null 2>&1; then
  xcrun notarytool submit "$DMG" --keychain-profile polpo-notary --wait
  xcrun stapler staple "$APP" && xcrun stapler staple "$DMG" && echo "  notarized + stapled"
else
  echo "  notarytool profile 'polpo-notary' missing — skipping (see gesture #3)."
fi
echo "DONE → $DMG"
