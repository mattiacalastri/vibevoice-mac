#!/bin/bash
# SPDX-License-Identifier: MIT
# Self-contained VibeVoice.app (py2app). Run from repo root: bash packaging/build_release.sh
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip show py2app >/dev/null || python3 -m pip install py2app
rm -rf packaging/build packaging/dist
python3 packaging/setup_py2app.py py2app --dist-dir packaging/dist --bdist-base packaging/build
APP="packaging/dist/VibeVoice.app"

echo "── graft: packages modulegraph cannot see ──"
# modulegraph's AST scan misses these at build time:
#   mlx, tiktoken_ext — PEP 420 namespace packages (no __init__.py)
#   tiktoken          — bundled PARTIALLY into the zip (py files reachable from
#                       annotations, but not core/__init__/the rust .so)
#   tqdm/filelock/certifi — behind huggingface_hub's lazy __getattr__ imports
#   llvmlite          — loaded by numba through ctypes/dylib indirection
#   onnxruntime       — DELIBERATELY excluded from the scan in setup_py2app.py
#                       (modulegraph crashes on the host's ML cluster); the F2
#                       Silero decider lazy-imports it, so graft the real tree
# Graft the full trees from the build machine's site-packages, and purge the
# partial tiktoken AND mlx from the zip so they cannot shadow the grafted
# copies. mlx is the worst case: it has no __init__.py (namespace package), so
# the zip's synthesized REGULAR mlx package (stub __init__.pyc + core.pyc
# extension stub, missing _reprlib_fix) beats the grafted tree at ANY sys.path
# position — regular packages always win over namespace portions.
BUNDLE_LIB="$APP/Contents/Resources/lib/python3.10"
# llvmlite too: its .pyc land in the zip, the zip's copy wins over the grafted
# tree, and it then looks for libllvmlite.dylib INSIDE the zip (where the purge
# below removes every binary). Only the bundle's own interpreter hits this —
# found by the engine-interpreter smoke on 1.1.0.
zip -d "$APP/Contents/Resources/lib/python310.zip" 'tiktoken/*' 'mlx/*' 'llvmlite/*' >/dev/null 2>&1 || true
# Purge EVERY Mach-O from the zip: CPython cannot dlopen an extension out of a
# zipfile, so any .so/.dylib in there is inert weight — and the notary service
# scans inside the zip, finds them unsigned (codesign cannot reach them), and
# rejects the whole DMG with "Invalid" (measured: submission a7917e47, 8 issues,
# all zip-internal binaries). The real llvmlite/objc binaries live grafted or
# in lib-dynload, both signable.
zip -d "$APP/Contents/Resources/lib/python310.zip" '*.so' '*.dylib' '*.so.dSYM/*' >/dev/null 2>&1 || true
python3 - "$BUNDLE_LIB" <<'PYEOF'
import importlib.util, os, shutil, sys
bundle = sys.argv[1]
for pkg in ["mlx", "tiktoken", "tiktoken_ext", "tqdm", "filelock", "certifi", "llvmlite",
            "onnxruntime"]:
    dst = os.path.join(bundle, pkg)
    if os.path.exists(dst):
        continue
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"graft: cannot locate package {pkg} on the build machine")
    shutil.copytree(list(spec.submodule_search_locations)[0], dst)
    print(f"grafted {pkg}")
# sounddevice's cffi binding (_sounddevice.py) and its PortAudio dylib
# (_sounddevice_data/portaudio-binaries) are reached through cffi and ctypes,
# which modulegraph cannot follow. Up to 1.1.0 neither was in the bundle and the
# build stayed green, because the smoke imported sounddevice with the HOST's
# site-packages behind it; the shipped engine died on «import sounddevice».
for mod in ["_sounddevice"]:
    spec = importlib.util.find_spec(mod)
    if spec is None or not spec.origin:
        sys.exit(f"graft: cannot locate module {mod} on the build machine")
    shutil.copy2(spec.origin, os.path.join(bundle, os.path.basename(spec.origin)))
    print(f"grafted {mod}")
for pkg in ["_sounddevice_data"]:
    dst = os.path.join(bundle, pkg)
    spec = importlib.util.find_spec(pkg)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"graft: cannot locate package {pkg} on the build machine")
    if not os.path.exists(dst):
        shutil.copytree(list(spec.submodule_search_locations)[0], dst)
    print(f"grafted {pkg}")
PYEOF

echo "── graft: Silero VAD model (op15 16k variant when equivalent) ──"
# engine._resolve_silero_model probes silero_vad/data/silero_vad.onnx next to a
# CONCRETE silero_vad/__init__.py (find_spec on a namespace package has
# origin=None and the cascade fails). Ship ONLY the model file: the real
# silero_vad __init__ imports torch, and its data dir carries ~7 MB of spare
# variants (.jit, half, op18) the engine never reads.
# GATE 1 decision: prefer the 1.3 MB 16k-only op15 variant, but only after
# proving it equivalent to the full 2.3 MB model ON THIS BUILD MACHINE —
# identical state-threaded probabilities (tol 1e-4) on a synthetic
# silence→speech→silence sweep AND identical SILERO_ONSET/SILERO_OFFSET
# hysteresis gate decisions. Any doubt or error → ship the full model.
python3 - "$BUNDLE_LIB" <<'PYEOF'
import ast, importlib.util, os, shutil, sys
import numpy as np
bundle = sys.argv[1]

# Hysteresis thresholds: read SILERO_ONSET/SILERO_OFFSET from engine.py (the
# single source of truth) — a literal copy here would keep silently validating
# the OLD gates if the engine constants ever change. Parsed via AST, not
# imported: importing engine.py would touch ~/.vibevoice/ at build time.
gates = {}
for node in ast.walk(ast.parse(open("engine.py").read())):
    if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("SILERO_ONSET", "SILERO_OFFSET")):
        gates[node.targets[0].id] = ast.literal_eval(node.value)
if set(gates) != {"SILERO_ONSET", "SILERO_OFFSET"}:
    sys.exit(f"graft: SILERO_ONSET/SILERO_OFFSET not found in engine.py ({gates})")
onset, offset = gates["SILERO_ONSET"], gates["SILERO_OFFSET"]

spec = importlib.util.find_spec("silero_vad")
if spec is None or not spec.origin:
    sys.exit("graft: cannot locate package silero_vad on the build machine")
data = os.path.join(os.path.dirname(spec.origin), "data")
full = os.path.join(data, "silero_vad.onnx")
op15 = os.path.join(data, "silero_vad_16k_op15.onnx")
if not os.path.exists(full):
    sys.exit(f"graft: {full} does not exist")

def probs(model_path):
    """State-threaded per-frame speech probabilities, engine-identical feeding
    (64-sample context + 512-sample frame, state (2,1,128), sr int64)."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros(64, dtype=np.float32)
    sr = np.array(16000, dtype=np.int64)
    rng = np.random.default_rng(9465)
    t = np.arange(512 * 40) / 16000.0
    speech = (0.3 * np.sin(2 * np.pi * 160 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
              + 0.05 * rng.standard_normal(t.size))
    quiet = 0.005 * rng.standard_normal(512 * 20)
    sig = np.concatenate([quiet, speech, quiet]).astype(np.float32)
    out = []
    for off in range(0, sig.size - 511, 512):
        x = np.concatenate([ctx, sig[off:off + 512]]).reshape(1, -1)
        o = sess.run(None, {"input": x, "state": state, "sr": sr})
        state = np.asarray(o[1], dtype=np.float32)
        ctx = x[0, -64:]
        out.append(float(np.asarray(o[0]).reshape(-1)[0]))
    return np.array(out)

chosen = full
if os.path.exists(op15):
    try:
        a, b = probs(full), probs(op15)
        same_probs = float(np.abs(a - b).max()) <= 1e-4
        same_gates = bool(((a >= onset) == (b >= onset)).all()
                          and ((a <= offset) == (b <= offset)).all())
        if same_probs and same_gates:
            chosen = op15
        else:
            print(f"silero: op15 NOT equivalent (max prob diff "
                  f"{float(np.abs(a - b).max()):.6f}) — shipping the full model")
    except Exception as err:
        print(f"silero: equivalence check failed ({err}) — shipping the full model")
dst_dir = os.path.join(bundle, "silero_vad", "data")
os.makedirs(dst_dir, exist_ok=True)
open(os.path.join(bundle, "silero_vad", "__init__.py"), "w").close()
shutil.copy2(chosen, os.path.join(dst_dir, "silero_vad.onnx"))
print(f"grafted silero model: {os.path.basename(chosen)} "
      f"({os.path.getsize(chosen)} bytes) -> silero_vad/data/silero_vad.onnx")
PYEOF

echo "── scrub: the build machine's home path out of the bundle ──"
# A .pyc records the absolute path of the .py it was compiled from, so every
# grafted package carried the builder's home — measured 27/09/2026 on the first
# shareable DMG: 1.523 files under Resources/lib/python3.10 and 28 inside
# python310.zip spelling out /Users/<builder>/Library/Python/3.10/... . Nothing
# secret, but a name we never chose to ship, in an artifact meant for other
# people. Two different cures, because the two halves differ:
#   * the grafted trees still have their .py sources → recompile in place with
#     the recorded path rewritten (the .pyc stays, so cold start does not regress)
#   * python310.zip holds .pyc without sources → read the path each one records,
#     recompile from THAT source on the build machine, and put the clean .pyc
#     back at the same entry
#
# What NOT to do, measured: deleting the `tests` directories looked like the
# obvious cure (dead weight AND the bulk of the leak) and broke the bundle —
# `numpy.testing` imports `numpy._core.tests._natype`, so a plain
# `import scipy` died with ModuleNotFoundError. A test directory is not
# automatically inert. The gate below caught it before the DMG existed.
python3 -m compileall -q -f -s "$BUNDLE_LIB" -p "/VibeVoice/lib/python3.10" \
    "$BUNDLE_LIB" >/dev/null 2>&1 || true
python3 - "$APP/Contents/Resources/lib/python310.zip" "$HOME" <<'PYEOF'
import importlib.util, marshal, os, py_compile, sys, tempfile, zipfile

zip_path, home = sys.argv[1], sys.argv[2]
HEADER = 16            # CPython 3.7+ pyc header: magic + flags + mtime + size
# marshal.loads here reads a code object this very script produced seconds ago,
# out of the bundle we are building — not third-party input. It is used only to
# read `co_filename`; the code is never executed. assert keeps that assumption
# honest if the interpreter's magic ever changes under us.
assert importlib.util.MAGIC_NUMBER, "no pyc magic — wrong interpreter?"
cleaned, skipped = [], []

with zipfile.ZipFile(zip_path) as z:
    names = [n for n in z.namelist() if n.endswith(".pyc")]
    for name in names:
        blob = z.read(name)
        try:
            code = marshal.loads(blob[HEADER:])
        except Exception:
            continue
        src = code.co_filename
        if home not in src:
            continue
        if not os.path.isfile(src):
            skipped.append(name)
            continue
        # dfile is the path the new .pyc will record — the whole point.
        dest = "/VibeVoice/lib/python310.zip/" + name
        tmp = tempfile.NamedTemporaryFile(suffix=".pyc", delete=False).name
        try:
            py_compile.compile(src, cfile=tmp, dfile=dest, doraise=True)
            cleaned.append((name, tmp))
        except Exception:
            skipped.append(name)

# zipfile cannot replace an entry in place: rewrite the archive once.
if cleaned:
    replacement = {n: t for n, t in cleaned}
    tmp_zip = zip_path + ".scrub"
    with zipfile.ZipFile(zip_path) as src_z, \
         zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src_z.infolist():
            if item.filename in replacement:
                with open(replacement[item.filename], "rb") as fh:
                    out.writestr(item, fh.read())
            else:
                out.writestr(item, src_z.read(item.filename))
    os.replace(tmp_zip, zip_path)
    for _, t in cleaned:
        os.unlink(t)

print(f"  zip: {len(cleaned)} .pyc recompiled clean, {len(skipped)} unresolved")
PYEOF

echo "── gate: no build-machine path may survive ──"
# The leak was invisible because nothing looked for it. Look, and name it — on
# BOTH halves: grepping the bundle alone would call the zip clean, since its
# .pyc are deflated and no plain-text search can reach them.
# `|| true` is load-bearing on both counts: grep exits 1 when it finds nothing,
# find propagates it, and under `set -euo pipefail` the clean case would kill
# the script — a gate that aborts the build precisely when it has nothing to
# report. Measured here: the first run died on this line, and the wrapper's exit
# code still read 0, so it looked like a build that simply stopped talking.
leaks=$(find "$APP/Contents" -type f -exec grep -l "$HOME" {} + 2>/dev/null | wc -l | tr -d ' ' || true)
ZTMP="$(mktemp -d -t vv-zipgate)"
( cd "$ZTMP" && unzip -q "$APP/Contents/Resources/lib/python310.zip" 2>/dev/null ) || true
zleaks=$(find "$ZTMP" -type f -exec grep -l "$HOME" {} + 2>/dev/null | wc -l | tr -d ' ' || true)
rm -rf "$ZTMP"
if [ "$leaks" != "0" ] || [ "$zleaks" != "0" ]; then
  echo "  $leaks files in the bundle and $zleaks inside python310.zip still carry $HOME — not shippable:"
  find "$APP/Contents" -type f -exec grep -l "$HOME" {} + 2>/dev/null | head -5
  exit 1
fi
echo "  build-machine paths: 0 in the bundle, 0 in python310.zip"

# Our own sources ship as readable .py next to the app. The comments used to
# carry who asked for what, internal session ids and the names of the
# author's other tools (scrubbed for 1.1.0); this keeps them from coming back.
# Third-party libraries are not scanned: the markers are ours, not theirs.
PRIVATE_MARKERS='[Mm]attia|[Pp]olpo|sess\.[0-9]{3,}|[Nn]otch [Aa]genti|ElevenLabs|Telegram|Astra Digital'
private=$(grep -E -l "$PRIVATE_MARKERS" "$APP"/Contents/Resources/*.py 2>/dev/null || true)
if [ -n "$private" ]; then
  echo "  personal or internal references in the shipped sources — not shippable:"
  grep -E -n "$PRIVATE_MARKERS" "$APP"/Contents/Resources/*.py | head -10
  exit 1
fi
echo "  personal references in our shipped sources: 0"

echo "── smoke: bundle is self-contained ──"
plutil -lint "$APP/Contents/Info.plist"
test -x "$APP/Contents/MacOS/VibeVoice"
test -d "$APP/Contents/Resources/lib" && echo "embedded python: OK"
# Import the ML stack from INSIDE the bundle and assert nothing leaks from the
# host's site-packages (python3 -I still sees the global site dir, so each
# module's __file__ must be checked explicitly).
python3 -I - "$BUNDLE_LIB" <<'PYEOF'
import os, sys
bundle = os.path.abspath(sys.argv[1])
sys.path.insert(0, bundle)
import mlx.core as mc
import mlx_whisper, numpy, sounddevice, tiktoken
for m in (mlx_whisper, tiktoken, numpy):
    assert m.__file__.startswith(bundle), f"{m.__name__} leaked: {m.__file__}"
assert mc.__file__.startswith(bundle), f"mlx.core leaked: {mc.__file__}"
enc = tiktoken.get_encoding("gpt2")
assert enc.decode(enc.encode("vibevoice")) == "vibevoice"
print("import smoke: OK —", mc.default_device())

# F2: onnxruntime grafted + the Silero model answers from INSIDE the bundle.
import onnxruntime
assert onnxruntime.__file__.startswith(bundle), f"onnxruntime leaked: {onnxruntime.__file__}"
model = os.path.join(bundle, "silero_vad", "data", "silero_vad.onnx")
assert os.path.isfile(model), f"silero model not grafted: {model}"
sess = onnxruntime.InferenceSession(model, providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": numpy.zeros((1, 576), dtype=numpy.float32),
                      "state": numpy.zeros((2, 1, 128), dtype=numpy.float32),
                      "sr": numpy.array(16000, dtype=numpy.int64)})
prob = float(numpy.asarray(out[0]).reshape(-1)[0])
assert 0.0 <= prob <= 1.0, f"silero inference out of range: {prob}"
print("onnxruntime smoke: OK —", onnxruntime.__version__)
print(f"silero smoke: OK — {os.path.getsize(model)} bytes, p(silence)={prob:.3f}")
PYEOF
# F1: AVFoundation importable from the bundle. py2app puts pyobjc pure modules
# in python310.zip and their extensions in lib-dynload — a SEPARATE -I
# invocation, because putting the zip on the ML smoke's path would shadow the
# host stdlib with the bundle's ssl.pyc, whose _ssl.so resolves
# @executable_path against the HOST interpreter and dlopen-fails (the real app
# resolves it against Contents/MacOS and is fine).
python3 -I - "$BUNDLE_LIB" <<'PYEOF'
import os, sys
bundle = os.path.abspath(sys.argv[1])
res = os.path.dirname(os.path.dirname(bundle))          # .../Contents/Resources
sys.path[:0] = [os.path.join(os.path.dirname(bundle), "python310.zip"),
                os.path.join(bundle, "lib-dynload")]
import AVFoundation
assert AVFoundation.__file__.startswith(res), f"AVFoundation leaked: {AVFoundation.__file__}"
print("AVFoundation smoke: OK")
PYEOF

echo "── smoke: the engine's interpreter, as the app spawns it ──"
# The checks above import with the BUILD machine's python, which quietly fills
# any gap from its own site-packages — that is how two releases shipped an
# engine that could not start. This one runs the bundle's own Contents/MacOS/
# python with an empty environment plus what vibevoice._clean_child_env sets,
# and requires every module to come from inside the app.
RES="$(cd "$APP/Contents/Resources" && pwd)"
env -i HOME="$HOME" PYTHONHOME="$RES" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$APP/Contents/MacOS/python" - "$RES" <<'PYEOF'
import os, sys
res = sys.argv[1]
import struct, queue, argparse
import numpy, sounddevice, _sounddevice, mlx.core, mlx_whisper, onnxruntime, AVFoundation
for m in (struct, queue, argparse, numpy, sounddevice, _sounddevice, mlx.core, mlx_whisper,
          onnxruntime, AVFoundation):
    f = os.path.abspath(getattr(m, "__file__", "") or "")
    assert f.startswith(res), f"{m.__name__} comes from outside the bundle: {f}"
# sounddevice tries a system PortAudio first (Homebrew's, on the builder) and
# falls back to its own copy; on a user's Mac that copy is the one that loads.
# So require the bundled copy to be present and loadable, whatever won here.
import _sounddevice_data
bundled = os.path.join(next(iter(_sounddevice_data.__path__)), "portaudio-binaries", "libportaudio.dylib")
assert os.path.abspath(bundled).startswith(res) and os.path.isfile(bundled), f"no bundled PortAudio: {bundled}"
_sounddevice.ffi.dlopen(bundled)
print("engine interpreter smoke: OK —", len(sounddevice.query_devices()), "audio devices")
PYEOF
echo "Built: $APP"
