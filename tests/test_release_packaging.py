#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Release-packaging shape tests (full-duplex task 6).

The real gate for the self-contained bundle is `bash packaging/build_release.sh`
(CLAUDE.md rule 9: raw py2app output is import-dead — the post-build graft plus
the origin-asserted in-bundle smoke are what make it real). That build takes
minutes, so — same convention as test_child_spawns_never_use_raw_sys_executable
— these tests lock the load-bearing SHAPE of the packaging scripts instead of
rebuilding the app on every pytest run:

 - AVFoundation ships via py2app "includes" (F1 voice-processing capture is a
   lazy import, invisible to modulegraph);
 - onnxruntime stays OUT of the modulegraph scan (deliberate anti-crash
   exclude) and is grafted back post-build for the F2 Silero decider;
 - exactly one Silero model file is grafted to silero_vad/data/silero_vad.onnx
   (the path engine._resolve_silero_model probes), chosen by the op15-vs-full
   equivalence gate (GATE 1 decision);
 - the in-bundle smoke exercises onnxruntime + AVFoundation + an
   InferenceSession on the grafted model, with origin-in-bundle asserts.
"""
from __future__ import annotations

from pathlib import Path

from setup_options import py2app_options

REPO = Path(__file__).resolve().parent.parent
SETUP = REPO / "packaging" / "setup_py2app.py"
RELEASE = (REPO / "packaging" / "build_release.sh").read_text()


def _py2app_options() -> dict:
    """Extract the OPTIONS dict via AST — importing setup_py2app.py would
    execute setup() at module level and kick off a real build.

    The parsing lives in tests/setup_options.py: OPTIONS now references the
    module-level VERSION constant, and a bare `ast.literal_eval` chokes on the
    Name node. One reader, so a change to the packaging file cannot break one
    test module and leave the other passing (sess.9767)."""
    return py2app_options(SETUP)


# ── py2app scan configuration ─────────────────────────────────────────────────

def test_avfoundation_is_included():
    """engine._ensure_avfoundation imports AVFoundation lazily — modulegraph
    cannot see it, so it must be an explicit py2app include."""
    assert "AVFoundation" in _py2app_options()["includes"]


def test_onnxruntime_stays_excluded_from_the_scan():
    """Deliberate anti-crash exclusion: modulegraph blows up scanning the
    host's heavyweight ML cluster. onnxruntime ships via the post-build graft
    in build_release.sh, never via the scan."""
    assert "onnxruntime" in _py2app_options()["excludes"]


# ── post-build graft (build_release.sh) ───────────────────────────────────────

def test_release_grafts_onnxruntime():
    """onnxruntime must be in the graft list (mlx/tiktoken pattern), or the
    bundled engine silently degrades to the RMS threshold forever."""
    assert '"onnxruntime"' in RELEASE, "onnxruntime missing from the graft list"


def test_release_grafts_exactly_one_silero_model():
    """Only data/silero_vad.onnx ships — never the whole silero_vad tree
    (its real __init__ imports torch; the data dir carries ~7 MB of spare
    variants). The op15 16k-only variant is preferred when equivalent."""
    assert "silero_vad_16k_op15.onnx" in RELEASE, "op15 equivalence gate missing"
    assert 'os.path.join(dst_dir, "silero_vad.onnx")' in RELEASE, (
        "grafted model must land exactly at silero_vad/data/silero_vad.onnx — "
        "the only path engine._resolve_silero_model probes"
    )
    assert '"silero_vad", "__init__.py"' in RELEASE, (
        "a concrete (non-namespace) silero_vad/__init__.py is required or "
        "find_spec().origin is None and the engine never finds the model"
    )


def test_release_equivalence_gate_reads_thresholds_from_engine():
    """The op15-vs-full gate must source SILERO_ONSET/SILERO_OFFSET from
    engine.py instead of duplicating the literals: a hardcoded 0.5/0.35 would
    silently keep validating the OLD hysteresis if the engine constants change."""
    assert '"SILERO_ONSET"' in RELEASE and '"SILERO_OFFSET"' in RELEASE, (
        "equivalence gate no longer reads the hysteresis thresholds from engine.py"
    )
    assert "(a >= onset) == (b >= onset)" in RELEASE, (
        "onset gate comparison must use the value read from engine.py"
    )
    assert "(a <= offset) == (b <= offset)" in RELEASE, (
        "offset gate comparison must use the value read from engine.py"
    )


def test_release_purges_partial_zip_shadows():
    """The zip's partial tiktoken AND mlx must be purged: the synthesized
    REGULAR mlx package in the zip (stub __init__.pyc, missing _reprlib_fix)
    beats the grafted namespace-package tree at ANY sys.path position, making
    `import mlx.core` die inside the bundle."""
    assert "'tiktoken/*'" in RELEASE
    assert "'mlx/*'" in RELEASE


# ── in-bundle smoke (build_release.sh) ────────────────────────────────────────

def test_release_smoke_covers_onnxruntime_avfoundation_and_silero():
    """The smoke must import onnxruntime + AVFoundation from INSIDE the bundle
    (origin asserts) and run one InferenceSession on the grafted model —
    a green build without this proves nothing (import-dead bundle scar)."""
    assert "import onnxruntime" in RELEASE
    assert "onnxruntime leaked" in RELEASE, "onnxruntime origin assert missing"
    assert "import AVFoundation" in RELEASE
    assert "AVFoundation leaked" in RELEASE, "AVFoundation origin assert missing"
    assert "InferenceSession" in RELEASE, "smoke never runs the grafted model"
