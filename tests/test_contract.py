#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Contract tests for VibeVoice.

These exercise the real modules (no reimplementation) and lock down the
state-file contract that decouples the engine from the pill — the invariants
documented in AGENTS.md. They are designed to run headless in CI: no
microphone, no GUI, no Whisper model download. State is redirected to a tmp
dir so the live ~/.vibevoice/ runtime is never touched.

Run:  pytest -q
"""
from __future__ import annotations

import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

import engine

# The pill (vibevoice.py) reads levels.bin with a hard-coded `struct.unpack("<60f", ...)`.
# That magic number lives on the reader side; this is the single source that must agree.
# `test_the_pill_really_unpacks_that_many_floats` checks this against the pill's
# own source, so the literal below cannot drift away from the reader unnoticed.
PILL_LEVELS_FORMAT = "<60f"
PILL_LEVELS_BYTES = 60 * 4


def _assert_only_swapped_in(live_path, write, seed=b"x"):
    """Run `write` twice and require that `live_path` was REPLACED, not rewritten.

    A writer that truncates the live file in place has a window in which a reader
    sees an empty or half-written file. Asserting "no tmp left behind" does not
    catch it — a direct write leaves no tmp either, because it never made one.

    The tell is the inode: `os.replace` swaps a different file into the name, so
    the inode changes; `write_text` reuses the same one. Asked of the filesystem
    rather than by patching `pathlib.Path` — patching the class reaches every
    thread in the process, and doing so took the suite from 13 s to over ten
    minutes (2026-08-06). A test must not cost more than the bug it guards.
    """
    live_path.write_bytes(seed)
    before = live_path.stat().st_ino
    write()
    after = live_path.stat().st_ino
    assert before != after, (
        f"{live_path.name} kept its inode — it was written in place, "
        "leaving a window where the pill reads a torn file"
    )


# ── levels.bin: the binary heartbeat (invariant #2) ───────────────────────────

@pytest.fixture
def engine_state(tmp_path, monkeypatch):
    """Redirect the engine's state files into a tmp dir, and disarm every path
    that reaches OUT of the process.

    EVERY module-level file the engine can write must be listed here: a full-flow
    test reaches process_utterance, and a path left unredirected leaks into the
    live ~/.vibevoice/ (scar sess.9685: seven fake entries in the real
    metrics.jsonl, written by this very suite before METRICS_FILE was patched).

    The same argument applies to the OUTBOUND switches, and it bites harder: a
    file written in the wrong place is recoverable, synthetic keystrokes posted
    into whatever app the user has in front of them are not. When the streaming
    paste landed (2026-08-02) `AUTOSEND` and `STREAM_PASTE` both defaulted to on,
    so the partial-pass tests typed "il polpo ha" into the user's screen and
    overwrote their clipboard. Off here by default: a test that wants to observe
    the paste turns it back on **and** replaces `type_text`/`autosend` with a
    recorder (see the `typed` fixture).
    """
    monkeypatch.setattr(engine, "AUTOSEND", False)
    monkeypatch.setattr(engine, "STREAM_PASTE", False)
    monkeypatch.setattr(engine, "LEVELS_FILE", tmp_path / "levels.bin")
    monkeypatch.setattr(engine, "LEVELS_TMP", tmp_path / "levels.tmp")
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state")
    monkeypatch.setattr(engine, "RAW_FILE", tmp_path / "raw.txt")
    monkeypatch.setattr(engine, "MUTED_FILE", tmp_path / "muted")
    monkeypatch.setattr(engine, "RETURN_HELD_FILE", tmp_path / "return_held")
    monkeypatch.setattr(engine, "HISTORY_FILE", tmp_path / "history.jsonl")
    monkeypatch.setattr(engine, "METRICS_FILE", tmp_path / "metrics.jsonl")
    monkeypatch.setattr(engine, "DICT_FILE", tmp_path / "dictionary.txt")
    monkeypatch.setattr(engine, "CORRECTIONS_FILE", tmp_path / "corrections.jsonl")
    monkeypatch.setattr(engine, "CORPUS_FILE", tmp_path / "corpus.jsonl")
    monkeypatch.setattr(engine, "PARTIAL_FILE", tmp_path / "partial.txt")
    monkeypatch.setattr(engine, "PARTIAL_TMP", tmp_path / "partial.tmp")
    monkeypatch.setattr(engine, "AUTOSEND_PAUSE_FLAG", tmp_path / "autosend_pause")
    return tmp_path


def _read_levels_as_pill(path) -> tuple[float, ...]:
    """Decode levels.bin exactly the way vibevoice.py's Controller does."""
    data = path.read_bytes()
    assert len(data) >= PILL_LEVELS_BYTES, "torn/short read — pill would skip the frame"
    return struct.unpack(PILL_LEVELS_FORMAT, data[:PILL_LEVELS_BYTES])


def test_levels_roundtrip_full(engine_state):
    """A full history writes 60 floats the pill can read back verbatim."""
    values = [i / 60.0 for i in range(engine.LEVELS_LEN)]
    engine.write_levels(deque(values, maxlen=engine.LEVELS_LEN))

    decoded = _read_levels_as_pill(engine.LEVELS_FILE)
    assert len(decoded) == 60
    assert decoded == pytest.approx(values, abs=1e-6)


def test_levels_left_padded_when_short(engine_state):
    """Fewer than 60 samples are left-padded with zeros — file is always 60 wide."""
    engine.write_levels(deque([0.5, 0.6, 0.7]))

    decoded = _read_levels_as_pill(engine.LEVELS_FILE)
    assert decoded[:-3] == pytest.approx([0.0] * 57)
    assert decoded[-3:] == pytest.approx([0.5, 0.6, 0.7])


def test_levels_keeps_last_60_when_long(engine_state):
    """More than 60 samples keep the most recent 60."""
    values = [float(i) for i in range(100)]
    engine.write_levels(deque(values))

    decoded = _read_levels_as_pill(engine.LEVELS_FILE)
    assert decoded == pytest.approx([float(i) for i in range(40, 100)])


def test_levels_write_is_atomic(engine_state):
    """The staging tmp file must not linger after an atomic os.replace."""
    engine.write_levels(deque([0.1] * engine.LEVELS_LEN))
    assert engine.LEVELS_FILE.exists()
    assert not engine.LEVELS_TMP.exists()


# ── Waveform cadence: the bars can only move as often as data arrives ────────

def test_levels_data_rate_is_at_least_twenty_per_second():
    """The pill redraws at 24 fps but can only *move* when a new RMS sample
    lands. At one block per 100 ms each bar was held for 2.4 frames and the
    scroll visibly stepped (reported from the live runtime, 2026-08-02).
    """
    blocks_per_sec = engine.SAMPLE_RATE / engine.BLOCKSIZE
    assert blocks_per_sec >= 20, f"only {blocks_per_sec:.0f} waveform samples/s"
    assert engine.LEVELS_HZ >= 20, "the writer would throttle away the extra samples"


def test_levels_write_rate_is_not_throttled_below_the_block_rate():
    """`_levels_every` must not silently drop back to one write per N blocks."""
    eng = engine.Engine()
    effective_hz = (engine.SAMPLE_RATE / engine.BLOCKSIZE) / eng._levels_every
    assert effective_hz >= 20, f"levels.bin written only {effective_hz:.0f} times/s"


def test_pre_roll_keeps_half_a_second_of_audio():
    """Pre-roll is a DURATION, not a block count: it is what saves the first
    syllable, and halving the block size must not halve it."""
    pre_roll_s = engine.PRE_ROLL_BLOCKS * engine.BLOCKSIZE / engine.SAMPLE_RATE
    assert pre_roll_s == pytest.approx(0.5, abs=0.06), f"{pre_roll_s:.2f}s of pre-roll"


def test_levels_len_matches_pill_magic():
    """Cross-side guard: the pill hard-codes 60; the engine must agree."""
    assert engine.LEVELS_LEN == 60


def test_the_pill_really_unpacks_that_many_floats():
    """Read the number out of the PILL, not out of this file.

    The guard above never opens vibevoice.py: it compares the engine against a
    `60` typed here. Change the pill to `struct.unpack("<48f", ...)` and both
    stay green while the VU bars read garbage — the writer and the reader
    disagreeing is the one thing the state-file contract exists to prevent.
    So parse every width the pill actually uses and require them all to match.
    """
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "vibevoice.py").read_text()
    widths = {int(n) for n in re.findall(r'struct\.unpack\(\s*"<(\d+)f"', src)}
    assert widths, "no struct.unpack of levels found in the pill — did it move?"
    assert widths == {engine.LEVELS_LEN}, (
        f"the pill unpacks {sorted(widths)} floats, the engine writes "
        f"{engine.LEVELS_LEN}"
    )

    # The length check guarding those unpacks must agree too: `len(data) < 60*4`
    # with a 48-float unpack would accept a short frame and read past it.
    guards = {int(n) for n in re.findall(r"len\(data\)\s*<\s*(\d+)\s*\*\s*4", src)}
    assert not guards - {engine.LEVELS_LEN}, (
        f"the pill guards on {sorted(guards)} samples, the engine writes "
        f"{engine.LEVELS_LEN}"
    )


# ── state / raw text files ────────────────────────────────────────────────────

def test_state_roundtrip(engine_state):
    for state in ("idle", "recording", "transcribing"):
        engine.write_state(state)
        assert engine.STATE_FILE.read_text() == state


def test_raw_roundtrip(engine_state):
    engine.write_raw("apri la dashboard")
    assert engine.RAW_FILE.read_text() == "apri la dashboard"


# ── mute control file: pill writes, engine reads (pause-not-kill contract) ─────

def test_is_muted_reflects_flag_file(engine_state):
    """`is_muted()` mirrors the presence of the muted control file."""
    assert engine.is_muted() is False
    engine.MUTED_FILE.touch()
    assert engine.is_muted() is True
    engine.MUTED_FILE.unlink()
    assert engine.is_muted() is False


def test_muted_engine_ignores_microphone(engine_state):
    """While muted, a loud block must NOT start recording — the mic is paused."""
    eng = engine.Engine()
    engine.MUTED_FILE.touch()
    loud = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)  # well above VAD
    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)
    assert eng._speaking is False
    assert engine.STATE_FILE.read_text() == "idle"


def test_unmuted_engine_starts_recording_on_speech(engine_state):
    """Without the mute flag, the same loud block starts an utterance — guards
    that the mute gate does not break the normal capture path."""
    eng = engine.Engine()
    loud = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)
    assert eng._speaking is True
    assert engine.STATE_FILE.read_text() == "recording"


def test_captured_blocks_survive_reused_callback_buffer(engine_state):
    """sounddevice hands the callback the SAME buffer every block. A stored
    block that is a view of it becomes the last block of the utterance — the
    silence at release — and Whisper answers «Grazie a tutti.» (measured
    2026-11-01: levels showed speech, every push-to-talk dictation dropped)."""
    eng = engine.Engine()
    shared = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
    eng._audio_callback(shared, engine.BLOCKSIZE, None, None)
    assert eng._speaking is True
    shared[:] = 0.0   # PortAudio overwrites the buffer with the next block
    assert float(np.abs(eng._buf[-1]).max()) == pytest.approx(0.5)


# ── Capture backend seam (F1: voice-processing capture + sounddevice fallback) ─

def test_capture_backend_falls_back_without_avfoundation(monkeypatch):
    """AVFoundation unavailable → the seam must select sounddevice."""
    monkeypatch.setattr(engine, "VP_ENABLED", True)
    monkeypatch.setattr(engine, "_ensure_avfoundation", lambda: False)
    backend = engine._select_capture_backend()
    assert backend is engine._SounddeviceCapture
    assert backend.name == "sounddevice"


def test_capture_backend_prefers_voice_processing(monkeypatch):
    """AVFoundation available + VIBEVOICE_VP on → voice-processing is selected."""
    monkeypatch.setattr(engine, "VP_ENABLED", True)
    monkeypatch.setattr(engine, "PTT", False)
    monkeypatch.setattr(engine, "_ensure_avfoundation", lambda: True)
    backend = engine._select_capture_backend()
    assert backend is engine._VoiceProcessingCapture
    assert backend.name == "voice-processing"


def test_push_to_talk_never_selects_voice_processing(monkeypatch):
    """Push-to-talk keeps its engine paused between dictations, and a paused
    voice-processing engine silences the built-in mic for every other app
    (measured 2026-10-31: peak 0.004 instead of 1.6 in another process, a
    wake-word listener heard nothing). So push-to-talk captures via sounddevice."""
    monkeypatch.setattr(engine, "VP_ENABLED", True)
    monkeypatch.setattr(engine, "PTT", True)
    monkeypatch.setattr(engine, "_ensure_avfoundation", lambda: True)
    assert engine._select_capture_backend() is engine._SounddeviceCapture


def test_vp_env_kill_switch_forces_sounddevice(monkeypatch):
    """VIBEVOICE_VP=0 must force sounddevice even when AVFoundation imports."""
    monkeypatch.setattr(engine, "VP_ENABLED", False)
    monkeypatch.setattr(engine, "_ensure_avfoundation", lambda: True)
    assert engine._select_capture_backend() is engine._SounddeviceCapture


def test_selection_survives_avfoundation_import_failure(monkeypatch, capsys):
    """Simulate a machine WITHOUT pyobjc-framework-AVFoundation: the real
    _ensure_avfoundation() must swallow the ImportError, print the install
    hint, and the seam must degrade to sounddevice (criterio 'è fatto' F1)."""
    import sys as _sys
    monkeypatch.setattr(engine, "_AVF", None)
    monkeypatch.setattr(engine, "_AVF_AVAILABLE", None)      # reset tri-state cache
    monkeypatch.setitem(_sys.modules, "AVFoundation", None)  # import → ImportError
    monkeypatch.setattr(engine, "VP_ENABLED", True)

    assert engine._ensure_avfoundation() is False
    assert engine._select_capture_backend() is engine._SounddeviceCapture
    assert "pyobjc-framework-AVFoundation" in capsys.readouterr().err


def test_run_falls_back_to_sounddevice_when_vp_open_fails(engine_state, monkeypatch, capsys):
    """A VP backend that raises at open time (mic permission, API error) must
    not kill the engine: run() logs the failure and retries via sounddevice."""
    class _BoomCapture:
        name = "voice-processing"

        def __init__(self, callback):
            pass

        def __enter__(self):
            raise RuntimeError("AVAudioEngine start failed")

        def __exit__(self, *exc):
            return False

    opened = {}

    class _FakeSounddevice:
        name = "sounddevice"

        def __init__(self, callback):
            opened["callback"] = callback

        def __enter__(self):
            opened["entered"] = True
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(engine, "_select_capture_backend", lambda: _BoomCapture)
    monkeypatch.setattr(engine, "_SounddeviceCapture", _FakeSounddevice)
    eng = engine.Engine()
    eng.stop()  # loop exits right after the stream opens — no blocking in CI
    eng.run()

    err = capsys.readouterr().err
    assert opened["entered"] is True
    assert opened["callback"] == eng._audio_callback
    assert "falling back to sounddevice" in err
    assert "capture: sounddevice" in err


def test_run_opens_capture_via_seam_and_logs_backend(engine_state, monkeypatch, capsys):
    """Engine.run() must open the mic through the selected backend and announce
    it on stderr ('capture: <name>') — headless-safe via a fake backend."""
    opened = {}

    class _FakeCapture:
        name = "fake"

        def __init__(self, callback):
            opened["callback"] = callback

        def __enter__(self):
            opened["entered"] = True
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(engine, "_select_capture_backend", lambda: _FakeCapture)
    eng = engine.Engine()
    eng.stop()  # loop exits right after the stream opens — no blocking in CI
    eng.run()

    assert opened["entered"] is True
    assert opened["callback"] == eng._audio_callback  # same signature/target as before
    assert "capture: fake" in capsys.readouterr().err


# ── Speech decider (F2: Silero VAD on a worker thread, RMS-threshold fallback) ─

class _FakeSileroSession:
    """Stand-in for onnxruntime.InferenceSession with a fixed speech
    probability. Echoes the recurrent state back incremented by 1 so the test
    can observe the state actually flowing through consecutive calls."""

    def __init__(self, prob: float) -> None:
        self.prob = prob
        self.feeds: list[dict] = []

    def run(self, _output_names, feeds):
        self.feeds.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return [
            np.array([[self.prob]], dtype=np.float32),
            np.asarray(feeds["state"], dtype=np.float32) + 1.0,
        ]


@pytest.fixture
def make_silero(monkeypatch):
    """Factory for a started SileroVad wired to a fake ONNX session; the
    fixture stops every started decider so no worker thread leaks."""
    started: list = []

    def _make(prob: float):
        fake = _FakeSileroSession(prob)
        monkeypatch.setattr(engine, "_resolve_silero_model", lambda: "fake.onnx")
        monkeypatch.setattr(engine, "_ensure_onnxruntime", lambda: True)
        monkeypatch.setattr(engine, "_create_silero_session", lambda path: fake)
        vad = engine.SileroVad().start()
        started.append(vad)
        return vad, fake

    yield _make
    for vad in started:
        vad.stop()


def _frames_after(n_blocks: int) -> int:
    """Frames the re-chunker must have consumed after `n_blocks` full blocks.

    Derived from the constants, never hard-coded: the law is "consume
    floor(samples / SILERO_FRAME), carry the rest", and it must keep holding
    when BLOCKSIZE changes (it went 1600 → 800 on 2026-08-02 to double the
    waveform's data rate, and four tests that had baked in 1600/512 = 3 went
    red — they were asserting an arithmetic coincidence, not the contract).
    """
    return (n_blocks * engine.BLOCKSIZE) // engine.SILERO_FRAME


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    """Poll `predicate` until true or timeout — the worker thread is async."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_silero_speech_probability_turns_speech_on(make_silero):
    """Fake session says speech (prob ≥ onset) → is_speech() flips True even on
    a silent block: with the model active, the model decides, not the energy."""
    vad, fake = make_silero(0.9)
    quiet = np.zeros(engine.BLOCKSIZE, dtype=np.float32)

    assert vad.is_speech() is False
    vad.submit(quiet, rms=0.0)
    vad.submit(quiet, rms=0.0)

    assert _wait_until(vad.is_speech), "worker never published a speech decision"
    # Re-chunk contract: whatever the block size, floor(samples/512) frames.
    assert _wait_until(lambda: len(fake.feeds) == _frames_after(2)), "worker did not re-chunk 2 blocks"
    for i, feed in enumerate(fake.feeds):
        # …each fed as context (64) + frame (512) at 16 kHz — the v5 model contract.
        assert feed["input"].shape == (1, engine.SILERO_CONTEXT + engine.SILERO_FRAME)
        assert int(feed["sr"]) == engine.SAMPLE_RATE
        # Recurrent state: call i must receive the state produced by call i-1.
        assert feed["state"].shape == (2, 1, 128)
        assert feed["state"] == pytest.approx(np.full((2, 1, 128), float(i)))


def test_silero_music_high_rms_low_prob_is_not_speech(make_silero):
    """Music: a block well above VAD_THRESHOLD but the model says non-speech →
    is_speech() stays False (the legacy energy VAD alone would have fired)."""
    vad, fake = make_silero(0.1)
    loud = np.full(engine.BLOCKSIZE, 0.5, dtype=np.float32)
    rms = 0.5
    assert rms >= engine.VAD_THRESHOLD  # sanity: the legacy threshold would say speech

    vad.submit(loud, rms=rms)
    assert _wait_until(lambda: len(fake.feeds) >= _frames_after(1)), "worker never consumed the block"
    assert vad.is_speech() is False


def test_silero_hysteresis_band_holds_previous_decision(make_silero):
    """Probabilities strictly inside the hysteresis band (SILERO_OFFSET < prob
    < SILERO_ONSET) must HOLD the previous decision in both directions — the
    anti-flap contract of SileroVad._infer (ARCHITECTURE.md §2.3). A regression
    collapsing either edge into the band (e.g. `elif prob <= SILERO_OFFSET` →
    `else`) flips these assertions."""
    vad, fake = make_silero(0.9)  # start above onset
    band = (engine.SILERO_OFFSET + engine.SILERO_ONSET) / 2
    assert engine.SILERO_OFFSET < band < engine.SILERO_ONSET  # sanity: strictly inside
    quiet = np.zeros(engine.BLOCKSIZE, dtype=np.float32)

    # Onset edge: prob ≥ SILERO_ONSET turns the decision ON.
    vad.submit(quiet, rms=0.0)
    assert _wait_until(vad.is_speech), "onset never turned speech on"
    # Wait for the block to be fully consumed before mutating the probability
    # (the worker is async); the frame count follows from the constants.
    assert _wait_until(lambda: len(fake.feeds) == _frames_after(1)), "first block not fully consumed"

    # Band while ON: the previous True must hold, not decay to False.
    fake.prob = band
    vad.submit(quiet, rms=0.0)
    assert _wait_until(lambda: len(fake.feeds) == _frames_after(2)), "second block not fully consumed"
    assert vad.is_speech() is True, "hysteresis band dropped the ON decision"

    # Offset edge: prob ≤ SILERO_OFFSET turns the decision OFF.
    fake.prob = 0.1
    vad.submit(quiet, rms=0.0)
    assert _wait_until(lambda: not vad.is_speech()), "offset never turned speech off"
    assert _wait_until(lambda: len(fake.feeds) == _frames_after(3)), "third block not fully consumed"

    # Band while OFF: the previous False must hold, not re-trigger True.
    fake.prob = band
    vad.submit(quiet, rms=0.0)
    assert _wait_until(lambda: len(fake.feeds) == _frames_after(4)), "fourth block not fully consumed"
    assert vad.is_speech() is False, "hysteresis band re-raised the OFF decision"


def test_silero_absent_is_bit_identical_to_rms_threshold(monkeypatch):
    """onnxruntime unavailable → the decider degrades to the energy threshold:
    is_speech() == (rms >= VAD_THRESHOLD) after every submit, exactly as today."""
    monkeypatch.setattr(engine, "_resolve_silero_model", lambda: "fake.onnx")
    monkeypatch.setattr(engine, "_ensure_onnxruntime", lambda: False)
    vad = engine.SileroVad().start()
    assert vad._worker is None  # no ONNX → no worker thread at all

    block = np.zeros(engine.BLOCKSIZE, dtype=np.float32)
    t = engine.VAD_THRESHOLD
    for rms in (0.0, t / 2, float(np.nextafter(t, 0.0)), t,
                float(np.nextafter(t, 1.0)), 2 * t, 0.5, 1.0):
        vad.submit(block, rms=rms)
        assert vad.is_speech() is (rms >= t), f"fallback drifted from threshold at rms={rms}"


class _ExplodingSileroSession(_FakeSileroSession):
    """_FakeSileroSession that raises after `explode_after` successful frames —
    drives the mid-stream inference-failure branch of SileroVad._run."""

    def __init__(self, prob: float, explode_after: int) -> None:
        super().__init__(prob)
        self.explode_after = explode_after

    def run(self, _output_names, feeds):
        if len(self.feeds) >= self.explode_after:
            raise RuntimeError("mid-stream inference failure")
        return super().run(_output_names, feeds)


def test_silero_midstream_failure_degrades_to_rms_fallback(monkeypatch, capsys):
    """Inference raising mid-stream (SileroVad._run's except branch): the
    decider must flip to the RMS fallback (_active=False) instead of freezing
    the last neural decision — the second leg of the ARCHITECTURE.md §2.3
    degradation contract (the onnxruntime-absent leg is locked by
    test_full_flow_without_onnxruntime_behaves_like_legacy)."""
    fake = _ExplodingSileroSession(prob=0.9, explode_after=_frames_after(1))
    monkeypatch.setattr(engine, "_resolve_silero_model", lambda: "fake.onnx")
    monkeypatch.setattr(engine, "_ensure_onnxruntime", lambda: True)
    monkeypatch.setattr(engine, "_create_silero_session", lambda path: fake)
    vad = engine.SileroVad().start()
    try:
        worker = vad._worker
        quiet = np.zeros(engine.BLOCKSIZE, dtype=np.float32)

        # Healthy stretch: the model (prob ≥ onset) decides speech on silence;
        # one block is exactly explode_after successful frames.
        vad.submit(quiet, rms=0.0)
        assert _wait_until(vad.is_speech), "model never published speech pre-failure"

        # The next frame raises inside the worker → degrade, announce, stop.
        vad.submit(quiet, rms=0.0)
        assert _wait_until(lambda: not worker.is_alive()), "worker survived the failure"
        assert vad._active is False
        assert "falling back to the RMS threshold" in capsys.readouterr().err

        # From here on the decision is exactly the energy threshold again: the
        # stale neural True must not leak through (frozen-decision scar).
        t = engine.VAD_THRESHOLD
        for rms in (0.0, t / 2, t, 2 * t):
            vad.submit(quiet, rms=rms)
            assert vad.is_speech() is (rms >= t), f"fallback drifted at rms={rms}"
    finally:
        vad.stop()


# ── Decider wired into the audio callback (F2, task 4) ────────────────────────

@pytest.fixture
def arm_engine_decider(monkeypatch):
    """Arm an Engine's own decider with a fake ONNX session (same seams as
    make_silero, but started on `eng._vad` so the callback wiring is exercised);
    stops every armed decider so no worker thread leaks."""
    armed: list = []

    def _arm(eng, prob: float):
        fake = _FakeSileroSession(prob)
        monkeypatch.setattr(engine, "_resolve_silero_model", lambda: "fake.onnx")
        monkeypatch.setattr(engine, "_ensure_onnxruntime", lambda: True)
        monkeypatch.setattr(engine, "_create_silero_session", lambda path: fake)
        eng._vad.start()
        armed.append(eng._vad)
        return fake

    yield _arm
    for vad in armed:
        vad.stop()


def test_callback_loud_music_not_recorded_when_model_says_nonspeech(
    engine_state, arm_engine_decider
):
    """Music: blocks well above VAD_THRESHOLD but the model says non-speech →
    the callback must NOT start an utterance. Proves the state machine is
    driven by the decider's decision, not by the raw RMS threshold."""
    eng = engine.Engine()
    fake = arm_engine_decider(eng, prob=0.1)
    loud = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
    assert 0.5 >= engine.VAD_THRESHOLD  # sanity: the legacy threshold would fire

    for _ in range(3):
        eng._audio_callback(loud, engine.BLOCKSIZE, None, None)
    assert _wait_until(lambda: len(fake.feeds) >= 3), "worker never consumed the blocks"
    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)  # decision now settled

    assert eng._speaking is False


def test_callback_quiet_speech_starts_recording_when_model_says_speech(
    engine_state, arm_engine_decider
):
    """The model hears speech in a block below the RMS threshold → the callback
    starts recording (with the legacy energy VAD this block was inaudible)."""
    eng = engine.Engine()
    arm_engine_decider(eng, prob=0.9)
    quiet = np.full((engine.BLOCKSIZE, 1), 0.001, dtype=np.float32)
    assert 0.001 < engine.VAD_THRESHOLD  # sanity: the legacy threshold stays silent

    # Pre-warm: inference is async — let the worker publish "speech" first.
    eng._vad.submit(quiet[:, 0], rms=0.001)
    assert _wait_until(eng._vad.is_speech), "worker never published a speech decision"

    eng._audio_callback(quiet, engine.BLOCKSIZE, None, None)

    assert eng._speaking is True
    assert engine.STATE_FILE.read_text() == "recording"


def test_levels_bin_identical_with_neural_vad_on_and_off(engine_state, arm_engine_decider):
    """Same audio → byte-identical levels.bin with the neural decider on or off:
    the decider replaces the speech DECISION only; the RMS→levels.bin pipeline
    must stay untouched (invariant #2)."""
    rng = np.random.default_rng(9465)
    blocks = [
        rng.normal(0.3, 0.05, (engine.BLOCKSIZE, 1)).astype(np.float32)
        for _ in range(12)
    ]  # loud enough that BOTH deciders say speech on every block

    def feed_all(eng) -> bytes:
        for b in blocks:
            eng._audio_callback(b, engine.BLOCKSIZE, None, None)
        assert eng._speaking is True  # guard: both paths really recorded
        return engine.LEVELS_FILE.read_bytes()

    # Neural ON (fake model: always speech; pre-warmed so the async worker
    # cannot skew the first decisions).
    eng_on = engine.Engine()
    arm_engine_decider(eng_on, prob=0.9)
    eng_on._vad.submit(blocks[0][:, 0], rms=0.5)
    assert _wait_until(eng_on._vad.is_speech), "worker never published a speech decision"
    with_neural = feed_all(eng_on)

    engine.LEVELS_FILE.unlink()

    # Neural OFF (unstarted decider → RMS fallback, exactly the legacy path).
    without_neural = feed_all(engine.Engine())

    assert with_neural == without_neural


def test_run_starts_and_stops_the_decider(engine_state, monkeypatch):
    """Engine.run() must bring the decider up before capture opens and stop its
    worker on the way out — the callback itself never manages lifecycles."""
    class _FakeCapture:
        name = "fake"

        def __init__(self, callback):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(engine, "_select_capture_backend", lambda: _FakeCapture)
    eng = engine.Engine()
    calls: list[str] = []
    monkeypatch.setattr(eng._vad, "start", lambda: calls.append("start"))
    monkeypatch.setattr(eng._vad, "stop", lambda: calls.append("stop"))
    eng.stop()  # loop exits right after the stream opens — no blocking in CI
    eng.run()
    assert calls == ["start", "stop"]


# ── Degradation lock-down (task 5: no onnxruntime / no AVFoundation) ──────────

def test_full_flow_without_onnxruntime_behaves_like_legacy(
    engine_state, monkeypatch, capsys
):
    """Machine WITHOUT onnxruntime (sys.modules poisoned, REAL
    _ensure_onnxruntime): the whole callback→finalize→transcribe flow must
    behave exactly as today — RMS-threshold decisions, same state-file
    transitions, transcription published to raw.txt + history."""
    import sys as _sys
    monkeypatch.setattr(engine, "_ORT", None)
    monkeypatch.setattr(engine, "_ORT_AVAILABLE", None)     # reset tri-state cache
    monkeypatch.setitem(_sys.modules, "onnxruntime", None)  # import → ImportError
    monkeypatch.setattr(engine, "_resolve_silero_model", lambda: "fake.onnx")
    # Deterministic finalize in CI: no real-time waits, no clipboard, no model.
    monkeypatch.setattr(engine, "SILENCE_SEC", 0.0)
    monkeypatch.setattr(engine, "MIN_DUR", 0.0)
    monkeypatch.setattr(engine, "AUTOSEND", False)
    monkeypatch.setattr(engine, "HISTORY_FILE", engine_state / "history.jsonl")
    monkeypatch.setattr(engine, "transcribe", lambda audio: "senza onnx")

    eng = engine.Engine()
    eng._vad.start()  # degrades in place: no worker thread, RMS fallback
    assert eng._vad._worker is None
    assert eng._vad._active is False
    assert "onnxruntime" in capsys.readouterr().err  # the real install hint fired

    loud = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
    quiet = np.zeros((engine.BLOCKSIZE, 1), dtype=np.float32)

    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)
    assert eng._speaking is True                            # RMS onset, exactly as today
    assert engine.STATE_FILE.read_text() == "recording"

    eng._audio_callback(quiet, engine.BLOCKSIZE, None, None)  # arms the silence timer
    eng._audio_callback(quiet, engine.BLOCKSIZE, None, None)  # SILENCE_SEC=0 → finalize

    assert eng._speaking is False
    assert _wait_until(
        lambda: engine.RAW_FILE.exists() and engine.RAW_FILE.read_text() == "senza onnx"
    ), "transcription never reached raw.txt on the degraded path"
    assert _wait_until(lambda: engine.STATE_FILE.read_text() == "idle")
    assert engine.HISTORY_FILE.exists()


def test_full_run_without_avfoundation_captures_via_sounddevice(
    engine_state, monkeypatch, capsys
):
    """Machine WITHOUT pyobjc-framework-AVFoundation (sys.modules poisoned,
    REAL _ensure_avfoundation): Engine.run() must degrade to sounddevice and
    the capture flow must keep working exactly as today under the fallback."""
    import sys as _sys
    monkeypatch.setattr(engine, "_AVF", None)
    monkeypatch.setattr(engine, "_AVF_AVAILABLE", None)      # reset tri-state cache
    monkeypatch.setitem(_sys.modules, "AVFoundation", None)  # import → ImportError
    monkeypatch.setattr(engine, "VP_ENABLED", True)
    monkeypatch.setattr(engine, "_resolve_silero_model", lambda: None)  # keep F2 out of the frame

    seen = {}

    class _FakeSounddevice:
        name = "sounddevice"

        def __init__(self, callback):
            self._callback = callback

        def __enter__(self):
            # Prove the degraded backend still drives the normal capture flow.
            loud = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
            self._callback(loud, engine.BLOCKSIZE, None, None)
            seen["state_after_loud"] = engine.STATE_FILE.read_text()
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(engine, "_SounddeviceCapture", _FakeSounddevice)
    eng = engine.Engine()
    eng.stop()  # loop exits right after the stream opens — no blocking in CI
    eng.run()

    err = capsys.readouterr().err
    assert "pyobjc-framework-AVFoundation" in err  # the real install hint fired
    assert "capture: sounddevice" in err
    assert seen["state_after_loud"] == "recording"  # callback flow intact under fallback
    assert engine.STATE_FILE.read_text() == "idle"  # run() closed the session cleanly




# ── transcription guard (no model in CI) ──────────────────────────────────────

def test_transcribe_returns_empty_when_mlx_unavailable(monkeypatch):
    """If mlx_whisper can't load, transcribe degrades to '' instead of crashing."""
    monkeypatch.setattr(engine, "_ensure_mlx_whisper", lambda: False)
    assert engine.transcribe(np.zeros(1600, dtype=np.float32)) == ""


# ── autosend daemon: arm flag + pause hook (one-shot semantics) ───────────────

@pytest.fixture
def autosend_mod(tmp_path, monkeypatch):
    import autosend
    monkeypatch.setattr(autosend, "STATE_FILE", tmp_path / "autosend")
    monkeypatch.setattr(autosend, "PAUSE_FLAG", tmp_path / "pause")
    return autosend


def test_autosend_defaults_to_disarmed(autosend_mod):
    """A missing flag file means NOT armed.

    It used to self-heal to "on", so a fresh state dir — a new machine, a new
    install, a cleaned ~/.vibevoice — armed the daemon with no gesture at all.
    TARGET_APPS is exactly the terminals (Terminal, iTerm2, Warp, Ghostty) and
    editors, so the first 0.8s pause while typing a command sent Return and RAN
    the half-written command line. The module's own docstring says "you arm it,
    you dictate one message", and invariant #7 exists to stop a zombie ON.

    The two errors do not cost the same: one extra click against a command
    nobody typed. Reviewed and changed 2026-08-02.
    """
    assert autosend_mod.is_enabled() is False
    assert not autosend_mod.STATE_FILE.exists(), "reading state must not create it"


def test_autosend_set_enabled_roundtrip(autosend_mod):
    autosend_mod.set_enabled(False)
    assert autosend_mod.is_enabled() is False
    autosend_mod.set_enabled(True)
    assert autosend_mod.is_enabled() is True


def test_pause_flag_fresh_suspends(autosend_mod):
    import time
    autosend_mod.PAUSE_FLAG.write_text(str(time.time()))
    assert autosend_mod.is_paused_by_flag() is True


def test_pause_flag_expired_self_clears(autosend_mod):
    import time
    stale = time.time() - (autosend_mod.PAUSE_TTL_SECONDS + 10)
    autosend_mod.PAUSE_FLAG.write_text(str(stale))
    assert autosend_mod.is_paused_by_flag() is False
    assert not autosend_mod.PAUSE_FLAG.exists()  # anti-deadlock cleanup


def test_pause_flag_absent_is_not_paused(autosend_mod):
    assert autosend_mod.is_paused_by_flag() is False


# ── history.jsonl: last transcriptions (settings window reads this) ───────────

@pytest.fixture
def history_state(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "HISTORY_FILE", tmp_path / "history.jsonl")
    return tmp_path


def test_history_appends_and_caps(history_state):
    import json
    for i in range(25):
        engine._append_history(f"frase {i}")
    lines = (history_state / "history.jsonl").read_text().splitlines()
    assert len(lines) == engine.HISTORY_MAX == 20
    assert json.loads(lines[-1])["text"] == "frase 24"   # newest last
    assert set(json.loads(lines[0])) == {"ts", "text"}


def test_history_write_failure_never_raises(history_state, monkeypatch):
    monkeypatch.setattr(engine, "HISTORY_FILE", history_state / "no" / "dir.jsonl")
    engine._append_history("must not raise")  # transcription path must survive


# ── AGENTS.md §6: the barge-in acceptance procedure is documented ─────────────
#
# The end-to-end criterion for the full-duplex jump (TTS-only / barge-in /
# music-only, judged on history.jsonl) runs on the LIVE runtime with a human
# voice at the mic — never in pytest (CLAUDE.md rule 2). What CAN be locked
# headless is that the ritual exists in AGENTS.md §6 with its measurable
# commands, so the doc and the runtime procedure cannot drift apart.

def _agents_section_6() -> str:
    text = (Path(engine.__file__).parent / "AGENTS.md").read_text()
    return text[text.index("## 6."):text.index("## 7.")]


def test_agents_documents_barge_in_acceptance():
    sec = _agents_section_6()
    low = sec.lower()
    # The three phases, by name.
    assert "tts-only" in low
    assert "barge-in" in low
    assert "music" in low
    # The measurable commands the criterion is judged by.
    assert "pgrep -f stt_bar.py" in sec              # legacy engine off → exit 1
    assert "L0=$(wc -l < ~/.vibevoice/history.jsonl" in sec
    assert "L0+1" in sec                             # barge-in grows it by one…
    assert "tail -1" in sec                          # …and the phrase is the last line
    assert "afplay" in sec                           # speakers open during the test
    # Measurable startup evidence that VP + Silero are the active paths.
    assert "VibeVoice: capture: voice-processing" in sec
    assert "VibeVoice: VAD: silero" in sec


# ── The learning corpus: a system that forgets cannot learn ──────────────────
# history.jsonl is capped at 20 lines because the pill shows it. That makes it a
# UI convenience, not a record — mining it for vocabulary yields nothing but the
# last five minutes. Learning needs a corpus, so the engine keeps one.

def test_every_utterance_is_kept_in_the_corpus(engine_state, monkeypatch):
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto tentacoli")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    for _ in range(3):
        engine.process_utterance(loud, t_end=None)

    lines = engine.CORPUS_FILE.read_text().splitlines()
    assert len(lines) == 3
    import json
    assert json.loads(lines[0])["text"] == "il polpo ha otto tentacoli"


def test_the_corpus_outlives_the_history_cap(engine_state, monkeypatch):
    """The whole point: history rolls at HISTORY_MAX, the corpus does not."""
    monkeypatch.setattr(engine, "transcribe", lambda audio: "una frase qualunque")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    for _ in range(engine.HISTORY_MAX + 5):
        engine.process_utterance(loud, t_end=None)

    assert len(engine.HISTORY_FILE.read_text().splitlines()) == engine.HISTORY_MAX
    assert len(engine.CORPUS_FILE.read_text().splitlines()) == engine.HISTORY_MAX + 5


def test_the_corpus_is_bounded_so_it_cannot_grow_without_end(engine_state, monkeypatch):
    """Unbounded is not a plan either: a year of dictation must not fill the disk."""
    monkeypatch.setattr(engine, "CORPUS_MAX", 10)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "frase")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    for _ in range(14):
        engine.process_utterance(loud, t_end=None)

    assert len(engine.CORPUS_FILE.read_text().splitlines()) == 10


def test_phantoms_never_enter_the_corpus(engine_state, monkeypatch):
    """Learning from a hallucination teaches the hallucination."""
    monkeypatch.setattr(engine, "transcribe", lambda audio: "Grazie a tutti.")
    silence = np.zeros(engine.SAMPLE_RATE, dtype=np.float32)

    engine.process_utterance(silence, t_end=None)

    assert not engine.CORPUS_FILE.exists() or engine.CORPUS_FILE.read_text().strip() == ""


# ── Telemetry of the streaming path (you can't improve what you don't log) ───
# metrics.jsonl measured the OLD architecture: how long the final decode took.
# It says nothing about what now governs the experience — when the first
# character reached the app, how much of the sentence the stream delivered, and
# whether the tail alignment found its anchor or gave up.

def test_a_busy_decoder_costs_one_block_not_a_whole_cadence(engine_state, monkeypatch):
    """LocalAgreement needs CONSECUTIVE passes, so a lost turn costs two.

    The due-time is advanced inside the audio lock, before the pass is known to
    be dispatchable. When the decoder was still busy the turn used to be dropped
    with the clock already moved on, so the retry waited a full interval — and
    since a word needs two agreeing hypotheses, one overrun pushed the first
    typed word out by two cadences. Live metrics 2026-08-30: first typed word at
    p50 1.83 s against a design floor near 1.1 s.

    Driven through the real audio callback: the bug lived in the ordering of
    that function, so a test that re-enacts the ordering by hand would prove
    nothing about it.
    """
    clock = {"t": 1000.0}
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock["t"])
    dispatched = []
    monkeypatch.setattr(engine.Engine, "_partial_worker",
                        lambda self, blocks, gen: dispatched.append(gen))

    eng = engine.Engine()
    loud = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)   # onset
    assert eng._speaking is True
    gen = eng._utt_gen

    # The decoder is still working on the previous pass when the next falls due.
    eng._partial_busy.acquire()
    clock["t"] += engine.partial_interval(False) + 0.01
    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)

    assert dispatched == [], "the slot was busy — nothing may be dispatched"
    assert eng._streamed_by_gen[gen]["busy_retries"] == 1, "the wait must be countable"

    # One audio block later the decoder frees. The pass must still be due.
    eng._partial_busy.release()
    clock["t"] += engine.BLOCKSIZE / engine.SAMPLE_RATE
    eng._audio_callback(loud, engine.BLOCKSIZE, None, None)

    assert dispatched == [gen], "the pass must start the moment the decoder frees"


def test_process_utterance_merges_extra_telemetry(engine_state, monkeypatch):
    """The engine knows the streaming numbers; process_utterance writes the line."""
    import json
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    engine.process_utterance(loud, t_end=None, extra={"stream_words": 3, "anchor": "ok"})

    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["stream_words"] == 3
    assert entry["anchor"] == "ok"
    assert "stt_ms" in entry, "the extra fields must not displace the existing ones"


def test_utterance_records_how_much_the_stream_delivered(engine_state, monkeypatch, typed):
    """The KPI that decides whether the tail latency still matters: if the
    stream delivers most of the sentence, the wait at the end is a detail."""
    import json
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto tentacoli")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)
    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)
    _drain_finals(eng)

    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["stream_words"] == 4, "four words typed, 'tentacoli' held on the edge"
    assert entry["tail_words"] == 1, "only the held-back word came with the final paste"
    assert entry["anchor"] == "ok"
    assert entry["partials"] >= 2
    assert entry["t_first_ms"] >= 0, "when the first character reached the app"


def test_a_stream_that_stands_down_is_counted(engine_state, monkeypatch, typed):
    """When an earlier utterance is still waiting to paste, the stream skips its
    turn — correct for ordering, but it is also the suspect for the immediacy
    p90 (5.18s live against 1.4s synthetic). Back-to-back dictation with MAX_DUR
    cuts is exactly when a predecessor is still pasting. Unmeasured, this is a
    guess; counted, it is an answer.
    """
    import json
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto")

    eng = engine.Engine()
    with eng._paste_cv:                    # a predecessor still owes a paste
        eng._seq_next = 1
        eng._paste_next = 0

    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)
    assert typed == [], "must not type over a pending paste"

    with eng._paste_cv:                    # predecessor done
        eng._paste_next = 1
    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)
    _drain_finals(eng)

    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["stood_down"] >= 1, "a skipped turn must be countable"


def test_refused_anchor_is_recorded_not_silently_swallowed(engine_state, monkeypatch):
    """A tail we chose not to paste is words the user lost. It must be countable."""
    import json
    monkeypatch.setattr(engine, "transcribe", lambda audio: "tutt'altra frase")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    eng = engine.Engine()
    with eng._agree_lock:
        eng._record_streamed(0, "una cosa completamente diversa")
    eng._transcribe_worker(loud, 0.0, 0, 0)

    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["anchor"] == "none"
    assert entry["tail_words"] == 0


# ── Hallucination guard: no audio, no words ──────────────────────────────────
# Whisper does not return "" on silence — it returns its training-set filler.
# Found in the user's own history.jsonl (2026-08-02 11:47:12): an utterance
# reading "Grazie a tutti." that was never spoken, and which the paste would
# have typed into whatever he had open.

def test_silence_that_transcribes_to_filler_is_dropped(engine_state, monkeypatch):
    """A near-silent blob whose text is short = hallucination, not speech."""
    monkeypatch.setattr(engine, "transcribe", lambda audio: "Grazie a tutti.")
    silence = np.zeros(engine.SAMPLE_RATE, dtype=np.float32)

    assert engine.process_utterance(silence, t_end=None) == ""
    assert not engine.RAW_FILE.exists(), "a phantom must not reach raw.txt"


def test_whisper_repetition_loops_never_reach_the_app(engine_state, monkeypatch):
    """Whisper degenerates into a repetition loop on non-speech audio. Captured
    live 2026-08-02 while the user was chewing: the mic opened an utterance and
    "trous" was typed 200 times into his editor, then "음" 110 times. The
    phantom guard could not help — it only judges SHORT texts, and a loop is
    long by definition.
    """
    monkeypatch.setattr(engine, "AUTOSEND", False)
    loud = np.full(engine.SAMPLE_RATE * 3, 0.3, dtype=np.float32)

    for loop in (" ".join(["trous"] * 200),
                 ", ".join(["음"] * 110),
                 " ".join(["Sottotitoli e revisione a cura di QTSS"] * 6)):
        monkeypatch.setattr(engine, "transcribe", lambda audio, t=loop: t)
        assert engine.process_utterance(loud, t_end=None) == "", f"leaked: {loop[:40]}"


def test_a_loop_after_real_speech_is_cut_and_the_speech_survives(engine_state, monkeypatch):
    """Control held a few seconds past the last word: Whisper fills the silent
    tail with a loop. Live 2026-10-31: a real two-sentence request followed by
    "Where will we go?" 24 times, typed and sent. The whole-text guard let it
    through because the real prefix kept the periodicity under the threshold."""
    monkeypatch.setattr(engine, "AUTOSEND", False)
    loud = np.full(engine.SAMPLE_RATE * 3, 0.3, dtype=np.float32)
    speech = ("Inoltre in questo menu ci sono alcune cose della versione precedente "
              "che andrebbero rimosse. Come proponi di ristrutturare il menu?")
    text = speech + " " + " ".join(["Where will we go?"] * 24)
    monkeypatch.setattr(engine, "transcribe", lambda audio: text)
    assert engine.process_utterance(loud, t_end=None) == speech


def test_real_speech_that_repeats_a_word_is_not_mistaken_for_a_loop(engine_state, monkeypatch):
    """Italian repeats words for emphasis and while thinking aloud. The guard
    must key on degeneracy, not on repetition."""
    monkeypatch.setattr(engine, "AUTOSEND", False)
    loud = np.full(engine.SAMPLE_RATE * 3, 0.3, dtype=np.float32)
    sentences = [
        "no no no non è quello che intendevo, aspetta un attimo che ti spiego meglio",
        "sì sì sì certo, allora dimmi pure come vuoi procedere con questa cosa",
        "cioè cioè non lo so, non lo so bene, forse dovremmo riprovare domani",
    ]
    for text in sentences:
        monkeypatch.setattr(engine, "transcribe", lambda audio, t=text: t)
        assert engine.process_utterance(loud, t_end=None) == text, f"eaten: {text[:40]}"


def test_canned_sign_offs_are_dropped_even_over_loud_audio(engine_state, monkeypatch):
    """The hallucination that survived the energy gate, and got auto-sent.

    `_is_phantom` only fires on near-silence, so a cough or a keystroke under
    the same hallucination let it through: 195 of 872 learned utterances on the
    live machine were "Grazie." or "Grazie a tutti.", none of them spoken, and
    with auto-send armed each one was typed AND sent as a message. Reported
    2026-08-30.
    """
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)
    for text in ("Grazie.", "Grazie a tutti.", "Grazie, grazie, grazie mille.",
                 "Hm Hm Hm Hm Hm Hm Grazie.",
                 "Sottotitoli e revisione a cura di QTSS"):
        monkeypatch.setattr(engine, "transcribe", lambda audio, t=text: t)
        assert engine.process_utterance(loud, t_end=None) == "", f"typed: {text}"


def test_canned_words_inside_real_speech_survive(engine_state, monkeypatch):
    """One real word anywhere saves the utterance — the guard covers, not matches.

    "grazie" is an ordinary Italian word. A list that deleted it wherever it
    appeared would be a worse bug than the one it fixes.
    """
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)
    for text in ("Grazie mille Marco, ci sentiamo domani",
                 "digli grazie da parte mia", "no grazie", "grazie no"):
        monkeypatch.setattr(engine, "transcribe", lambda audio, t=text: t)
        assert engine.process_utterance(loud, t_end=None) == text, f"eaten: {text}"


def test_dropped_canned_phrases_are_counted(engine_state, monkeypatch):
    """Same rule as every other silent deletion: it must show in the ledger."""
    import json
    monkeypatch.setattr(engine, "transcribe", lambda audio: "Grazie.")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    assert engine.process_utterance(loud, t_end=None) == ""
    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["canned"] is True
    assert entry["chars"] == 0, "nothing was published"


def test_quietly_spoken_short_words_survive_the_phantom_guard(engine_state, monkeypatch):
    """The guard re-litigates speech-vs-silence with the crude RMS threshold,
    even when the neural VAD (which is more sensitive by design, and is why it
    was added) is what opened this utterance. A soft "sì" just under the RMS
    floor was deleted with no trace anywhere. The floor is now well below the
    VAD threshold, so only true silence qualifies. Found by review 2026-08-02.
    """
    monkeypatch.setattr(engine, "transcribe", lambda audio: "sì")
    quiet = np.full(engine.SAMPLE_RATE, engine.VAD_THRESHOLD * 0.9, dtype=np.float32)

    assert engine.process_utterance(quiet, t_end=None) == "sì"


def test_dropped_phantoms_are_counted(engine_state, monkeypatch):
    """A silent deletion nobody can see is worse than the phantom it prevents:
    if the guard ever starts eating real speech, the ledger must show it."""
    import json
    monkeypatch.setattr(engine, "transcribe", lambda audio: "Grazie a tutti.")
    silence = np.zeros(engine.SAMPLE_RATE, dtype=np.float32)

    assert engine.process_utterance(silence, t_end=None) == ""
    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["phantom"] is True
    assert entry["chars"] == 0, "nothing was published"


def test_quiet_but_long_transcription_is_kept(engine_state, monkeypatch):
    """The guard keys on BOTH quiet audio and a short text: a real sentence
    recorded quietly must survive, or the guard becomes a censor."""
    # A real sentence, not a repeated token: a degenerate string would now be
    # caught by the loop guard and this test would be measuring the wrong thing.
    long_text = "questa è una frase intera detta piano ma perfettamente reale e comprensibile"
    monkeypatch.setattr(engine, "transcribe", lambda audio: long_text)
    silence = np.zeros(engine.SAMPLE_RATE, dtype=np.float32)

    assert engine.process_utterance(silence, t_end=None) == long_text


def test_audible_speech_is_never_dropped_however_short(engine_state, monkeypatch):
    """Audio well above the VAD threshold is speech: shortness alone never drops it.

    The example used to be "Grazie a tutti." — which is the one thing loudness
    does NOT vouch for, and the corpus proved it (2026-08-30: 195 unspoken
    sign-offs learned as if they were dictation). Loudness still settles the
    question for everything outside `_CANNED`; that list is the exception, and
    it is enumerated rather than inferred.
    """
    monkeypatch.setattr(engine, "transcribe", lambda audio: "Va bene.")
    loud = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)

    assert engine.process_utterance(loud, t_end=None) == "Va bene."


# ── Streaming: LocalAgreement-2 stabilizer (F3) ───────────────────────────────
# A word is published only once two successive hypotheses agree on it. That is
# what makes live text stable instead of flickering: Whisper re-decodes the whole
# buffer every pass and freely rewrites its own tail, so the tail is never
# trustworthy — the agreed prefix is.

def test_local_agreement_publishes_nothing_on_the_first_hypothesis():
    """One hypothesis has nothing to agree with — it is all still tentative."""
    agree = engine.LocalAgreement()
    assert agree.update("il polpo ha otto") == ""
    assert agree.confirmed == ""


def test_local_agreement_confirms_only_the_agreed_prefix():
    """Two hypotheses agree on 'il polpo ha' — the divergent tail stays tentative."""
    agree = engine.LocalAgreement()
    agree.update("il polpo ha otto")
    assert agree.update("il polpo ha molti tentacoli") == "il polpo ha"
    assert agree.confirmed == "il polpo ha"


def test_local_agreement_emits_each_word_exactly_once():
    """Successive updates return only the delta — never re-emit the prefix."""
    agree = engine.LocalAgreement()
    agree.update("il polpo")
    agree.update("il polpo ha otto")          # confirms "il polpo"
    delta = agree.update("il polpo ha otto tentacoli")  # confirms "ha otto"
    assert delta == "ha otto"
    assert agree.confirmed == "il polpo ha otto"


def test_local_agreement_never_retracts_a_confirmed_word():
    """A later hypothesis that contradicts confirmed text must not un-say it."""
    agree = engine.LocalAgreement()
    agree.update("il polpo ha")
    agree.update("il polpo ha")               # confirms all three
    assert agree.update("il") == ""           # shorter/divergent: no retraction
    assert agree.confirmed == "il polpo ha"


def test_local_agreement_matches_across_punctuation_and_case_drift():
    """Whisper re-punctuates as context grows; agreement is on words, not glyphs."""
    agree = engine.LocalAgreement()
    agree.update("Ciao mondo")
    assert agree.update("ciao, mondo come stai") == "ciao, mondo"


def test_local_agreement_repunctuates_confirmed_words_as_context_grows():
    """A word confirmed at the truncation edge carries a full stop that is wrong
    mid-sentence ("autonomo." → "autonomo,"). Measured live on 2026-08-02: the
    draft read "modo autonomo. mentre il cervello". The word order is what must
    never change; the glyphs must converge on what the final text will say.
    """
    agree = engine.LocalAgreement()
    agree.update("pensa in modo autonomo.")
    agree.update("pensa in modo autonomo. mentre")      # confirms four words
    agree.update("pensa in modo autonomo, mentre il")   # full context: it was a comma
    assert agree.confirmed == "pensa in modo autonomo, mentre"


def test_local_agreement_keeps_confirmed_words_when_the_tail_shrinks():
    """Re-rendering from the newest hypothesis must not lose confirmed words if
    that hypothesis comes back shorter than what is already published."""
    agree = engine.LocalAgreement()
    agree.update("il polpo ha otto")
    agree.update("il polpo ha otto")   # confirms all four
    agree.update("il polpo")           # a bad pass: shorter
    assert agree.confirmed == "il polpo ha otto"


def test_local_agreement_survives_a_word_appearing_at_the_HEAD():
    """Whisper re-decodes the whole buffer, so it can also revise the START:
    "ciao amico mio" → "oh ciao amico mio". The committed words were tracked by
    POSITION, so the shift silently misaligned every later commit and published
    "ciao amico mio mio caro" — a duplicated word, typed straight to the screen.
    Found by review 2026-08-02 and reproduced verbatim here.
    """
    agree = engine.LocalAgreement()
    published: list[str] = []
    for hypothesis in ("ciao amico mio",
                       "ciao amico mio caro",
                       "oh ciao amico mio caro",
                       "oh ciao amico mio caro davvero"):
        published.extend(agree.update(hypothesis).split())

    # THE invariant, and the one the streaming paste stands on: what is
    # confirmed must be exactly what was published, word for word. Anything else
    # means published words were silently rewritten underneath — and those words
    # are already on the user's screen, where they cannot be taken back.
    # (Asserting only the final string was not enough: a mutation that neuters
    # _locate produces a plausible-looking final string while rewriting the
    # published prefix. Caught by mutation testing 2026-08-02.)
    assert [engine._agreement_key(w) for w in agree.confirmed.split()] == \
        [engine._agreement_key(w) for w in published], \
        f"published {published} but confirmed {agree.confirmed!r}"
    assert "mio mio" not in agree.confirmed, f"duplicated word: {agree.confirmed!r}"
    # Whatever it publishes must be an in-order subsequence of the real sentence:
    # no invented words, no repeats, no reordering.
    truth = "oh ciao amico mio caro davvero".split()
    pos = -1
    for word in agree.confirmed.split():
        assert word in truth[pos + 1:], f"out of order or invented: {agree.confirmed!r}"
        pos = truth.index(word, pos + 1)


def test_local_agreement_publishes_nothing_when_it_cannot_locate_itself():
    """If the new hypothesis no longer contains what was already published, the
    only safe move is to publish nothing — never to guess a position."""
    agree = engine.LocalAgreement()
    agree.update("il polpo ha")
    agree.update("il polpo ha")            # confirms three words
    assert agree.update("tutt'altra frase") == ""
    assert agree.update("tutt'altra frase") == ""
    assert agree.confirmed == "il polpo ha"


def test_local_agreement_reset_starts_a_new_utterance_clean():
    """Utterance N+1 must not inherit the confirmed prefix of utterance N."""
    agree = engine.LocalAgreement()
    agree.update("prima frase")
    agree.update("prima frase")
    agree.reset()
    assert agree.confirmed == ""
    assert agree.update("seconda") == ""


# ── Streaming: the partial.txt contract file ──────────────────────────────────

def test_partial_roundtrip(engine_state):
    """partial.txt is plain text, like raw.txt — the live sentence so far."""
    engine.write_partial("il polpo ha otto")
    assert engine.PARTIAL_FILE.read_text() == "il polpo ha otto"


def test_partial_write_is_atomic(engine_state):
    """Written via tmp + os.replace: the pill never sees a torn sentence."""
    engine.write_partial("una frase intera")
    assert not engine.PARTIAL_TMP.exists()


def test_partial_is_never_written_straight_onto_the_live_file(engine_state):
    """"No tmp left behind" is also true of a plain `write_text` — which leaves
    no tmp because it never made one.

    Verified 2026-08-06: swap `write_partial` for a direct truncate-in-place and
    the assertion above still passes. So watch the mechanism instead — the live
    file must only ever appear via os.replace, because the pill reads it on a
    timer and a truncate window is exactly when it would catch half a sentence.
    """
    _assert_only_swapped_in(engine.PARTIAL_FILE,
                            lambda: engine.write_partial("una frase intera"))


def test_levels_are_never_written_straight_onto_the_live_file(engine_state):
    """Same guard for the binary heartbeat: the pill unpacks it 60 floats at a
    time and a partial file is a torn frame."""
    from collections import deque
    _assert_only_swapped_in(
        engine.LEVELS_FILE,
        lambda: engine.write_levels(deque([0.1] * engine.LEVELS_LEN)))


def test_clear_partial_removes_the_file(engine_state):
    """No file = nothing live to show. Absence is the 'no partial' signal."""
    engine.write_partial("qualcosa")
    engine.clear_partial()
    assert not engine.PARTIAL_FILE.exists()


def test_clear_partial_is_idempotent(engine_state):
    """Clearing twice (or before any write) must never raise."""
    engine.clear_partial()
    engine.clear_partial()
    assert not engine.PARTIAL_FILE.exists()


# ── Streaming: the engine publishes partials WHILE speech is still going ──────

def _speech_block(level: float = 0.5) -> np.ndarray:
    return np.full((engine.BLOCKSIZE, 1), level, dtype=np.float32)


def _drain_partials(eng, timeout: float = 5.0) -> None:
    """Wait for the single partial slot to be free again (worker finished)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if eng._partial_busy.acquire(blocking=False):
            eng._partial_busy.release()
            return
        time.sleep(0.01)
    raise AssertionError("partial worker never finished")


def _drain_finals(eng, timeout: float = 5.0) -> None:
    """Wait for the final transcription AND its paste to have run their course."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with eng._paste_cv:
            settled = eng._paste_next >= eng._seq_next
        if settled and eng._busy._value == 2:
            time.sleep(0.05)   # let the paste thread's last statement land
            return
        time.sleep(0.01)
    raise AssertionError("final transcription/paste never settled")


def test_engine_publishes_a_partial_while_still_speaking(engine_state, monkeypatch):
    """The whole point: text exists BEFORE the utterance is finalized."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)  # every block
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    for _ in range(3):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert eng._speaking is True, "utterance must still be open"
    assert engine.STATE_FILE.read_text() == "recording"
    assert engine.PARTIAL_FILE.read_text() == "il polpo ha"


def test_partial_pass_never_writes_raw_or_history(engine_state, monkeypatch):
    """Partials are provisional: only a finalized utterance may touch raw.txt."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "provvisorio")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert not engine.RAW_FILE.exists()
    assert not engine.HISTORY_FILE.exists()


def test_finalize_clears_the_partial(engine_state, monkeypatch):
    """When the real text lands in raw.txt the live draft must disappear."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "frase")
    monkeypatch.setattr(engine, "AUTOSEND", False)

    eng = engine.Engine()
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    _drain_partials(eng)
    assert engine.PARTIAL_FILE.exists()

    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)
    assert not engine.PARTIAL_FILE.exists()


def test_new_utterance_does_not_inherit_the_previous_partial(engine_state, monkeypatch):
    """Utterance N+1 starts from an empty draft, not from N's confirmed prefix."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "prima")
    monkeypatch.setattr(engine, "AUTOSEND", False)

    eng = engine.Engine()
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    _drain_partials(eng)
    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)

    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    assert eng._agree.confirmed == "", "stabilizer must reset at utterance onset"


def test_streaming_off_behaves_exactly_like_the_legacy_engine(engine_state, monkeypatch):
    """Degradation is contract: with STREAMING off, no partial work happens at all."""
    monkeypatch.setattr(engine, "STREAMING", False)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)

    def _boom(audio):
        raise AssertionError("no transcription may run during capture when streaming is off")

    monkeypatch.setattr(engine, "transcribe", _boom)

    eng = engine.Engine()
    for _ in range(3):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)

    assert eng._speaking is True
    assert not engine.PARTIAL_FILE.exists()


# ── Time to the first character: the number the product is sold on ───────────
# Everything above proves the streaming path is WIRED. None of it proves it is
# FAST, because every one of those tests sets PARTIAL_INTERVAL to 0 and skips
# the cadence policy entirely — which is exactly how a 1.4 s wait survived 190
# green tests. These tests measure the wait instead.

_MEASURED_DECODE_S = 0.17      # one partial pass, M5 Max / whisper-turbo (AGENTS.md);
                               # flat in buffer length — Whisper pads to 30 s anyway

# What whisper-turbo ACTUALLY returns for the smoke sentence, buffer length by
# buffer length, recorded 2026-08-03 by decoding each prefix once (pre-roll
# included, exactly as the engine assembles it). Verbatim and not a rate model:
# the first attempt here assumed words arrive at a steady 3/s, which flattered
# short buffers and predicted a win the live tool then refuted. Whisper on a
# sliver of speech does not return fewer words — it returns WRONG ones, and it
# keeps changing its mind ("Il Polo" → "Il Poltino" → "Il polpo"). That churn is
# the whole reason LocalAgreement needs two passes, so a model that smooths it
# away cannot be used to reason about latency.
_HEARD_AT: dict[float, str] = {
    0.05: "Grazie a tutti.",          # the hallucination on near-silence, verbatim
    0.10: "Is",
    0.15: "E",
    0.20: "E ba ba ba ba ba ba ba ba",
    0.25: "Il po'",
    0.30: "Il Polo",
    0.35: "Il Poltino",
    0.40: "Il polpo",
    0.45: "Il polpo è",
    0.50: "Il polpo a",
    0.55: "Il polpo a ove.",
    0.60: "Il Polpo è",
    0.65: "Il Polpo è ottenuto.",
    0.70: "Il polpo è 8.",
    0.75: "Il polpo è 8.",
    0.80: "Il polpo è 8 tessere.",
    0.85: "Il polpo a 8 tess.",
    0.90: "Il polpo a 8 tenei.",
    0.95: "Il polpo a 8 tentative.",
    1.00: "Il polpo a 8 tentativi.",
    1.05: "Il polpo a 8 tentacoli.",
    1.10: "Il polpo a 8 tentacoli.",
    1.15: "Il polpo a 8 tentacoli.",
    1.20: "Il polpo a 8 tentacoli.",
}


def _heard_after(t: float) -> str:
    """The recognizer's answer for a buffer holding `t` seconds of speech."""
    grid = min(_HEARD_AT, key=lambda g: abs(g - t))
    return _HEARD_AT[grid] if t <= max(_HEARD_AT) else _HEARD_AT[max(_HEARD_AT)]


def _seconds_to_first_typed_character(
    *,
    decode_s: float = _MEASURED_DECODE_S,
    horizon: float = 8.0,
) -> float | None:
    """Replay the streaming scheduler on a fake clock; return when the app would
    receive its first character, in seconds after speech onset.

    A model, and it says so — but every part of it that could lie is real. It
    runs the **real** `engine.partial_interval` and the **real**
    `engine.LocalAgreement` over the **recorded** answers of the real
    recognizer, and it mirrors the three rules the audio callback follows: one
    decode slot, the timer resetting on dispatch whether or not the slot was
    free, and the one-word lag before anything is typed.

    Calibrated against the live tool, not asserted into existence — see
    `test_the_scheduler_model_reproduces_the_measured_baselines`.
    """
    agree = engine.LocalAgreement()
    block_s = engine.BLOCKSIZE / engine.SAMPLE_RATE
    t = 0.0
    t_partial = 0.0             # the engine seeds this at utterance onset
    in_flight: tuple[float, str] | None = None

    while t < horizon:
        if in_flight is not None and t >= in_flight[0]:
            agree.update(in_flight[1])
            in_flight = None
            # The stream types the confirmed prefix minus its last word, so two
            # confirmed words are what it takes to put one on screen.
            if len(agree.confirmed.split()) >= 2:
                return round(t, 3)
        if t - t_partial >= engine.partial_interval(len(agree.confirmed.split()) >= 2):
            t_partial = t
            if in_flight is None:
                in_flight = (t + decode_s, _heard_after(t))
        t += block_s
    return None


def test_the_scheduler_model_reproduces_the_measured_baselines(monkeypatch):
    """The model is only worth its assertions if it agrees with the real thing.

    `tools/smoke_streaming.py` — real audio, real Whisper, wall clock — was run
    interleaved on both cadences, three times each on 2026-08-03: the flat 0.6 s
    schedule put the first character at 1.4 s every single time, and the 0.45 s
    warm-up at 1.0/1.1/1.1 s. The model must land on both. If this ever drifts,
    the budget below is measuring a fiction and must not be trusted until the
    model is re-recorded against the tool.
    """
    monkeypatch.setattr(engine, "PARTIAL_WARMUP", engine.PARTIAL_INTERVAL)  # the old flat cadence
    flat = _seconds_to_first_typed_character()
    assert flat is not None and abs(flat - 1.4) <= 0.05, (
        f"model says {flat}s where the live tool measured 1.4s flat — re-record it"
    )

    monkeypatch.undo()
    warmed = _seconds_to_first_typed_character()
    assert warmed is not None and abs(warmed - 1.05) <= 0.10, (
        f"model says {warmed}s where the live tool measured 1.0–1.1s — re-record it"
    )


def test_the_first_character_arrives_a_third_of_a_second_sooner():
    """The product claim, as a number.

    The competitor transcribes while you talk; this engine made you wait 1.4 s
    for the first character on a clean synthetic sentence — and a p50 of 2.31 s
    on the real dictation ledger. The wait is not the decode (170 ms): it is
    `2 × cadence`, because LocalAgreement-2 publishes nothing until two passes
    have agreed. Moving the first pass earlier — without adding passes, which
    measurement showed makes it worse — is what buys the difference between text
    that follows the voice and text that trails it.
    """
    first = _seconds_to_first_typed_character()

    assert first is not None, "no character was ever typed within the horizon"
    assert first <= 1.15, f"first character at {first}s — the wait is back"


def test_the_warmup_does_not_buy_the_first_character_with_more_passes(monkeypatch):
    """The scar of this fix, kept executable.

    The obvious move — decode far more often while the line is empty — was tried
    and measured, and it lost: 1.1–2.1 s live against a steady 1.4 s. A cadence
    below the decode cost plus its jitter loses whole turns (the timer resets on
    dispatch, free slot or not), and agreement needs *consecutive* hypotheses.
    So the warm-up must stay comfortably above one decode. Whoever tunes this
    down next should have to delete this test on purpose.
    """
    assert engine.PARTIAL_WARMUP >= 2 * _MEASURED_DECODE_S, (
        f"a {engine.PARTIAL_WARMUP}s warm-up leaves under "
        f"{engine.PARTIAL_WARMUP - _MEASURED_DECODE_S:.2f}s of slack on a "
        f"{_MEASURED_DECODE_S}s decode — jitter will eat the turns"
    )

    monkeypatch.setattr(engine, "PARTIAL_WARMUP", 0.25)
    assert _seconds_to_first_typed_character() is not None


def test_the_cadence_is_tighter_only_until_the_first_word_is_confirmed():
    """Two regimes, one knob. Spending passes is cheap while the line is empty
    and wasteful once text is flowing — where extra passes buy only freshness
    and add pressure on the paste queue, which the ledger already blames for 58
    stood-down turns out of 113."""
    assert engine.partial_interval(False) == engine.PARTIAL_WARMUP
    assert engine.partial_interval(True) == engine.PARTIAL_INTERVAL
    assert engine.PARTIAL_WARMUP < engine.PARTIAL_INTERVAL, (
        "a warm-up at or above the steady cadence is the old behaviour wearing a new name"
    )


def test_the_warmup_can_only_make_the_engine_more_eager(monkeypatch):
    """A floor, not a second cadence. Someone who asks for passes every 0.1 s
    everywhere must not get a 0.25 s wait for the word they are waiting on."""
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.1)
    monkeypatch.setattr(engine, "PARTIAL_WARMUP", 0.25)

    assert engine.partial_interval(False) == 0.1


def test_the_engine_leaves_the_warmup_cadence_once_a_word_reaches_the_app(
    engine_state, monkeypatch
):
    """The policy has to be the one the audio thread actually consults.

    A pure function nobody calls is the classic way for this fix to rot. So:
    make the two regimes impossible to confuse — warm-up fires on every block,
    steady fires never — and count the passes. Two passes are what
    LocalAgreement needs to confirm anything; the third block must find the
    engine already back on the steady cadence and stay silent.
    """
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_WARMUP", 0.0)      # always due
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 10.0)   # never due

    # Hypotheses chosen so that "confirmed" and "delivered" happen on DIFFERENT
    # passes — otherwise the test cannot tell the two apart, and a version that
    # ends the warm-up one pass early passes it just as happily. (It did: the
    # mutation check called that out before this file was committed.)
    #   pass 1  "il polpo"       → nothing yet, no previous hypothesis
    #   pass 2  "il gatto"       → agrees on "il" alone: confirmed, but the
    #                              one-word lag means the app still has nothing
    #   pass 3  "il gatto nero"  → "gatto" agrees too: now a word comes free
    hypotheses = iter(["il polpo", "il gatto", "il gatto nero"])
    monkeypatch.setattr(engine, "transcribe",
                        lambda audio: next(hypotheses, "il gatto nero"))

    eng = engine.Engine()
    for _ in range(4):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert eng._has_delivered is True, "the third pass frees a word for the app"
    assert eng._partials == 3, (
        f"{eng._partials} passes: the warm-up ended on the CONFIRMED word instead "
        "of the delivered one, so the fourth block found the steady cadence and "
        "the user waits a full interval for a character that was one pass away"
    )


def test_a_new_utterance_starts_blind_again(engine_state, monkeypatch):
    """The warm-up is per-utterance. If the flag survived the close, the second
    sentence of a dictation would pay the full steady cadence for its first
    word — and back-to-back utterances are the common case, not the edge."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_WARMUP", 0.0)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 10.0)
    monkeypatch.setattr(engine, "SILENCE_SEC", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    for _ in range(3):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)
    assert eng._has_delivered is True

    # Silence closes the utterance, then speech opens the next one.
    eng._audio_callback(_speech_block(0.0), engine.BLOCKSIZE, None, None)
    eng._audio_callback(_speech_block(0.0), engine.BLOCKSIZE, None, None)
    _drain_finals(eng)
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)

    assert eng._has_delivered is False, "the next sentence must start on the warm-up cadence"


# ── Streaming paste: typing the confirmed words while the sentence is open ────
# The confirmed prefix never retracts — that is exactly what makes it safe to
# type. What must never happen is typing the same words twice: the final paste
# may only add the tail the stream never reached.

def test_unstreamed_tail_is_the_whole_text_when_nothing_was_streamed():
    assert engine.unstreamed_tail("", "il polpo ha otto") == "il polpo ha otto"


def test_unstreamed_tail_is_empty_when_the_stream_already_said_everything():
    assert engine.unstreamed_tail("il polpo ha otto", "il polpo ha otto") == ""


def test_unstreamed_tail_returns_only_the_words_the_stream_never_reached():
    assert engine.unstreamed_tail("il polpo ha", "il polpo ha otto tentacoli") == "otto tentacoli"


def test_unstreamed_tail_aligns_across_punctuation_drift():
    """The stream typed "autonomo"; the final says "autonomo," — same word, no repeat."""
    assert engine.unstreamed_tail(
        "pensa in modo autonomo", "Pensa in modo autonomo, mentre il"
    ) == "mentre il"


def test_unstreamed_tail_anchors_on_the_END_of_what_was_typed():
    """Verbatim from the wild, 2026-08-02. The stream typed a mangled early
    reading; the final decode fixed it. Prefix alignment broke at word 1 and the
    whole sentence was pasted a second time — the user's screen read
    "…che pattern e risolvili tranquillamente Ti dico i ragionamenti che fai…".

    What is already on screen ENDS somewhere inside the final text, so that is
    where to look: anchor on the last words typed, not on the first.
    """
    streamed = "Però ti dico i ragionamenti che pattern e risolvili tranquillamente"
    final = ("Ti dico i ragionamenti che fai, individua i pattern e risolvili "
             "tranquillamente con tutta la tua conoscenza")
    assert engine.unstreamed_tail(streamed, final) == "con tutta la tua conoscenza"


def test_unstreamed_tail_picks_the_junction_where_the_stream_actually_stopped():
    """Italian repeats short phrases when thinking aloud ("non lo so, non lo so
    bene"). Anchoring on the LAST occurrence deleted everything in between —
    silently, since nothing on screen looks broken. The junction belongs where
    the stream actually reached, so among equal matches take the one nearest to
    the number of words already typed. Found by review 2026-08-02.
    """
    assert engine.unstreamed_tail(
        "non lo so", "non lo so, non lo so bene"
    ) == "non lo so bene"
    assert engine.unstreamed_tail(
        "allora", "allora dimmi allora cosa vuoi"
    ) == "dimmi allora cosa vuoi"


def test_unstreamed_tail_still_anchors_late_when_the_stream_kept_pace():
    """The nearest-junction rule must not undo the original fix: when the typed
    text really does reach deep into the sentence, the late anchor is correct."""
    assert engine.unstreamed_tail(
        "il polpo ha otto tentacoli", "il polpo ha otto tentacoli e pensa"
    ) == "e pensa"


def test_agreement_key_treats_both_apostrophes_as_the_same_word():
    """Whisper emits U+2019 for Italian elisions as often as U+0027. Treating
    them as different words broke the agreement chain exactly at l', dell',
    quest', un' — the most frequent words in fluent Italian, so the live draft
    stalled precisely where it should flow."""
    assert engine._agreement_key("l'inter") == engine._agreement_key("l’inter")
    assert engine._agreement_key("dell'anno") == engine._agreement_key("dell’anno")
    assert engine._agreement_key("quest'anno") == engine._agreement_key("quest’anno")


def test_unstreamed_tail_refuses_to_paste_a_sentence_twice():
    """No anchor anywhere = we cannot tell what is already on screen. Repeating
    a whole sentence makes the text unusable; a missing tail is visible and
    re-dictatable. Say nothing rather than say it twice."""
    assert engine.unstreamed_tail(
        "una cosa completamente diversa", "tutt'altra frase senza rapporto"
    ) == ""


def test_unstreamed_tail_survives_one_revised_word_in_the_middle():
    """The common case: full context fixes a single word already typed. The
    anchor is the tail, so the paste is still just the remainder."""
    assert engine.unstreamed_tail(
        "il polpo ha molti tentacoli", "il polpo ha otto tentacoli e pensa"
    ) == "e pensa"


@pytest.fixture
def typed(monkeypatch):
    """Record what the engine types instead of hitting the real keyboard."""
    calls: list[str] = []
    monkeypatch.setattr(engine, "type_text", lambda text: calls.append(text))
    monkeypatch.setattr(engine, "autosend", lambda text: calls.append(text))
    return calls


def test_stream_paste_types_the_confirmed_delta_while_still_speaking(engine_state, monkeypatch, typed):
    """The whole point: words reach the app BEFORE the trailing silence expires."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert eng._speaking is True, "must have typed while the utterance was open"
    # "ha" is still the newest confirmed word — held back by one word of lag.
    assert "".join(typed) == "il polpo"


def test_stream_paste_holds_back_the_word_on_the_truncation_edge(engine_state, monkeypatch, typed):
    """Whisper closes every truncated buffer with a full stop, so the newest
    confirmed word carries punctuation that full context will drop. The pill's
    draft re-punctuates itself; text already typed cannot. Seen in the wild
    2026-08-02: "lo sto testando. e sembra molto bello".

    So a word is typed only once ANOTHER word has been confirmed after it — by
    then its punctuation is settled. The draft still shows everything.
    """
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)

    # Pass 1+2 confirm "sto testando." — the period is a truncation artefact.
    # Pass 3+4 confirm "e sembra" and reveal it was never a sentence end.
    hypotheses = iter([
        "lo sto testando.",
        "lo sto testando.",
        "lo sto testando e sembra",
        "lo sto testando e sembra",
    ])
    monkeypatch.setattr(engine, "transcribe", lambda audio: next(hypotheses, "lo sto testando e sembra"))

    eng = engine.Engine()
    for _ in range(4):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    landed = "".join(typed)
    assert "testando." not in landed, f"typed the truncation-edge full stop: {landed!r}"
    assert landed == "lo sto testando e", "one word of lag: 'sembra' is still on the edge"
    # The draft is not held back — only the typing is.
    assert engine.PARTIAL_FILE.read_text() == "lo sto testando e sembra"


def test_stream_paste_separates_successive_chunks_with_a_space(engine_state, monkeypatch, typed):
    """Two deltas must not weld into "il polpoha"."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)

    hypotheses = iter(["il polpo", "il polpo ha", "il polpo ha otto", "il polpo ha otto"])
    monkeypatch.setattr(engine, "transcribe", lambda audio: next(hypotheses, "il polpo ha otto"))

    eng = engine.Engine()
    for _ in range(4):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    # "otto" is the newest confirmed word: held back, it arrives with the final paste.
    assert "".join(typed) == "il polpo ha"


def test_final_paste_adds_only_the_tail_after_streaming(engine_state, monkeypatch, typed):
    """No duplication: the stream said "il polpo", the final adds " ha otto"."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)
    assert "".join(typed) == "il", "one word of lag: 'polpo' is still on the edge"

    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto")
    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)
    _drain_finals(eng)

    assert "".join(typed) == "il polpo ha otto", "the tail only, never the whole sentence again"


def test_engine_state_fixture_disarms_the_outbound_switches(engine_state):
    """The argine itself: no test may reach the user's keyboard or clipboard by
    default. If this fails, every other test in this file became dangerous."""
    assert engine.AUTOSEND is False
    assert engine.STREAM_PASTE is False


def test_stream_paste_off_still_pastes_the_whole_sentence_once(engine_state, monkeypatch, typed):
    """Degradation is contract: with streaming paste off the behaviour is the old one."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", False)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto")

    eng = engine.Engine()
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    _drain_partials(eng)
    assert typed == [], "nothing may be typed mid-utterance when the feature is off"

    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)
    _drain_finals(eng)
    assert typed == ["il polpo ha otto"]


def test_stream_paste_stands_down_while_a_previous_utterance_is_still_pasting(engine_state, monkeypatch, typed):
    """Ordering beats latency: typing into the middle of a pending paste would
    interleave two sentences. The stream skips its turn and the final paste
    delivers the whole thing."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "seconda frase")

    eng = engine.Engine()
    # Simulate an earlier utterance whose paste has not had its turn yet.
    with eng._paste_cv:
        eng._seq_next = 1
        eng._paste_next = 0

    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert typed == [], "must not type over a pending paste"


# ── Streaming paste vs the standalone auto-Return daemon ─────────────────────
# autosend.py fires Return after AUTO_SEND_DELAY (0.8s) of typing silence. The
# streaming paste types in bursts ~0.65s apart — close enough that one slow pass
# would send the message mid-sentence. The engine therefore holds the daemon's
# pause flag while it is streaming and releases it once the sentence is whole.

def test_stream_paste_holds_the_auto_return_daemon(engine_state, monkeypatch, typed):
    """A pause between two typed chunks must not be read as 'the user stopped'."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert engine.AUTOSEND_PAUSE_FLAG.exists(), "the daemon must be held while streaming"
    # Line 1 is the timestamp, line 2 names the holder — this test is a third
    # reader of that contract, and parsing the whole file made it fail exactly
    # the way the daemon's parser did before it was fixed.
    held_at = float(engine.AUTOSEND_PAUSE_FLAG.read_text().splitlines()[0])
    assert abs(held_at - time.time()) < 5, "a stale timestamp would let the TTL expire mid-sentence"


def test_finished_sentence_releases_the_auto_return_daemon(engine_state, monkeypatch, typed):
    """Once the sentence is whole the Return must be allowed to fire."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)
    assert engine.AUTOSEND_PAUSE_FLAG.exists()

    eng._finalize(eng._t_start + engine.MIN_DUR + 0.1)
    _drain_finals(eng)

    assert not engine.AUTOSEND_PAUSE_FLAG.exists(), "the daemon must be freed after the paste"


def test_no_hold_when_the_streaming_paste_is_off(engine_state, monkeypatch, typed):
    """Nothing is typed mid-sentence, so nothing needs holding — leave the flag alone."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", False)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    for _ in range(2):
        eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
        _drain_partials(eng)

    assert not engine.AUTOSEND_PAUSE_FLAG.exists()


def test_a_chunk_typed_while_the_sentence_closes_is_not_repeated(engine_state, monkeypatch):
    """The race seen in the wild (2026-08-02): a partial pass had claimed its
    chunk and was still typing when the utterance finalized, so the final paste
    computed its tail from a stale `streamed` and typed the chunk a second time.
    The user's screen read "...funzionando bene. parola.Anche la trascrizione".

    Here `type_text` blocks mid-flight while the utterance is finalized, exactly
    reproducing that window.
    """
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo")

    typing_started = threading.Event()
    may_finish_typing = threading.Event()
    landed: list[str] = []

    def slow_type(text: str) -> bool:
        landed.append(text)
        typing_started.set()
        may_finish_typing.wait(timeout=5)
        return True

    monkeypatch.setattr(engine, "type_text", slow_type)
    monkeypatch.setattr(engine, "autosend", lambda text: landed.append(text))

    eng = engine.Engine()
    # First pass has nothing to agree with yet — let it finish, or the second
    # callback finds the single partial slot busy and silently skips its turn.
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    _drain_partials(eng)
    # Second pass confirms "il polpo" and starts typing it — then blocks.
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    assert typing_started.wait(timeout=5), "the partial pass never reached type_text"

    # The sentence closes while that chunk is still on the wire.
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha otto")
    closer = threading.Thread(
        target=eng._finalize, args=(eng._t_start + engine.MIN_DUR + 0.1,), daemon=True
    )
    closer.start()
    time.sleep(0.2)
    may_finish_typing.set()
    closer.join(timeout=5)
    _drain_finals(eng)

    assert "".join(landed) == "il polpo ha otto", f"duplicated: {landed!r}"


def test_muting_stops_a_partial_from_typing_afterwards(engine_state, monkeypatch, typed):
    """🔇 is a pause, and the user expects it to take effect at once. The mute
    branch dropped the in-flight utterance but did NOT bump the generation, so a
    partial already on the wire passed its checks and typed into the frontmost
    app after the mic was paused. It also left partial.txt claiming a live
    utterance while state said idle. Found by review 2026-08-02.
    """
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "il polpo ha")

    eng = engine.Engine()
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    gen_before = eng._utt_gen
    _drain_partials(eng)

    engine.MUTED_FILE.touch()
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)

    assert eng._utt_gen != gen_before, "muting must invalidate the open utterance"
    assert not engine.PARTIAL_FILE.exists(), "a paused mic must not claim a live draft"
    assert not engine.AUTOSEND_PAUSE_FLAG.exists(), "the Return daemon must be freed"


def test_an_utterance_too_short_to_transcribe_still_frees_the_return_daemon(engine_state, monkeypatch, typed):
    """The flag is raised before typing and released after the paste. An
    utterance dropped for being under MIN_DUR never reaches a paste, so it left
    the daemon held — and autosend then swallows the Return of the sentence the
    user actually dictated, silently, for up to PAUSE_TTL_SECONDS."""
    eng = engine.Engine()
    engine.hold_autosend()
    assert engine.AUTOSEND_PAUSE_FLAG.exists()

    eng._t_start = time.monotonic()
    eng._finalize(eng._t_start + engine.MIN_DUR / 2)   # too short: dropped

    assert not engine.AUTOSEND_PAUSE_FLAG.exists()


def test_engine_and_daemon_agree_on_the_pause_flag_path():
    """Two processes, one path, no imports between them — assert they still match."""
    daemon = (Path(engine.__file__).parent / "autosend.py").read_text()
    assert str(engine.AUTOSEND_PAUSE_FLAG) in daemon or \
        "/tmp/vibevoice_autosend_pause" in daemon


def test_the_next_utterance_cannot_erase_the_previous_one_s_typed_record(engine_state):
    """The duplication that plagued the live runtime all of 2026-08-02.

    `_streamed` was a single slot, cleared at the next onset. But the closing
    utterance's worker reads it AFTER finalize — and while it waits for an
    in-flight partial, the user is still talking, so the next utterance opens
    and wipes the slot. The worker then reads "", concludes the stream typed
    nothing, and pastes the whole sentence a second time:

        "…e nuota nel mare" + "il polpo ha otto tentacoli e nuota nel mare profondo"

    MAX_DUR is the realistic trigger, not luck: at 15s an utterance is force-cut
    WHILE speech continues, so a partial is plausibly still decoding and the next
    block opens the successor 50ms later. 32 of the user's 145 utterances hit
    that cut.
    """
    eng = engine.Engine()
    closing_gen = 1
    with eng._agree_lock:                      # utterance A typed something
        eng._streamed = "il polpo ha otto tentacoli"
        eng._streamed_gen = closing_gen
        eng._record_streamed(closing_gen, eng._streamed)

    # A closes; B opens before the worker gets to read A's record.
    with eng._lock:
        eng._utt_gen = 3
    with eng._agree_lock:
        eng._streamed = ""                     # B's onset reset
        eng._streamed_gen = -1

    assert eng._collect_streamed(closing_gen)["typed"] == "il polpo ha otto tentacoli"


def test_a_collected_record_is_not_kept_forever(engine_state):
    """The per-utterance records must not accumulate for the life of the process."""
    eng = engine.Engine()
    for gen in range(1, 60):
        with eng._agree_lock:
            eng._record_streamed(gen, f"frase {gen}")
        # Collect only every other one: the records nobody ever collects (an
        # utterance dropped as too short, or one whose slot was full) are
        # exactly the ones that would accumulate.
        if gen % 2:
            eng._collect_streamed(gen)
    assert len(eng._streamed_by_gen) <= 8


def test_pill_reads_the_partial_file(engine_state):
    """Writer + reader in the same commit: a draft nobody renders is dead code.

    The pill needs AppKit, so it cannot be imported headless — this locks the
    contract at source level, the same way the levels.bin magic number is.
    """
    pill = (Path(engine.__file__).parent / "vibevoice.py").read_text()
    assert "partial.txt" in pill, "the pill must read the live draft"
    # And it must prefer the draft only while the utterance is open, otherwise
    # it would keep showing a stale draft after the real text landed.
    assert "_read_partial" in pill


def test_partial_worker_failure_does_not_break_capture(engine_state, monkeypatch):
    """A partial is a bonus, never a dependency — it may fail silently."""
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "PARTIAL_INTERVAL", 0.0)

    def _boom(audio):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(engine, "transcribe", _boom)

    eng = engine.Engine()
    eng._audio_callback(_speech_block(), engine.BLOCKSIZE, None, None)
    _drain_partials(eng)

    assert eng._speaking is True
    assert engine.STATE_FILE.read_text() == "recording"
    assert not engine.PARTIAL_FILE.exists()


# --- voice-processing output ducking (sess. 25/09/2026) -----------------------

def test_relax_vp_output_ducking_applies_min_level(monkeypatch):
    """VP must not duck the rest of the Mac's audio: level pinned to Min."""
    import engine as eng

    class _Cfg:
        def __init__(self, advanced, level):
            self.advanced, self.level = advanced, level

    class _AVFStub:
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration = _Cfg
        AVAudioVoiceProcessingOtherAudioDuckingLevelMin = 10

    class _Node:
        def __init__(self):
            self.cfg = None

        def setVoiceProcessingOtherAudioDuckingConfiguration_(self, cfg):
            self.cfg = cfg

    monkeypatch.setattr(eng, "_AVF", _AVFStub)
    node = _Node()
    assert eng._relax_vp_output_ducking(node) is True
    assert node.cfg.level == 10
    assert node.cfg.advanced is False


def test_relax_vp_output_ducking_degrades_without_symbol(monkeypatch):
    """Older macOS / PyObjC without the symbol: no crash, capture still runs."""
    import engine as eng

    class _AVFStub:
        pass

    class _Node:
        def setVoiceProcessingOtherAudioDuckingConfiguration_(self, cfg):
            raise AssertionError("must not be called when the symbol is missing")

    monkeypatch.setattr(eng, "_AVF", _AVFStub)
    assert eng._relax_vp_output_ducking(_Node()) is False


def test_relax_vp_output_ducking_swallows_setter_failure(monkeypatch, capsys):
    """A HAL refusal is reported on stderr and never propagates."""
    import engine as eng

    class _Cfg:
        def __init__(self, advanced, level):
            pass

    class _AVFStub:
        AVAudioVoiceProcessingOtherAudioDuckingConfiguration = _Cfg
        AVAudioVoiceProcessingOtherAudioDuckingLevelMin = 10

    class _Node:
        def setVoiceProcessingOtherAudioDuckingConfiguration_(self, cfg):
            raise RuntimeError("boom")

    monkeypatch.setattr(eng, "_AVF", _AVFStub)
    assert eng._relax_vp_output_ducking(_Node()) is False
    assert "could not relax voice-processing ducking" in capsys.readouterr().err


# ── the widget's play/stop key: paste yes, Return no ─────────────────────────
def _record_keys(monkeypatch):
    """Replace every outbound effect of engine.autosend() with a recorder."""
    import subprocess as _sp
    keys = []
    monkeypatch.setattr(engine, "_press_key_cg",
                        lambda code, with_command=False: keys.append(code) or True)
    monkeypatch.setattr(_sp, "run", lambda *a, **k: None)
    monkeypatch.setattr(engine, "RETURN_DELAY", 0.0)
    monkeypatch.setattr(engine, "AUTOSEND_RETURN", True)
    # Where the paste lands is read from the live desktop; a test must not
    # depend on which app happens to be in front while it runs.
    monkeypatch.setattr(engine, "_paste_target", lambda: ("com.apple.terminal", None, True))
    monkeypatch.setattr(engine, "_secure_input", lambda: False)
    return keys


def test_is_return_held_reflects_flag_file(engine_state):
    assert engine.is_return_held() is False
    engine.RETURN_HELD_FILE.touch()
    assert engine.is_return_held() is True
    engine.RETURN_HELD_FILE.unlink()
    assert engine.is_return_held() is False


def test_autosend_presses_return_when_not_held(engine_state, monkeypatch):
    keys = _record_keys(monkeypatch)
    engine._REAL_autosend("ciao")
    assert keys == [9, 36]  # Cmd+V, then Return


def test_held_return_pastes_but_never_sends(engine_state, monkeypatch):
    """Stop on the widget: the words still land, the Return never fires."""
    keys = _record_keys(monkeypatch)
    engine.RETURN_HELD_FILE.touch()
    engine._REAL_autosend("ciao")
    assert keys == [9]


# ── The display is live while LISTENING, not only while recording ────────────
# The engine used to stream levels.bin only inside an utterance. The frame it
# left behind at finalize still held the utterance's energy, and nothing ever
# refreshed it: the hardware widget (always on screen, unlike the pill) kept
# showing red bars in the middle of a silent room (Mattia, 2026-10-23).

def test_levels_keep_streaming_while_idle_so_the_last_utterance_fades(engine_state):
    eng = engine.Engine()
    engine.write_state("idle")   # as Engine.run() does before the first block
    # an utterance just ended: the history still carries its energy
    eng._rms_history.extend([0.6] * engine.LEVELS_LEN)
    engine.write_levels(eng._rms_history)
    assert max(_read_levels_as_pill(engine.LEVELS_FILE)) > 0.5
    assert eng._speaking is False

    silence = np.zeros((engine.BLOCKSIZE, 1), dtype=np.float32)
    for _ in range(engine.LEVELS_LEN + eng._levels_every):
        eng._audio_callback(silence, engine.BLOCKSIZE, None, None)

    decoded = _read_levels_as_pill(engine.LEVELS_FILE)
    assert max(decoded) < 0.01, "idle silence must flatten the display, not freeze it"
    assert eng._speaking is False
    assert engine.STATE_FILE.read_text() == "idle"
