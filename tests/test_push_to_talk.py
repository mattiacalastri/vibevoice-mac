#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Contract tests for push-to-talk (hold Control+Option, talk, let go) and the
house fixes applied to every transcription.

In push-to-talk mode the speech decision is the key, not the VAD: the engine
records exactly while the pill says the chord is held (`~/.vibevoice/ptt`) and
closes the utterance a short tail after release — one press, one message, never
split on a pause and never opened by room noise.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import engine


LOUD = np.full((engine.BLOCKSIZE, 1), 0.5, dtype=np.float32)
QUIET = np.zeros((engine.BLOCKSIZE, 1), dtype=np.float32)


@pytest.fixture
def engine_state(tmp_path, monkeypatch):
    """Every writable engine path in tmp, every outbound switch off (as in
    test_contract.py)."""
    monkeypatch.setattr(engine, "AUTOSEND", False)
    monkeypatch.setattr(engine, "STREAM_PASTE", False)
    for name in ("LEVELS_FILE", "LEVELS_TMP", "STATE_FILE", "RAW_FILE", "MUTED_FILE",
                 "RETURN_HELD_FILE", "HISTORY_FILE", "METRICS_FILE", "DICT_FILE",
                 "CORRECTIONS_FILE", "CORPUS_FILE", "PARTIAL_FILE", "PARTIAL_TMP",
                 "AUTOSEND_PAUSE_FLAG", "FIXES_FILE", "LEARNED_FIXES_FILE"):
        monkeypatch.setattr(engine, name, tmp_path / name.lower())
    return tmp_path


@pytest.fixture
def ptt(engine_state, monkeypatch):
    monkeypatch.setattr(engine, "PTT", True)
    monkeypatch.setattr(engine, "PTT_FILE", engine_state / "ptt")
    return engine_state


def _feed(eng, block, n=1):
    for _ in range(n):
        eng._audio_callback(block, engine.BLOCKSIZE, None, None)


def test_room_noise_never_opens_an_utterance_without_the_key(ptt):
    eng = engine.Engine()
    _feed(eng, LOUD, 5)
    assert eng._speaking is False


def test_holding_the_key_records_even_in_silence(ptt):
    eng = engine.Engine()
    eng._ptt_held = True
    _feed(eng, QUIET)
    assert eng._speaking is True
    assert engine.STATE_FILE.read_text() == "recording"


def test_a_pause_while_held_does_not_split_the_message(ptt, monkeypatch):
    eng = engine.Engine()
    closed = []
    monkeypatch.setattr(eng, "_finalize", lambda now: closed.append(now))
    eng._ptt_held = True
    _feed(eng, LOUD, 3)
    _feed(eng, QUIET, 400)          # ~4 s of silence, key still down
    assert closed == []
    assert eng._speaking is True


def test_release_closes_the_utterance_after_the_tail(ptt, monkeypatch):
    monkeypatch.setattr(engine, "PTT_TAIL", 0.0)
    eng = engine.Engine()
    closed = []
    monkeypatch.setattr(eng, "_finalize", lambda now: closed.append(now))
    eng._ptt_held = True
    _feed(eng, LOUD, 3)
    eng._ptt_held = False
    _feed(eng, QUIET, 2)
    assert len(closed) == 1


def test_a_long_hold_is_not_cut_at_max_dur(ptt, monkeypatch):
    eng = engine.Engine()
    closed = []
    monkeypatch.setattr(eng, "_finalize", lambda now: closed.append(now))
    eng._ptt_held = True
    _feed(eng, LOUD)
    eng._t_start -= engine.MAX_DUR + 5.0   # held well past the VAD-mode cap
    _feed(eng, LOUD)
    assert closed == []


def test_the_hard_cap_still_exists_for_a_stuck_key(ptt, monkeypatch):
    eng = engine.Engine()
    closed = []
    monkeypatch.setattr(eng, "_finalize", lambda now: closed.append(now))
    eng._ptt_held = True
    _feed(eng, LOUD)
    eng._t_start -= engine.PTT_MAX_DUR + 1.0
    _feed(eng, LOUD)
    assert len(closed) == 1


def test_while_held_the_draft_is_written_but_never_typed(ptt, monkeypatch):
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "STREAM_PASTE", True)
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "apri il terminale adesso")
    typed = []
    monkeypatch.setattr(engine, "type_text", lambda t: typed.append(t) or True)
    eng = engine.Engine()
    eng._ptt_held = True
    _feed(eng, LOUD)
    gen = eng._utt_gen
    for _ in range(2):                       # two agreeing passes confirm the words
        assert eng._partial_busy.acquire(blocking=False)
        eng._partial_worker([LOUD[:, 0]], gen)
    assert engine.PARTIAL_FILE.read_text().startswith("apri il terminale")
    assert typed == []


def test_streaming_passes_run_while_held(ptt, monkeypatch):
    monkeypatch.setattr(engine, "STREAMING", True)
    eng = engine.Engine()
    passes = []
    monkeypatch.setattr(eng, "_partial_worker", lambda *a: passes.append(a) or eng._partial_busy.release())
    eng._ptt_held = True
    _feed(eng, LOUD)
    eng._t_partial -= 10.0
    _feed(eng, LOUD)
    assert len(passes) == 1


def test_ptt_reads_the_flag_file(ptt):
    eng = engine.Engine()
    eng._poll_ptt_once()
    assert eng._ptt_held is False
    engine.PTT_FILE.touch()
    eng._poll_ptt_once()
    assert eng._ptt_held is True


def test_vad_mode_is_untouched(engine_state, monkeypatch):
    monkeypatch.setattr(engine, "PTT", False)
    eng = engine.Engine()
    _feed(eng, LOUD)
    assert eng._speaking is True


# ── house fixes: known mishearings corrected after every transcription ─────────

@pytest.fixture
def fixes(tmp_path, monkeypatch):
    path = tmp_path / "fixes.json"
    monkeypatch.setattr(engine, "FIXES_FILE", path)
    monkeypatch.setattr(engine, "LEARNED_FIXES_FILE", tmp_path / "learned_fixes.json")
    return path


def test_fixes_correct_cloud_into_claude(fixes):
    fixes.write_text(json.dumps([[r"\bcloud\s+code\b", "Claude Code"],
                                 [r"\bcloud\b", "Claude"]]))
    assert engine.apply_fixes("usa Cloud Code e chiedi a cloud") == \
        "usa Claude Code e chiedi a Claude"


def test_fixes_read_the_house_dictionary_shape(fixes):
    fixes.write_text(json.dumps({"termini": ["x"], "correzioni": [
        [r"\b(di|da)\s+dude\b", r"\1 OOTDOT"]]}))
    assert engine.apply_fixes("parla di dude") == "parla di OOTDOT"


def test_fixes_missing_or_broken_change_nothing(fixes):
    assert engine.apply_fixes("ciao") == "ciao"
    fixes.write_text("{not json")
    assert engine.apply_fixes("ciao") == "ciao"
    fixes.write_text(json.dumps([["(unclosed", "x"], [r"\bciao\b", "salve"]]))
    assert engine.apply_fixes("ciao") == "salve"


def test_transcribe_applies_the_fixes(fixes, monkeypatch):
    class _W:
        def transcribe(self, audio, **kw):
            return {"text": " apri cloud code "}
    monkeypatch.setattr(engine, "_MLX_WHISPER", _W())
    monkeypatch.setattr(engine, "_MLX_AVAILABLE", True)
    monkeypatch.setattr(engine, "DICT_FILE", fixes.parent / "dictionary.txt")
    fixes.write_text(json.dumps([[r"\bcloud\s+code\b", "Claude Code"]]))
    assert engine.transcribe(np.zeros(16000, dtype=np.float32)) == "apri Claude Code"


def test_an_edited_fixes_file_counts_from_the_next_sentence(fixes):
    fixes.write_text(json.dumps([[r"\bcloud\b", "Claude"]]))
    assert engine.apply_fixes("cloud") == "Claude"
    fixes.write_text(json.dumps([[r"\bcloud\b", "Claude"], [r"\bfathom\b", "Fathom"],
                                 [r"\bgo level\b", "GoHighLevel"]]))
    assert engine.apply_fixes("cloud go level") == "Claude GoHighLevel"


def test_a_stuck_key_closes_and_reopens_without_losing_audio(ptt, monkeypatch):
    eng = engine.Engine()
    closed = []
    monkeypatch.setattr(eng, "_finalize", lambda now: closed.append(now) or setattr(eng, "_speaking", False))
    eng._ptt_held = True
    _feed(eng, LOUD)
    eng._t_start -= engine.PTT_MAX_DUR + 1.0
    _feed(eng, LOUD)
    _feed(eng, LOUD)
    assert len(closed) == 1 and eng._speaking is True


def test_warm_up_loads_the_model_without_publishing(engine_state, monkeypatch):
    calls = []
    monkeypatch.setattr(engine, "transcribe", lambda audio: calls.append(len(audio)) or "Grazie.")
    eng = engine.Engine()
    eng._warm_up()
    assert calls == [engine.SAMPLE_RATE]
    assert not engine.RAW_FILE.exists() and not engine.HISTORY_FILE.exists()
    assert eng._busy.acquire(blocking=False) and eng._busy.acquire(blocking=False)


# ── did the text land? ─────────────────────────────────────────────────────────

def test_paste_verdict():
    v = engine.paste_verdict
    assert v(False, False, "com.apple.terminal", None, True) == (False, "paste")
    assert v(True, True, "com.apple.terminal", "AXTextArea", True) == (False, "secure")
    assert v(True, False, "com.vibevoice.app", "AXTextArea", True) == (False, "vibevoice")
    # terminals/editors expose no reliable text role (Antigravity says AXGroup)
    assert v(True, False, "com.google.antigravity-ide", "AXGroup", True) == (True, "target")
    assert v(True, False, "com.google.chrome", "AXTextArea", True) == (True, "text")
    assert v(True, False, "com.apple.finder", "AXList", True) == (False, "role:AXList")
    assert v(True, False, "com.apple.finder", None, True) == (False, "nofocus")
    assert v(True, False, "com.apple.finder", None, False) == (True, "unknown")
    # a chat app's composer that is not a classic text role: benefit of the doubt
    assert v(True, False, "net.whatsapp.whatsapp", "AXWebArea", True) == (True, "maybe:AXWebArea")


def test_a_paste_that_missed_is_reported_and_never_sent(engine_state, monkeypatch):
    keys = []
    monkeypatch.setattr(engine, "PASTE_FILE", engine_state / "paste.json")
    monkeypatch.setattr(engine, "AUTOSEND_RETURN", True)
    monkeypatch.setattr(engine, "RETURN_DELAY", 0.0)
    monkeypatch.setattr(engine, "_press_key_cg", lambda code, with_command=False: keys.append(code) or True)
    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_paste_target", lambda: ("com.apple.finder", None, True))
    monkeypatch.setattr(engine, "_secure_input", lambda: False)
    engine._REAL_autosend("ciao mondo")
    report = json.loads((engine_state / "paste.json").read_text())
    assert report["ok"] is False and report["reason"] == "nofocus" and report["text"] == "ciao mondo"
    assert keys == [9]                      # Cmd+V yes, Return no


def test_a_good_paste_is_reported_and_sent(engine_state, monkeypatch):
    keys = []
    monkeypatch.setattr(engine, "PASTE_FILE", engine_state / "paste.json")
    monkeypatch.setattr(engine, "AUTOSEND_RETURN", True)
    monkeypatch.setattr(engine, "RETURN_DELAY", 0.0)
    monkeypatch.setattr(engine, "_press_key_cg", lambda code, with_command=False: keys.append(code) or True)
    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_paste_target", lambda: ("com.google.antigravity-ide", "AXGroup", True))
    monkeypatch.setattr(engine, "_secure_input", lambda: False)
    engine._REAL_autosend("ciao")
    assert json.loads((engine_state / "paste.json").read_text())["ok"] is True
    assert keys == [9, 36]


def test_the_chip_sees_the_newest_words_even_when_agreement_stalls(ptt, monkeypatch):
    monkeypatch.setattr(engine, "STREAMING", True)
    hyps = iter(["Ok, funziona correttamente.", "ok funziona correttamente e adesso",
                 "Ok. Funziona correttamente, e adesso parlo ancora"])
    monkeypatch.setattr(engine, "transcribe", lambda audio: next(hyps))
    eng = engine.Engine()
    eng._ptt_held = True
    _feed(eng, LOUD)
    gen = eng._utt_gen
    for _ in range(3):
        assert eng._partial_busy.acquire(blocking=False)
        eng._partial_worker([LOUD[:, 0]], gen)
    assert engine.PARTIAL_FILE.read_text().endswith("parlo ancora")


def test_the_draft_never_opens_on_a_hallucinated_grazie(ptt, monkeypatch):
    monkeypatch.setattr(engine, "STREAMING", True)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "Grazie.")
    eng = engine.Engine()
    eng._ptt_held = True
    _feed(eng, LOUD)
    assert eng._partial_busy.acquire(blocking=False)
    eng._partial_worker([QUIET[:, 0]], eng._utt_gen)
    assert not engine.PARTIAL_FILE.exists() or engine.PARTIAL_FILE.read_text() == ""


def test_leading_artifacts_are_dropped_from_the_draft_only_at_the_start():
    f = engine._strip_canned_lead
    assert f("Grazie. Grazie a tutti. Ecco qua, sto provando.") == "Ecco qua, sto provando."
    assert f("Ecco qua. Grazie a tutti.") == "Ecco qua. Grazie a tutti."
    assert f("Grazie mille Marco, ci sentiamo.") == "Grazie mille Marco, ci sentiamo."


def test_the_mic_closes_when_idle_and_reopens_on_the_key(ptt, monkeypatch):
    """Push-to-talk must not keep the microphone open: macOS showed the orange
    mic dot all day although nothing was recorded without the key (Mattia,
    2026-10-31). The stream opens on the press and closes after an idle grace;
    a reopened stream must not prepend audio from the previous session."""
    import threading
    import time

    monkeypatch.setattr(engine, "MIC_IDLE_CLOSE", 0.05)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "")
    events = []
    eng = engine.Engine()

    class _FakeCapture:
        name = "fake"

        def __init__(self, callback):
            self._cb = callback

        def __enter__(self):
            events.append(("open", len(eng._pre)))
            for _ in range(3):
                self._cb(QUIET, engine.BLOCKSIZE, None, None)
            return self

        def __exit__(self, *exc):
            events.append(("close", None))
            return False

    monkeypatch.setattr(engine, "_select_capture_backend", lambda: _FakeCapture)
    t = threading.Thread(target=eng.run, daemon=True)
    t.start()

    def _wait(pred, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not pred():
            time.sleep(0.01)
        return pred()

    try:
        assert _wait(lambda: ("close", None) in events), "the mic never closed when idle"
        assert [e for e in events if e[0] == "open"] == [("open", 0)]
        time.sleep(0.1)
        assert len([e for e in events if e[0] == "open"]) == 1, "reopened with no key"
        engine.PTT_FILE.write_text("")
        assert _wait(lambda: len([e for e in events if e[0] == "open"]) == 2), \
            "the key did not reopen the mic"
        assert [e for e in events if e[0] == "open"][1] == ("open", 0), \
            "stale pre-roll from the previous session"
    finally:
        eng.stop()
        t.join(timeout=2)


def test_a_pausable_backend_is_paused_not_rebuilt(ptt, monkeypatch):
    """Rebuilding voice processing takes ~0.9 s to the first audio block, and
    the first word of the sentence was lost (measured 2026-10-31: «Cosa dicono»
    arrived as «Dicono»). A stopped AVAudioEngine restarts in ~150 ms and still
    releases the mic, so a backend that can pause is opened once and paused."""
    import threading
    import time

    monkeypatch.setattr(engine, "MIC_IDLE_CLOSE", 0.05)
    monkeypatch.setattr(engine, "transcribe", lambda audio: "")
    events = []
    eng = engine.Engine()

    class _PausableCapture:
        name = "fake-pausable"

        def __init__(self, callback):
            pass

        def __enter__(self):
            events.append("open")
            return self

        def __exit__(self, *exc):
            events.append("close")
            return False

        def pause(self):
            events.append("pause")

        def resume(self):
            events.append("resume")

    monkeypatch.setattr(engine, "_select_capture_backend", lambda: _PausableCapture)
    t = threading.Thread(target=eng.run, daemon=True)
    t.start()

    def _wait(pred, timeout=3.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end and not pred():
            time.sleep(0.01)
        return pred()

    try:
        assert _wait(lambda: "pause" in events), "the mic never paused when idle"
        engine.PTT_FILE.write_text("")
        assert _wait(lambda: "resume" in events), "the key did not resume the mic"
        assert events.count("open") == 1 and "close" not in events, events
    finally:
        eng.stop()
        t.join(timeout=2)
    assert events[-1] == "close"
