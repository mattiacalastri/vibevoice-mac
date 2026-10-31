#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Contract tests for the dictation-quality layer: the personal dictionary that
biases Whisper decoding via `initial_prompt`.

Like test_contract.py these exercise the real engine module headless — no mic,
no model download — and redirect every runtime file into tmp_path so the live
~/.vibevoice/ is never touched.
"""
from __future__ import annotations

import numpy as np
import pytest

import engine


@pytest.fixture
def quality_state(tmp_path, monkeypatch):
    """Redirect the dictation-quality runtime files into a tmp dir."""
    monkeypatch.setattr(engine, "DICT_FILE", tmp_path / "dictionary.txt")
    monkeypatch.setattr(engine, "RAW_FILE", tmp_path / "raw.txt")
    monkeypatch.setattr(engine, "HISTORY_FILE", tmp_path / "history.jsonl")
    monkeypatch.setattr(engine, "CORPUS_FILE", tmp_path / "corpus.jsonl")
    monkeypatch.setattr(engine, "METRICS_FILE", tmp_path / "metrics.jsonl")
    monkeypatch.setattr(engine, "STATE_FILE", tmp_path / "state")
    monkeypatch.setattr(engine, "LEVELS_FILE", tmp_path / "levels.bin")
    monkeypatch.setattr(engine, "LEVELS_TMP", tmp_path / "levels.tmp")
    return tmp_path


class _FakeWhisper:
    """Stand-in for the mlx_whisper module: records transcribe() arguments."""

    def __init__(self, text: str = " ciao mondo "):
        self.text = text
        self.calls: list[dict] = []
        self.audio_args: list = []

    def transcribe(self, audio, **kwargs):
        self.audio_args.append(audio)
        self.calls.append(kwargs)
        return {"text": self.text}


@pytest.fixture
def fake_whisper(monkeypatch):
    fake = _FakeWhisper()
    monkeypatch.setattr(engine, "_MLX_WHISPER", fake)
    monkeypatch.setattr(engine, "_MLX_AVAILABLE", True)
    return fake


# ── load_dictionary ───────────────────────────────────────────────────────────

def test_load_dictionary_reads_terms_skipping_comments_and_blanks(quality_state):
    engine.DICT_FILE.write_text(
        "# clienti\nMoonline\n\nFathom\n  GoHighLevel  \n# fine\n"
    )
    assert engine.load_dictionary() == ["Moonline", "Fathom", "GoHighLevel"]


def test_load_dictionary_missing_file_is_empty(quality_state):
    assert engine.load_dictionary() == []


def test_load_dictionary_caps_terms(quality_state):
    engine.DICT_FILE.write_text("\n".join(f"termine{i}" for i in range(500)))
    assert len(engine.load_dictionary()) == engine.DICT_MAX_TERMS


# ── transcribe() ──────────────────────────────────────────────────────────────

def test_transcribe_passes_the_array_not_a_wav_path(quality_state, fake_whisper):
    """mlx_whisper decodes file paths through ffmpeg, which is NOT on the
    launchd PATH (scar sess.9685: every dictation died with 'No such file or
    directory: ffmpeg'). Passing the float32 array skips ffmpeg entirely."""
    audio = np.full(1600, 0.25, dtype=np.float32)

    engine.transcribe(audio)

    assert len(fake_whisper.audio_args) == 1
    arg = fake_whisper.audio_args[0]
    assert isinstance(arg, np.ndarray)
    assert arg is audio  # the buffer itself, no temp-file roundtrip

def test_transcribe_passes_dictionary_as_initial_prompt(quality_state, fake_whisper):
    engine.DICT_FILE.write_text("Moonline\nFathom\n")
    audio = np.zeros(1600, dtype=np.float32)

    text = engine.transcribe(audio)

    assert text == "ciao mondo"
    assert len(fake_whisper.calls) == 1
    prompt = fake_whisper.calls[0].get("initial_prompt")
    assert prompt is not None
    assert "Moonline" in prompt and "Fathom" in prompt


def test_transcribe_without_dictionary_omits_initial_prompt(quality_state, fake_whisper):
    audio = np.zeros(1600, dtype=np.float32)

    engine.transcribe(audio)

    assert len(fake_whisper.calls) == 1
    assert fake_whisper.calls[0].get("initial_prompt") is None


def test_transcribe_survives_unreadable_dictionary(quality_state, fake_whisper, monkeypatch):
    """A broken dictionary must never break transcription (degradation is contract)."""
    monkeypatch.setattr(
        engine, "load_dictionary", lambda: (_ for _ in ()).throw(OSError("boom"))
    )
    audio = np.zeros(1600, dtype=np.float32)

    assert engine.transcribe(audio) == "ciao mondo"


# ── metrics.jsonl: the latency telemetry ─────────────────────────────────────

def test_append_metrics_caps_lines(quality_state):
    import json

    for i in range(engine.METRICS_MAX + 30):
        engine._append_metrics({"i": i})

    lines = engine.METRICS_FILE.read_text().splitlines()
    assert len(lines) == engine.METRICS_MAX
    assert json.loads(lines[-1])["i"] == engine.METRICS_MAX + 29  # newest last


def test_process_utterance_writes_metrics_with_latency_fields(quality_state, monkeypatch):
    import json
    import time as _time

    monkeypatch.setattr(engine, "transcribe", lambda audio: "ciao mondo")
    # Audible on purpose: this test is about the latency fields, and a silent
    # buffer with a short text is now dropped as a hallucination (_is_phantom).
    audio = np.full(engine.SAMPLE_RATE, 0.4, dtype=np.float32)  # 1s of audio

    # t_end is on the audio loop's clock (time.monotonic), not time.time().
    text = engine.process_utterance(audio, t_end=_time.monotonic() - 0.5)

    assert text == "ciao mondo"
    assert engine.RAW_FILE.read_text() == "ciao mondo"
    assert "ciao mondo" in engine.HISTORY_FILE.read_text()
    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert entry["audio_s"] == pytest.approx(1.0)
    assert entry["chars"] == len("ciao mondo")
    assert entry["stt_ms"] >= 0
    assert entry["total_ms"] >= 500  # end-of-speech was 0.5s ago


def test_process_utterance_empty_transcription_writes_nothing(quality_state, monkeypatch):
    monkeypatch.setattr(engine, "transcribe", lambda audio: "")
    audio = np.zeros(1600, dtype=np.float32)

    assert engine.process_utterance(audio, t_end=None) == ""
    assert not engine.RAW_FILE.exists()
    assert not engine.METRICS_FILE.exists()


# ── paste ordering (long-dictation quality) ──────────────────────────────────

def test_pastes_stay_in_utterance_order_even_when_second_finishes_first(
    quality_state, monkeypatch
):
    """Two blobs in flight (Semaphore(2), invariant #4): the SECOND may finish
    transcribing first, but pastes must land in speech order — out-of-order
    pastes scramble long dictations (AGENTS.md §9.3, the documented sharp edge:
    the correct fix sequences PASTE order while keeping concurrent transcribe)."""
    import threading
    import time as _t

    eng = engine.Engine()
    pasted = []
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "autosend", lambda text: pasted.append(text))

    # Signature must mirror the real process_utterance, `extra` included:
    # a double that drifts from it stops testing the thing it stands for.
    def fake_process(audio, t_end=None, extra=None):
        _t.sleep(0.30 if len(audio) > 100 else 0.02)  # first slow, second fast
        return "prima frase" if len(audio) > 100 else "seconda frase"

    monkeypatch.setattr(engine, "process_utterance", fake_process)

    slow = np.zeros(1000, dtype=np.float32)
    fast = np.zeros(10, dtype=np.float32)
    eng._busy.acquire(blocking=False)
    eng._busy.acquire(blocking=False)  # both slots taken, as _finalize does
    t1 = threading.Thread(target=eng._transcribe_worker, args=(slow, 0.0, 0), daemon=True)
    t2 = threading.Thread(target=eng._transcribe_worker, args=(fast, 0.0, 1), daemon=True)
    t1.start()
    _t.sleep(0.05)
    t2.start()

    deadline = _t.monotonic() + 3.0
    while len(pasted) < 2 and _t.monotonic() < deadline:
        _t.sleep(0.02)
    assert pasted == ["prima frase", "seconda frase"]


def test_empty_transcription_does_not_dam_the_paste_queue(quality_state, monkeypatch):
    """An utterance that transcribes to '' must still advance the sequence, or
    every following paste waits for a turn that never comes."""
    import threading
    import time as _t

    eng = engine.Engine()
    pasted = []
    monkeypatch.setattr(engine, "AUTOSEND", True)
    monkeypatch.setattr(engine, "autosend", lambda text: pasted.append(text))
    monkeypatch.setattr(
        engine, "process_utterance",
        lambda audio, t_end=None, extra=None: "" if len(audio) > 100 else "dopo il vuoto",
    )

    eng._busy.acquire(blocking=False)
    eng._busy.acquire(blocking=False)
    t1 = threading.Thread(
        target=eng._transcribe_worker,
        args=(np.zeros(1000, dtype=np.float32), 0.0, 0), daemon=True)
    t2 = threading.Thread(
        target=eng._transcribe_worker,
        args=(np.zeros(10, dtype=np.float32), 0.0, 1), daemon=True)
    t1.start()
    t2.start()

    deadline = _t.monotonic() + 3.0
    while len(pasted) < 1 and _t.monotonic() < deadline:
        _t.sleep(0.02)
    assert pasted == ["dopo il vuoto"]


# ── LLM cleanup pass (env-gated; degradation is contract) ────────────────────

class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _chat_response(content: str) -> bytes:
    import json

    return json.dumps({"choices": [{"message": {"content": content}}]}).encode()


@pytest.fixture
def cleanup_enabled(monkeypatch):
    monkeypatch.setattr(engine, "CLEANUP_ENABLED", True)
    monkeypatch.setattr(engine, "CLEANUP_API_KEY", "test-key")


def test_cleanup_disabled_by_default_makes_no_network_call(quality_state, monkeypatch):
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("network call with cleanup disabled")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert engine.CLEANUP_ENABLED is False  # default OFF: zero behavior change
    assert engine.cleanup_text("ehm ciao a tutti") == "ehm ciao a tutti"


def test_cleanup_success_returns_polished_text(quality_state, cleanup_enabled, monkeypatch):
    import json
    import urllib.request

    seen = {}

    def _fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["payload"] = json.loads(req.data.decode())
        return _FakeHTTPResponse(_chat_response("Ciao a tutti."))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    out = engine.cleanup_text("ehm ciao a tutti")

    assert out == "Ciao a tutti."
    assert seen["auth"] == "Bearer test-key"
    assert seen["payload"]["model"] == engine.CLEANUP_MODEL
    assert "ehm ciao a tutti" in json.dumps(seen["payload"])


def test_cleanup_key_falls_back_to_key_file(quality_state, monkeypatch):
    import urllib.request

    monkeypatch.setattr(engine, "CLEANUP_ENABLED", True)
    monkeypatch.setattr(engine, "CLEANUP_API_KEY", "")  # no env key
    monkeypatch.setattr(engine, "CLEANUP_KEY_FILE", engine.DICT_FILE.parent / "cleanup_key")
    engine.CLEANUP_KEY_FILE.write_text("file-key\n")
    seen = {}

    def _fake_urlopen(req, timeout=None):
        seen["auth"] = req.get_header("Authorization")
        return _FakeHTTPResponse(_chat_response("Pulito."))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    assert engine.cleanup_text("grezzo qui") == "Pulito."
    assert seen["auth"] == "Bearer file-key"


def test_cleanup_without_any_key_skips_silently(quality_state, monkeypatch):
    import urllib.request

    monkeypatch.setattr(engine, "CLEANUP_ENABLED", True)
    monkeypatch.setattr(engine, "CLEANUP_API_KEY", "")
    monkeypatch.setattr(engine, "CLEANUP_KEY_FILE", engine.DICT_FILE.parent / "missing_key")

    def _boom(*a, **k):
        raise AssertionError("network call without a key")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert engine.cleanup_text("testo") == "testo"


def test_cleanup_any_failure_falls_back_to_raw(quality_state, cleanup_enabled, monkeypatch):
    import urllib.request

    def _fail(*a, **k):
        raise OSError("timeout")

    monkeypatch.setattr(urllib.request, "urlopen", _fail)
    assert engine.cleanup_text("testo grezzo") == "testo grezzo"


def test_cleanup_hallucination_guard_rejects_bloated_output(
    quality_state, cleanup_enabled, monkeypatch
):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _FakeHTTPResponse(_chat_response("bla " * 200)),
    )
    assert engine.cleanup_text("breve") == "breve"


def test_cleanup_prompt_includes_dictionary_terms(quality_state):
    engine.DICT_FILE.write_text("Moonline\nFathom\n")
    prompt = engine._build_cleanup_prompt()
    assert "Moonline" in prompt and "Fathom" in prompt


def test_cleanup_prompt_includes_recent_corrections_as_examples(quality_state, monkeypatch):
    import json

    monkeypatch.setattr(engine, "CORRECTIONS_FILE", engine.DICT_FILE.parent / "corrections.jsonl")
    engine.CORRECTIONS_FILE.write_text(
        json.dumps({"ts": 1.0, "raw": "ciao mun line", "corrected": "ciao Moonline"}) + "\n"
    )
    prompt = engine._build_cleanup_prompt()
    assert "ciao mun line" in prompt and "ciao Moonline" in prompt


def test_load_corrections_returns_newest_last_capped(quality_state, monkeypatch):
    import json

    monkeypatch.setattr(engine, "CORRECTIONS_FILE", engine.DICT_FILE.parent / "corrections.jsonl")
    engine.CORRECTIONS_FILE.write_text(
        "\n".join(
            json.dumps({"ts": i, "raw": f"r{i}", "corrected": f"c{i}"}) for i in range(20)
        )
        + "\n"
    )
    pairs = engine._load_corrections(max_n=5)
    assert len(pairs) == 5
    assert pairs[-1]["raw"] == "r19"


def test_process_utterance_records_cleanup_ms_when_enabled(
    quality_state, cleanup_enabled, monkeypatch
):
    import json

    monkeypatch.setattr(engine, "transcribe", lambda audio: "ehm ciao")
    monkeypatch.setattr(engine, "cleanup_text", lambda text: "Ciao.")
    # Audible on purpose: this test is about the metrics fields, and a silent
    # buffer with a short text is now dropped as a hallucination (_is_phantom).
    audio = np.full(1600, 0.4, dtype=np.float32)

    text = engine.process_utterance(audio, t_end=None)

    assert text == "Ciao."
    assert engine.RAW_FILE.read_text() == "Ciao."  # what you see is what was pasted
    entry = json.loads(engine.METRICS_FILE.read_text().splitlines()[-1])
    assert "cleanup_ms" in entry
