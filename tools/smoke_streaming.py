#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""End-to-end smoke for the streaming path (F3) — real audio, real Whisper.

`pytest` proves the wiring with a fake `transcribe`; it cannot prove that live
text actually appears while a human is still talking. This does: it replays a
real WAV through the capture seam at wall-clock speed with the real mlx_whisper
model, and watches ~/.vibevoice/partial.txt grow.

It is a tool, not a pytest, for the same reason as smoke_settings_window.py:
it needs the model on disk and takes tens of seconds — CI has neither.

State is redirected to a throwaway directory: the live ~/.vibevoice/ runtime is
never touched, and AUTOSEND is forced off so nothing is pasted into whatever app
you have in front of you.

Usage:
    python3 tools/smoke_streaming.py                 # generates speech with `say`
    python3 tools/smoke_streaming.py path/to.wav     # 16 kHz mono float-able WAV

Exit code 0 = the criteria below all held.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine  # noqa: E402

PHRASE = (
    "Il polpo ha otto tentacoli e ognuno di essi pensa in modo autonomo, "
    "mentre il cervello centrale si limita a coordinare la direzione generale "
    "del movimento senza decidere ogni singolo gesto."
)


def _redirect_state(tmp: Path) -> None:
    """Point every file the engine can write at a throwaway dir (CLAUDE.md rule 2)."""
    for attr, name in (
        ("STATE_FILE", "state"), ("LEVELS_FILE", "levels.bin"), ("LEVELS_TMP", "levels.tmp"),
        ("RAW_FILE", "raw.txt"), ("HISTORY_FILE", "history.jsonl"),
        ("METRICS_FILE", "metrics.jsonl"), ("DICT_FILE", "dictionary.txt"),
        ("CORRECTIONS_FILE", "corrections.jsonl"), ("MUTED_FILE", "muted"),
        ("PARTIAL_FILE", "partial.txt"), ("PARTIAL_TMP", "partial.tmp"),
    ):
        setattr(engine, attr, tmp / name)


# What the engine WOULD have typed, with timestamps. The real keyboard is never
# touched: synthetic keystrokes posted into whatever app the user has in front
# of them are not recoverable, unlike a file written in the wrong place.
TYPED: list[tuple[float, str]] = []
_T0 = [0.0]


def _arm_paste_recorder() -> None:
    """Enable the paste paths but redirect them into TYPED."""
    engine.AUTOSEND = True
    engine.STREAM_PASTE = True
    engine.type_text = lambda text: TYPED.append((time.monotonic() - _T0[0], text)) or True
    engine.autosend = lambda text: TYPED.append((time.monotonic() - _T0[0], text))


def _speech_wav(dest: Path) -> Path:
    """Render PHRASE to a 16 kHz mono WAV with the system voice."""
    aiff = dest.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff), PHRASE], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(dest)],
        check=True,
    )
    return dest


def _load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        if w.getframerate() != engine.SAMPLE_RATE:
            raise SystemExit(f"need {engine.SAMPLE_RATE} Hz, got {w.getframerate()}")
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class ReplayCapture:
    """Capture backend that replays a buffer in real time through the callback.

    Same contract as _SounddeviceCapture: a context manager that feeds
    (indata, frames, time_info, status) blocks. Real time matters — the whole
    point is to observe what exists *while* the utterance is still open.
    """

    name = "replay"
    audio: np.ndarray = np.zeros(0, dtype=np.float32)
    done = threading.Event()

    def __init__(self, callback) -> None:
        self._cb = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "ReplayCapture":
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return False

    def _pump(self) -> None:
        block_s = engine.BLOCKSIZE / engine.SAMPLE_RATE
        # Trailing silence long enough to close the utterance (SILENCE_SEC) plus margin.
        tail = np.zeros(int((engine.SILENCE_SEC + 1.5) * engine.SAMPLE_RATE), dtype=np.float32)
        stream = np.concatenate([self.audio, tail])
        t0 = time.monotonic()
        for i, off in enumerate(range(0, len(stream) - engine.BLOCKSIZE, engine.BLOCKSIZE)):
            if self._stop.is_set():
                return
            block = stream[off:off + engine.BLOCKSIZE].reshape(-1, 1)
            self._cb(block, engine.BLOCKSIZE, None, None)
            # Pace to wall clock so "while still speaking" means what it says.
            slack = t0 + (i + 1) * block_s - time.monotonic()
            if slack > 0:
                time.sleep(slack)
        type(self).done.set()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="vv_smoke_"))
    _redirect_state(tmp)

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else _speech_wav(tmp / "speech.wav")
    audio = _load(src)
    speech_s = len(audio) / engine.SAMPLE_RATE
    print(f"▸ sorgente: {src.name}  ({speech_s:.1f}s di parlato)")
    print(f"▸ streaming={engine.STREAMING}  interval={engine.PARTIAL_INTERVAL}s  "
          f"silence={engine.SILENCE_SEC}s")

    if not engine._ensure_mlx_whisper():
        print("✗ mlx_whisper non disponibile — impossibile provare il sistema vero")
        return 2
    engine.transcribe(audio[:engine.SAMPLE_RATE])  # warm-up: model load out of the measurement

    ReplayCapture.audio = audio
    ReplayCapture.done = threading.Event()
    eng = engine.Engine()

    timeline: list[tuple[float, str]] = []   # (t since speech start, confirmed draft)
    raw_at: list[float] = []

    def watch() -> None:
        last = None
        while not ReplayCapture.done.is_set() or (time.monotonic() - t_start) < speech_s + 6:
            try:
                draft = engine.PARTIAL_FILE.read_text()
            except OSError:
                draft = None
            if draft is not None and draft != last:
                timeline.append((time.monotonic() - t_start, draft))
                last = draft
            if engine.RAW_FILE.exists() and not raw_at:
                raw_at.append(time.monotonic() - t_start)
            time.sleep(0.05)

    t_start = time.monotonic()
    _T0[0] = t_start
    _arm_paste_recorder()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()

    runner = threading.Thread(target=lambda: eng._capture_loop(ReplayCapture), daemon=True)
    runner.start()
    ReplayCapture.done.wait(timeout=speech_s + 30)
    time.sleep(2.0)   # let the final transcription land
    eng.stop()
    watcher.join(timeout=3)

    print("\n▸ timeline della bozza viva (t = secondi dall'inizio del parlato)")
    for t, draft in timeline:
        shown = draft if draft else "(vuota — nessuna parola ancora stabile)"
        print(f"   {t:5.1f}s  {shown}")

    final = engine.RAW_FILE.read_text() if engine.RAW_FILE.exists() else ""
    print(f"\n▸ testo finale (raw.txt): {final!r}")

    # ── Criteria ─────────────────────────────────────────────────────────────
    words = [(t, d) for t, d in timeline if d.strip()]
    first_word_t = words[0][0] if words else None
    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"   {'✓' if passed else '✗'} {label}{('  — ' + detail) if detail else ''}")

    print("\n▸ criteri")
    check("una bozza con parole è comparsa", first_word_t is not None,
          f"prima parola a {first_word_t:.1f}s" if first_word_t else "nessuna bozza con testo")
    check("è comparsa MENTRE si parlava ancora",
          first_word_t is not None and first_word_t < speech_s,
          f"{first_word_t:.1f}s < {speech_s:.1f}s di parlato" if first_word_t else "")
    check("la bozza è cresciuta più volte", len(words) >= 2, f"{len(words)} aggiornamenti")
    monotonic = all(
        words[i][1].startswith(words[i - 1][1][:len(words[i - 1][1])])
        or words[i - 1][1] in words[i][1]
        for i in range(1, len(words))
    )
    check("nessuna parola già mostrata è stata ritirata", monotonic)
    check("il testo finale è arrivato", bool(final.strip()))

    # ── Streaming paste: what actually reached the app, and when ─────────────
    print("\n▸ cosa è stato digitato nell'app (t = secondi dall'inizio del parlato)")
    for t, chunk in TYPED:
        print(f"   {t:5.1f}s  {chunk!r}")
    landed = "".join(chunk for _, chunk in TYPED)
    keys_typed = [engine._agreement_key(w) for w in landed.split()]
    keys_final = [engine._agreement_key(w) for w in final.split()]

    first_typed_t = TYPED[0][0] if TYPED else None
    check("qualcosa è stato digitato mentre parlavi ancora",
          first_typed_t is not None and first_typed_t < speech_s,
          f"primo carattere a {first_typed_t:.1f}s" if first_typed_t is not None else "nulla digitato")
    check("nessuna parola digitata due volte, nessuna persa",
          keys_typed == keys_final,
          f"{len(keys_typed)} parole digitate vs {len(keys_final)} finali")
    check("le parole non si sono saldate fra loro", "  " not in landed and landed == landed.strip())

    if first_typed_t is not None:
        # The old engine could not paste before SILENCE_SEC had elapsed after the
        # last word — that is the number this whole feature exists to beat.
        old_paste_t = speech_s + engine.SILENCE_SEC
        print(f"\n▸ guadagno sull'incolla: primo testo nell'app a {first_typed_t:.1f}s "
              f"contro {old_paste_t:.1f}s del motore batch → {old_paste_t - first_typed_t:.1f}s prima")
    if first_word_t is not None and raw_at:
        print(f"▸ guadagno sulla bozza: {first_word_t:.1f}s vs testo finale a {raw_at[0]:.1f}s "
              f"→ {raw_at[0] - first_word_t:.1f}s di anticipo")

    print(f"\n{'PASS' if ok else 'FAIL'}   (stato in {tmp})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
