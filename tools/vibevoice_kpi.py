#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""KPI report for VibeVoice — the numbers that decide what to fix next.

`vibevoice_metrics.py` reports the latency of the final decode, which was the
whole story when the engine was pure batch. It is not any more: what governs
the experience now is when the first character reaches the app, how much of the
sentence the stream delivers before the silence, and whether the reconciliation
found its anchor or dropped words on the floor.

Reads ~/.vibevoice/metrics.jsonl (per-utterance ledger, written by the engine)
and ~/.vibevoice/history.jsonl (the final texts). Read-only.

Usage:
    python3 tools/vibevoice_kpi.py            # everything on record
    python3 tools/vibevoice_kpi.py --today    # only utterances from today
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

STATE = Path.home() / ".vibevoice"
METRICS = STATE / "metrics.jsonl"
HISTORY = STATE / "history.jsonl"

# Defaults mirrored from engine.py — only used to explain the tail wait.
SILENCE_SEC = float(__import__("os").environ.get("VIBEVOICE_SILENCE", "1.5"))
PARTIAL_INTERVAL = float(__import__("os").environ.get("VIBEVOICE_PARTIAL_INTERVAL", "0.6"))


def _load(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return []


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _word_key(word: str) -> str:
    return re.sub(r"[^\w']+", "", word, flags=re.UNICODE).casefold()


def _bar(fraction: float, width: int = 24) -> str:
    filled = max(0, min(width, round(fraction * width)))
    return "█" * filled + "·" * (width - filled)


def main() -> int:
    only_today = "--today" in sys.argv
    rows = _load(METRICS)

    # The engine never finalises an utterance shorter than MIN_DUR (0.4s), so a
    # row below that was not dictated — it is a test buffer that leaked into the
    # live ledger through a worker outliving its test (13 such rows on
    # 2026-08-02, each 0.1s of audio transcribed as "Grazie a tutti."). They made
    # slow decodes look three times more common than they are. The leak is
    # closed in tests/conftest.py; this keeps the history readable regardless.
    strays = len(rows)
    rows = [r for r in rows if r.get("audio_s", 1.0) >= 0.4]
    strays -= len(rows)

    if only_today:
        midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        rows = [r for r in rows if r.get("ts", 0) >= midnight]
    if not rows:
        print("Nessuna utterance registrata — parla un po' e riprova.")
        return 1

    print(f"╭─ VibeVoice · KPI su {len(rows)} utterance"
          f"{' (oggi)' if only_today else ''}")
    if strays:
        print(f"│   ({strays} righe sotto MIN_DUR ignorate — non sono dettature)")

    # ── 1. Immediacy: when did the first character reach the app? ────────────
    first = [r["t_first_ms"] for r in rows if r.get("t_first_ms", -1) >= 0]
    print("│")
    print("├─ IMMEDIATEZZA — dal primo suono al primo carattere nell'app")
    if first:
        print(f"│   p50 {_pct(first, .5)/1000:5.2f}s   p90 {_pct(first, .9)/1000:5.2f}s"
              f"   n={len(first)}")
        # Why the slow ones are slow: the stream stands down whenever an earlier
        # utterance still owes a paste, and back-to-back dictation with MAX_DUR
        # cuts is exactly when that happens.
        downs = [r.get("stood_down", 0) for r in rows if "stood_down" in r]
        if downs:
            blocked = len([d for d in downs if d])
            print(f"│   turni saltati per un incolla ancora in coda: "
                  f"{sum(downs)} su {len(downs)} utterance ({blocked} utterance toccate)")
            if blocked:
                print(f"│   → ognuno costa ~{PARTIAL_INTERVAL:.1f}s di attesa in più"
                      f" sul primo carattere")
    else:
        print("│   (nessun dato: la digitazione progressiva era spenta)")

    # ── 2. Coverage: how much arrived before the silence? ────────────────────
    covered = [(r["stream_words"], r.get("final_words") or r["stream_words"] + r.get("tail_words", 0))
               for r in rows if "stream_words" in r and r.get("stream_words", 0) > 0]
    print("│")
    print("├─ COPERTURA — quanta parte della frase arriva PRIMA del silenzio")
    if covered:
        ratios = [s / t for s, t in covered if t]
        mean = sum(ratios) / len(ratios)
        print(f"│   {_bar(mean)}  {mean*100:.0f}% in media   n={len(ratios)}")
        print(f"│   più bassa {min(ratios)*100:.0f}%   più alta {max(ratios)*100:.0f}%")
        print(f"│   → il {(1-mean)*100:.0f}% resta in coda e paga l'attesa qui sotto")
    else:
        print("│   (nessuna utterance con digitazione progressiva)")

    # ── 3. The tail wait, and how much of it is imposed silence ─────────────
    stt = [r["stt_ms"] for r in rows if "stt_ms" in r]
    print("│")
    print("├─ ATTESA SULLA CODA — fine parlato → ultime parole")
    if stt:
        wait50 = SILENCE_SEC + _pct(stt, .5) / 1000
        wait90 = SILENCE_SEC + _pct(stt, .9) / 1000
        share = SILENCE_SEC / wait50
        print(f"│   p50 {wait50:5.2f}s   p90 {wait90:5.2f}s")
        print(f"│   silenzio imposto {_bar(share)}  {share*100:.0f}%")
        print(f"│   decodifica       {_bar(1-share)}  {(1-share)*100:.0f}%"
              f"  (p50 {_pct(stt,.5):.0f}ms · p99 {_pct(stt,.99):.0f}ms)")
        slow = [v for v in stt if v > 1000]
        if slow:
            print(f"│   ⚠ {len(slow)}/{len(stt)} decodifiche oltre 1s (max {max(stt)/1000:.1f}s)"
                  f" — la coda di quelle frasi è molto più lenta della mediana")

    # ── 4. Fidelity: did the reconciliation lose or repeat anything? ────────
    anchors = [r.get("anchor") for r in rows if "anchor" in r]
    print("│")
    print("├─ FEDELTÀ — la giuntura fra ciò che è stato digitato e la coda finale")
    if anchors:
        lost = anchors.count("none")
        print(f"│   ancora trovata {anchors.count('ok')}/{len(anchors)}")
        if lost:
            print(f"│   ⚠ {lost} volte nessuna ancora → coda NON incollata (parole perse)")
        else:
            print("│   nessuna coda scartata")
    else:
        print("│   (nessun dato: serve una versione con la telemetria della giuntura)")

    # ── 5. What the final text itself looks like ────────────────────────────
    texts = [r["text"] for r in _load(HISTORY)]
    print("│")
    print(f"├─ TESTO FINALE — ultime {len(texts)} frasi in archivio")
    if texts:
        dup = 0
        edge = 0
        for t in texts:
            words = [_word_key(w) for w in t.split() if _word_key(w)]
            if any(words[i:i+2] == words[i+2:i+4] for i in range(len(words) - 3)):
                dup += 1
            edge += len(re.findall(r"\w\.\s+[a-zà-ù]", t))
        print(f"│   ripetizioni interne {dup}/{len(texts)}   punti seguiti da minuscola {edge}")

    # ── 6. Throughput ───────────────────────────────────────────────────────
    audio = [r["audio_s"] for r in rows if "audio_s" in r]
    chars = [r["chars"] for r in rows if "chars" in r]
    if audio and chars:
        speech = sum(audio)
        print("│")
        print("├─ RITMO")
        print(f"│   {sum(chars)/max(speech,1)*60/5.5:.0f} parole/minuto dettate"
              f"   ({sum(chars)} caratteri in {speech/60:.0f} minuti di parlato)")
        capped = len([a for a in audio if a >= 14.5])
        if capped:
            print(f"│   {capped}/{len(audio)} frasi tagliate a MAX_DUR — periodi lunghi,"
                  f" ogni taglio è una giuntura in più")
    print("╰─")
    return 0


if __name__ == "__main__":
    sys.exit(main())
