#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
vibevoice_correct.py — corrections loop v0: never make the same mistake twice.

Pairs the LAST dictation (history.jsonl, the text as pasted) with the version
the user actually wanted, then:
  1. records the pair in ~/.vibevoice/corrections.jsonl — the engine feeds the
     most recent pairs to the cleanup LLM as few-shot examples;
  2. adds the corrected words the engine missed to ~/.vibevoice/dictionary.txt —
     which biases Whisper's initial_prompt on every following dictation.

Usage:
  python3 tools/vibevoice_correct.py "il testo come doveva essere"
  python3 tools/vibevoice_correct.py --clipboard    # corrected text from pbpaste
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

STATE_DIR = Path.home() / ".vibevoice"
HISTORY_FILE = STATE_DIR / "history.jsonl"
CORRECTIONS_FILE = STATE_DIR / "corrections.jsonl"
DICT_FILE = STATE_DIR / "dictionary.txt"
CORRECTIONS_MAX = 200

_WORD = re.compile(r"[\w'À-ÿ]+")


def _tokens(text: str) -> list:
    return _WORD.findall(text)


def extract_new_terms(raw: str, corrected: str) -> list:
    """Words present in the corrected text but absent from the raw dictation.

    These are the spellings the recognizer missed — exactly what belongs in the
    dictionary. Case-only and punctuation-only changes are not vocabulary.
    """
    raw_lower = {t.lower() for t in _tokens(raw)}
    terms, seen = [], set()
    for tok in _tokens(corrected):
        low = tok.lower()
        if len(tok) < 3 or low in raw_lower or low in seen:
            continue
        if not any(c.isalpha() for c in tok):
            continue
        seen.add(low)
        terms.append(tok)
    return terms


def record_correction(
    corrected: str,
    history_file: Path = HISTORY_FILE,
    corrections_file: Path = CORRECTIONS_FILE,
    dict_file: Path = DICT_FILE,
) -> dict:
    """Pair `corrected` with the last dictation and persist the lesson."""
    try:
        last = json.loads(history_file.read_text().splitlines()[-1])
        raw = last["text"]
    except Exception:
        sys.exit(f"no dictation to correct: cannot read {history_file}")

    lines = []
    try:
        lines = corrections_file.read_text().splitlines()
    except OSError:
        pass
    lines.append(json.dumps({"ts": time.time(), "raw": raw, "corrected": corrected}))
    corrections_file.write_text("\n".join(lines[-CORRECTIONS_MAX:]) + "\n")

    new_terms = extract_new_terms(raw, corrected)
    if new_terms:
        try:
            existing = dict_file.read_text().splitlines()
        except OSError:
            existing = []
        known = {t.strip().lower() for t in existing if t.strip()}
        added = [t for t in new_terms if t.lower() not in known]
        if added:
            body = "\n".join(existing + added).strip("\n")
            dict_file.write_text(body + "\n")
        new_terms = added

    return {"raw": raw, "corrected": corrected, "new_terms": new_terms}


def main() -> int:
    args = sys.argv[1:]
    if args == ["--clipboard"]:
        corrected = subprocess.run(
            ["pbpaste"], capture_output=True, timeout=3
        ).stdout.decode("utf-8").strip()
    else:
        corrected = " ".join(args).strip()
    if not corrected:
        sys.exit((__doc__ or "").strip())

    result = record_correction(corrected)
    print(f'raw:       "{result["raw"]}"')
    print(f'corrected: "{result["corrected"]}"')
    if result["new_terms"]:
        print(f"dictionary += {', '.join(result['new_terms'])}")
    else:
        print("dictionary unchanged (no new terms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
