#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Grow the personal dictionary from what you actually dictate — safely.

The dictionary biases Whisper through `initial_prompt`, and it was hand-written:
68 terms, last touched in July. The correction tool that was supposed to grow it
has never been run, because it asks you to type a command after every mistake.
Nobody does that. So the loop was dead.

THE RULE THAT MAKES THIS SAFE: a term is only added when it appears in something
**you wrote** — your vault, your repos, your notes. Whisper's own output is
evidence that a word was SPOKEN, never evidence of how it is SPELLED. Learning
spelling from the transcriber teaches it its own mistakes, and the mistake then
becomes self-reinforcing through the prompt.

So the corpus (`~/.vibevoice/corpus.jsonl`) supplies candidates, and your own
writing supplies the truth. A candidate nobody corroborates is reported, never
added.

Usage:
    python3 tools/vibevoice_learn.py                 # propose, change nothing
    python3 tools/vibevoice_learn.py --apply         # append the corroborated ones
    python3 tools/vibevoice_learn.py --corpus DIR    # extra corpus of your writing
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

STATE = Path.home() / ".vibevoice"
CORPUS = STATE / "corpus.jsonl"
HISTORY = STATE / "history.jsonl"
DICT = STATE / "dictionary.txt"

# Where the user's own writing lives — the only admissible source of spelling.
WRITING = [
    Path.home() / "Obsidian",
    Path.home() / "Desktop",
    Path.home() / "projects" / "brand-kits",
]
WRITING_SUFFIXES = {".md", ".txt", ".json", ".py", ".ts", ".tsx"}
WRITING_MAX_FILES = 4000

MIN_OCCURRENCES = 3     # below this it is noise, not vocabulary
MIN_LENGTH = 4          # short tokens are function words or initials
WORD = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'’-]{2,}")


SENTENCE_SPLIT = re.compile(r"(?<=[.!?…:;])\s+|\n+")


def _proper_nouns(text: str) -> list[str]:
    """Words capitalised MID-sentence — the only cheap proper-noun signal.

    "Capitalised" alone is not one: the first word of every sentence is, so the
    first run of this tool proposed adding "Grazie" to the dictionary. It was
    even corroborated, being an ordinary Italian word the user has obviously
    written — and it came from the phantom hallucination. Skipping the opening
    token of each sentence removes that whole class.
    """
    found = []
    for sentence in SENTENCE_SPLIT.split(text):
        words = WORD.findall(sentence)
        for word in words[1:]:          # never the sentence opener
            if len(word) >= MIN_LENGTH and word[0].isupper():
                found.append(word)
    return found


def _spoken_terms() -> Counter:
    """Candidate terms from what was dictated. Evidence of speech, not spelling."""
    counts: Counter = Counter()
    for path in (CORPUS, HISTORY):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                text = json.loads(line)["text"]
            except Exception:
                continue
            for word in _proper_nouns(text):
                counts[word] += 1
        if path is CORPUS and counts:
            break   # prefer the corpus; history is only a fallback
    return counts


def _known_terms() -> set[str]:
    try:
        lines = DICT.read_text().splitlines()
    except OSError:
        return set()
    known = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            known.update(w.casefold() for w in line.split())
    return known


def _written_by_the_user(roots: list[Path]) -> set[str]:
    """Every word the user has written, case-folded. The spelling authority."""
    seen: set[str] = set()
    files = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if files >= WRITING_MAX_FILES:
                return seen
            if not path.is_file() or path.suffix.lower() not in WRITING_SUFFIXES:
                continue
            if any(part.startswith(".") or part in {"node_modules", "venv", "__pycache__"}
                   for part in path.parts):
                continue
            try:
                text = path.read_text(errors="ignore")[:200_000]
            except OSError:
                continue
            files += 1
            seen.update(w.casefold() for w in WORD.findall(text))
    return seen


def main() -> int:
    apply = "--apply" in sys.argv
    roots = list(WRITING)
    if "--corpus" in sys.argv:
        roots.append(Path(sys.argv[sys.argv.index("--corpus") + 1]))

    spoken = _spoken_terms()
    if not spoken:
        # Say WHICH of the two it is: an empty corpus and a corpus with nothing
        # to learn are different situations, and only one of them is a problem.
        if not CORPUS.exists():
            print(f"Il corpus non esiste ancora ({CORPUS}).")
            print("Si riempie da solo mentre detti — riprova fra qualche giorno.")
            return 1
        print("Corpus letto: nessun nome proprio nuovo da imparare.")
        return 0
    known = _known_terms()
    candidates = {w: n for w, n in spoken.items()
                  if n >= MIN_OCCURRENCES and w.casefold() not in known}
    if not candidates:
        print(f"{sum(spoken.values())} occorrenze esaminate — il dizionario le copre già.")
        return 0

    print("Cerco conferma nei tuoi scritti…")
    written = _written_by_the_user(roots)

    corroborated = {w: n for w, n in candidates.items() if w.casefold() in written}
    unconfirmed = {w: n for w, n in candidates.items() if w.casefold() not in written}

    print(f"\n▸ CONFERMATI dai tuoi scritti — sicuri da aggiungere ({len(corroborated)})")
    for word, n in sorted(corroborated.items(), key=lambda kv: -kv[1]):
        print(f"   {n:3d}×  {word}")

    print(f"\n▸ NON confermati — dettati ma mai scritti da te ({len(unconfirmed)})")
    print("   (possono essere errori di Whisper: aggiungerli insegnerebbe lo sbaglio)")
    for word, n in sorted(unconfirmed.items(), key=lambda kv: -kv[1])[:20]:
        print(f"   {n:3d}×  {word}")

    if not corroborated:
        print("\nNiente da aggiungere.")
        return 0
    if not apply:
        print(f"\n{len(corroborated)} termini pronti. Rilancia con --apply per scriverli.")
        return 0

    import time
    block = [f"\n# — appreso dal parlato, confermato dagli scritti "
             f"({time.strftime('%Y-%m-%d')}) —"]
    block += sorted(corroborated, key=lambda w: -corroborated[w])
    with DICT.open("a") as fh:
        fh.write("\n".join(block) + "\n")
    print(f"\nAggiunti {len(corroborated)} termini a {DICT}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
