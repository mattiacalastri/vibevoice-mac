#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Learn the recognizer's recurring mishearings — and correct them from then on.

The companion of vibevoice_learn.py, which grows the glossary. This one grows the
CORRECTIONS: «crowdcode» heard where you said «Claude Code».

THE SAME SAFETY RULE, turned around: the recognizer's output tells us what it
HEARD, never how a word is spelled. So the right spelling must come from a list
you curated — the glossary (~/.vibevoice/dictionary.txt) and the house voice
dictionary (`termini`, `nomi`). A rule is learned only when, in your recent
dictations, the SAME odd form shows up at least MIN_SEEN times and:

  * at least one of its words is unknown to the Italian spell checker (an
    ordinary word like «online» is never rewritten into «Moonline») — or it is
    a phrase of real words almost identical to a term («cloud code»);
  * it sounds like a glossary term (same first sound, phonetic similarity
    ≥ MIN_SIMILARITY on a crude Italian-ear normalisation);
  * no existing correction already fixes it.

Learned rules go to ~/.vibevoice/learned_fixes.json (the engine reads it next
to fixes.json), each with its evidence, and one line per run to learn.log.
Delete a rule from the file and it is gone; add its form to "never" and it is
never proposed again.

Usage:
    python3 tools/vibevoice_learn_fixes.py            # propose, change nothing
    python3 tools/vibevoice_learn_fixes.py --apply    # write the new rules
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

STATE = Path.home() / ".vibevoice"
CORPUS = STATE / "corpus.jsonl"
DICT = STATE / "dictionary.txt"
FIXES = STATE / "fixes.json"
LEARNED = STATE / "learned_fixes.json"
LOG = STATE / "learn.log"
HOUSE = FIXES  # the house voice dictionary: `termini` and `nomi` beside `correzioni`

MIN_SEEN = 2
MIN_SIMILARITY = 0.82
STRONG = 0.9            # a phrase of real words this close is still a mishearing
WINDOW_DAYS = 60
WORD = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9'.]*")


def sound(s: str) -> str:
    """How an Italian ear files a word: no spaces or dashes, k→c, y→i, w→v,
    ph→f, silent h dropped, doubled letters single."""
    s = s.lower().replace(" ", "").replace("-", "").replace(".", "")
    for a, b in (("ph", "f"), ("ow", "o"), ("k", "c"), ("y", "i"), ("w", "v"), ("h", ""), ("q", "c"),
                 ("ou", "o"), ("au", "o")):
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s)


def rule_for(form: str) -> str:
    """Case-insensitive regex for the misheard form, tolerant to spacing."""
    return r"\b" + r"\s*".join(re.escape(w) for w in form.split()) + r"\b"


def candidates(texts, glossary, is_unknown, already_fixed, never=()):
    """[(form, term, seen, example)] — the pure heart, testable without AppKit."""
    terms = [(t, sound(t)) for t in glossary if len(sound(t)) >= 4]
    gl = {t.casefold() for t in glossary}
    # Every word you put in the glossary is a word you SAY, spelled right:
    # «brain» (Second Brain) is never a botched «branch», and «roadmap di» is
    # the right word followed by a preposition, not a new mishearing.
    gl_words = {w.casefold() for t in glossary for w in WORD.findall(t)}
    seen, example = Counter(), {}
    for text in texts:
        words = WORD.findall(text)
        for n in (1, 2):
            for i in range(len(words) - n + 1):
                form = " ".join(words[i:i + n]).strip(".").casefold()
                if form and form not in gl:
                    seen[form] += 1
                    example.setdefault(form, text[:100])
    out = []
    for form, count in seen.items():
        if count < MIN_SEEN or form in never or already_fixed(form):
            continue
        nwords = len(form.split())
        if nwords == 1 and form in gl_words:
            continue
        fs = sound(form)
        best, score = None, 0.0
        for term, ts in terms:
            if fs[:1] != ts[:1] or abs(len(fs) - len(ts)) > 3 or fs == ts:
                continue
            # «roadmap di» is roadmap plus a preposition: two words only ever
            # stand for a two-word term; one glued word may stand for either.
            if nwords > 1 and nwords != len(term.split()):
                continue
            r = SequenceMatcher(None, fs, ts).ratio()
            if r > score:
                best, score = term, r
        if not best or score < MIN_SIMILARITY:
            continue
        # A known word is rewritten only inside a near-identical phrase:
        # «cloud code» (0.94 to Claude Code) yes, «online» alone never.
        if any(is_unknown(w) for w in form.split()) or (nwords > 1 and score >= STRONG):
            out.append((form, best, count, example[form]))
    return sorted(out, key=lambda c: -c[2])


# ── the real world ────────────────────────────────────────────────────────────

def _glossary() -> list:
    terms = []
    try:
        for line in DICT.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    except OSError:
        pass
    try:
        d = json.loads(HOUSE.read_text())
        terms += d.get("termini", []) + d.get("nomi", [])
    except Exception:
        pass
    return sorted(set(terms))


def _recent_texts() -> list:
    cutoff = time.time() - WINDOW_DAYS * 86400
    out = []
    try:
        for line in CORPUS.read_text().splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            if float(row.get("ts", 0)) >= cutoff and row.get("text"):
                out.append(row["text"])
    except OSError:
        pass
    return out


def _spell_unknown():
    from AppKit import NSSpellChecker
    sc = NSSpellChecker.sharedSpellChecker()
    cache = {}

    def unknown(word: str) -> bool:
        if word not in cache:
            r = sc.checkSpellingOfString_startingAt_language_wrap_inSpellDocumentWithTag_wordCount_(
                word, 0, "it", False, 0, None)
            cache[word] = r[0].length > 0
        return cache[word]
    return unknown


def _existing_rules() -> list:
    rules = []
    for path in (FIXES, LEARNED):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        pairs = d.get("correzioni") if isinstance(d, dict) else d
        for p in pairs or []:
            try:
                rules.append(re.compile(p[0], re.IGNORECASE))
            except Exception:
                pass
    return rules


def _load_learned() -> dict:
    try:
        d = json.loads(LEARNED.read_text())
        if isinstance(d, dict):
            d.setdefault("correzioni", [])
            d.setdefault("prove", {})
            d.setdefault("never", [])
            return d
    except Exception:
        pass
    return {"_perche": "Storpiature imparate da sole da tools/vibevoice_learn_fixes.py. "
                       "Togli una riga per annullarla; metti la forma in «never» per non "
                       "riproporla.", "correzioni": [], "prove": {}, "never": []}


def main() -> int:
    apply = "--apply" in sys.argv
    learned = _load_learned()
    rules = _existing_rules()
    found = candidates(_recent_texts(), _glossary(), _spell_unknown(),
                       lambda form: any(r.search(form) for r in rules),
                       never=set(learned["never"]))
    stamp = time.strftime("%Y-%m-%d %H:%M")
    if not found:
        line = f"{stamp} nessuna storpiatura nuova"
    else:
        line = f"{stamp} " + "; ".join(f"{f!r}→{t} ×{n}" for f, t, n, _ in found)
    print(line)
    for form, term, n, ex in found:
        print(f"   {n:3d}×  {form!r} → {term}   | {ex}")
    if apply and found:
        for form, term, n, ex in found:
            learned["correzioni"].append([rule_for(form), term])
            learned["prove"][form] = {"termine": term, "visto": n, "esempio": ex, "quando": stamp}
        tmp = LEARNED.with_suffix(".tmp")
        tmp.write_text(json.dumps(learned, ensure_ascii=False, indent=1) + "\n")
        tmp.replace(LEARNED)
    try:
        with LOG.open("a") as fh:
            fh.write(("applicate: " if apply else "proposte: ") + line + "\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
