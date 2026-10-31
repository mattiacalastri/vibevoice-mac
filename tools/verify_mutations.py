#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Do the tests actually catch the bugs they were written for?

A green suite says the code passes its tests. It does not say the tests would
notice if the code stopped working. This removes each cure from a throwaway copy
of the repo and runs the test that claims to guard it: the test must FAIL. One
that passes anyway is not a guard, it is a reassurance — and it earned its place
here by being found in the wild on 2026-08-02, hiding among 190 green tests.

Each entry is a scar with a name: the mutation reverts a specific fix, and the
named test is the one that should scream. Add an entry whenever you fix a defect
that could plausibly come back.

Read-only against the repo — every mutation happens in a copy under the system
temp dir.

Usage:
    python3 tools/verify_mutations.py            # all of them
    python3 tools/verify_mutations.py apostrofo  # only entries matching a word
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (label, file, find, replace, test that must catch it)
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    ("warm-up cadence: go back to one flat interval",
     "engine.py",
     "    return PARTIAL_INTERVAL if has_delivered else min(PARTIAL_INTERVAL, PARTIAL_WARMUP)",
     "    return PARTIAL_INTERVAL",
     "test_the_first_character_arrives_a_third_of_a_second_sooner"),

    ("warm-up: end it on confirmed instead of delivered",
     "engine.py",
     "                if len(draft.split()) >= 2:\n                    self._has_delivered = True",
     "                if draft:\n                    self._has_delivered = True",
     "test_the_engine_leaves_the_warmup_cadence_once_a_word_reaches_the_app"),

    ("stabilizer: go back to trusting positions",
     "engine.py",
     "        here = _locate(self._committed, words)\n        there = _locate(self._committed, prev)",
     "        here = 0\n        there = 0",
     "test_local_agreement_survives_a_word_appearing_at_the_HEAD"),

    ("typed record: go back to a single slot",
     "engine.py",
     "return self._streamed_by_gen.pop(closing_gen, dict(_EMPTY_UTT))",
     'return dict(_EMPTY_UTT, typed=(self._streamed if self._streamed_gen == closing_gen else ""))',
     "test_the_next_utterance_cannot_erase_the_previous_one_s_typed_record"),

    ("loop guard: removed",
     "engine.py",
     "if text and (_is_loop(text) or _is_phantom(audio, text)):",
     "if text and _is_phantom(audio, text):",
     "test_whisper_repetition_loops_never_reach_the_app"),

    ("anchor: go back to the last occurrence",
     "engine.py",
     "        best = min(matches, key=lambda start: (abs(start - expected), start))",
     "        best = matches[-1]",
     "test_unstreamed_tail_picks_the_junction_where_the_stream_actually_stopped"),

    ("mute: stop invalidating the open utterance",
     "engine.py",
     '                        self._utt_gen += 1\n                    write_state("idle")\n                    clear_partial()',
     '                    write_state("idle")\n                    clear_partial()',
     "test_muting_stops_a_partial_from_typing_afterwards"),

    ("apostrophe: go back to distinguishing the two glyphs",
     "engine.py",
     "word.translate(_APOSTROPHES)",
     "word",
     "test_agreement_key_treats_both_apostrophes_as_the_same_word"),

    ("one word of lag: removed",
     "engine.py",
     "typeable = draft.split()[:-1]",
     "typeable = draft.split()",
     "test_stream_paste_holds_back_the_word_on_the_truncation_edge"),

    ("Return daemon: not freed on utterances that are too short",
     "engine.py",
     "            release_autosend()\n            return\n\n        # Up to two transcriptions",
     "            return\n\n        # Up to two transcriptions",
     "test_an_utterance_too_short_to_transcribe_still_frees_the_return_daemon"),
]


def main() -> int:
    wanted = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    entries = [m for m in MUTATIONS if wanted in m[0].lower()]
    if not entries:
        print(f"No mutation matches {wanted!r}")
        return 2

    print(f"Mutation check — {len(entries)} cure(s) removed one at a time\n")
    caught = missing = 0
    for label, filename, find, replace, test in entries:
        work = Path(tempfile.mkdtemp(prefix="vv_mut_"))
        dest = work / "repo"
        shutil.copytree(REPO, dest, ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".ruff_cache", "dist", "build"))
        target = dest / filename
        source = target.read_text()
        if find not in source:
            # The anchor moved: the entry is stale, which is itself a finding —
            # it means this scar is no longer being checked at all.
            print(f"  ⚠  {label}\n     anchor not found — entry is stale, nothing was verified")
            missing += 1
            shutil.rmtree(work, ignore_errors=True)
            continue
        target.write_text(source.replace(find, replace, 1))

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_contract.py",
             "-q", "-p", "no:cacheprovider", "-k", test, "--no-header", "-x"],
            cwd=dest, capture_output=True, text=True, timeout=900,
        )
        failed = result.returncode != 0
        caught += failed
        print(f"  {'✓' if failed else '✗ THEATRE'}  {label}")
        print(f"     {test} → {'fails, as it must' if failed else 'PASSES ANYWAY'}")
        shutil.rmtree(work, ignore_errors=True)

    print(f"\n{caught}/{len(entries)} cures are really guarded"
          f"{f' ({missing} stale entries)' if missing else ''}.")
    return 0 if caught == len(entries) and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
