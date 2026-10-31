#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The three JSONL ledgers must survive a kill and two concurrent writers.

history.jsonl, corpus.jsonl and metrics.jsonl are read-modify-written: read all
lines, append one, write the lot back capped. Two holes followed from that, and
the pill walks straight into both — it `pkill`s the engine on every restart and
every engine-visible settings change, and up to two transcriptions are in flight
at once (invariant #4).

  * `write_text` truncates before it writes. A kill inside that window leaves an
    empty or half-written ledger. corpus.jsonl is a learning corpus; nothing can
    reconstruct it.
  * Two workers that both read, then both write, keep only the second's entry.

levels.bin and partial.txt were already atomic. The files that outlive the
process were the ones left exposed.
"""
from __future__ import annotations

import json
import threading

import pytest

import engine


@pytest.fixture
def ledgers(tmp_path, monkeypatch):
    for attr in ("HISTORY_FILE", "CORPUS_FILE", "METRICS_FILE"):
        monkeypatch.setattr(engine, attr, tmp_path / f"{attr.lower()}.jsonl")
    return tmp_path


@pytest.mark.parametrize("append, target", [
    (lambda: engine._append_history("ciao"), "HISTORY_FILE"),
    (lambda: engine._append_corpus("ciao"), "CORPUS_FILE"),
    (lambda: engine._append_metrics({"ok": 1}), "METRICS_FILE"),
])
def test_a_ledger_is_never_truncated_in_place(ledgers, monkeypatch, append, target):
    """The durable file is only ever swapped in whole, via os.replace."""
    written_directly = []
    real_write_text = engine.Path.write_text

    def spy(self, *a, **kw):
        if self.suffix != ".tmp":
            written_directly.append(self)
        return real_write_text(self, *a, **kw)

    monkeypatch.setattr(engine.Path, "write_text", spy)
    replaced = []
    real_replace = engine.os.replace
    monkeypatch.setattr(engine.os, "replace",
                        lambda src, dst: replaced.append(dst) or real_replace(src, dst))

    append()

    assert getattr(engine, target) not in written_directly, (
        f"{target} is truncated in place — a kill mid-write loses it"
    )
    assert getattr(engine, target) in replaced, f"{target} was not swapped in atomically"


# `count` stays under each ledger's own cap (HISTORY_MAX is only 20): this test
# is about entries lost to a race, not about the cap doing its job.
@pytest.mark.parametrize("append, target, count", [
    (lambda i: engine._append_history(f"u{i}"), "HISTORY_FILE", 16),
    (lambda i: engine._append_corpus(f"u{i}"), "CORPUS_FILE", 40),
    (lambda i: engine._append_metrics({"i": i}), "METRICS_FILE", 40),
])
def test_concurrent_writers_do_not_lose_entries(ledgers, append, target, count):
    """Two transcription workers can append at the same time (invariant #4)."""
    def run(lo):
        for i in range(lo, lo + count // 2):
            append(i)

    threads = [threading.Thread(target=run, args=(0,)),
               threading.Thread(target=run, args=(count // 2,))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = [ln for ln in getattr(engine, target).read_text().splitlines() if ln.strip()]
    assert len(lines) == count, f"{count - len(lines)} entries lost to a concurrent write"
    for ln in lines:
        json.loads(ln)      # every surviving line is whole


def test_no_tmp_file_is_left_behind(ledgers):
    engine._append_history("ciao")
    engine._append_corpus("ciao")
    engine._append_metrics({"ok": 1})
    assert not list(ledgers.glob("*.tmp")), "os.replace should consume the temp file"
