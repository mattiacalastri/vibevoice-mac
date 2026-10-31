#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
vibevoice_metrics.py — p50/p90/p99 report over ~/.vibevoice/metrics.jsonl.

The engine appends one JSONL line per utterance (see engine.process_utterance).
This reads them back and prints the latency percentiles that matter: the
product target is the p99 of total_ms (end-of-speech → text ready), not the
mean — a dictation tool is judged on its worst common case.

Usage:  python3 tools/vibevoice_metrics.py [path-to-metrics.jsonl]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_FILE = Path.home() / ".vibevoice" / "metrics.jsonl"
FIELDS = ("stt_ms", "cleanup_ms", "total_ms", "audio_s")
PERCENTILES = (50, 90, 99)


def percentile(values: list, pct: float):
    """Linear-interpolated percentile; None on empty input."""
    if not values:
        return None
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    rank = (pct / 100.0) * (len(data) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(data) - 1)
    frac = rank - lo
    return data[lo] + (data[hi] - data[lo]) * frac


def report(path: Path) -> dict:
    """Summarize a metrics.jsonl into {count, <field>: {p50, p90, p99}}."""
    entries = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    out: dict = {"count": len(entries)}
    for field in FIELDS:
        values = [e[field] for e in entries if isinstance(e.get(field), (int, float))]
        if values:
            out[field] = {f"p{p}": round(percentile(values, p) or 0.0, 1) for p in PERCENTILES}
    return out


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE
    summary = report(path)
    if not summary["count"]:
        print(f"no metrics in {path}")
        return 1
    print(f"utterances: {summary['count']}  ({path})")
    for field in FIELDS:
        if field in summary:
            p = summary[field]
            print(f"  {field:<10} p50={p['p50']:>8}  p90={p['p90']:>8}  p99={p['p99']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
