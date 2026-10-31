#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Tests for tools/vibevoice_metrics.py — the p50/p90/p99 latency report."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "vibevoice_metrics.py"
spec = importlib.util.spec_from_file_location("vibevoice_metrics", _TOOL)
metrics_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_tool)


def test_percentile_interpolates_and_clamps():
    values = [float(i) for i in range(1, 101)]  # 1..100
    assert metrics_tool.percentile(values, 50) == pytest.approx(50.5)
    assert metrics_tool.percentile(values, 99) == pytest.approx(99.01, abs=0.5)
    assert metrics_tool.percentile([42.0], 99) == 42.0


def test_percentile_empty_is_none():
    assert metrics_tool.percentile([], 50) is None


def test_report_reads_jsonl_and_summarizes(tmp_path):
    f = tmp_path / "metrics.jsonl"
    f.write_text(
        "\n".join(
            json.dumps({"stt_ms": 100.0 + i, "total_ms": 400.0 + i, "audio_s": 2.0})
            for i in range(10)
        )
        + "\n"
    )
    report = metrics_tool.report(f)
    assert report["count"] == 10
    assert report["stt_ms"]["p50"] == pytest.approx(104.5)
    assert report["total_ms"]["p99"] <= 409.0 + 1
