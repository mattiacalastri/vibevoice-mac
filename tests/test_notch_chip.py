#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pure logic of the notch chip: the Control gate, the chip geometry,
the bars and the outcome cue after release. No window, no keyboard."""
from __future__ import annotations

import notch_chip as nc


CTRL, OPT, SHIFT, CMD = nc.FLAG_CONTROL, nc.FLAG_OPTION, nc.FLAG_SHIFT, nc.FLAG_COMMAND


def test_the_chord_is_control_alone():
    assert nc.is_chord(CTRL)
    assert nc.is_chord(CTRL | (1 << 8))               # device-dependent bits ignored
    assert not nc.is_chord(0)
    assert not nc.is_chord(CTRL | OPT)                # Notch Agenti's «parla col Polpo»
    assert not nc.is_chord(CTRL | CMD)
    assert not nc.is_chord(CTRL | SHIFT)


def test_a_shortcut_typed_with_control_never_dictates():
    g = nc.PttGate(hold=0.25)
    assert g.update(True, 0.00) is None
    g.cancel()                                        # Ctrl+C: a key went down
    assert g.update(True, 0.40) is None               # held long, still blocked
    assert g.update(False, 0.50) is None              # silent release
    assert g.update(True, 0.60) is None               # a fresh hold works again
    assert g.update(True, 0.90) == "press"


def test_a_key_after_the_press_does_not_cancel_the_dictation():
    g = nc.PttGate(hold=0.25)
    g.update(True, 0.0)
    assert g.update(True, 0.3) == "press"
    g.cancel()
    assert g.update(False, 2.0) == "release"


def test_a_quick_tap_never_starts_a_dictation():
    g = nc.PttGate(hold=0.15)
    assert g.update(True, 0.00) is None
    assert g.update(True, 0.10) is None
    assert g.update(False, 0.12) is None              # let go before the hold


def test_hold_then_release_gives_one_press_and_one_release():
    g = nc.PttGate(hold=0.15)
    events = [g.update(True, t / 100) for t in range(0, 40, 3)]
    assert events.count("press") == 1
    assert g.update(False, 0.5) == "release"
    assert g.update(False, 0.6) is None


def test_the_hold_restarts_after_a_release():
    g = nc.PttGate(hold=0.15)
    g.update(True, 0.0)
    g.update(True, 0.2)
    g.update(False, 0.3)
    assert g.update(True, 0.31) is None
    assert g.update(True, 0.40) is None
    assert g.update(True, 0.47) == "press"


def test_chip_sits_on_the_notch_and_grows_sideways():
    x, y, w, h = nc.chip_rect(0, 0, 1512, 982, notch_w=200, notch_h=32, wing=70)
    assert (w, h) == (340, 32)
    assert x + w / 2 == 756                           # centred on the notch
    assert y + h == 982                               # flush with the top edge
    cx, cy, cw, ch = nc.chip_rect(0, 0, 1512, 982, notch_w=200, notch_h=32, wing=0)
    assert (cw, cx) == (200, 656)


def test_bars_follow_the_latest_levels():
    levels = [0.0] * 55 + [0.0, 0.01, 0.02, 0.04, 0.2]
    bars = nc.chip_bars(levels, n=5)
    assert len(bars) == 5
    assert all(0.0 <= b <= 1.0 for b in bars)
    assert bars[-1] >= bars[0]
    assert nc.chip_bars(None, n=5) == [0.0] * 5


def test_outcome_after_release():
    # still working
    assert nc.release_outcome(10.0, 10.0, "transcribing", 0.8) is None
    assert nc.release_outcome(10.0, 10.0, "recording", 0.2) is None
    # a new sentence reached raw.txt → it went to the terminal
    assert nc.release_outcome(10.0, 12.5, "idle", 0.9) == "sent"
    # idle again with nothing new → nothing heard
    assert nc.release_outcome(10.0, 10.0, "idle", 1.5) == "empty"
    # an engine that never answers does not leave the chip up forever
    assert nc.release_outcome(10.0, 10.0, "transcribing", nc.RESULT_TIMEOUT + 1) == "empty"


def test_engine_log_is_rotated_past_the_cap(tmp_path):
    import vibevoice
    log = tmp_path / "engine.err"
    log.write_bytes(b"x" * 50)
    vibevoice.rotate_log(log, max_bytes=100)
    assert log.exists()
    log.write_bytes(b"x" * 150)
    vibevoice.rotate_log(log, max_bytes=100)
    assert not log.exists() and (tmp_path / "engine.err.1").stat().st_size == 150


def test_a_missed_paste_is_offered_back(tmp_path):
    import json
    rep = tmp_path / "paste.json"
    assert nc.read_missed(rep, since=0) is None                       # no report yet
    rep.write_text(json.dumps({"ts": 100.0, "ok": True, "reason": "target", "text": "ciao"}))
    assert nc.read_missed(rep, since=50) is None                      # it landed
    rep.write_text(json.dumps({"ts": 100.0, "ok": False, "reason": "nofocus", "text": "ciao"}))
    assert nc.read_missed(rep, since=150) is None                     # an older paste
    assert nc.read_missed(rep, since=50) == ("ciao", "nessun campo di testo in primo piano")
    rep.write_text("{broken")
    assert nc.read_missed(rep, since=0) is None


def test_reason_labels_are_plain_words():
    assert nc.reason_label("secure") == "un campo protetto blocca la tastiera"
    assert nc.reason_label("role:AXList") == "nessun campo di testo in primo piano"
    assert nc.reason_label("boh") == "non sono sicuro che sia arrivato"


def test_preview_keeps_the_end_of_a_long_sentence():
    assert nc.preview("ciao", 10) == "ciao"
    out = nc.preview("una frase molto lunga che non entra nella chip", 20)
    assert out.startswith("…") and len(out) <= 21 and out.endswith("chip")


def test_the_pill_asks_pgrep_at_most_once_a_second(monkeypatch):
    import vibevoice
    calls = []
    monkeypatch.setattr(vibevoice, "_engine_running", lambda: calls.append(1) or True)
    monkeypatch.setattr(vibevoice, "_ENGINE_SEEN", {"t": -1e9, "on": False})
    for i in range(60):                               # one second of frames
        assert vibevoice._engine_running_cached(now=100.0 + i / 60) is True
    assert len(calls) == 1
    vibevoice._engine_running_cached(now=101.5)
    assert len(calls) == 2


def test_edge_glow_fades_in_and_out():
    import edge_glow as eg
    assert eg.glow_fade(None, None, 5.0) == 0.0
    assert eg.glow_fade(0.0, None, eg.FADE_IN / 2) == 0.5
    assert eg.glow_fade(0.0, None, 3.0) == 1.0
    assert abs(eg.glow_fade(0.0, 3.0, 3.0 + eg.FADE_OUT / 2) - 0.5) < 1e-9
    assert eg.glow_fade(0.0, 3.0, 9.0) == 0.0


def test_edge_glow_style_is_the_one_mattia_chose():
    import edge_glow as eg
    assert eg.RINGS == [(50.0, 0.45, 22.0), (20.0, 0.7, 8.0), (6.0, 0.95, 2.0), (2.0, 1.0, 0.0)]
    assert eg.voice_level(None) == 0.0 and 0.0 < eg.voice_level([0.0] * 59 + [0.05]) <= 1.0


def test_reports_with_accents_are_read_under_an_ascii_locale(tmp_path, monkeypatch):
    """The app runs with no UTF-8 locale: a report holding «è» was unreadable,
    so every verdict waited 1.5 s and a miss could never be shown (30/10)."""
    import json
    import locale
    monkeypatch.setattr(locale, "getpreferredencoding", lambda *a, **k: "ascii")
    rep = tmp_path / "paste.json"
    rep.write_text(json.dumps({"ts": 10.0, "ok": False, "reason": "nofocus",
                               "text": "non è stata incollata"}, ensure_ascii=False),
                   encoding="utf-8")
    assert nc.read_missed(rep, since=0)[0] == "non è stata incollata"


def test_the_glow_runs_out_of_the_notch_and_back():
    import edge_glow as eg
    assert eg.glow_reveal(None, None, 1.0) == 0.0
    assert eg.glow_reveal(0.0, None, 0.0) == 0.0
    half = eg.glow_reveal(0.0, None, eg.REVEAL_IN / 2)
    assert abs(half - 0.5) < 1e-9                         # constant speed: no slow join
    assert eg.glow_reveal(0.0, None, 1.0) == 1.0
    assert eg.glow_reveal(0.0, 2.0, 2.0) == 1.0
    assert eg.glow_reveal(0.0, 2.0, 2.0 + eg.REVEAL_S) < 1e-9
