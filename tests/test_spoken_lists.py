#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""A list dictated out loud lands as a list.

«Ti voglio dire tre cose. La prima è… la seconda è… e la terza è…» used to
arrive as one paragraph. Asked for on 2026-10-31 («come fa Wispr Flow»): when
the ordinals are clearly there, one item per line, numbered. When they are not,
the text must come through untouched — a wrong list is worse than no list.
"""
from __future__ import annotations

import pytest

import engine


def test_the_dictation_that_asked_for_it():
    said = ("Ti voglio dire tre cose. La prima è che il sistema sembra che funzioni, "
            "la seconda è che il sistema sembra che funziona benissimo e la terza è "
            "il guantone da boxe tira un pugno e spacca tutto.")
    assert engine.format_spoken_list(said) == (
        "Ti voglio dire tre cose:\n"
        "1. Il sistema sembra che funzioni.\n"
        "2. Il sistema sembra che funziona benissimo.\n"
        "3. Il guantone da boxe tira un pugno e spacca tutto."
    )


@pytest.mark.parametrize("said, expected", [
    ("Primo, controlla la campagna. Secondo, scrivi al cliente. Terzo, aggiorna il report.",
     "1. Controlla la campagna.\n2. Scrivi al cliente.\n3. Aggiorna il report."),
    ("Due cose: punto uno, il budget. Punto due, le creatività.",
     "Due cose:\n1. Il budget.\n2. Le creatività."),
    ("In primo luogo serve il pixel, in secondo luogo serve il catalogo.",
     "1. Serve il pixel.\n2. Serve il catalogo."),
    ("Ecco il piano: 1. apri Meta Ads 2. duplica la campagna 3. cambia il pubblico",
     "Ecco il piano:\n1. Apri Meta Ads.\n2. Duplica la campagna.\n3. Cambia il pubblico."),
    ("Three things. First, ship it. Second, measure it. Third, fix it.",
     "Three things:\n1. Ship it.\n2. Measure it.\n3. Fix it."),
])
def test_spoken_lists_become_numbered_lines(said, expected):
    assert engine.format_spoken_list(said) == expected


@pytest.mark.parametrize("said", [
    # an ordinal that is not an item marker
    "La prima volta che l'ho visto era a Verona, la seconda volta a Milano.",
    # a single marker is not a list
    "La prima è che non ho tempo oggi.",
    # out of order: not a list someone is dictating
    "La seconda è che manca il budget, la prima è che manca il tempo.",
    # ordinary prose with numbers
    "Ho speso 1.200 euro in 3 giorni e ne restano 2.",
    "Dammi un secondo, controllo e ti dico.",
    "",
])
def test_everything_else_comes_through_untouched(said):
    assert engine.format_spoken_list(said) == said


def test_push_to_talk_pastes_the_formatted_list(monkeypatch, tmp_path):
    """Wired into the final text of a push-to-talk dictation (raw.txt is what
    gets pasted)."""
    import numpy as np

    monkeypatch.setattr(engine, "AUTOSEND", False)
    monkeypatch.setattr(engine, "PTT", True)
    for name in ("RAW_FILE", "HISTORY_FILE", "METRICS_FILE", "CORPUS_FILE", "DICT_FILE",
                 "CORRECTIONS_FILE", "FIXES_FILE", "LEARNED_FIXES_FILE", "STATE_FILE"):
        monkeypatch.setattr(engine, name, tmp_path / name.lower())
    said = "Primo, apri il CRM. Secondo, chiama il lead."
    monkeypatch.setattr(engine, "transcribe", lambda audio: said)
    loud = np.full(engine.SAMPLE_RATE * 2, 0.3, dtype=np.float32)
    out = engine.process_utterance(loud, t_end=None)
    assert out == "1. Apri il CRM.\n2. Chiama il lead."
    assert engine.RAW_FILE.read_text() == out


# How Mattia really dictates a list (2026-10-31): «la prima», then «poi…» with a
# false start corrected by «come secondo punto», then «e il terzo punto».
REAL_LIST = (
    "Allora, ti voglio dire tre cose. La prima, dobbiamo fare un follow-up a Luca "
    "Bianchi, come lui stesso ci ha chiesto. Poi, dobbiamo aggiungere delle creatività "
    "statiche, come secondo punto, dobbiamo aggiungere delle creatività statiche per "
    "Moonline, nella campagna di Moonline. E il terzo punto, dobbiamo finalmente realizzare "
    "il clone digitale, ok? Il tutto con l'obiettivo di costruire il nostro nuovo prodotto."
)


def test_the_way_mattia_really_dictates_a_list():
    assert engine.format_spoken_list(REAL_LIST) == (
        "Allora, ti voglio dire tre cose:\n"
        "1. Dobbiamo fare un follow-up a Luca Bianchi, come lui stesso ci ha chiesto.\n"
        "2. Dobbiamo aggiungere delle creatività statiche per Moonline, nella campagna di Moonline.\n"
        "3. Dobbiamo finalmente realizzare il clone digitale, ok? Il tutto con l'obiettivo di "
        "costruire il nostro nuovo prodotto."
    )


def test_poi_continues_a_list_already_opened():
    said = "Due cose. La prima è il budget. Poi, le creatività nuove."
    assert engine.format_spoken_list(said) == "Due cose:\n1. Il budget.\n2. Le creatività nuove."


@pytest.mark.parametrize("said", [
    # «poi» alone never opens a list
    "Poi, vediamo domani. Poi, ne parliamo con calma.",
    # a restated ordinal that is not a false start keeps all the words
    "Come prima cosa, guardiamo i numeri.",
])
def test_poi_and_come_never_invent_a_list(said):
    assert engine.format_spoken_list(said) == said


def test_a_restated_ordinal_without_a_false_start_loses_only_the_marker():
    said = ("La prima, chiama il cliente. Poi, prepara il report, come secondo punto, "
            "e mandalo entro sera.")
    assert engine.format_spoken_list(said) == (
        "1. Chiama il cliente.\n2. Prepara il report, e mandalo entro sera."
    )
