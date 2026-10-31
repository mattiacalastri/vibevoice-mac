#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Synthesize the push-to-talk cues into assets/sounds/ (our own, MIT).

    python3 tools/make_sounds.py

Four cues, one family: soft, muffled thuds — low-passed noise plus a short
sagging sine — like punching a block of wool. No notes.

  press    a soft thud            the chip lit, talk now
  release  a darker thud          heard, transcribing
  sent     two light quick thuds  the text is in the terminal
  empty    a low dull thud        nothing was heard, nothing was sent
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

RATE = 44100
OUT = Path(__file__).resolve().parent.parent / "assets" / "sounds"


def lowpass(x: np.ndarray, cutoff: float, passes: int = 3) -> np.ndarray:
    """One-pole low-pass run `passes` times: the felt, the wool, the muffle."""
    a = np.exp(-2 * np.pi * cutoff / RATE)
    for _ in range(passes):
        y = np.empty_like(x)
        acc = 0.0
        for i, v in enumerate(x):
            acc = (1 - a) * v + a * acc
            y[i] = acc
        x = y
    return x


def puff(cutoff: float, thump: float, dur: float, amp: float, start: float,
         total: float, seed: int) -> np.ndarray:
    """A soft hit on something spongy: low-passed noise for the body, a short
    sine whose pitch sags for the weight, both with a 4 ms attack and a fast
    exponential tail. No pitch to hum along to — a thud, not a note."""
    n = int(total * RATE)
    t = np.arange(n) / RATE - start
    on = (t >= 0) & (t < dur)
    env = np.where(on, (1 - np.exp(-t.clip(0) / 0.004)) * np.exp(-t.clip(0) / (dur / 3.2)), 0.0)
    noise = lowpass(np.random.default_rng(seed).standard_normal(n), cutoff)
    noise /= np.abs(noise).max() + 1e-9
    freq = thump * (1 - 0.35 * (t.clip(0) / dur).clip(0, 1))
    body = np.sin(2 * np.pi * np.cumsum(np.where(on, freq, 0)) / RATE)
    return amp * env * (0.75 * noise + 0.45 * body)


# Spongy, like punching a block of wool (Mattia, 30/10: «più spugnoso»): muffled
# noise thuds instead of notes. Same four roles, told apart by how dark and how
# doubled the thud is.
CUES = {
    # 30/10 second pass: «leggermente più alto il volume… un po' più acuto».
    "press":   ([(1300.0, 250.0, 0.09, 0.55, 0.000, 1)], 0.14),
    "release": ([(900.0, 200.0, 0.10, 0.50, 0.000, 2)], 0.15),
    "sent":    ([(1600.0, 280.0, 0.06, 0.40, 0.000, 3), (1600.0, 300.0, 0.06, 0.34, 0.075, 4)], 0.16),
    "empty":   ([(560.0, 150.0, 0.14, 0.52, 0.000, 5)], 0.18),
}


def cue(parts, total: float) -> np.ndarray:
    sig = sum(puff(c, th, d, a, s, total, seed) for c, th, d, a, s, seed in parts)
    n = int(0.008 * RATE)
    sig[-n:] *= np.linspace(1, 0, n)
    return sig


def write(name: str, sig: np.ndarray) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    path = OUT / f"{name}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    for name, (notes, total) in CUES.items():
        print(write(name, cue(notes, total)))
