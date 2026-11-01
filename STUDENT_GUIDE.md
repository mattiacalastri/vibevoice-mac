# VibeVoice — student guide

VibeVoice is a small real macOS application for learning Claude Code through
spoken instructions. It listens on-device, turns speech into text with Whisper,
and types the result into the frontmost app.

## Prerequisites

- macOS 12 or newer;
- Apple Silicon is recommended for the local MLX Whisper model;
- Python 3.10 or newer;
- Microphone and Accessibility permissions only for the live app.

A Mac with a notch is useful for the Dynamic Island UI, but is not required for
the demo or for studying the engine.

## Five-minute setup

```bash
git clone https://github.com/mattiacalastri/vibevoice-mac.git
cd vibevoice-mac
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest ruff pynput
ruff check .
pytest
```

The tests are headless: they do not open the microphone, download a model, or
write to your real `~/.vibevoice` directory.

## Safe first run

Preview the interface without microphone access:

```bash
python vibevoice.py --demo
```

For live dictation, use two terminals from the activated environment:

```bash
VIBEVOICE_AUTOSEND=0 python engine.py
python vibevoice.py
```

With autosend disabled, the engine transcribes but does not type into the
frontmost application. Enable it only after understanding the flow and granting
Accessibility permission.

## Learning path with Claude Code

Ask Claude Code to inspect `AGENTS.md`, `README.md`, and the relevant tests
before editing. Good first exercises:

1. change demo copy in `vibevoice.py` and add a focused test;
2. trace one utterance through `engine.py` and the state-file contract;
3. improve a troubleshooting step and verify the command still works;
4. run `tools/vibevoice_metrics.py` against a local metrics file and explain it.

Do not begin by changing the state-file contract. If it changes, update the
writer, every reader, and its tests in the same change.

## Privacy

The default transcription path is local. The optional cleanup pass sends
transcribed text to the configured OpenAI-compatible endpoint only when
`VIBEVOICE_CLEANUP=1`; it is off by default. Never commit API keys, personal
dictionary files, runtime state, or `.env` files.

See `AGENTS.md` for engineering invariants and `CONTRIBUTING.md` for the
smallest acceptable pull request.
