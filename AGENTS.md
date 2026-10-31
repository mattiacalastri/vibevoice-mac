# AGENTS.md — VibeVoice

Orientation for AI coding agents working in this repository. Read this before
editing. It captures the architecture, the **invariants you must not break**, and
how to run and verify changes. (Human contributors: see `README.md` first; this
file is the engineering contract underneath it.)

---

## 0. Repo layout

```
vibevoice/
├── engine.py                     # capture (voice-processing|sounddevice) → speech decider (Silero|RMS) → Whisper → paste (sole writer of state files)
├── vibevoice.py                  # the pill UI + menu-bar master switch (reads state files)
├── autosend.py                   # standalone one-shot auto-Return daemon (pynput)
├── build_app.sh                  # assembles a double-clickable VibeVoice.app (own TCC identity) -> dist/
├── CLAUDE.md                     # short agent rules → points here
├── AGENTS.md                     # this file: the engineering contract
├── README.md                     # human-facing intro, install, troubleshooting
├── requirements.txt              # pyobjc (+AVFoundation), mlx-whisper, sounddevice, numpy
├── pyproject.toml                # ruff + pytest config (no [project]: it's an app, not a package)
├── LICENSE                       # MIT
├── com.vibevoice.pill.plist      # LaunchAgent template for the pill   (replace __HOME__)
├── com.vibevoice.autosend.plist  # LaunchAgent template for autosend   (replace __HOME__)
├── docs/
│   └── ARCHITECTURE.md           # deep runtime reference (threads, VAD, geometry, constants)
├── tools/
│   ├── smoke_streaming.py        # end-to-end smoke for the streaming path: real audio, real model, no mic
│   ├── vibevoice_kpi.py          # KPI report: immediacy, coverage, tail wait, junction fidelity
│   └── verify_mutations.py       # removes each cure and checks its test screams
├── tests/
│   ├── conftest.py               # session net: autosend/type_text are no-ops for the whole run
│   ├── test_contract.py          # headless contract tests (no mic/GUI/model)
│   └── test_app_bundle.py        # headless: builds VibeVoice.app, locks its shape
└── .github/workflows/ci.yml      # macOS CI: ruff check + pytest
```

Runtime files (created at run time, **not** in the repo) live under `~/.vibevoice/`
— see §2. Tests must never touch them; they redirect to `tmp_path`.

---

## 1. What this is

VibeVoice is a macOS speech-to-text utility with a "Dynamic Island" UI. You speak,
it transcribes on-device with Whisper (Apple Silicon / MLX), and it pastes the text
into whatever app is frontmost — optionally pressing Return so a dictated sentence is
*sent*.

It is built as **three decoupled processes** that never import each other. They
communicate **only through small files** under `~/.vibevoice/`. This decoupling is
the single most important design property of the codebase — preserve it.

```
  microphone ──► engine.py ──► ~/.vibevoice/{state,levels.bin,raw.txt} ──► vibevoice.py (the pill UI)
                    │                                                          ▲
                    └──► pbcopy + Cmd+V into frontmost app                     │ menu-bar icon
                                                                               │ launches / kills engine.py
  global keystrokes ──► autosend.py ──► simulated Return  (independent, shares nothing but an optional pause flag)
```

| File | Role | Process kind |
|------|------|--------------|
| `engine.py` | Mic capture → energy VAD → Whisper transcription → paste. **Sole writer** of the state files. | headless daemon |
| `vibevoice.py` | The "pill": borderless NSPanel under the notch. **Reads** the state files and draws. Menu-bar icon is the master switch that launches/kills `engine.py`. | AppKit GUI |
| `autosend.py` | Standalone `pynput` daemon that presses Return after typing goes quiet. **One-shot** by design. | headless daemon |

---

## 2. The state-file contract (the spine)

Everything flows through `~/.vibevoice/`. **`engine.py` is the only writer** of the
first three files; `vibevoice.py` is strictly a reader. `autosend.py` owns only its
own `autosend` flag.

| Path | Format | Writer | Reader |
|------|--------|--------|--------|
| `~/.vibevoice/state` | text: `idle` \| `recording` \| `transcribing` | engine | pill |
| `~/.vibevoice/levels.bin` | **exactly 60 × float32 little-endian**, RMS 0..1, written atomically (`tmp` + `os.replace`) | engine | pill |
| `~/.vibevoice/raw.txt` | last transcription, plain text (the sentence only) | engine | pill |
| `~/.vibevoice/partial.txt` | **live draft** of the utterance being spoken, plain text, written atomically (`tmp` + `os.replace`). **Presence = an utterance is in flight; content = the words confirmed so far** (may legitimately be empty). Absent = nothing live. | engine | pill |
| `~/.vibevoice/autosend` | text: `on` \| `off` (armed state) | autosend.py, pill | autosend.py, pill |
| `/tmp/vibevoice_autosend_pause` | unix timestamp; suspends autosend for `PAUSE_TTL_SECONDS` (60s, anti-deadlock). **The engine raises it on every streamed chunk and clears it once the sentence is whole** (`hold_autosend` / `release_autosend`): the daemon fires Return after 0.8s of typing silence, and the streaming paste types in bursts ~`PARTIAL_INTERVAL` apart — without the hold, one slow pass sends the message mid-sentence | external tools, engine | autosend.py |
| `~/.vibevoice/muted` | presence = mic paused: engine stays alive but ignores audio (a pause, not a kill) | pill | engine (`is_muted()`) |
| `~/.vibevoice/dictionary.txt` | personal terms, one per line, `#` comments — biases Whisper via `initial_prompt` (max `DICT_MAX_TERMS`) | user, external tools | engine (`load_dictionary()`) |
| `~/.vibevoice/metrics.jsonl` | per-utterance latency telemetry (`stt_ms`, `total_ms`, …), JSONL capped at `METRICS_MAX` | engine | `tools/vibevoice_metrics.py` (p50/p90/p99 report) |
| `~/.vibevoice/corrections.jsonl` | user corrections `{ts,raw,corrected}`, JSONL capped — few-shot examples for the cleanup prompt; new terms flow into `dictionary.txt` | `tools/vibevoice_correct.py` | engine (`_load_corrections()`) |
| `~/.vibevoice/locked` | presence = pill stays visible (no auto-hide) | pill | pill |
| `~/.vibevoice/robot_pos` | text: `x,y` — saved position of the floating robot widget (drag) | pill | pill |
| `~/.vibevoice/widget` | presence = hardware-look floating voice widget shown (menu toggle 🎛; independent of `SHOW_PILL`) | pill | pill |
| `~/.vibevoice/widget_pos` | text: `x,y` — saved position of the hardware widget (drag) | pill | pill |
| `~/.vibevoice/cleanup_key` | bearer key for the cleanup endpoint (chmod 600) — fallback when the env vars are absent (LaunchAgent plists can't read the shell env) | user | engine (`cleanup_text()`) |
| `~/.vibevoice/tts` | presence = TTS-reactivity hook enabled (optional) | external TTS | pill |
| `~/.vibevoice/tts.txt` | line 1 `<start_epoch> <duration_s>`, line 2+ spoken text — the pill types it out in sync, tinted red (optional) | external TTS | pill |
| `~/.vibevoice/tts_levels.bin` | **exactly 60 × float32 little-endian** RMS of the TTS audio (optional) | external TTS | pill |

The control files (`muted`, `locked`, `robot_pos`) are **not** engine-owned state: the
pill writes them and the engine (or the pill itself) honors them — the same
external-control pattern as the autosend pause flag. The `autosend` flag is co-owned —
`autosend.py` owns its semantics, and the pill's **🔁 Auto-send loop** toggle writes
`on`/`off` and spawns the daemon when it isn't already running. The `tts*` files are an
**optional self-contained reactivity hook**: any external text-to-speech may write them,
the pill only reads them (when `tts` is present, the pill turns red and mirrors the
spoken sentence). None of these violate invariant #1, which governs only `state` /
`levels.bin` / `raw.txt`. The `60` in `tts_levels.bin` mirrors `levels.bin` — both are
read with the pill's `struct.unpack("<60f", ...)`.

If you change this contract, you must change **both** the writer and every reader in
the same commit. The `60` in `levels.bin` is duplicated as `LEVELS_LEN` (engine) and a
hard-coded `60` in the pill's `struct.unpack("<60f", ...)` — keep them in lockstep.

**The full-duplex jump (F1 capture backend + F2 Silero decider) changed NOTHING in this
contract.** Same writers, same readers, same formats: `levels.bin` is still exactly 60
float32 LE fed by the raw RMS (whoever makes the speech decision), and `state` still
cycles `idle → recording → transcribing → idle`. Both new pieces are engine-internal and
optional — without `onnxruntime` the decision degrades to the RMS threshold, without
`AVFoundation` capture degrades to sounddevice — and the degraded flow is bit-identical
to the legacy one (locked by the degradation tests in `tests/test_contract.py`).

**`muted` stays a manual master switch.** It is written by the pill (the 🔇 toggle) and
read by the engine — full-duplex did **not** turn it into an auto-ducking channel, and no
component may start writing it automatically. Echo suppression while the Mac speaks is
the job of the voice-processing capture backend (Apple AEC at the source), not of this
flag. Verified at GATE 1 (2026-07-16): no script outside this repo writes
`~/.vibevoice/muted` — the legacy namesake (`voice_briefing.py`) touches only
`~/.local/run/jarvis/` — so no out-of-repo patch is needed or wanted.

---

## 3. Hard invariants — DO NOT break these

These are load-bearing. Violating one produces a regression that is hard to spot
because the code keeps "working" in the happy path.

1. **Engine is the sole writer of `state` / `levels.bin` / `raw.txt`.** Never make the
   pill write them. The pill only reads + draws.
2. **`levels.bin` is exactly 60 float32 LE, written atomically.** The pill guards
   against torn reads (`if len(data) < 60*4: skip frame`). Keep the atomic
   `tmp + os.replace` write and keep both sides agreeing on `60`.
3. **The audio callback (`Engine._audio_callback`) must never raise — and never block.**
   It runs on the realtime capture thread (sounddevice or the voice-processing worker);
   it swallows all exceptions on purpose. File I/O and thread spawning are deferred to
   *outside* the lock. Do not add work that can throw or block inside the lock.
   **This extends to the speech decider (F2):** the only decider calls allowed on the
   audio thread are `SileroVad.submit()` (bounded `put_nowait`, drops on overflow,
   swallows everything) and `is_speech()` (a single attribute read). Silero ONNX
   inference runs exclusively on the `silero-vad` worker thread — never move it into
   the callback, and never make `submit()` blocking.
4. **Keep `self._busy = threading.Semaphore(2)`.** Two transcriptions may be in flight
   so the tail of a long utterance isn't dropped while the previous blob is still being
   transcribed. Reverting to `Semaphore(1)` reintroduces the dropped-monologue bug
   (fixed in commit `9e6ee0e`, "sustain rhythm on long dictation").
5. **`MAX_DUR = 15.0` is deliberate, not arbitrary.** It force-finalizes an utterance so
   each audio blob stays within the recognizer's comfort window and long dictation keeps
   a steady cadence. Don't bump it back up to 30 without re-testing long monologues.
6. **Paste uses `CGEventPost` at `kCGHIDEventTap`** (key codes V=9, Return=36). This is
   what lets the keystroke reach **sandboxed Electron-based editors**. Keep the
   `osascript` path as the no-PyObjC fallback — don't delete it.
7. **`autosend.py` is one-shot.** After it fires one Return it disarms itself
   (`set_enabled(False)`). This prevents a "zombie ON" state from pressing Return while
   the user later types by hand. Do not make it persistent-by-default.
8. **The master switch finds the engine by its absolute path** — `pgrep -f` /
   `pkill -f` in the pill match `str(ENGINE_PATH)`, i.e. the exact argv the pill
   spawns, not the bare name. If you rename `engine.py`, you still break
   start/stop/“is it running” detection in `vibevoice.py` (the path ends in that
   name); update all three call sites if you must rename. Never widen the pattern
   back to the bare name: `pgrep -f engine.py` matches any stranger carrying that
   string in argv, and the `pkill` half then kills it (locked by
   `tests/test_lifecycle.py`).
9. **The three processes share no Python imports.** Coupling is via files only. Do not
   "simplify" by importing `engine` into `vibevoice` (or vice-versa) — it would couple
   their lifecycles and defeat the crash-isolation the file contract buys.
10. **`engine.py` and `vibevoice.py` must stay siblings.** The pill resolves the engine
    as `Path(__file__).parent / "engine.py"`. Moving one without the other breaks launch.

---

## 4. Two independent Return mechanisms (common confusion)

There are **two** separate ways a Return can be pressed. They do not know about each
other and can both fire:

- **In-engine** (`engine.autosend`): pastes via Cmd+V and, *only if*
  `VIBEVOICE_AUTOSEND_RETURN=1`, presses Return after `RETURN_DELAY` (1.5s).
- **Standalone** (`autosend.py`): a `pynput` daemon that watches *all* typing and fires
  Return after `AUTO_SEND_DELAY` (0.8s) of silence in a target app, then disarms.

If you are debugging "Return fired twice" or "Return fired unexpectedly," check whether
both are active. They are intentionally orthogonal — `autosend.py` works with any STT,
not just this engine.

---

## 5. Run & develop

```bash
pip install -r requirements.txt        # pyobjc (+AVFoundation), mlx-whisper, sounddevice, numpy
pip install pynput                     # only needed for autosend.py
pip install onnxruntime silero-vad     # optional: neural speech decider (F2) — without
                                       # them the decision degrades to the RMS threshold

python3 vibevoice.py                   # live pill (reads engine state files)
python3 vibevoice.py --demo            # animated preview, no mic — use to iterate on UI
python3 vibevoice.py --place           # placement mode: pill stays visible

python3 engine.py                      # run the capture/transcription engine standalone
python3 autosend.py --delay 1.0        # standalone auto-Return daemon
```

LaunchAgents: `com.vibevoice.pill.plist`, `com.vibevoice.autosend.plist`.

### Environment variables (engine)
| Var | Default | Meaning |
|-----|---------|---------|
| `VIBEVOICE_LANG` | `it` | Whisper language code |
| `VIBEVOICE_MODEL` | `mlx-community/whisper-turbo` | mlx_whisper model id (downloaded on first use) |
| `VIBEVOICE_AUTOSEND` | `1` | paste transcription into frontmost app |
| `VIBEVOICE_AUTOSEND_RETURN` | `0` | press Return after pasting |
| `VIBEVOICE_VP` | `1` | capture via macOS voice processing (AVAudioEngine + Apple AEC/NS/AGC — the full-duplex prerequisite); `0` forces sounddevice. Auto-falls back to sounddevice when AVFoundation is missing or the VP path fails at open time |
| `VIBEVOICE_SILERO_MODEL` | *(unset)* | explicit path to a Silero VAD ONNX model; overrides the copy shipped with the `silero_vad` package. If the path doesn't exist it warns and cascades to the package model; with neither the model nor `onnxruntime` available the speech decision falls back to the RMS threshold (`VAD_THRESHOLD`) |
| `VIBEVOICE_CLEANUP` | `0` | `1` enables the LLM cleanup pass after transcription (fillers out, punctuation, glossary spelling). Any failure — no key, timeout, bad reply — falls back to the raw text: the pass is a bonus, never a dependency |
| `VIBEVOICE_CLEANUP_URL` | Groq chat completions | OpenAI-compatible endpoint for the cleanup pass |
| `VIBEVOICE_CLEANUP_MODEL` | `llama-3.1-8b-instant` | model id sent to the cleanup endpoint |
| `VIBEVOICE_CLEANUP_TIMEOUT` | `2.5` | seconds before the cleanup call is abandoned (raw text pasted instead) |
| `VIBEVOICE_CLEANUP_API_KEY` | *(unset)* | bearer key for the endpoint; falls back to `GROQ_API_KEY` |
| `VIBEVOICE_STREAMING` | `1` | re-decode the **open** utterance while it is still being spoken and publish the stable prefix to `partial.txt`, so text exists before `SILENCE_SEC` expires. `0` restores the pure batch engine (no partial pass, no `partial.txt`) — locked by `test_streaming_off_behaves_exactly_like_the_legacy_engine` |
| `VIBEVOICE_PARTIAL_INTERVAL` | `0.6` | minimum seconds between two streaming passes **once text is flowing**. Lower = fresher live text and more CPU; a pass that is still decoding when the next is due simply skips its turn. Headroom is real: a partial pass measured **~170 ms** on an M5 Max (whisper-turbo, MLX, 2–12 s buffers — the cost is flat because Whisper pads to its 30 s window anyway; measured 2026-08-02) |
| `VIBEVOICE_PARTIAL_WARMUP` | `0.45` | the same cadence **before the first word has reached the app**, where the whole wait lives: LocalAgreement-2 publishes nothing until two passes agree, so the first character costs `2 × cadence + one decode` and nothing else. Applied as a floor — it can only make the engine more eager, never less, so lowering `PARTIAL_INTERVAL` below it still wins. It ends on *delivered*, not on *confirmed*: the one-word lag puts those a full pass apart, and ending it on `confirmed` moved the draft earlier while moving the typed character **later** (1.4 s → 1.8 s, measured). Do **not** tune it below ~2× the decode cost — see the trap below |
| `VIBEVOICE_STREAM_PASTE` | `1` | type each confirmed chunk into the frontmost app **as it is confirmed** (via `type_text`, direct unicode keystrokes — no clipboard), instead of pasting the whole sentence after the trailing silence. The final paste then adds only `unstreamed_tail()`. Safe only because the confirmed prefix never retracts. `0` restores the single atomic paste |
| `VIBEVOICE_SILENCE` | `1.5` | trailing silence that closes an utterance. With `VIBEVOICE_STREAM_PASTE=1` this no longer gates the text reaching the app — only the last unconfirmed word or two wait for it |

### Environment variables (pill)
| Var | Default | Meaning |
|-----|---------|---------|
| `VIBEVOICE_ENGINE_AUTOSTART` | `0` | Read by `vibevoice.py` (not the engine). `1` makes the pill spawn `engine.py` on launch — the all-in-one path so one LaunchAgent runs the whole stack. Gated so the default (manual 🎙 toggle) is unchanged; spawned in the pill's GUI/TCC context so the mic permission resolves. Set to `1` in `com.vibevoice.pill.plist`. |

### macOS permissions (changes here are usually permission problems, not code bugs)
- **Microphone** → `engine.py` (System Settings ▸ Privacy & Security ▸ Microphone).
- **Accessibility** → `autosend.py` (pynput global listener + synthetic keys) and the
  CGEvent paste in the engine. The *launching app* (Terminal/editor) needs the grant.

---

## 6. How to verify a change

**Contract tests:** `pytest` (config in `pyproject.toml`). They run headless — no mic,
no GUI, no model download — and lock the state-file contract + pure helpers against the
real modules. CI (`.github/workflows/ci.yml`) runs `ruff check .` + `pytest` on macOS for
every push/PR. Run both locally before you commit.

Tests cover the contract and pure logic, **not** the realtime audio/GUI paths — those are
still verified behaviorally. After any change, also exercise the path you touched:

- **UI / pill changes** → `python3 vibevoice.py --demo` and watch the waveform, the
  fade in/out, the typewriter text, the ✕/⧉ hit-targets. No mic needed.
- **Settings window changes** → `python3 tools/smoke_settings_window.py`. It builds the
  real window against a throwaway `HOME`, drives the controls and closes it — the AppKit
  calls in `openSettings_` only fail at runtime, so `pytest` passing proves nothing about
  them. It is not a pytest because creating a window needs a GUI session and CI is headless.
- **Engine / VAD / transcription** → run `python3 engine.py`, speak a short phrase and a
  long monologue; confirm `~/.vibevoice/state` cycles `idle→recording→transcribing→idle`,
  `raw.txt` updates, and the long monologue is not truncated (invariant #4/#5).
- **Streaming / live text (headless, no mic)** → `python3 tools/smoke_streaming.py`. It
  replays a `say`-generated WAV through the capture seam **at wall-clock speed** with
  the real model, redirects state to a throwaway dir, forces `AUTOSEND=0`, and prints
  the draft timeline plus pass/fail criteria (a draft appeared, it appeared *before*
  the speech ended, it grew, it never retracted, the final text landed). This is what
  `pytest` cannot prove: the suite fakes `transcribe`, so it locks the wiring, not the
  behaviour. It is a tool and not a pytest because it needs the model on disk and takes
  ~30 s. Baseline 2026-08-03, M5 Max, interleaved A/B over three runs a side:
  first character in the app at **1.0–1.1 s** on a 10.3 s sentence (the flat-cadence
  engine before it: 1.4 s, three times out of three), ~11 s ahead of the final text,
  one update every ~0.65 s. Run it **interleaved** when comparing two settings —
  back-to-back runs of one then the other measure the machine's load, not the change.

  **Trap — a tighter warm-up cadence makes the first character LATER.** The obvious
  optimisation is to decode much more often while the line is empty. Measured, it
  loses: at 0.25 s the live first character went to 1.1–2.1 s against a rock-steady
  1.4 s. Two reasons, both structural. The dispatch timer resets whether or not the
  decode slot was free, so a cadence with less than ~2× the decode cost in slack
  loses whole turns to jitter — and LocalAgreement needs *consecutive* hypotheses,
  so a lost turn costs an entire agreement. And short buffers are where Whisper is
  least trustworthy: 0.20 s of speech decodes to "E ba ba ba ba…", 0.35 s to
  "Il Poltino". The recorded per-buffer answers live in `tests/test_contract.py`
  (`_HEARD_AT`) so the scheduler can be reasoned about without the model on disk.
- **Streaming / live text (live mic)** → `VIBEVOICE_AUTOSEND=0 python3 engine.py`, then in another
  shell `while :; do printf '\r%-90s' "$(cat ~/.vibevoice/partial.txt 2>/dev/null)"; sleep 0.2; done`
  and speak a long sentence **without stopping**. The draft must grow *while you talk*
  and never un-say a word already shown; it disappears when the utterance finalizes and
  `raw.txt` takes over. `VIBEVOICE_STREAMING=0` must produce no `partial.txt` at all.
- **Paste / autosend** → focus a terminal *and* an Electron editor; confirm the text
  lands in both (invariant #6) and Return behaves as configured (section 4).
- **Is the suite real?** → `python3 tools/verify_mutations.py`. A green suite says
  the code passes its tests; it does not say the tests would notice if the code
  stopped working. This removes each fix from a throwaway copy and runs the test
  that claims to guard it — the test must fail. It earned its keep on 2026-08-02
  by finding a guard that passed with its own defect reinstated, hiding among 190
  green tests. **Add an entry whenever you fix a defect that could come back**; a
  stale entry (anchor no longer found) is itself reported, because it means that
  scar stopped being checked.
- **Contract changes** → grep for every reader before editing a writer:
  `grep -rn "levels.bin\|raw.txt\|\.vibevoice/state" .`
- **Degradation paths are contract, not best-effort** → the no-`onnxruntime` and
  no-`AVFoundation` flows are locked headless in `tests/test_contract.py`
  (`test_full_flow_without_onnxruntime_behaves_like_legacy`,
  `test_full_run_without_avfoundation_captures_via_sounddevice`). If you touch the
  decider or a capture backend, those tests must stay green unmodified — a machine
  without the optional deps must behave exactly like the pre-full-duplex engine.
- **Full-duplex changes (VP capture / Silero decider)** → run the barge-in
  acceptance procedure below on the live runtime.

### End-to-end acceptance: barge-in with open speakers (the full-duplex criterion)

The definitive behavioral check for the full-duplex jump. It runs on the **live
runtime** — never in pytest (tests must not touch `~/.vibevoice/`, CLAUDE.md rule 2)
— and it needs a human voice at the mic: the barge-in phrase cannot be automated,
because anything played through the speakers is exactly what Apple's AEC is there to
cancel. Success = the engine transcribes *you* while the Mac is talking/playing, and
transcribes *nothing* when only the Mac is talking/playing. The judge is
`~/.vibevoice/history.jsonl` (one JSONL line per utterance, newest last).

```bash
# 0. Legacy engine OFF for the whole test (the namesake trap in CLAUDE.md — it
#    fights over the mic). Expected: no output, exit code 1. If it is running,
#    stop it first: launchctl bootout gui/$UID/com.vibevoice.dictation
#    (and re-enable it after the test — it is the daily driver).
pgrep -f stt_bar.py

# 1. Engine up with VP + Silero. Both prove themselves on stderr at startup —
#    do not proceed until you have seen BOTH lines:
#      VibeVoice: capture: voice-processing
#      VibeVoice: VAD: silero (…/silero_vad.onnx)
VIBEVOICE_AUTOSEND=0 python3 engine.py   # autosend off: the test only observes
                                         # history.jsonl, nothing gets pasted mid-run

# 2. Baseline. history.jsonl caps at HISTORY_MAX=20 — if L0 is already 20 the
#    "+1" below saturates; judge by `tail -1` (ts + text) instead.
L0=$(wc -l < ~/.vibevoice/history.jsonl 2>/dev/null || echo 0)

# 3. TTS-only — ~30 s of spoken TTS from the speakers, nobody talks:
say -o /tmp/vv_tts.aiff "$(printf 'Questa è una prova di sintesi vocale del sistema. %.0s' {1..12})"
afplay /tmp/vv_tts.aiff
wc -l < ~/.vibevoice/history.jsonl       # expected: == L0 (AEC ate the far-end audio)

# 4. Barge-in — replay the TTS and SPEAK a phrase over it while it plays
#    (e.g. "il polpo ha otto tentacoli"):
afplay /tmp/vv_tts.aiff &
wc -l < ~/.vibevoice/history.jsonl       # expected: == L0+1
tail -1 ~/.vibevoice/history.jsonl       # expected: contains the phrase you spoke

# 5. Music-only — ~30 s of music from the speakers, nobody talks:
afplay /path/to/music.mp3                # any ~30 s instrumental clip
wc -l < ~/.vibevoice/history.jsonl       # expected: still == L0+1, no spurious utterance

# 6. Legacy still off (nothing resurrected it mid-test). Expected: 1.
pgrep -f stt_bar.py; echo $?
```

Triage when a step fails: step 3 grows the file → the capture backend is not
cancelling the speaker signal; re-check the startup stderr really said
`voice-processing` (with `VIBEVOICE_VP=0` or a silent fallback you are on
sounddevice, where TTS *will* leak into the mic). Step 4 doesn't grow it → speak
louder/closer, then look at `SILERO_*` thresholds. Step 5 grows it → the decider is
letting music through; verify `VAD: silero` was on stderr (the RMS fallback fires on
any loud audio, music included — that is expected degraded behavior, not a bug).
The command sequence above is locked by `test_agents_documents_barge_in_acceptance`
in `tests/test_contract.py` so doc and ritual cannot drift.

Style: the repo is `ruff`-clean (a `.ruff_cache` is present). Run `ruff check .` if
available and keep it green. Match the existing comment density — the code favors short
"why" comments over "what" comments; follow that.

---

## 7. Map: where to look for what

| If you're touching… | Go to |
|---------------------|-------|
| VAD thresholds, silence/duration tuning, transcription | `engine.py` → `Engine`, module constants |
| Streaming / live text while speaking, hypothesis stabilization | `engine.py` → `LocalAgreement`, `Engine._partial_worker`, `write_partial`, `STREAMING` / `PARTIAL_INTERVAL` · reader: `vibevoice.py` → `_read_partial` |
| Capture backend (voice processing / Apple AEC, sounddevice fallback) | `engine.py` → `_VoiceProcessingCapture`, `_SounddeviceCapture`, `_select_capture_backend`, `_ensure_avfoundation` |
| Speech/non-speech decision (Silero VAD worker, RMS fallback, hysteresis) | `engine.py` → `SileroVad`, `_resolve_silero_model`, `_ensure_onnxruntime`, `SILERO_*` constants |
| The paste mechanism / Electron compatibility | `engine.py` → `_press_key_cg`, `autosend` |
| Pill geometry, notch detection, animation | `vibevoice.py` → `Controller._build_window`, `_animate_` |
| Waveform / text rendering | `vibevoice.py` → `PillView.drawRect_` |
| Menu-bar master switch, engine start/stop | `vibevoice.py` → `_engine_running`, `_start_engine`, `_stop_engine` |
| Auto-Return timing, target-app gating, one-shot logic | `autosend.py` → `AutoSendDaemon` |
| Personal dictionary, LLM cleanup pass, latency telemetry | `engine.py` → `load_dictionary`, `cleanup_text`, `process_utterance`, `CLEANUP_*` constants |
| Correcting the last dictation / growing the dictionary | `tools/vibevoice_correct.py` · report: `tools/vibevoice_metrics.py` |
| Deciding what to optimise next (immediacy, coverage, tail wait, lost tails) | `tools/vibevoice_kpi.py` — reads the per-utterance ledger; `metrics.jsonl` now carries `t_first_ms` / `stream_words` / `tail_words` / `final_words` / `anchor` / `partials` alongside `stt_ms` |
| Deeper data-flow / threading model | `docs/ARCHITECTURE.md` |
| `.app` bundle layout, Info.plist identity, mic/AppleEvents usage strings | `build_app.sh` (locked by `tests/test_app_bundle.py`) |

---

## 8. Commit & PR conventions

- **Conventional commits.** Prefix with `feat:`, `fix:`, `docs:`, `test:`,
  `refactor:`, `chore:` (+ optional scope, e.g. `fix(engine): …`). Match the
  existing history (`git log --oneline`).
- **Keep CI green.** Every push/PR runs `ruff check .` + `pytest` on macOS. Run
  both locally first; do not commit red.
- **Contract changes are atomic.** Any change to a `~/.vibevoice/` file format
  must update the writer *and every reader* in the **same commit** (see §2).
  Add or update a contract test in `tests/test_contract.py` to lock the new shape.
- **One concern per commit.** UI tweaks, engine/VAD changes, and autosend changes
  are independent surfaces — keep them in separate commits.
- **Definition of done** (before you open a PR):
  1. `ruff check .` clean · 2. `pytest` green · 3. contract writer+readers in
  sync · 4. you exercised the touched path behaviorally (§6) · 5. docs updated
  if you changed an invariant, constant, or the file map.

---

## 9. Known sharp edges (intentional — do not "fix" blindly)

These are deliberate trade-offs, documented so an agent doesn't "repair" them
into a regression. Improve them only with a design that preserves the invariants
in §3 — and update this section if you do.

1. **Process match for start/stop is by path, not by PID.** The pill matches
   `pgrep -f "$ENGINE_PATH"` / `pkill -f "$ENGINE_PATH"` (§3 invariant #8). It
   still cannot distinguish two instances launched from the *same* path — that
   would need PID tracking, and the single-user design does not earn it. What it
   no longer does is answer for strangers: the pattern used to be the bare
   `engine.py`, which matched an editor open on the file, a `tail -f`, or another
   project's `engine.py` entirely (reproduced 2026-08-06 — the pill saw a
   stranger, refused to start its own engine, and `pkill` would have killed the
   stranger). Anchoring cost nothing: no PID bookkeeping, the filename contract
   intact. If you do make it PID-tracked, preserve start / stop / "is it running".
2. **Clipboard is overwritten, not restored.** `engine.autosend()` `pbcopy`s the
   transcription and pastes it; the user's previous clipboard is lost. Restoring
   it is possible but races with fast successive dictations — left simple on
   purpose. Don't add a naive save/restore without handling overlap.
3. **Two transcriptions may finish out of order — pastes may NOT.** `Semaphore(2)`
   (invariant #4) keeps the tail of a long monologue from being dropped; since
   sess.9685 the paste step is sequenced (`_paste_in_order`: each finalized
   utterance takes a sequence number, the paste waits its turn, bounded by
   `_PASTE_ORDER_TIMEOUT` so a wedged predecessor can't dam the queue — a late
   paste beats a lost one). Transcription stays concurrent. Do **not** revert to
   `Semaphore(1)` — that reintroduces the dropped-monologue bug (commit `9e6ee0e`);
   an empty transcription must still advance the sequence (locked by tests).
4. **Exceptions are swallowed widely** (`except Exception: pass` / writes to
   `stderr`). This is required for the realtime audio callback (#3) and keeps the
   daemons crash-proof, but it hides systematic failures. When debugging, add
   temporary logging — don't make the swallow conditional in a way that can let
   the audio callback raise.
5. **`autosend.py` spawns `osascript` per keystroke.** `get_frontmost_signature()`
   shells out to AppleScript on the listener thread while you type in a target
   app. It's fine in practice but is the place to look for input latency or
   process churn; any optimization must keep the window-signature check (it's
   what prevents a Return firing into a window you switched away from).
6. **The streaming paste cannot take a word back, so it would rather say too
   little than too much.** It types confirmed words as they are confirmed; the
   final transcription, which has full context, may disagree with what is
   already on screen, and backspaces were rejected (synthetic deletes in
   terminals and autocomplete fields destroy more than they fix). Three rules
   contain it, all of them written after live damage on 2026-08-02:
   - `unstreamed_tail()` anchors on the **end** of what was typed, not its
     beginning. Prefix alignment was the first attempt: one corrected word near
     the start dropped the alignment to zero and the whole sentence was pasted a
     second time ("…che pattern e risolvili tranquillamente Ti dico i
     ragionamenti che fai…"). No anchor found anywhere → paste **nothing**. A
     sentence printed twice makes the text unusable; a missing tail is visible
     and re-dictatable.
   - **One word of lag.** The newest confirmed word still sits on the truncation
     edge, where Whisper puts a full stop that full context will drop — the
     screen read "lo sto testando. e sembra molto bello". A word is typed only
     once another has been confirmed after it. The pill's draft shows everything;
     only the typing waits.
   - Punctuation already typed can still differ from the final text: the draft
     re-punctuates itself, the app cannot. `VIBEVOICE_STREAM_PASTE=0` trades the
     latency back for exactness.
7. **Tests must disarm the outbound switches, not just redirect the files.** The
   `engine_state` fixture forces `AUTOSEND` and `STREAM_PASTE` off. When the
   streaming paste landed, both defaulted to on and the partial-pass tests typed
   into the user's frontmost app and overwrote their clipboard. A file written in
   the wrong place is recoverable; posted keystrokes are not. Locked by
   `test_engine_state_fixture_disarms_the_outbound_switches`.
