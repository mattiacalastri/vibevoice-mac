# Architecture

Deep reference for VibeVoice's runtime. For the editing contract and invariants, read
`AGENTS.md` at the repo root first; this document explains *how* the pieces work so you
can change them safely.

---

## 1. Process topology

Three processes, zero shared imports, file-only IPC:

```
                       ┌──────────────────────────────────────────────┐
                       │                ~/.vibevoice/                  │
                       │   state        levels.bin        raw.txt      │
                       └──────────────────────────────────────────────┘
                              ▲   ▲          ▲   ▲           ▲   ▲
                  writes      │   │          │   │           │   │   reads
                              │   └──────────┼───┴───────────┼───┘
           ┌──────────────────┴──┐          │               │   ┌──────────────────────┐
           │      engine.py      │          └───────────────┴──►│     vibevoice.py     │
           │  (capture + ASR +   │                              │   (the pill UI)      │
           │   paste daemon)     │◄── pgrep/pkill -f $ENGINE_PATH┤  menu-bar switch     │
           └─────────┬───────────┘                              └──────────────────────┘
                     │ pbcopy + Cmd+V (CGEvent @ HID tap)
                     ▼
            frontmost application

           ┌─────────────────────┐      reads/writes
           │     autosend.py     │◄───── ~/.vibevoice/autosend  ("on"|"off")
           │  (pynput one-shot   │◄───── /tmp/vibevoice_autosend_pause (optional)
           │   auto-Return)      │─────► synthetic Return into target app
           └─────────────────────┘
```

Why file-IPC instead of sockets/imports: it gives **crash isolation** (the GUI can die
and respawn without losing the engine, and vice-versa), makes each process independently
runnable and testable, and lets `autosend.py` work with *any* STT, not just this engine.

---

## 2. The engine pipeline (`engine.py`)

### 2.1 Threads

| Thread | Created by | Job |
|--------|-----------|-----|
| main | process start | opens the capture backend (voice-processing or sounddevice), then blocks on `self._stop.wait(0.25)` |
| audio callback | capture backend | one call per ~100 ms block; feeds the decider + drives the state machine; **must never raise or block** |
| VP tap (CoreAudio) | `_VoiceProcessingCapture` | copy + enqueue **only** (holding the GIL longer would starve the audio HAL) |
| VP worker | `_VoiceProcessingCapture` | drains the tap queue, resamples native rate → 16 kHz mono (`AVAudioConverter`), invokes the audio callback |
| `silero-vad` worker | `SileroVad.start()` | drains `submit()`'s queue, re-chunks into 512-sample frames, runs ONNX inference, publishes the speech flag with hysteresis |
| transcription worker(s) | `_finalize` | up to **2** at once (`Semaphore(2)`); runs Whisper, writes `raw.txt`, spawns paste |
| paste | `_transcribe_worker` | `autosend()` off-thread so state returns to `idle` promptly |

Locking: a single `threading.Lock` guards the VAD/capture state (`_speaking`, `_buf`,
`_pre`, timers, `_rms_history`). The rule the code follows religiously: **decide inside
the lock, do I/O and spawn threads outside it.** The callback computes a `do_finalize`
flag under the lock, releases, then performs `write_state` / `_finalize` outside. Keep
this discipline — file writes inside the audio lock would risk realtime glitches. The
decider calls (`submit` / `is_speech`) stay *outside* the lock too: both are
non-blocking and never raise.

### 2.2 Capture backends (F1)

Capture is a seam: both backends feed the same callback with float32 mono blocks of
`BLOCKSIZE` samples at `SAMPLE_RATE`, using the sounddevice signature
`(indata, frames, time_info, status)`, `indata` shaped `(N, 1)`.

- **`_VoiceProcessingCapture`** (default, `VIBEVOICE_VP=1`): `AVAudioEngine` with
  `setVoiceProcessingEnabled_(True)` on the input node — the AEC/NS/AGC stack FaceTime
  uses. The Mac's own speaker output is subtracted from the mic signal *at the source*:
  this is the full-duplex prerequisite (the pill can speak while listening without
  transcribing itself). The tap block runs on a CoreAudio thread and only copies +
  enqueues; a dedicated worker resamples and calls the engine callback.
- **`_SounddeviceCapture`**: the legacy PortAudio path, kept as the universal fallback.

Selection (`_select_capture_backend`) prefers voice processing unless `VIBEVOICE_VP=0`
or `AVFoundation` is not importable (`_ensure_avfoundation`, lazy tri-state — prints an
install hint once). Failures at *open time* (mic permission, VP API errors) degrade to
sounddevice inside `Engine.run()` instead of killing the engine. Both degradations are
locked by `test_full_run_without_avfoundation_captures_via_sounddevice`.

### 2.3 Speech decision (F2) + utterance state machine

The speech/non-speech **decision** comes from `SileroVad` — the Silero VAD ONNX model
(onnxruntime, CPU, single-threaded) running on its own worker thread. The audio callback
only calls `submit(block, rms)` (bounded `put_nowait`; drops blocks when the queue is
full) and `is_speech()` (one attribute read). The worker re-chunks the ~100 ms capture
blocks into 512-sample model frames (64-sample context, recurrent state threaded
through), and publishes the flag with onset/offset hysteresis (`SILERO_ONSET` /
`SILERO_OFFSET`; in between, the previous decision holds) so borderline frames don't
flap.

**Degradation contract**: without `onnxruntime` or a model (`_resolve_silero_model`
cascade: `VIBEVOICE_SILERO_MODEL` env path → `silero_vad` package data → None), or after
a mid-stream inference failure, `is_speech()` returns `rms >= VAD_THRESHOLD` —
bit-identical to the legacy energy VAD (the missing-dependency leg is locked by
`test_full_flow_without_onnxruntime_behaves_like_legacy`, the mid-stream failure leg by
`test_silero_midstream_failure_degrades_to_rms_fallback`). The raw RMS keeps feeding
`levels.bin` regardless of who decides — the waveform is capture-driven, not
decision-driven.

The utterance state machine on top of that decision is unchanged:

```
          decider says speech                  decider says silence
     (Silero flag; rms ≥ VAD_THRESHOLD    (Silero flag off; rms < VAD_THRESHOLD
            on the fallback)                      on the fallback)
   ┌────────┐  onset: start utterance,   ┌───────────┐  start silence timer
   │  idle  │ ───────────────────────►   │ recording │ ─────────────────────┐
   │        │  include PRE_ROLL_BLOCKS    │           │                      │
   └────────┘                            └───────────┘                       │
        ▲                                   │      ▲                          │
        │                                   │      │ speech resumes           │
        │       silence ≥ SILENCE_SEC       │      └── (timer reset)          │
        │       OR elapsed ≥ MAX_DUR        ▼                                 │
        │                              ┌──────────────┐                       │
        └──────────────────────────────│ finalize +   │◄──────────────────────┘
            (worker done → idle)        │ transcribe   │
                                        └──────────────┘
```

- **Pre-roll**: `PRE_ROLL_BLOCKS` (5 ≈ 0.5 s) of pre-speech audio is retained in a
  `deque` and prepended on onset, so the first phoneme isn't clipped.
- **Finalize triggers**: trailing silence ≥ `SILENCE_SEC` (1.5 s) **or** utterance
  length ≥ `MAX_DUR` (15 s, mid-speech force-flush — see invariant #5 in `AGENTS.md`).
- **Drop rule**: utterances shorter than `MIN_DUR` (0.4 s) are discarded silently.
- **Backpressure**: if both transcription slots are busy at finalize, that utterance is
  dropped and state returns to `idle` rather than queuing unbounded work.

### 2.4 Transcription & paste

`transcribe()` writes the float32 buffer to a temp 16 kHz/16-bit WAV and calls
`mlx_whisper.transcribe(path_or_hf_repo=MODEL, language=LANG)`. `mlx_whisper` is imported
**lazily** (`_ensure_mlx_whisper`) so the process starts and the UI is usable even before
the model is present; the user gets actionable install/download guidance instead of a
crash.

Paste (`autosend()` in the engine): `pbcopy` to the clipboard, then `Cmd+V` via
`CGEventCreateKeyboardEvent` posted at `kCGHIDEventTap`. The HID tap is below the app
sandbox, which is what makes paste land in Electron editors. If Quartz/PyObjC is missing,
it falls back to `osascript ... keystroke "v"`. Return (key 36) is pressed only when
`VIBEVOICE_AUTOSEND_RETURN=1`, after `RETURN_DELAY`.

### 2.5 Streaming: live text while the utterance is still open (F3)

Batch transcription has a floor that no tuning removes: the first word cannot appear
before `SILENCE_SEC` of trailing silence, because the utterance is not closed until
then. F3 removes the floor by transcribing the **open** buffer as it grows.

Every `PARTIAL_INTERVAL` (0.6 s) the audio callback snapshots the block **list**
(references only — the `np.concatenate` happens on the worker, never on the audio
thread) and hands it to a single-slot worker: `Semaphore(1)`, separate from the two
final-transcription slots, so a partial can never starve a real one. If the previous
pass is still decoding the turn is simply skipped.

The worker re-decodes the whole open utterance, which means Whisper freely rewrites
its own tail every pass. Raw partials would flicker, so they go through
`LocalAgreement` (LocalAgreement-2, whisper-streaming): **a word is published only
once two successive hypotheses agree on it.** Agreement compares words with
punctuation stripped and case folded — Whisper re-punctuates as context grows, and
comparing glyphs would find almost no agreement. The spelling published is the one
from the pass that saw the word *first*: the draft the user is reading must not
re-punctuate itself under their eyes. Nothing is ever retracted.

The confirmed prefix goes to `partial.txt` (atomic write). A generation counter,
bumped on every utterance open **and** close, makes an in-flight partial stale so it
cannot publish a draft for an utterance that is already finished. `_finalize` clears
the file; the pill only renders it while `state == "recording"`, which is the real
guard against a late partial.

Failure is contained by design: the pass is wrapped whole, and any error costs one
pass of live text — never the sentence. `VIBEVOICE_STREAMING=0` removes the pass
entirely and the engine is bit-identical to the pre-F3 one.

### 2.6 Streaming paste: the words land in the app as they are confirmed

Streaming fixed the *visible* latency; the paste still waited for `SILENCE_SEC`.
Measured on the author's live telemetry (32 utterances, 2026-08-02): `stt_ms`
p50 = **185 ms**, so of the ~1.7 s between the last word and the paste, ~1.5 s
was dead waiting.

`STREAM_PASTE` types each confirmed delta the moment LocalAgreement commits it.
This is only safe because the confirmed prefix never retracts — the stabilizer
is the precondition, not a nicety. Three details make it survivable:

- **Direct keystrokes, not the clipboard.** `type_text()` posts the characters
  with `CGEventKeyboardSetUnicodeString` at the HID tap, in 16-character chunks.
  Going through `pbcopy` a dozen times per sentence would shred the pasteboard.
- **The chunk is claimed under the stabilizer's lock.** `update()` returns the
  delta and `_streamed` is advanced in the same critical section, so two partial
  passes can never type overlapping text.
- **The stream stands down when an earlier utterance is still pasting**
  (`_paste_queue_is_clear`). Interleaving two sentences is worse than a slow one;
  in that case the final paste delivers the whole thing.

At finalize, `unstreamed_tail(streamed, final)` aligns what was typed against the
authoritative transcription — punctuation-insensitively, since the stream saw
less context — and only the remainder is pasted, prefixed with a space so it
cannot weld onto the last typed word. See sharp edge #6 for what happens when
the two genuinely disagree.

Baseline 2026-08-02, M5 Max, 10.3 s sentence: first characters in the app at
**1.3 s**, against 11.8 s for the batch engine — 10.5 s earlier, 31 words typed
against 31 in the final text (no duplication, no loss).

---

## 3. The pill (`vibevoice.py`)

A borderless, non-activating `NSPanel` pinned flush under the notch. It is a pure
**consumer** of the state files plus a process supervisor for the engine.

### 3.1 Render loop
A 24 fps `NSTimer` (`tick:`) drives everything:
1. read `levels.bin` → 60 floats, downsample to `N_BARS` (32), apply `GAIN` and a
   perceptual curve `v**0.6` so quiet speech is still visible;
2. read `state` and `raw.txt`;
3. decide visibility — appears immediately when raw RMS clears `VOICE_THRESH` or state is
   `recording`/`transcribing`; hides after `IDLE_HIDE_S` (1.5 s) of silence, or 2.5 s
   while transcribed text is shown (so the user can click ⧉ to copy);
4. on a visibility *change* only, trigger the native AppKit expand/collapse animation
   (the "Dynamic Island" grow-from-notch / shrink-into-notch).

### 3.2 Geometry
`_build_window` locates the screen **with the notch** via `safeAreaInsets().top > 0`
(not `mainScreen`), computes two frames — `exp` (full pill, flush top edge) and `col`
(exact notch footprint from `auxiliaryTopLeft/RightArea`) — and animates between them.

### 3.3 Hit-targets
`drawRect_` paints the waveform LEDs, the blinking caret, the transcript (Menlo 11,
Matrix green), an inline ⧉ copy glyph at the end of the text, and a ✕ stop glyph
top-right. `mouseDown_` maps clicks: top-right rect → `_stop_engine()`; the recorded
`copy_rect` → re-copy the sentence to the pasteboard. Hover state is polled via
`mouseLocationOutsideOfEventStream()` each tick (no event stream needed for a
non-activating panel).

### 3.4 Master switch
The menu-bar item (`🎙`/`🔇`) toggles the engine by shelling out:
`pgrep -f "$ENGINE_PATH"` (running?), `Popen([python, ENGINE_PATH],
start_new_session=True)` (start), `pkill -f "$ENGINE_PATH"` (stop) — the pattern is
the **absolute path**, so the switch answers for the engine it spawned and not for
any process that merely carries the string `engine.py` in argv. The engine's
**filename** is still part of the contract (invariant #8): it is the tail of that path.

---

## 4. The autosend daemon (`autosend.py`)

A `pynput` global keyboard listener, independent of the engine. Behavior:
- **Arming**: `Cmd+Shift+Space` toggles the `~/.vibevoice/autosend` flag (`tink` = armed,
  `submarine` = disarmed) and posts a desktop notification when armed.
- **Silence timer**: any non-modifier key in a **target app** (re)starts a
  `threading.Timer(delay)`; manual Return or Esc cancels it.
- **Window locking**: when the timer is scheduled it snapshots a *window signature*
  (`get_frontmost_signature` — window **id** for Terminal/iTerm, position for
  Electron/Ghostty/Warp, title prefix otherwise). At fire time it re-checks the signature
  and the app name and **skips** if either changed — so a Return never fires into a window
  you've since switched away from.
- **One-shot**: after firing, it calls `set_enabled(False)`. Re-arm per dictated message.
- **Pause hook**: an external tool can write a timestamp to
  `/tmp/vibevoice_autosend_pause` to suspend firing for up to `PAUSE_TTL_SECONDS` (60 s),
  with TTL auto-clear to prevent a stuck deadlock.

---

## 5. Tunable constants (and why they're set where they are)

| Constant | File | Value | Rationale |
|----------|------|-------|-----------|
| `SAMPLE_RATE` | engine | 16000 | mlx_whisper expects 16 kHz mono |
| `BLOCKSIZE` | engine | 800 | 50 ms/block. Also the **waveform's data rate**: the pill redraws at 24 fps but can only MOVE when a new RMS sample lands, so at 100 ms each bar was held for 2.4 frames and the scroll visibly stepped (reported live 2026-08-02). Silero re-chunks to its own 512-sample frames with carry, so its throughput is unchanged |
| `VAD_THRESHOLD` | engine | 0.015 | RMS gate between noise floor and speech (fallback decision + levels feed) |
| `SILERO_ONSET` / `SILERO_OFFSET` | engine | 0.5 / 0.35 | speech-probability hysteresis: ≥ onset → ON, ≤ offset → OFF, in between → hold |
| `SILERO_FRAME` / `SILERO_CONTEXT` | engine | 512 / 64 | Silero v5 ONNX contract: samples per inference frame / context samples prepended |
| `SILERO_QUEUE_MAX` | engine | 32 | submit()→worker backlog cap; beyond this, blocks are dropped (never block the callback) |
| `SILENCE_SEC` | engine | 1.5 | trailing silence that ends an utterance |
| `MIN_DUR` | engine | 0.4 | below this = noise, dropped |
| `MAX_DUR` | engine | 15.0 | force-flush to keep blobs in the recognizer's comfort window + steady cadence on long dictation |
| `PRE_ROLL_BLOCKS` | engine | 10 | 0.5 s pre-speech so the first word isn't clipped — a **duration**, so it tracks any BLOCKSIZE change (locked by `test_pre_roll_keeps_half_a_second_of_audio`) |
| `Semaphore` | engine | 2 | concurrent transcriptions → don't drop a long utterance's tail |
| `RETURN_DELAY` | engine | 1.5 | gap between paste and the auto-Return |
| `LEVELS_LEN` / `LEVELS_HZ` | engine | 60 / 20 | waveform ring buffer size / write rate. `LEVELS_HZ` must not throttle below the block rate or the extra samples are discarded before the pill sees them |
| `PARTIAL_INTERVAL` / `STREAM_PASTE` | engine | 0.6 / on | seconds between streaming passes / type each confirmed chunk as it lands |
| `N_BARS` | pill | 32 | waveform columns (downsampled from the 60 stored levels) |
| `GAIN` / `VOICE_THRESH` | pill | 12.0 / 0.018 | waveform sensitivity / immediate-onset threshold |
| `IDLE_HIDE_S` / `TICK` | pill | 1.5 / 1/24 | fade-out delay / 24 fps render |
| `AUTO_SEND_DELAY` | autosend | 0.8 | typing-silence before auto-Return |
| `PAUSE_TTL_SECONDS` | autosend | 60 | max suspension from the pause flag (anti-deadlock) |

---

## 6. Failure modes & where they surface

| Symptom | Likely cause | Where to look |
|---------|-------------|---------------|
| Pill never appears | engine not running, or no `levels.bin` | menu-bar switch; `~/.vibevoice/state` |
| Pill flickers / blank waveform | torn read of `levels.bin` | pill `_read_levels` guard; engine atomic write |
| "Transcription disabled" on stderr | `mlx_whisper` not installed / model not downloaded | `_ensure_mlx_whisper` |
| Text not pasted | missing Accessibility grant, or sandbox blocking the keystroke | `_press_key_cg` (HID tap) + osascript fallback |
| Return fires twice / unexpectedly | both engine `AUTOSEND_RETURN` and `autosend.py` active | `AGENTS.md` §4 |
| Long monologue truncated | `Semaphore` reduced to 1, or `MAX_DUR` too low | engine `_finalize`, invariant #4/#5 |
| Mic permission error at start | Microphone privacy grant | `Engine.run` error message |
| "voice-processing capture disabled" on stderr | `pyobjc-framework-AVFoundation` not installed (or `VIBEVOICE_VP=0`) — engine keeps running on sounddevice, just without Apple AEC | `_ensure_avfoundation` |
| "capture 'voice-processing' failed … falling back" | VP open-time error (mic permission, VP API) — degraded to sounddevice, not fatal | `Engine.run` fallback chain |
| "Silero VAD disabled … RMS threshold" on stderr | `onnxruntime` / model missing — decision degrades to the legacy energy VAD, flow unchanged | `_ensure_onnxruntime`, `_resolve_silero_model` |
| Speech flag frozen mid-session | Silero inference failed mid-stream — decider flips itself to the RMS fallback and logs it | `SileroVad._run` |
```
