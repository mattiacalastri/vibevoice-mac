# Contributing to VibeVoice

VibeVoice is a teaching repository as well as an application. Small, explained
pull requests are preferred over broad rewrites.

Before opening a pull request:

```bash
ruff check .
pytest
```

If the change affects the app bundle, run the relevant packaging test. If it
affects the live microphone path, describe the manual check and macOS version;
CI cannot grant microphone or Accessibility permissions.

Every pull request should explain the outcome, files changed, tests run, and any
manual verification not possible in CI. Do not include model downloads,
`dist/` bundles, `~/.vibevoice` state, personal dictionaries, private logs, or
API keys. Use a focused branch and a conventional commit (`fix:`, `feat:`,
`docs:`, or `test:`).
