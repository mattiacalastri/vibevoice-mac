# Security and privacy

Do not report suspected vulnerabilities in a public issue with credentials,
private transcripts, or personal data attached. Contact the maintainer privately
through the repository owner before publishing disclosure details.

VibeVoice writes runtime state under `~/.vibevoice/`; keep it out of commits.
API keys for the optional cleanup endpoint belong in the environment or local key
file described in `AGENTS.md`, never in source code or issues.

The default speech-to-text path is on-device. The optional cleanup feature is a
separate network operation and is disabled unless explicitly enabled.
