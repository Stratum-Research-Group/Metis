# Build Prosopon — production-grade local text and voice interface

You are building **Prosopon from scratch** as a new Python backend plus single-page front end. Read `STACK_BUILD_CONTRACT.md`, the Prosopon v2 `.txt` brief and the relevant Kybernetes/Phylax/Hestia docs. Deliver a phone-friendly first release for one owner. Prosopon talks only to the versioned Kybernetes API, not directly to models, memory or tools. Inspect the real API before implementation; use a local mock only for UI tests and identify it as such.

## Product outcome

Provide fast, resumable text chat with cited evidence, honest route/outcome status and owner approvals. Add push-to-talk voice using local `whisper.cpp` and `piper` after the text and approval path is dependable. Runtime assets and processing remain local; no cloud STT/TTS, hosted scripts, remote fonts, telemetry or external calls.

## Non-negotiable constraints

- Add read-only LTHCM and Dianoia submodules at the exact v2 commits as requested; verify pins, never modify/copy their contents. These are source/integration references only: the Prosopon runtime must still call Kybernetes exclusively.
- Default server bind is loopback. Phone use requires an explicit LAN mode with authenticated device pairing or token, protected transport appropriate for browser microphone access, origin/CSRF defenses, session expiry and revocation. Do not publish an unauthenticated LAN chat or approval endpoint.
- Approval cards show the exact Phylax action, target, argument summary, policy rule and expiry. An approval resolves only that pending handle; voice never auto-approves actions.
- `show work` means a redacted execution summary—route, components, cited evidence, timing, verification and approval statuses—not raw hidden prompts or private internal reasoning.
- Keep text chat fully usable without microphone or speakers. Support keyboard navigation, visible focus, captions/transcripts, readable mobile controls and explicit playback controls.

## Build in this order

### 1. Contracts and security setup

Implement a typed Kybernetes client for versioned turn, session, stream, citation and approval APIs. Validate incompatible schema versions at startup. Specify localhost and phone LAN modes, pairing/revocation, allowed origins, transport, secret storage and rate limits. Bundle all front-end assets locally. Document the trust boundary and threat cases: stolen device, another LAN client, malicious page origin and stale approval card.

### 2. Text chat and streaming

Create a mobile-first single-page UI with accessible message list, composer, streaming deltas, cancellation, retry state, errors, citation links and a route/evidence panel. Use ordered event IDs/sequence numbers to reconnect without duplicated text or repeated tools. Never display a response as verified until Kybernetes has supplied a final verified status; mark draft/conflicted/unavailable states clearly. Show model/index identity in a details view when useful for debugging.

### 3. Sessions and data lifecycle

Persist multiple named sessions locally with schema migration and atomic writes/transactions. Define whether audio is stored (default: transient), text retention, export, deletion and backup/restore. Respect Kybernetes/Phylax redaction labels. Session deletion must propagate to owned stores and record any external log/training deletion obligation; do not promise deletion from systems Prosopon cannot control.

### 4. Approval UX

Render pending `ASK` events with exact action, target, arguments, reason/rule and countdown. Provide allow-once and deny. Handle expiry, changed arguments, duplicate click, lost connection and simultaneous cards. Only show success after the server confirms execution; preserve denial and timeout in session history. Never let a stale UI card grant a new action.

### 5. Local voice

Use browser push-to-talk to capture bounded audio, transcribe with local `whisper.cpp`, display/edit the transcript, send the confirmed text to Kybernetes, and synthesize/play the reply with local `piper`. Support permission denial, silence, unsupported codec, cancellation, offline model not installed, TTS failure and long answers. Before any side-effecting tool action derived from speech, require the same explicit Phylax approval as text. No always-on mic or wake word.

### 6. Measurement and operations

Instrument microphone-stop to transcript, transcript to first answer text, answer completion, first audio and complete playback. Measure p50/p95 on the target Android browser and available GPU/CPU with warm and cold models; include failed turns. Treat “a few seconds” as a target to validate, not a promise. Provide setup/doctor for browser secure context, `whisper.cpp` and `piper` binaries/models, local assets, Kybernetes API, LAN pairing and storage permissions. Document startup, backup, update and rollback.

## Required tests and demo

- UI tests: streaming order, cancellation, reconnect, long response, citation navigation, route/outcome display, accessible keyboard path and small-screen layout.
- Security tests: unauthenticated LAN access, invalid origin, CSRF, expired/revoked device, stale/replayed approval, changed arguments and restricted evidence redaction.
- Voice tests: mic permission denial, empty/silent audio, transcript correction, model unavailable, STT/TTS timeout, no speaker, and action requiring approval.
- Live demo: authenticated Android browser text conversation with streaming and exact citation; one allow and one deny approval; session resume; then a full local voice round trip with phase timing. If target hardware or browser is unavailable, deliver the app and automated tests but mark the live/mobile latency gate unverified.

## Definition of done

A clean checkout can start Prosopon with local assets, pair an authorized phone, chat and resume safely, inspect citations and route status, approve exact actions, and complete a measured local voice turn. No interface success claim is based only on mocks, and no unpaired client can access sessions or actions.
