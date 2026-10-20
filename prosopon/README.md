# Prosopon (Πρόσωπον)

**Prosopon** (Greek: *face* / *presence*) is the single-owner conversational interface for the Metis sovereign local AI stack.

It connects authenticated browser clients to the versioned **Kybernetes** orchestrator, providing fast streaming markdown, cited evidence badges, honest deliberative route status, interactive **Phylax** security approvals, and push-to-talk local voice with zero external network calls.

---

## 1. Architecture & Design Principles

```
  +-------------------------------------------------------------------+
  |               Mobile / Desktop Browser (Vanilla SPA)              |
  |  - Audio Push-to-Talk (MediaRecorder WebM/Opus)                   |
  |  - Interactive Phylax Approval Cards (countdown, allow-once/deny) |
  |  - Streaming Markdown with Citation Badges & "Show Work" Drawer   |
  +---------------------------------+---------------------------------+
                                    |
                  HTTPS / W3C Secure Context (Bearer Token)
                                    |
                                    v
  +-------------------------------------------------------------------+
  |                   Prosopon FastAPI Backend                        |
  |  - Auth & Single-Owner Device Pairing (6-digit PIN)               |
  |  - Dynamic Self-Signed TLS (for mobile microphone permissions)   |
  |  - Audio Transcoder (ffmpeg WebM -> 16kHz mono WAV)               |
  |  - Local Voice Engines: whisper.cpp (STT) + piper (TTS)           |
  |  - Atomic Session Store & Retention Manager                       |
  +---------------------------------+---------------------------------+
                                    |
               Kybernetes HTTP / SSE API (/v1/chat/turn)
                                    |
                                    v
  +-------------------------------------------------------------------+
  |              Kybernetes Orchestrator & Phylax Gateway             |
  +-------------------------------------------------------------------+
```

### Core Invariants

1. **Strict API Boundary**: Prosopon communicates **exclusively** with the versioned Kybernetes API (`--kybernetes-url`). It never bypasses the orchestrator to call models (Hestia), memory (LTHCM), or tools (Phylax) directly.
2. **Single-Owner Authentication & Pairing**:
   - By default, Prosopon binds to loopback (`127.0.0.1`).
   - For LAN access from a phone (`--lan`), single-owner authentication is enforced via a 6-digit terminal pairing PIN generated on demand (`prosopon pair`).
   - Automatic lockout after 5 consecutive incorrect attempts.
3. **Secure Context & Mobile Microphone (W3C Secure Context)**:
   - Modern mobile browsers (Chrome on Android, Safari on iOS) require HTTPS (`W3C Secure Context`) to allow microphone access.
   - Built-in on-the-fly TLS generation (`--tls`) allows secure local phone voice interactions over home Wi-Fi.
4. **Phylax Approval Cards**:
   - Side-effecting actions (`ASK` events) render dedicated interactive approval cards with exact targets, argument summaries, policy rules, and expiration countdowns.
   - Approvals are strictly one-use; voice commands never auto-approve actions.
5. **Redacted Execution Trace ("Show Work")**:
   - Displays deliberative routes, component timings, and verifier status without leaking private system prompts or internal scratchpads.
6. **Zero External CDNs or Trackers**:
   - All CSS, JavaScript, and fonts are bundled locally. Zero external network calls.

---

## 2. Quick Start & CLI Usage

### Setup Virtual Environment
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run Diagnostics (Doctor Check)
Verify Python, `ffmpeg`, audio tools, and secure context:
```bash
prosopon doctor
```

### Start the Server (Local Desktop Loopback)
```bash
# Connect to live Kybernetes orchestrator
prosopon serve --port 8000 --kybernetes-url http://127.0.0.1:8081 --no-mock

# Or in standalone mock/testing mode
prosopon serve --port 8000 --mock
```
Open `http://127.0.0.1:8000` in your browser.

### Start the Server for Phone Use (LAN + TLS)
```bash
prosopon serve --lan --tls --port 8000 --kybernetes-url http://127.0.0.1:8081
```
- Binds to `0.0.0.0:8000` with self-signed TLS.
- Prints a 5-minute 6-digit device pairing PIN to the terminal.
- Open `https://<host-ip>:8000` on your phone browser, enter the PIN, and enjoy authenticated chat and voice push-to-talk!

### Secure Desktop Launch (`prosopon open`)
Open Prosopon in the default browser using a secure single-use bootstrap ticket (or print URL with `--no-browser`):
```bash
prosopon open
```

### Device Pairing & Remote Access (`prosopon device` / `prosopon pair`)
Generate a fresh 6-digit PIN pairing code on demand:
```bash
prosopon pair
```

Manage registered devices, WebAuthn passkeys, and remote device revocations:
```bash
# List all registered devices and passkeys
prosopon device list

# Revoke a specific device and terminate its active sessions
prosopon device revoke <device-id>

# Immediate panic lockdown: revoke all remote devices
prosopon device lockdown
```

Revoke all paired tokens across the system immediately:
```bash
prosopon revoke
```

---

## 3. Audio Pipeline & Local Voice

- **Recording**: Browser captures push-to-talk microphone audio via `MediaRecorder` in WebM/Opus format.
- **Transcoding**: Server transcodes incoming WebM to 16kHz mono 16-bit PCM WAV using local `ffmpeg`.
- **Speech-to-Text (STT)**: Transcribed via local `whisper.cpp` (or OpenAI-compatible STT endpoint).
- **Text-to-Speech (TTS)**: Synthesized via local `piper` voice model into fast audio waveforms streamed back to the client.

---

## 4. Running Tests

```bash
uv run pytest prosopon/tests -v --cov=prosopon
```
All tests validate contract schemas, PIN pairing, brute-force lockout, session persistence, SSE streaming, Phylax approvals, audio transcoding, and CLI commands.
