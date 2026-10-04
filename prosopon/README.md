# Prosopon: Production-Grade Local Text and Voice Interface

Prosopon (Greek: *face / presence*) is the single-owner conversational interface for the Metis local AI stack. It connects authenticated browser clients to the versioned Kybernetes orchestrator, providing fast streaming text, cited evidence, honest deliberative route status, interactive Phylax security approvals, and push-to-talk local voice.

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
                    Kybernetes HTTP / SSE API (/v1)
                                    |
                                    v
  +-------------------------------------------------------------------+
  |              Kybernetes Orchestrator & Phylax Gateway             |
  +-------------------------------------------------------------------+
```

### Core Invariants
1. **Strict API Boundary:** Prosopon communicates **exclusively** with the versioned Kybernetes API. It never bypasses the orchestrator to call models (Hestia), memory (LTHCM), or tools (Phylax) directly.
2. **Phone Security & W3C Secure Context:** 
   - By default, Prosopon binds to loopback (`127.0.0.1`).
   - For phone access over LAN (`--lan`), single-owner authentication is enforced via a 6-digit terminal pairing PIN.
   - Built-in TLS generation (`--tls`) provides HTTPS so mobile browsers (Android Chrome, Firefox) activate the W3C Secure Context and permit microphone access.
3. **Phylax Approval Cards:** 
   - Side-effecting actions (`ASK` events) render dedicated interactive approval cards with exact targets, argument summaries, policy rules, and expiration countdowns.
   - Approvals are strictly one-use; voice never auto-approves actions.
4. **Redacted Execution Trace ("Show Work"):**
   - Displays the high-level deliberative route, components used, timings, and verifier status without leaking private system prompts or raw internal scratchpads.
5. **Completely Local & Self-Contained:**
   - Zero external CDNs, fonts, or tracking scripts. All CSS and JavaScript assets are bundled locally.

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
prosopon serve --port 8000
```
Open `http://127.0.0.1:8000` in your browser.

### Start the Server for Phone Use (LAN + TLS)
```bash
prosopon serve --lan --tls --port 8000
```
- Binds to `0.0.0.0:8000` with self-signed TLS.
- Prints a 5-minute 6-digit device pairing PIN to the terminal.
- Open `https://<host-ip>:8000` on your phone browser, enter the PIN, and enjoy authenticated chat and voice push-to-talk!

### Device Pairing & Revocation
Generate a fresh pairing PIN on demand:
```bash
prosopon pair
```

Revoke all paired tokens immediately:
```bash
prosopon revoke
```

---

## 3. Running the Test Suite

Execute the complete asynchronous unit and integration test suite:
```bash
.venv/bin/pytest -v
```

All 22 test cases validate contract schemas, PIN pairing, brute-force lockout, session persistence, SSE streaming, Phylax approvals, audio transcoding, and CLI commands.
