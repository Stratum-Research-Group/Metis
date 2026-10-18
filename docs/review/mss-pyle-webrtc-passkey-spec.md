# MSS Pyle: Modern WebRTC DataChannels & WebAuthn/Passkey Architecture Spec

**Document ID:** `METIS-SPEC-PYLE-WEBRTC-001`  
**File:** `docs/review/mss-pyle-webrtc-passkey-spec.md`  
**Status:** Approved Architectural Upgrade to `mss-pyle-remote-access-devices-spec-c949cfb0.txt`  
**Subsystems:** `prosopon` (Presentation, WebRTC Gateway, WebAuthn RP), `phylax` (Cryptographic Action Gating), `kybernetes` (Device Attribution)

---

## 1. Executive Summary & Paradigm Shift

The original Pyle specification was modeled after **T3 Code** (`pingdotgg/t3code`), utilizing standard HTTP session cookies, bearer tokens, origin-bound WebSockets, and self-signed TLS certificates.

While functional on desktop browsers over loopback, this legacy pattern introduces fatal friction on modern mobile devices:
1. **Self-Signed TLS Warnings & Secure Context Denial**: iOS Safari and Android Chrome display aggressive warning interstitials on self-signed certificates and often **disable browser Secure Contexts**, breaking microphone capture for Prosopon's real-time voice pipeline.
2. **Bearer Token Fragility**: Shared secret tokens stored in `localStorage` or cookies remain vulnerable to cross-site scripting (XSS), session hijacking, and reverse-proxy header leaks.
3. **Origin Boundary Siloing**: Cookies issued on a home LAN origin (`https://192.168.1.50:8000`) cannot be used when switching to a Tailscale origin (`https://mss.tailnet.ts.net`), forcing redundant re-logins.
4. **NAT Traversal Headaches**: Connecting from mobile cellular requires either manual router port forwarding, external dynamic DNS, or active VPN client daemons on every phone.

### The Modern Alternative: WebRTC DataChannels + Hardware Passkeys (WebAuthn/FIDO2)

By pairing **W3C WebAuthn / Passkeys** with **WebRTC DataChannels** (powered by `aiortc` in Python), MSS achieves:
* **Zero Self-Signed Certificate Warnings**: WebRTC negotiates DTLS 1.3 in-band using ephemeral self-generated certificates that browsers accept natively without security prompts.
* **Hardware-Bound Identity**: Asymmetric P-256 / Ed25519 keypairs generated inside the phone's **Secure Enclave** (Apple Secure Enclave / Android KeyStore). Private keys never leave the device.
* **Hardware Biometric Approvals**: Dangerous Phylax actions (T2/T3 mutating tools, shell execution) prompt native FaceID / TouchID / Fingerprint on the remote device before emitting a signed cryptographic approval.
* **Automatic P2P NAT Traversal**: ICE/STUN traverses home firewalls and cellular networks without port forwarding or third-party cloud relays.
* **Native Opus Voice Streaming**: Real-time microphone and speaker audio flows directly over WebRTC MediaStreams with sub-50ms latency.

---

## 2. Architecture & Protocol Stack

```
+-----------------------------------------------------------------------------------------+
| Remote Client (iOS Safari / Android Chrome / Remote Laptop)                             |
|                                                                                         |
|  +-----------------------------------------------------------------------------------+  |
|  | Hardware Secure Enclave (FIDO2 / WebAuthn Authenticator)                          |  |
|  |  - Asymmetric Keypair (P-256) per Device ID                                      |  |
|  |  - Biometric User Verification (FaceID, TouchID, Android Biometrics)              |  |
|  +-----------------------------------------------------------------------------------+  |
|                                           |                                             |
|  +-----------------------------------------------------------------------------------+  |
|  | Native Browser WebRTC PeerConnection (RTCPeerConnection)                          |  |
|  |  - In-Band DTLS 1.3 Handshake (No CA Warnings, No Port Forwarding)                |  |
|  |  - ICE / STUN NAT Traversal (Peer-to-Peer direct connection)                      |  |
|  |  +-----------------------+ +-----------------------+ +-------------------------+  |  |
|  |  | DataChannel: "control" | | DataChannel: "stream" | | MediaStream: "voice"    |  |  |
|  |  | (JSON-RPC / Auth)     | | (SSE / Thought Graph) | | (Opus Audio 48kHz Bidirectional) |
|  |  +-----------------------+ +-----------------------+ +-------------------------+  |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
                                            ▲
                                            │ Direct P2P Encrypted DTLS/SCTP
                                            ▼
+-----------------------------------------------------------------------------------------+
| Host Server: Prosopon Gateway (Python aiortc + FastAPI)                                 |
|                                                                                         |
|  +-------------------------------------+   +-----------------------------------------+  |
|  | WebAuthn Relying Party (RP) Engine  |   | aiortc PeerConnection Manager           |  |
|  | (py_webauthn)                       |   | - SCTP Multiplexer                      |  |
|  | - Challenge generation & verification|  | - DTLS Session Anchor                   |  |
|  | - Public Key & Signature Validation |   | - Channel Dispatcher                    |  |
|  +-------------------------------------+   +-----------------------------------------+  |
|                                           |                                             |
|  +-----------------------------------------------------------------------------------+  |
|  | Device Identity & SQLite Access Store (prosopon/data/access.db)                   |  |
|  | - devices (device_id, label, public_key_pem, preset, scopes)                      |  |
|  | - passkeys (credential_id, public_key, sign_count, uv_flags)                      |  |
|  | - access_audit (SHA-256 hash-chained immutable audit ledger)                      |  |
|  +-----------------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
             |                                              |
             v                                              v
+--------------------------+                   +--------------------------+
| Phylax Capability Gateway|                   | Kybernetes Orchestrator  |
| (Cryptographic Signature |                   | (Attributed Turn Context |
| Verification for T2/T3)  |                   | with Signed Device ID)   |
+--------------------------+                   +--------------------------+
```

---

## 3. Subsystem Specifications

### 3.1 Identity Subsystem: Hardware Passkeys (WebAuthn/FIDO2)

MSS acts as a local **WebAuthn Relying Party (RP)** using the standard W3C Web Authentication API:

#### 1. Registration Ceremony (Device Pairing)
1. Host operator creates a pairing request via CLI (`prosopon device pair --label "Pixel 8"`) or the Workbench UI.
2. The host generates a cryptographically random, single-use 32-byte challenge: `registration_challenge`.
3. The host displays a pairing QR code containing:
   ```json
   {
     "rp_id": "metis.local",
     "pairing_id": "pair_9f82a1...",
     "challenge": "<BASE64_CHALLENGE>",
     "signaling_url": "webrtc-offer-payload"
   }
   ```
4. The remote device scans the QR code. The browser invokes:
   ```javascript
   const credential = await navigator.credentials.create({
     publicKey: {
       rp: { name: "Metis Sovereign Studio", id: window.location.hostname },
       user: { id: deviceIdBytes, name: "Pixel 8", displayName: "Pixel 8 (Companion)" },
       challenge: base64ToBuffer(challenge),
       pubKeyCredParams: [{ alg: -7, type: "public-key" }], // ES256 (P-256)
       authenticatorSelection: {
         authenticatorAttachment: "platform",
         userVerification: "preferred"
       },
       timeout: 60000
     }
   });
   ```
5. The device's Secure Enclave generates a P-256 keypair. The public key and attestation object are returned to the host.
6. Prosopon verifies the attestation signature using `py_webauthn`, records the public key in `access.db`, and activates the device.

#### 2. Authentication Ceremony (Session Bootstrap)
1. Whenever a device reconnects, the host issues an authentication challenge.
2. The client signs the challenge using `navigator.credentials.get()`.
3. Prosopon verifies the signature against the registered public key and increments the monotonic sign counter (`sign_count`) to prevent replay attacks.

---

### 3.2 Transport Subsystem: WebRTC DataChannels (`aiortc`)

WebRTC provides an encrypted, peer-to-peer transport with zero reliance on browser CA trust stores:

#### 1. In-Band DTLS 1.3 Encryption
- Standard HTTPS requires the client browser to trust the server's certificate. Self-signed certs fail this test.
- WebRTC, by contrast, uses **DTLS** (Datagram Transport Layer Security) negotiated in-band between peer endpoints. Browsers do not validate WebRTC DTLS certificates against the OS root CA store; instead, peer certificate fingerprints are verified in the signaling handshake.
- Result: **Zero browser security warnings, zero custom profile installations on mobile Safari.**

#### 2. Multiplexed SCTP DataChannels
Once the `RTCPeerConnection` is open, Prosopon multiplexes communication across dedicated channels:

| Channel Name | Protocol / Format | Purpose | Reliability |
| :--- | :--- | :--- | :--- |
| `control` | JSON-RPC 2.0 | Authentication, scope queries, device rename, heartbeats. | Ordered, reliable |
| `stream` | SSE-framed JSON | Streaming agent thought graphs, token deltas, Lochos squad state. | Ordered, reliable |
| `terminal` | Binary (UTF-8) | Interactive terminal session I/O, diff rendering, log streaming. | Ordered, reliable |
| `audio` | RTP (Opus 48kHz) | Bidirectional streaming voice for Prosopon speech recognition and synthesis. | Unreliable / Real-time |

#### 3. P2P NAT Traversal (ICE / STUN)
- Prosopon includes a built-in lightweight local STUN responder.
- For connections across cellular or external networks, an optional public STUN server (e.g. `stun:stun.l.google.com:19302`) can be configured without leaking any user payload data.

---

### 3.3 Cryptographic Action Signing (Phylax Integration)

Pyle integrates directly with Phylax's point-of-action capability mediation:

#### The Remote Approval Flow
1. The sovereign coding agent proposes a mutating action (e.g., `sandboxed_bash` executing `pytest`).
2. Phylax intercepts the call and creates a pending approval record with:
   - `approval_id`: Unique token (UUIDv4).
   - `action_digest`: SHA-256 of tool parameters and target workspace path.
   - `risk_tier`: `T1` (Workspace Mutating) or `T2` / `T3` (High Risk / System Mutating).
3. Prosopon pushes an approval card over the WebRTC `stream` channel to the remote device.
4. On the remote phone:
   - For `T1` actions: User taps "Approve" (device signs challenge with passkey).
   - For `T2`/`T3` actions: Browser triggers **biometric verification** (FaceID/Fingerprint):
     ```javascript
     const assertion = await navigator.credentials.get({
       publicKey: {
         challenge: sha256(approval_id + action_digest),
         userVerification: "required"
       }
     });
     ```
5. The signature and authenticator data are transmitted over the `control` channel.
6. Prosopon verifies that:
   - The signing key belongs to an active device holding `approvals:resolve` scope.
   - The user verification flag (`uv == 1`) is set for high-tier actions.
   - The signed hash matches the exact pending Phylax action digest.
7. Phylax consumes the approval token and logs the device's public key identifier into `.phylax/audit.jsonl`.

---

## 4. SQLite Data Schema (`prosopon/data/access.db`)

All device identities and cryptographic material are stored in a dedicated SQLite database:

```sql
-- Devices Table (Hardware identities)
CREATE TABLE devices (
    device_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    paired_via TEXT NOT NULL, -- 'qr_passkey' | 'cli' | 'local_bootstrap'
    preset TEXT NOT NULL,      -- 'viewer' | 'companion' | 'operator' | 'owner'
    scopes TEXT NOT NULL,      -- JSON Array of granted scopes
    expires_at_utc TEXT,
    revoked_at_utc TEXT,
    revoked_reason TEXT
);

-- Passkeys Table (WebAuthn credentials)
CREATE TABLE passkeys (
    credential_id BLOB PRIMARY KEY,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    public_key_cbor BLOB NOT NULL,
    sign_count INTEGER NOT NULL DEFAULT 0,
    aaguid BLOB,
    created_at_utc TEXT NOT NULL,
    last_used_at_utc TEXT NOT NULL,
    transports TEXT -- JSON Array: ['internal', 'hybrid']
);

-- Active WebRTC Sessions
CREATE TABLE webrtc_sessions (
    session_id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    connected_at_utc TEXT NOT NULL,
    peer_address TEXT NOT NULL,
    last_heartbeat_utc TEXT NOT NULL
);

-- Access Audit Chain (Cryptographically linked)
CREATE TABLE access_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    actor_device_id TEXT,
    action TEXT NOT NULL,
    target_device_id TEXT,
    detail_json TEXT NOT NULL,
    prev_row_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
);
```

---

## 5. Implementation Roadmap & Milestones

### Phase 0: Local Perimeter Hardening (Immediate Priority)
- [ ] Remove loopback exemption in `prosopon/prosopon/server.py`.
- [ ] Remove `/api/auth/local-token`.
- [ ] Implement host-owner credential file (`.metis/host_owner.token`, `mode 0600`).
- [ ] Implement `prosopon open` CLI command for local browser bootstrap.
- [ ] Enforce ticket authentication on `/api/lochos/ws`.

### Phase 1: WebAuthn Engine & SQLite Store
- [ ] Add `py_webauthn` dependency to `prosopon/pyproject.toml`.
- [ ] Implement `WebAuthnManager` in `prosopon/auth/webauthn.py` (registration and authentication ceremonies).
- [ ] Create SQLite schema in `prosopon/data/access.db`.
- [ ] Add CLI device management commands (`prosopon device list`, `prosopon device pair --qr`, `prosopon device revoke`).

### Phase 2: WebRTC Gateway & Multiplexed DataChannels
- [ ] Add `aiortc` dependency to `prosopon/pyproject.toml`.
- [ ] Implement `WebRTCGateway` in `prosopon/transport/webrtc.py`.
- [ ] Implement signaling endpoint (`/api/webrtc/signal`) for exchanging SDP offer/answers via QR code scan.
- [ ] Build client WebRTC controller (`prosopon/web/static/js/webrtc_client.js`) handling channel multiplexing (`control`, `stream`, `terminal`).

### Phase 3: Biometric Phylax Approvals & Remote Voice
- [ ] Implement biometric action challenge signing for T2/T3 Phylax actions.
- [ ] Connect WebRTC `audio` media stream to Prosopon's Piper/Whisper pipeline for mobile push-to-talk voice interaction.
- [ ] Build the updated **Manage Devices** UI in Prosopon frontend with passkey credential details and live WebRTC status dots.
- [ ] Implement active agent task termination on device revocation.

---

## 6. Security Invariants & Guarantees

1. **Zero Secret Exposure**: No passwords, PINs, or bearer tokens are transmitted over the network or stored in cookies/localStorage.
2. **Private Keys Unextractable**: All cryptographic keys reside in hardware platform authenticators (Secure Enclaves).
3. **No Unauthenticated Listeners**: WebRTC DataChannels are rejected at the DTLS layer if the peer fails WebAuthn challenge verification.
4. **Immediate Revocation**: Revoking a device in SQLite closes its active `aiortc` PeerConnection within 500ms and halts any running agent tasks.
5. **Fail-Closed Auditability**: All pairings, authentications, and remote approvals are chained via SHA-256 in `access_audit`.
