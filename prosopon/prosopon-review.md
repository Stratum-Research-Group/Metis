# Prosopon: local interface review and improved build brief

Reviewed 2026-09-29. Sources: both `9 prosopon*.txt` prompts and the supplied stack documents. This is a design review; no Prosopon implementation was inspected.

## Verdict

The prompt describes a useful UI, but “local” alone does not define a safe or usable phone connection. Specify the browser-to-Kybernetes boundary, approval UX, voice states, and end-to-end latency budget before building the visual layer.

## Highest-impact gaps and changes

| Priority | Gap in the prompt | Concrete change | Acceptance gate |
| --- | --- | --- | --- |
| P0 | Phone access implies a network listener, yet no authentication, transport, origin, or pairing model is specified. | Default to loopback for desktop; for phone use explicit LAN bind, TLS or a trusted local tunnel, device pairing/token, origin/CSRF checks, session expiry, and clear revocation. | An unpaired phone or cross-origin page cannot read chats, invoke tools, or resolve approvals. |
| P0 | Phylax `ASK` is not in the UI plan. | Show exact action, target and argument summary, policy rule, expiry, allow-once/deny controls, and post-action status. Do not auto-approve during voice mode. | Changed or expired request cannot be approved from a stale card; UI records the decision. |
| P0 | “Show work” can be mistaken for raw model reasoning and may leak private documents or tool arguments. | Show an auditable route/evidence summary: components used, citations, timing, verifier status, errors, and approval events. Redact according to Phylax data labels. | The route view matches the execution trace without exposing hidden prompts or restricted content. |
| P0 | “Under a few seconds” lacks a start/end definition and may be unrealistic with cold model load, STT, reasoning walk, and TTS on 16 GB. | Define p50/p95 budgets for microphone stop-to-transcript, transcript-to-first text, answer completion, and first/complete audio, with warm/cold states. Measure on the actual phone and hardware. | Latency report includes all phases and failed/aborted turns; target is revised from measured baseline. |
| P1 | Browser push-to-talk details are missing. | Specify capture permissions, secure-context requirement on mobile browsers, codec/container support, upload cap, VAD/end-of-turn, cancellation, and a visible transcription edit/confirm step before tool actions. | Android browser voice path works after permission denial, silence, network loss, and accidental transcript. |
| P1 | Sessions are persisted but have no retention, deletion, encryption, or backup plan. | Define local database schema, migration, bounded retention, export/delete, backup/restore, and whether audio is ever stored. | Deleting a session removes text/audio and corresponding derived records subject to the declared policy. |
| P1 | Streaming, reconnect and overlapping turns are unspecified. | Use versioned Kybernetes events with turn ID, sequence number, status and cancellation; prevent duplicate submissions and restore only committed messages after reconnect. | Mid-stream disconnect/retry does not execute a tool twice or corrupt session order. |
| P1 | Audio playback and mic permissions need accessible alternatives. | Keep full keyboard/text path, captions, explicit playback control, reduced motion, large controls, and clear error states. | Core flow remains usable without microphone or speakers. |

## Revised build order

1. Agree on Kybernetes event schema, authentication boundary, Phylax approval contract, and evidence/redaction fields.
2. Build text chat with streaming, cancellation, session persistence, and route/evidence summary; test phone network and reconnect behavior.
3. Add approval UI and denial/timeout states. Test exact-operation binding.
4. Add push-to-talk STT, editable transcript, then TTS; instrument each phase before setting a latency target.
5. Run end-to-end usability and fault tests on the target Android browser and local hardware.

## First milestone

Demonstrate a paired phone session with authenticated text chat, a cited retrieval answer, a denied and an approved Phylax action, then a voice round trip. Record real phase timings and a clear failure state for unavailable STT/TTS.

See [stack integration plan](stack-integration.md) for shared schemas and rollout order.
