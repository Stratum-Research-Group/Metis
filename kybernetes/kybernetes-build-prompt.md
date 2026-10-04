# Build Kybernetes — production-grade local assistant orchestrator

You are building **Kybernetes from scratch** as a new Python project. Read `STACK_BUILD_CONTRACT.md`, the Kybernetes v2 `.txt` brief and the existing component docs/improvement files. Deliver a dependable first release with a public API and CLI. Inspect actual dependency APIs before adapting them. Do not claim an unavailable dependency or a mock path is a working live integration. Make routine design choices independently; document deviations and unresolved external blockers.

## Product outcome

Given a user turn, Kybernetes selects a bounded execution plan, obtains authorization for capabilities, invokes local models via Hestia, retrieves evidence via LTHCM/Stoma, deliberates via Dianoia when useful, executes approved tools, and returns a cited answer or explicit failure/abstention state. A turn may combine retrieval, deliberation and a tool; `DIRECT`, `MEMORY`, `DELIBERATE`, `TOOL` are route labels for reporting, **not mutually exclusive execution boxes**.

## Non-negotiable constraints

- Runtime stays local. Hestia serves models; LTHCM/Dianoia remain read-only dependencies; Stoma owns ingestion; Phylax guards actions; Dokime evaluates results. No paid APIs.
- Add read-only LTHCM and Dianoia submodules at the exact v2 commits; verify pins, do not modify/copy their contents. Maintain adapters rather than importing arbitrary private internals.
- Every route, invocation, outcome, latency and failure is logged with a trace ID and resolved dependency/model/index versions. “Outcome” means observed status unless an independent scorer or explicit feedback supplies a correctness label.
- Set per-turn deadlines, maximum model tokens, retrieval budget, walk steps, tool calls, retries and queue wait. Propagate cancellation. A dependency outage or malformed response ends in a typed, bounded failure or documented degraded route.
- No file read, tool call or outbound action bypasses Phylax's guarded execution path. Treat retrieved content and tool output as untrusted data, never as instructions to change permissions or system behavior.

## Build in this order

### 1. Scaffold and compatibility layer

Create an installable package, versioned config, CLI, HTTP API, schema definitions, dependency adapters, tests and operations guide. Define adapter contracts for Hestia model discovery/chat/embeddings, LTHCM retrieval with explicit `as_of`, Dianoia walk/result, Phylax authorization and Stoma index generation/provenance. At startup, check dependency versions and model availability; report actionable incompatibilities. Specifically test Hestia `/v1/models`, Stoma/LTHCM timestamp units, and zero-edge Dianoia behavior.

### 2. Plan representation and rules

Implement a typed, inspectable plan with optional stages: authorize, retrieve, deliberate, tool, synthesize, verify/abstain, respond. Start with deterministic rules for evidence need, currentness, tool need, multi-hop complexity and uncertainty. Include direct, retrieval-only, retrieval-plus-deliberation, and tool paths. Permit an explicit flat retrieval fallback when graph walking is unavailable; surface that fallback in the trace and answer status. Do not silently replace a tool or model.

### 3. Safe tool interface

Define a plugin specification with name/version, JSON input/output schemas, declared capabilities, timeout, side-effect class, idempotency behavior and error codes. Ship calculator and local file search. File search must respect allowed roots, canonical paths and Phylax decisions. Bind any approval to exact arguments. Sandbox or disable untrusted third-party plugins; do not execute arbitrary callable code merely because it advertises a schema.

### 4. Session and conversation loop

Persist versioned turns, stable turn IDs, source citations and outcome states. Support cancellation, reconnect without repeated side effects, prompt size limits and privacy/retention/deletion settings. Use bounded model context built from trusted instructions and clearly delimited untrusted evidence. For factual retrieved claims, return citation IDs that resolve to the exact Stoma passage/index generation. If Dianoia reports conflict, insufficient evidence or verifier unavailable, represent it honestly; never stream a conflicted draft as verified.

### 5. API, streaming and logs

Expose a versioned API used by Prosopon only, with authenticated local/LAN access as configured, request validation, ordered stream events, backpressure and typed errors. `show_work` returns route, components, citations, timings, verifier/approval statuses and reasons; it must not expose private prompts or unrestricted raw reasoning traces. Write structured JSONL/event logs with redaction, rotation, schema version, route candidate features and observed outcomes. Keep Paideia training exports opt-in and label quality explicit.

### 6. Evaluation and operations

Add `doctor`, health/readiness, configuration validation, and startup/shutdown cleanup. Create a small deterministic offline fixture and Dokime adapter for real held-out comparisons. Compare the rule router to always-direct, always-retrieve and always-deliberate where applicable using equal answer model/evidence budgets. Measure answer correctness, citation support, abstention, failure rate, p50/p95 latency, token use and peak VRAM on the actual available hardware.

## Required tests and demo

- Unit tests: route plan composition, token/step/deadline budgets, malformed dependency responses, schema version mismatch, trace correlation and sensitive-log redaction.
- Integration tests: Hestia model discovery and streaming, LTHCM retrieval with explicit epoch `as_of`, Dianoia result states and zero-edge fallback, Phylax allow/deny/ask, calculator/file search, client cancellation and restart/reconnect.
- Fault tests: unavailable Hestia, missing model, corrupt index, contradictory Dianoia result, expired approval, slow tool, overload and mid-stream disconnect. Assert bounded completion and no duplicate side effects.
- Demo: direct response; Stoma-ingested factual question with exact citation; multi-hop graph-enabled Dianoia walk; calculator action; denied file access; zero-edge flat fallback. Log and display the path actually used.

## Definition of done

A clean checkout can start the service against the pinned local dependencies, run the fixture/demo and show one coherent trace through each stage. The public API and schema are documented for Prosopon. Every failure has a typed outcome and no unapproved side effect occurs. Real integration and benchmark results are distinguished from mocked tests and unverified hardware paths.
