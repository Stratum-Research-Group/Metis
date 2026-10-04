# Kybernetes (Κυβερνήτης)

**Kybernetes** (Greek: *steersman* / *navigator*) is the versioned local assistant orchestrator and deliberation dispatcher for the Metis ecosystem.

It receives incoming user turns, plans and coordinates capability authorization via **Phylax**, retrieves memory via **LTHCM/Stoma**, performs multi-hop reasoning over concept graphs via **Dianoia**, executes guarded tools, calls local models via **Hestia**, validates factual grounding via **Krisis**, monitors resource burn rates via an in-process **Budget Engine**, and streams verified results and "Show Work" metadata to **Prosopon**.

---

## Key Architectural Principles

1. **De-Siloed Execution Pipeline**:
   The old 4-route dispatcher (`DIRECT | MEMORY | DELIBERATE | TOOL`) is rejected in favor of an ordered, inspectable, multi-stage pipeline:
   `AUTHORIZE -> RETRIEVE -> DELIBERATE -> TOOL -> SYNTHESIZE -> VERIFY -> RESPOND`
   `DIRECT`, `MEMORY`, `DELIBERATE`, and `TOOL` are observability tags describing activated capabilities, not exclusive silos. A single turn can combine retrieval, deliberation, and tools.

2. **Strict Local-First Operation**:
   No external telemetry or paid APIs. Local SQLite stores predictions, usage events, outcomes, and Bayesian calibration.

3. **Phylax Capability Gateway Integration**:
   Every file access, tool execution, and outbound action is mediated through Phylax. Untrusted retrieved content and tool output are treated as untrusted data, never as prompt instructions to change permissions or system behavior.

4. **Zero-Edge Dianoia Fallback**:
   If Dianoia's thought graph walk finds 0 edges (cold index or disconnected concept nodes), Kybernetes falls back explicitly to flat passage retrieval, logging the degraded fallback in the trace rather than fabricating an artificial walk.

5. **Integrated Budget Loop Bolt-On**:
   - Pre-dispatch Bayesian forecasting and adaptive conformal intervals (90% target coverage).
   - In-flight Kalman filter burn-rate tracking.
   - Earned Value Management (EVM) metrics: Cost Performance Index ($CPI = EV / AC$) and Estimate at Completion ($EAC = \text{Budget} / CPI$).
   - Overrun probability distribution emitting `ALLOW`, `WARN`, or `BLOCK` decisions before dispatch and at phase boundaries.
   - Export of bounded decisions (`allow`, `warn`, `block`) to Phylax without leaking private task text.

6. **Honest Factual Grounding**:
   Raw generation completion is strictly distinguished from factual verification. Synthesized claims are mapped to `verified`, `conflicted`, `insufficient_evidence`, `abstained`, or `draft`.

---

## HTTP & Server-Sent Events (SSE) API

Kybernetes exposes a versioned API used by **Prosopon**:

- `GET /v1/health`: Checks status of all local subsystems (Hestia, LTHCM, Dianoia, Phylax, Krisis).
- `POST /v1/chat/turn`: Primary conversational endpoint streaming `TurnEvent` objects over SSE (`Accept: text/event-stream`).
- `POST /v1/chat/stream`: Alias streaming endpoint.
- `POST /v1/approvals/{approval_id}/resolve`: Resolves a pending Phylax interactive capability approval (`allow_once` or `deny`).
- `POST /v1/chat/turns/{turn_id}/cancel`: Cancels an in-flight conversational turn.
- `GET /v1/traces/{turn_id}`: Retrieves complete audit trace record including component versions and stage latencies.
- `GET /v1/budget/report/{task_id}`: Returns post-task reconciliation of forecast vs actual spend.

---

## Safe Tools

- **Calculator (`calculator`)**: AST-based mathematical expression evaluator. Strictly zero use of `eval()` or `exec()`. Supports standard arithmetic, powers, and math functions (`sqrt`, `abs`, `round`, `sin`, `cos`, `log`, etc.).
- **File Search (`file_search`)**: Bounded search strictly confined to authorized workspace roots. Resolves canonical paths to prevent path traversal (`../`) attacks.

---

## Command-Line Interface (CLI)

```bash
# Verify stack dependencies and subsystem readiness
kybernetes doctor

# Run the 6 canonical demonstration scenarios
kybernetes demo

# Run a single turn interactively
kybernetes chat -q "What is Metis architecture?"

# Start the FastAPI SSE server
kybernetes serve --host 127.0.0.1 --port 8080 --db-path kybernetes_budget.db
```

---

## Test Suite

Run the full automated test suite with coverage:

```bash
.venv/bin/pytest -v --cov=kybernetes
```

The test suite covers:
- Data contracts and Prosopon schema compatibility
- Rule-based route planning
- Budget engine: Kalman filter, Bayesian priors, adaptive conformal calibration, EVM, and SQLite persistence
- Safe AST calculator and bounded file search
- Adapters (Hestia, LTHCM with epoch timestamps, Dianoia zero-edge fallback, Phylax approvals, Krisis verification, Dokime priors)
- Pipeline stages, session management, reconnection, secret redaction, and client cancellation
- FastAPI SSE endpoints
