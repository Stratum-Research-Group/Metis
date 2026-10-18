# Kybernetes (Κυβερνήτης)

**Kybernetes** (Greek: *steersman* / *navigator*) is the sovereign local assistant orchestrator, deliberation dispatcher, and capability coordinator for the Metis ecosystem.

It receives incoming conversational turns, plans and coordinates capability authorization via **Phylax**, retrieves hierarchical memory via **LTHCM/Stoma**, performs multi-hop reasoning over concept graphs via **Dianoia**, executes guarded tools, calls local models via **Hestia**, validates factual grounding via **Krisis**, monitors resource burn rates via an in-process **Budget Engine**, and streams verified results and "Show Work" metadata to **Prosopon**.

---

## 1. Architectural Pipeline & Principles

Kybernetes replaces rigid, siloing single-route dispatchers (`DIRECT | MEMORY | DELIBERATE | TOOL`) with an inspectable, ordered, multi-stage pipeline:

```
User Turn
   │
   ├── 1. AUTHORIZE (Phylax capability checks for requested tools/files)
   │
   ├── 2. RETRIEVE (Query Stoma/LTHCM memory within Lorentzian past light cones)
   │
   ├── 3. DELIBERATE (Execute Dianoia thought-graph walks with Kleroterion stochastic selection)
   │
   ├── 4. TOOL (Execute guarded tools: AST calculator, bounded file search)
   │
   ├── 5. SYNTHESIZE (Generate answer candidates via Hestia OpenAI-compatible endpoints)
   │
   ├── 6. VERIFY (Evaluate factual grounding, citations, and claim support via Krisis & Sheaf)
   │
   └── 7. RESPOND (Stream verified tokens, citations, and ShowWork metadata to Prosopon)
```

### Core Invariants

1. **De-Siloed Execution**: A single conversational turn can seamlessly combine memory retrieval, multi-hop deliberation, and tool execution without artificial mode boundaries.
2. **Action-Bound Phylax Mediation**: Outbound actions (system modifications, file writes, network sends) generate cryptographic, single-use approval tokens mediated through Phylax. Once resolved (`allow_once`), the token is consumed atomically.
3. **Zero-Edge Dianoia Fallback**: If Dianoia's thought-graph walk finds zero graph edges (e.g. cold index or unlinked concept nodes), Kybernetes falls back explicitly to flat passage retrieval, recording the degraded fallback in the execution trace.
4. **Local-First & Offline**: No cloud APIs, no external telemetry. All audit logs, budget metrics, and session traces are stored locally in SQLite (`kybernetes_budget.db`).

---

## 2. Integrated Budget Engine & Resource Kalman Filter

To prevent runaway generation loops and enforce strict latency/token budgets, Kybernetes incorporates a real-time Bayesian-Kalman budget loop:

### Mathematical Formulation

1. **Pre-Dispatch Bayesian Forecasting**:
   Maintains conjugate prior distributions over expected token generation ($T$) and execution latency ($L$). Computes conformal prediction intervals providing $\ge 90\%$ empirical coverage.
2. **In-Flight Kalman Filter State Updates**:
   Continuously estimates remaining resource consumption during multi-step deliberation:
   $$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1})$$
   $$P_{k|k} = (I - K_k H) P_{k|k-1}$$
3. **Earned Value Management (EVM)**:
   - **Cost Performance Index**: $CPI = \frac{EV}{AC} = \frac{\text{Earned Value (Progress)}}{\text{Actual Cost (Consumed Tokens/Time)}}$
   - **Estimate at Completion**: $EAC = \frac{\text{Budget}}{CPI}$
4. **Policy Enforcement Decisions**:
   - $\text{Decision} \in \{\text{ALLOW}, \text{WARN}, \text{BLOCK}\}$
   - If projected overrun probability $P(\text{Cost} > \text{Budget}) > 0.85$, the pipeline blocks further execution and emits a graceful degradation response.

---

## 3. Server-Sent Events (SSE) Protocol

Kybernetes streams real-time conversational responses and telemetry to Prosopon over HTTP SSE (`text/event-stream`):

### Event Types

| Event | Payload | Description |
| :--- | :--- | :--- |
| `stage_start` | `{"stage": "RETRIEVE", "timestamp": ...}` | Signals entry into an orchestration pipeline stage. |
| `stage_end` | `{"stage": "RETRIEVE", "duration_ms": 42}` | Signals completion of a stage with timing metrics. |
| `content_delta` | `{"delta": "According to [1]..."}` | Streaming token chunks for user display. |
| `approval_request` | `{"approval_id": "appr-...", "action": "send", ...}` | Prompts user in Prosopon UI for single-use approval. |
| `trace_complete` | `{"trace_id": "...", "show_work": {...}}` | Final trace bundle containing full ShowWork metadata. |
| `error` | `{"code": "BUDGET_EXCEEDED", "message": ...}` | Typed error and abstention notifications. |

### ShowWork Metadata Drawer

The `trace_complete` event delivers structured transparency without exposing internal system instructions:
- **Component Traces**: Exact wall-clock latencies and token usage across LTHCM, Dianoia, and Hestia.
- **Verification Audit**: Breakdown of claims evaluated by Krisis with status (`supported`, `refuted`), confidence scores, and citation mappings.
- **Budget Report**: Planned vs actual token consumption and EVM performance indices.

---

## 4. Safe Tool Execution

Kybernetes includes sandboxed built-in tools:

- **Calculator (`calculator`)**:
  - Python AST-based evaluator with strict operator whitelisting (`Add`, `Sub`, `Mult`, `Div`, `Pow`, `BitXor`).
  - Whitelisted safe math functions (`sqrt`, `abs`, `round`, `sin`, `cos`, `log`, etc.).
  - **Zero use of `eval()` or `exec()`**.
- **File Search (`file_search`)**:
  - Confined strictly to authorized workspace directory trees.
  - Resolves canonical symlinks and paths, preventing directory traversal (`../`) vulnerabilities.

---

## 5. CLI Commands & Service Control

For complete operational reference and advanced usage recipes, see [docs/MSS.md](../docs/MSS.md).

### 5.1 Orchestrator & Server Operations
```bash
# Stack readiness check
kybernetes doctor

# Run canonical multi-hop and capability demos
kybernetes demo

# Interactive command-line turn
kybernetes chat -q "What is Metis architecture?"

# Start FastAPI SSE server
kybernetes serve --host 127.0.0.1 --port 8081 --db-path kybernetes_budget.db
```

### 5.2 Sovereign Coding Agent (`kybernetes code`)
```bash
# Autonomous coding session with automated test rollback
kybernetes code \
  --goal "Implement feature and add unit tests" \
  --workspace . \
  --test-cmd "pytest kybernetes/tests" \
  --model "qwen2.5-coder-32b-instruct" \
  --temperature 0.15

# Auto-approved unattended run
kybernetes code --goal "Refactor imports" --auto-approve --test-cmd "pytest"
```

### 5.3 Lochos Multi-Agent Squad (`kybernetes lochos`)
```bash
# View active seats and current task batons
kybernetes lochos ps

# Inspect durable SQLite queue
kybernetes lochos queue

# Dispatch a mission slice to a seat
kybernetes lochos dispatch --mission "v0.6.0" --slice "001" --title "Task title" --seat coder

# Transfer task baton from coder to reviewer
kybernetes lochos transfer --task-id "<TASK_ID>" --from-seat coder --to-seat reviewer

# Execute claimed task on a seat
kybernetes lochos run --seat coder --auto-approve
```

### 5.4 Pinax Tool Catalogue (`kybernetes pinax`)
```bash
# Passive PATH scan to discover system tools (no binary execution)
kybernetes pinax scan

# Search tools by keyword and risk tier
kybernetes pinax lookup "git" --tier T1

# Sliced documentation retrieval (size-capped)
kybernetes pinax docs git --max-bytes 4096

# Evaluate invocation arguments against security policies
kybernetes pinax evaluate git commit -m "feat: new feature"
```

---

## 6. Running Tests

```bash
uv run pytest kybernetes/tests -v --cov=kybernetes
```
All unit tests validate route planning, Kalman filter burn-rate tracking, EVM bounds, safe AST calculator, SSE streaming, TerminalCodingAgent, Lochos squads, Pinax catalogue, and Phylax capability integration.

