# Metis Sovereign Stack (MSS) & Autonomous Coding Agent

**Document ID:** `METIS-DOCS-MSS-001`  
**File:** `docs/MSS.md`  
**Status:** Canonical User Manual & Operational Reference  
**Subsystems:** `kybernetes` (Coding Agent, Lochos, Pinax), `phylax` (Guardrails), `hestia` (Inference), `prosopon` (Workbench & UI)

---

## 1. Executive Overview

The **Metis Sovereign Stack (MSS)** is an autonomous, local-first coding environment and multi-agent execution framework built into the Metis ecosystem. Designed to operate **100% offline with zero external API dependencies, telemetry, or recurring cloud costs**, MSS transforms local Large Language Models (LLMs) into reliable, self-verifying software engineers.

Rather than relying on naive prompt loops or unverified diffs, MSS synthesizes the proven architectural principles of the industry's premier coding harnesses:
- **Pi Agent**: A minimalist, high-signal tool surface (`read_file_range`, `edit_file_block`, `write_file`, `sandboxed_bash`) preserving $>95\%$ of context tokens for source code and compiler outputs.
- **Nous Research Hermes Agent**: Dynamic, self-evolving skills written to `.metis/skills/<domain>/SKILL.md` upon passing tests and automatically indexed for subsequent runs.
- **Odyssey**: Git milestone state checkpointing and automatic rollback upon test failure, completely preventing runaway workspace corruption.
- **Aider / SWE-agent**: Exact-anchor search/replace block editing with FNV-1a hashline validation and in-process AST syntax diagnostics, eliminating broken unified patch failures.
- **Metis Native Superpowers**: Action-bound single-use cryptographic approvals via **Phylax**, multi-agent squad orchestration via **Lochos**, and system tool discovery via **Pinax**.

---

## 2. Core Architecture

```
                                    +-----------------------------+
                                    |        Operator / CLI       |
                                    +-----------------------------+
                                                   |
                                                   v
+---------------------------------------------------------------------------------------------------+
| Kybernetes Engine                                                                                 |
|                                                                                                   |
|  +---------------------------------------------------------------------------------------------+  |
|  | TerminalCodingAgent (kybernetes code)                                                       |  |
|  |                                                                                             |  |
|  |  +--------------------+   +-----------------------+   +----------------------------------+  |  |
|  |  | DynamicSkillLearner|   |GitMilestoneCheckpointer|  | AgentStateMachine                |  |  |
|  |  | (.metis/skills)    |   | (Pre-turn git commit) |  | (IDLE -> ACTING -> VERIFYING)    |  |  |
|  |  +--------------------+   +-----------------------+   +----------------------------------+  |  |
|  |                                                                                             |  |
|  |  +---------------------------------------------------------------------------------------+  |  |
|  |  | ToolRegistry & Sandboxed Coding Tools                                                 |  |  |
|  |  |  - read_file_range   (1-indexed, hashline anchors)                                    |  |  |
|  |  |  - edit_file_block   (Exact match, AST syntax check, unified diff preview)            |  |  |
|  |  |  - write_file        (Atomic write with overwrite confirmation)                       |  |  |
|  |  |  - sandboxed_bash    (Path-restricted, timeout-bounded shell runner)                  |  |  |
|  |  |  - code_mode         (In-process AST sandboxed batch Python script executor)          |  |  |
|  |  |  - file_search       (Scoped workspace file search)                                   |  |  |
|  |  |  - calculator        (AST-whitelisted safe math evaluator)                           |  |  |
|  |  |  - catalogue_lookup  (Pinax Level 0/1 tool catalogue search)                         |  |  |
|  |  |  - catalogue_docs    (Pinax Level 2 size-capped tool documentation slicer)            |  |  |
|  |  +---------------------------------------------------------------------------------------+  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                                                                   |
|  +------------------------------------+          +---------------------------------------------+  |
|  | Lochos Multi-Agent Squad           |          | Pinax Tool Catalogue Subsystem              |  |
|  |  - Durable Queue (lochos.sqlite)   |          |  - Passive PATH Scanner (No execution)      |  |
|  |  - Agent Seats (coder, reviewer)   |          |  - SQLite FTS5 Index & Category Classifier  |  |
|  |  - Baton Handover State Machine    |          |  - Risk Tier Classifier (T0 to T3)          |  |
|  |  - Attention Blocks & Resolutions  |          |  - Policy Override Engine & Probes          |  |
|  +------------------------------------+          +---------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
             |                                              |                             |
             v                                              v                             v
+--------------------------+                   +--------------------------+  +--------------------------+
| Hestia Model Server      |                   | Phylax Capability Gateway|  | Local Git Workspace      |
| (Local OpenAI /v1 API)   |                   | (Action-bound approvals) |  | (.git snapshot rollback) |
+--------------------------+                   +--------------------------+  +--------------------------+
```

---

## 3. Configuration & Settings Reference

MSS provides granular configuration via CLI options, environment variables, configuration files, and the Prosopon GUI Settings Panel.

### 3.1 Sovereign Coding Agent Options (`kybernetes code`)

| Option | Short Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--goal` | `-g` | String | **Required** | The engineering goal, feature request, or bug description to resolve. |
| `--workspace` | `-w` | Path | `.` (CWD) | Absolute or relative path to the target workspace root directory. |
| `--model` | `-m` | String | *From Hestia* | LLM model identifier on the Hestia server (e.g., `qwen2.5-coder-32b-instruct`). |
| `--base-url` | `-u` | String | `http://127.0.0.1:8080/v1` | Base URL for the Hestia OpenAI-compatible endpoint. |
| `--temperature` | | Float | `0.15` | Sampling temperature for code generation. Lower values provide more deterministic output. |
| `--test-cmd` | `-t` | String | `None` | Automated verification command (e.g. `pytest tests/`). **Triggers git rollback on failure.** |
| `--max-steps` | `-s` | Integer | `20` | Maximum autonomous turns before halting. |
| `--auto-approve` | `-y` | Flag | `False` | Auto-approve mutating file/shell operations without interactive prompt cards. |

### 3.2 Orchestrator & Server CLI Commands (`kybernetes`)

Beyond the coding agent, `kybernetes` provides full service control commands:

| Command | Arguments / Flags | Description |
| :--- | :--- | :--- |
| `kybernetes doctor` | None | Verify readiness of Hestia, LTHCM/Stoma, Dianoia, Phylax, and Krisis subsystems. |
| `kybernetes demo` | None | Run the 5 canonical multi-hop reasoning, capability gating, and budget scenarios. |
| `kybernetes chat` | `-q, --query TEXT` | Run an interactive single-turn query through the de-siloed 7-stage orchestrator pipeline. |
| `kybernetes serve` | `--host TEXT` (default `127.0.0.1`)<br/>`--port INT` (default `8081`)<br/>`--db-path PATH` (default `kybernetes_budget.db`)<br/>`--memory-root PATH`<br/>`--config PATH`<br/>`--admin-token TEXT`<br/>`--service-token TEXT` | Start the FastAPI SSE orchestrator server and `/v1/admin` control plane. |
| `kybernetes lochos` | `ps`, `queue`, `dispatch`, `transfer`, `resolve`, `run` | Manage multi-agent squad seats and durable work queue (see Section 7). |
| `kybernetes pinax` | `scan`, `lookup`, `docs`, `detail`, `evaluate` | Tool inventory discovery, FTS5 lookups, and risk evaluation (see Section 8). |
| `kybernetes heuresis` | `search`, `writeup`, `crawl`, `eval` | Combinatorial concept space discovery, OOV synthesis, and peer-review boundary crawling. |

### 3.3 Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HESTIA_BASE_URL` | `http://127.0.0.1:8080/v1` | Endpoint URL for the local Hestia model server. |
| `HESTIA_DEFAULT_MODEL` | `llama-3.1-8b-instruct` | Fallback model name when not specified via `--model`. |
| `HESTIA_SERVICE_TOKEN` | `None` | Optional Bearer token for authenticated Hestia endpoints. |
| `PHYLAX_ALLOWED_ROOTS` | Current Workspace | Colon-separated directory paths permitted for file access. |
| `LOCHOS_DB_PATH` | `lochos.sqlite` | Path to the SQLite database storing the Lochos work queue and seat state. |
| `PINAX_INDEX_PATH` | `~/.metis/pinax_catalogue.db` | Path to the SQLite FTS5 database storing the Pinax tool catalogue. |

### 3.4 Prosopon Workbench Settings Panel

In the Prosopon Web Workbench (`http://127.0.0.1:8000`), pressing `Ctrl+,` opens the centralized Settings Panel containing categories aligned with the MSS runtime:
- **General & Workbench**: Theme (`theme-obsidian`, `theme-linear`), editor font size, minimap, and breadcrumbs.
- **Hestia (Local Models)**: Active coding model, context length (4k–128k tokens), VRAM budget limit (MB), and idle unload timeout.
- **Kybernetes (Engine & Budget)**: Kalman filter budget smoothing $\alpha$, tool timeout bounds, max Dianoia tree depth, and milestone checkpoint toggle.
- **Phylax (Guardrails & Policies)**: Strict default-deny toggle, approval token TTL (seconds), auto-allow safe reads, and cryptographic audit hash-chaining.
- **Memory (LTHCM & Stoma)**: Light-cone horizon (seconds), Poincaré manifold dimension (default 384), and Stoma AST chunk token bound ($\le 254$).

### 3.5 Perimeter Hardening, Filesystem Authority & Pyle Remote Architecture

To guarantee sovereign isolation against unauthorized local processes and establish friction-free companion access:
- **Zero-Loopback Security Model**: Loopback (`127.0.0.1` / `::1`) exemptions are completely eliminated across Prosopon and Kybernetes. All HTTP endpoints, REST control routes, and WebSocket gateways (`/api/lochos/ws`) strictly mandate bearer token or host-owner credential verification before admitting requests or accepting connections. Unauthorized WebSocket connections are rejected pre-handshake with `WS_1008_POLICY_VIOLATION`.
- **Filesystem Authority (`.host_owner_token`)**: The server initializes `data_dir / ".host_owner_token"` with strict POSIX `0600` permissions on startup. Local CLI commands (`prosopon open`, test suites, and internal daemons) authenticate seamlessly via filesystem ownership without storing shared static passwords or hardcoded keys.
- **Single-Use Browser Launch (`prosopon open`)**: Operators launch the workbench via `prosopon open`, which reads the filesystem token, generates an ephemeral 60-second single-use bootstrap ticket (`/?bootstrap_token=<ticket>`), and opens the default browser. The client frontend (`api.js`) exchanges this ticket for an in-memory session token and immediately strips the ticket from the browser address bar.
- **Pyle WebRTC + Passkey Remote Architecture**: As specified in [`docs/review/mss-pyle-webrtc-passkey-spec.md`](file:///home/akarta/Projects/Metis/docs/review/mss-pyle-webrtc-passkey-spec.md) (`METIS-SPEC-PYLE-WEBRTC-001`), remote companion access (iOS/Android/remote laptops) replaces brittle self-signed TLS certificates and static bearer tokens with:
  1. **Hardware-Bound Passkeys (W3C WebAuthn / FIDO2)**: Asymmetric P-256 keypairs generated inside device Secure Enclaves; private keys never leave the hardware.
  2. **WebRTC DataChannels (`aiortc`)**: In-band DTLS 1.3 encryption and SCTP multiplexing (`control`, `stream`, `terminal`, `audio`) avoiding browser CA trust warnings and cellular NAT hurdles.
  3. **Biometric Phylax Approvals**: Remote biometric challenge-signing (FaceID/TouchID/Android Biometrics) for high-risk T2/T3 agent tool invocations.

### 3.6 Local Zero-Friction vs. Remote MSS Void Peer Topology

Metis differentiates strictly between the local loopback trust boundary and external network boundaries:

1. **Local Machine Zero-Friction (Default)**:
   - When running on the host machine (`127.0.0.1`), MSS terminal commands (`kybernetes code`), workbench browser launches (`prosopon open`), and background daemons (Stoma watcher) require **zero manual token exports, zero PIN prompts, and zero interactive pairing friction**.
   - Local tools automatically resolve authorization from the POSIX `0600` filesystem authority ([`deploy/metis.env`](file:///home/akarta/Projects/Metis/deploy/metis.env) or `data/.host_owner_token`).
   - The UI contains no mobile pairing modals or padlock prompts on localhost.

2. **Remote MSS Void / Remote Peer Connection**:
   - When a remote copy of MSS (e.g. running in an isolated VM, laptop, or remote container) connects to this Metis instance, it authenticates at the network perimeter:
     ```bash
     # Remote MSS Void environment configuration:
     export KYBERNETES_BASE_URL="http://<metis-host-ip>:8081"
     export KYBERNETES_SERVICE_TOKEN="<service-token-from-deploy/metis.env>"
     export HESTIA_BASE_URL="http://<metis-host-ip>:8080/v1"
     export HESTIA_SERVICE_TOKEN="<service-token-from-deploy/metis.env>"
     ```
   - All network calls over HTTP require `Authorization: Bearer <token>` or WebRTC DTLS 1.3 encryption.

3. **Sandboxed Agent Execution (`BubblewrapBackend`)**:
   - All tool commands executed by coding agents run within Bubblewrap kernel namespaces:
     - Read-only host filesystem mount (`--ro-bind / /`)
     - Writable binds strictly confined to the project workspace (`--bind <cwd> <cwd>`)
     - Private ephemeral tmpfs (`--tmpfs /tmp`)
     - Confined environment: `HOME` points to workspace, `TMPDIR` points to `/tmp`
     - Isolated PID, IPC, UTS, and Network namespaces (`--unshare-net`) by default, blocking rogue outbound sockets and exfiltration unless explicitly permitted.

---

## 4. The Tool Surface

All agent operations are mediated through the `ToolRegistry` with strict root bounding:

### 1. `read_file_range`
Reads lines from a file with 1-indexed line numbers and line-level FNV-1a hashline anchors:
```json
{
  "tool_name": "read_file_range",
  "arguments": {
    "path": "kybernetes/engine/terminal_agent.py",
    "start_line": 80,
    "end_line": 120
  }
}
```
*Output format*: `line_num:hash| content`. Hashline anchors ensure subsequent edits reference the exact lines inspected.

### 2. `edit_file_block`
Precisely replaces a search block with a replacement block:
```json
{
  "tool_name": "edit_file_block",
  "arguments": {
    "path": "kybernetes/engine/terminal_agent.py",
    "search_block": "temperature: float = 0.2",
    "replace_block": "temperature: float = 0.15",
    "allow_syntax_errors": false
  }
}
```
*Invariants*:
- Performs an in-process AST syntax check on Python files. Rejects invalid syntax with detailed line/column diagnostic messages before writing to disk.
- Renders an interactive colorized unified diff preview in the terminal before applying changes.

### 3. `write_file`
Atomically creates a new file or completely replaces an existing file:
```json
{
  "tool_name": "write_file",
  "arguments": {
    "path": "docs/NEW_FEATURE.md",
    "content": "# New Feature\nDocumentation...",
    "overwrite": true
  }
}
```

### 4. `sandboxed_bash`
Executes non-interactive shell commands confined to the workspace root:
```json
{
  "tool_name": "sandboxed_bash",
  "arguments": {
    "command": "pytest tests/test_agent.py -v",
    "timeout_seconds": 30
  }
}
```
*Guards*:
- Prevents commands attempting directory traversal or modifying files outside permitted roots.
- Head and tail truncation bounds prevent runaway terminal output from exhausting context windows.

### 5. `code_mode`
Executes in-process sandboxed Python scripts for high-speed multi-file analysis:
```json
{
  "tool_name": "code_mode",
  "arguments": {
    "script": "import glob\nfiles = glob.glob('**/*.py', recursive=True)\nprint(f'Total python files: {len(files)}')",
    "timeout_seconds": 5.0
  }
}
```

### 6. `catalogue_lookup` & `catalogue_docs` (Pinax Integration)
Enables the coding agent to query the system tool catalogue and retrieve tool documentation without consuming excessive context tokens:
```json
{
  "tool_name": "catalogue_lookup",
  "arguments": {
    "query": "ripgrep file search",
    "category": "search",
    "limit": 5
  }
}
```

---

## 5. Milestone Checkpoints & Automated Rollback

MSS incorporates the Odyssey state checkpointing pattern via `GitMilestoneCheckpointer`:

1. **Pre-Turn Snapshot**: Before the agent executes any mutating tool (`edit_file_block`, `write_file`, or mutating `sandboxed_bash`), a Git milestone commit is recorded on a hidden reference.
2. **Post-Turn Verification (`--test-cmd`)**: If a verification command is specified (e.g. `--test-cmd "pytest kybernetes/tests"`), MSS executes the command immediately after the turn completes.
3. **Automatic Rollback**:
   - If the test command exits with code `0`, the milestone is preserved as a verified checkpoint.
   - If the test command fails (non-zero exit), MSS **instantly resets the working tree** back to the pre-turn milestone commit.
   - The test failure output is injected into the model's context as a constraint for the next step, preventing runaway regressions.

---

## 6. Dynamic Self-Improving Skills

MSS learns from successful task resolutions using the Nous Hermes skill synthesis pattern:

1. **Storage Location**: `.metis/skills/<domain>/SKILL.md` within the workspace root.
2. **Structure**:
   ```markdown
   ---
   name: pytest-asyncio-configuration
   description: Guidelines for running pytest with asyncio fixtures in Kybernetes
   tags: [testing, asyncio, pytest]
   ---
   # Pytest Asyncio Configuration
   When running async tests in Kybernetes, always ensure `anyio` or `asyncio_mode = auto` is configured...
   ```
3. **Retrieval**: At the start of every MSS session, `DynamicSkillLearner` indexes available skill markdown files and injects relevant skills into the system prompt based on semantic keyword matching against the `--goal`.

---

## 7. Lochos: Multi-Agent Squad Orchestration

Lochos manages collaborative, multi-seat agent workflows backed by a durable SQLite queue (`lochos.sqlite`):

### 7.1 Agent Seats & Roles

| Seat Address | Role | Default Model | Permissions |
| :--- | :--- | :--- | :--- |
| `squad-coder` | Software Engineer | `qwen2.5-coder-32b-instruct` | Full workspace read/write, test execution. |
| `squad-reviewer` | Code Reviewer | `llama-3.1-8b-instruct` | Read-only analysis, diff inspection. |

### 7.2 Lochos CLI Commands

```bash
# 1. View all seats, current states, and active batons
kybernetes lochos ps

# 2. List queued, in-progress, and completed tasks
kybernetes lochos queue

# 3. Dispatch a new task slice to a seat
kybernetes lochos dispatch --mission "v0.6.0" --slice "001" --title "Add temperature setting" --seat coder

# 4. Transfer the task baton from coder to reviewer
kybernetes lochos transfer --task-id "task-abc123" --from-seat coder --to-seat reviewer --note "Ready for review"

# 5. Execute the claimed task using the terminal agent
kybernetes lochos run --seat coder --auto-approve

# 6. Resolve attention blocks (e.g. manual operator approvals)
kybernetes lochos resolve --task-id "task-abc123" --seat coder --resolution "approved_by_operator"
```

---

## 8. Pinax: System Tool Catalogue Subsystem

Pinax maintains an inspectable, risk-tiered inventory of all system binaries and utilities:

### 8.1 Risk Tiers

- **T0 (Read-Only / Safe)**: `cat`, `ls`, `grep`, `file`, `wc`. Pre-approved without prompts.
- **T1 (Workspace Mutating)**: `git commit`, `sed`, `touch`, `cp`. Requires local approval.
- **T2 (System Wide Mutating)**: `apt install`, `systemctl`, `pip install`. Requires explicit confirmation.
- **T3 (Dangerous / Network / Unsafe)**: `rm -rf`, `curl`, `ssh`, `dd`. Requires strict operator authorization.

### 8.2 Pinax CLI Commands

```bash
# 1. Run passive PATH scan (discovers executables without executing them)
kybernetes pinax scan

# 2. Search tool inventory by keyword, category, or tier
kybernetes pinax lookup "search" --category search --tier T0

# 3. Retrieve size-capped documentation slice for a tool
kybernetes pinax docs git --max-bytes 2048

# 4. View Level 1 specification, capabilities, and overrides
kybernetes pinax detail rg

# 5. Evaluate invocation arguments against risk override policies
kybernetes pinax evaluate git commit -m "feat: new feature"
```

---

## 9. Practical Usage Recipes

### Recipe 1: Standard Interactive Coding Session
```bash
kybernetes code \
  --goal "Fix division by zero in calculator tool and add unit tests" \
  --workspace . \
  --test-cmd "pytest kybernetes/tests/test_tools.py"
```

### Recipe 2: Unattended Automated CI Run
```bash
kybernetes code \
  --goal "Update deprecated HTTP_422_UNPROCESSABLE_ENTITY constants to HTTP_422_UNPROCESSABLE_CONTENT" \
  --auto-approve \
  --max-steps 10 \
  --test-cmd "pytest kybernetes/tests/"
```

### Recipe 3: Custom Model & Temperature
```bash
kybernetes code \
  --goal "Refactor Hestia model swap state machine" \
  --model "qwen2.5-coder-32b-instruct" \
  --base-url "http://127.0.0.1:8080/v1" \
  --temperature 0.1 \
  --test-cmd "pytest hestia/tests/test_swap.py"
```

---

## 10. Operational & Security Checklist

Before deploying MSS on production repositories:
- [ ] Ensure `git status` is clean before launching `kybernetes code` so milestone rollbacks have an immutable baseline.
- [ ] Always provide a reliable `--test-cmd` so the agent cannot exit successfully while leaving breaking changes.
- [ ] For sensitive projects, run without `--auto-approve` to interactively inspect every unified diff before it is applied to disk.
- [ ] Check `.phylax/audit.jsonl` to inspect the complete cryptographic audit trail of all actions executed by the agent.

---

## 11. Metis Monorepo Subsystem CLI Master Directory

In addition to `kybernetes` (the orchestrator and home of MSS, Lochos, and Pinax), the Metis monorepo contains 8 other standalone subsystem CLIs. All 9 CLIs can be invoked directly from their virtual environments or when installed in editable mode:

### 11.1 `hestia` — Local Model Serving Layer
```bash
# Start OpenAI-compatible API server on port 8080
hestia serve --host 127.0.0.1 --port 8080 --config models.yaml

# List registered models and their residency state
hestia list

# Show detailed GPU utilisation and VRAM consumption per model
hestia status

# Explicitly load or unload model weights from GPU memory
hestia load qwen2.5-coder-32b-instruct
hestia unload qwen2.5-coder-32b-instruct
```

### 11.2 `phylax` — Guarded Capability Gateway
```bash
# Validate and lint YAML capability policies
phylax validate policies/coding_agent.yaml

# Run health diagnostics on policy files, directories, and audit hash chains
phylax doctor

# Inspect and verify cryptographic integrity of the audit ledger
phylax audit verify

# List and resolve pending interactive human approvals
phylax approvals list
phylax approvals approve <APPROVAL_ID>
```

### 11.3 `stoma` — Memory Mouth & Ingestion
```bash
# Ingest single file or recursive directory into LTHCM memory
stoma ingest /path/to/docs/

# Query LTHCM topological memory directly with provenance verification
stoma query "What is Dianoia sheaf contradiction re-walk?" --top-k 5

# Start real-time filesystem watcher for automated incremental indexing
stoma watch /path/to/code/ --debounce 0.5

# Reconcile disk files against Stoma generational index manifest
stoma reconcile

# Display index status, active generation, and model metadata
stoma status

# Verify direct index interchange compatibility with Dianoia
stoma interchange
```

### 11.4 `dianoia` — Deliberative Thought-Graph Walks
```bash
# Execute a deliberative graph walk over documents or pre-built index
dianoia --docs /path/to/corpus/ -q "Trace token approval lifecycle in Phylax"
dianoia --index /path/to/lthcm.idx -q "Explain Kalman budget engine" --max-steps 12
```

### 11.5 `dokime` — Evaluation & Release Gates
```bash
# List discovered evaluation benchmarks and tasks
dokime tasks

# Execute benchmark evaluation across models
dokime run --model qwen2.5-coder-32b-instruct --task arc_easy_25 --seed 42

# Compare candidate model against baseline with 10k paired bootstrap CIs
dokime compare <BASELINE_RUN_ID> <CANDIDATE_RUN_ID>
```

### 11.6 `paideia` — Training & Policy Evidence Pipeline
```bash
# Verify environment, CUDA toolchains, and dependencies
paideia doctor

# Audit and validate LTHCM router datasets
paideia dataset validate-router /path/to/router_data.jsonl

# Validate Dianoia walk logs and extract gold policy trajectories
paideia dataset validate-walk ~/.dianoia/walks.jsonl

# Run baseline ladder (Majority -> Regex -> Lexical -> Embedding) on frozen splits
paideia baseline /path/to/router_data.jsonl --task best_route --seed 42
```

### 11.7 `krisis` — Candidate Answer Selector & Verification Gate
```bash
# Validate policy JSON against JSON Schema
krisis validate-schema --policy krisis-policy-0.1.json --schema krisis-policy.schema.json

# Replay deterministic pure-policy fixtures and verify Half-Even scores
krisis replay --fixtures krisis-fixtures-0.1.json --policy krisis-policy-0.1.json

# Evaluate candidate answer JSON bundle directly
krisis evaluate request_turn.json --policy krisis-policy-0.1.json

# Inspect rules, tau thresholds, and reason precedence in a policy manifest
krisis inspect-policy --policy krisis-policy-0.1.json
```

### 11.8 `prosopon` — Web Workbench & Sovereign Voice Client
```bash
# Start Prosopon web server (http://127.0.0.1:8000)
prosopon serve --host 127.0.0.1 --port 8000

# Open workbench in default browser with single-use bootstrap ticket (zero-loopback auth)
prosopon open [--host 127.0.0.1] [--port 8000]

# Start in LAN mode with self-signed TLS and phone pairing enabled
prosopon serve --lan --tls

# Generate 6-digit phone device pairing PIN (valid 5 min)
prosopon pair

# Immediately revoke all paired mobile browser sessions
prosopon revoke

# Diagnostic check for local STT (Whisper), TTS (Piper), and audio devices
prosopon doctor
```

