# Metis Architecture, Modules & Interconnect Guide

**Document ID:** `METIS-DOCS-ARCH-001`  
**File:** `docs/ARCHITECTURE_AND_INTERCONNECT.md`  
**Status:** Canonical Reference Manual  
**Last Updated:** November 2026  
**Scope:** Complete architectural specification, module catalog, CLI reference, interconnect protocols, and runtime dataflows across all 11 Metis subsystems.

---

## 1. Executive Monorepo Topology

Metis is an autonomous, single-owner, local-first sovereign AI cognitive architecture. Designed to operate **100% offline with zero cloud API dependencies, zero telemetry, and zero subscription costs**, Metis unifies non-Euclidean memory, MCTS deliberation, local model inference, cryptographic capability guardrails, conformal answer verification, empirical release gating, and multi-agent squad orchestration.

### 1.1 Monorepo Subsystem Map

The monorepo contains 11 core subsystems, an evaluation suite, and a specialized IDE client:

```
Metis/
├── hestia/        # Local GGUF Inference Serving (Port 8080)
├── kybernetes/    # Central Orchestrator & Multi-Agent Squads (Port 8081)
├── prosopon/      # Web SPA, WebRTC Pyle Hub & Voice Client (Port 8000)
├── LTHCM/         # Long-Term Hyperbolic Concept Memory (Poincaré & ScaNN)
├── stoma/         # Ingestion, Semantic Chunking & Deterministic Grounding
├── dianoia/       # Deliberative MCTS Thought-Graph Walks & Anadrome Replay
├── kleroterion/   # Stochastic Decision Policy & Adaptive Entropy Feedback
├── krisis/        # Post-Hoc Claim Grounding & Conformal Prediction Sets
├── phylax/        # Capability Security Gateway & Bubblewrap Sandboxing
├── paideia/       # Boundary-Aware QLoRA Training & Retraining Loop
├── dokime/        # Preregistered Statistical Evaluation & Release Gates
├── vscode/        # Forked Void IDE Client with Sovereign Dashboard & Pyle
├── deploy/        # Systemd user services, environment configs & installers
└── docs/          # Operational manuals, reviews, registers, and benchmarks
```

### 1.2 Network Allocations & Security Boundaries

Metis enforces strict host and inter-process boundaries with zero external network leakage:

| Service / Subsystem | Transport / Protocol | Host Port | Auth Invariant | Managed Unit |
| :--- | :--- | :---: | :--- | :--- |
| **Hestia** | HTTP / REST (`/v1`, `/manage`) | `8080` | Service Bearer Token | `metis-hestia.service` |
| **Kybernetes** | HTTP / SSE (`/v1`, `/api`) | `8081` | Service Bearer Token / Host Owner | `metis-kybernetes.service` |
| **Prosopon** | HTTP / WebRTC / WS (`/api`) | `8000` | Session Token / WebAuthn / Host Token | `metis-prosopon.service` |
| **Stoma Watcher** | Inotify / File System | N/A | Local File Authority (`0600`) | `metis-stoma-watcher.service` |
| **Paideia Retrainer** | Background Daemon | N/A | Local File Authority + Hestia Token | `metis-paideia-retrainer.service` |

> [!IMPORTANT]
> **The Zero-Loopback Security Invariant (`docs/MSS.md` Section 3.5 & Finding F-04)**:  
> Metis completely prohibits unauthenticated loopback exemptions on `127.0.0.1`. Every internal REST or WebSocket request must supply a valid bearer token (`KYBERNETES_SERVICE_TOKEN`, `HESTIA_SERVICE_TOKEN`, or `PROSOPON_HOST_OWNER_TOKEN`). Automated local tooling authenticates exclusively via filesystem authority (`data/.host_owner_token` created with mode `0600`).

---

## 2. Complete Module Catalog & CLI Usage Reference

Each subsystem is an independently maintainable and testable package with isolated dependencies, dedicated test suites, and standardized CLI interfaces.

### 2.1 Hestia (`hestia/`) — Local Model Serving & Speculative Cascades
- **Architectural Role**: Local LLM and embedding inference engine using `llama.cpp`/`llama-server`, providing high-throughput generation with GPU admission bounds, speculative draft-model cascades, and generation-checked CAS hot-swapping.
- **Key Python Modules**: `hestia.server`, `hestia.backend.llamacpp`, `hestia.cascades`, `hestia.model_swap`.
- **CLI Commands**:
  ```bash
  # Launch inference server
  hestia serve --host 127.0.0.1 --port 8080 --models-file models.yaml
  
  # Inspect active model registry and VRAM consumption
  hestia models list
  
  # Test atomic model swap
  hestia test-swap --slot primary --new-model qwen2.5-coder-32b
  ```
- **Test Suite**: `./hestia/.venv/bin/pytest hestia/tests -q` (**83 passed, 1 skipped**).

### 2.2 Kybernetes (`kybernetes/`) — Orchestrator & Multi-Agent Squads
- **Architectural Role**: Central cognitive coordinator executing the 7-stage pipeline (`route`, `retrieve`, `deliberate`, `synthesize`, `execute`, `verify`, `reflect`). Houses the sovereign coding agent (MSS), Lochos multi-agent squads, and the Pinax system tool catalogue.
- **Key Python Modules**: `kybernetes.engine.pipeline`, `kybernetes.engine.coding_agent`, `kybernetes.lochos`, `kybernetes.tools.pinax`.
- **CLI Commands**:
  ```bash
  # Start orchestrator SSE server
  kybernetes serve --host 127.0.0.1 --port 8081 --db-path kybernetes.db
  
  # Launch autonomous coding agent
  kybernetes code --goal "Implement feature X" --workspace . --test-cmd "pytest"
  
  # Inspect Lochos multi-agent squad fleet
  kybernetes lochos squad
  kybernetes lochos dispatch --seat coder --mission M1 --title "Task 1"
  kybernetes lochos complete --task-id task-123 --seat reviewer
  
  # Query Pinax system tool catalogue
  kybernetes pinax scan
  kybernetes pinax list --risk-tier T1
  ```
- **Test Suite**: `./.venv/bin/pytest kybernetes/tests -q` (**216 passed**).

### 2.3 Prosopon (`prosopon/`) — Web SPA, WebRTC Pyle Hub & Voice Client
- **Architectural Role**: Sovereign presentation layer providing text chat, push-to-talk voice (Piper TTS & Whisper STT), device pairing via WebAuthn/Passkey, interactive Phylax approval cards, and SCTP WebRTC DataChannel control (Pyle).
- **Key Python Modules**: `prosopon.server`, `prosopon.transport.webrtc`, `prosopon.storage.access_store`, `prosopon.auth.webauthn`.
- **CLI Commands**:
  ```bash
  # Start web server
  prosopon serve --host 127.0.0.1 --port 8000
  
  # Mint 60-second single-use ticket and open default browser
  prosopon open
  
  # Manage paired companion devices
  prosopon device list
  prosopon device pair --ttl 300
  prosopon device revoke <device-id>
  prosopon device lockdown   # Emergency sub-500ms disconnect of all remote peers
  ```
- **Test Suite**: `./prosopon/.venv/bin/pytest prosopon/tests -q` (**76 passed**).

### 2.4 LTHCM (`LTHCM/`) — Poincaré Topological Context Memory
- **Architectural Role**: Continuous, hierarchical non-Euclidean memory representing knowledge as points on the Poincaré disk ($d=384$). Features Lorentzian light-cone attention, Cellular Sheaf Laplacian contradiction filtering, and Google ScaNN anisotropic Tree-AH vector indexing.
- **Key Python Modules**: `lthcm.engine`, `lthcm.topology.sheaf`, `lthcm.index.scann`, `lthcm.embeddings`.
- **CLI Commands**:
  ```bash
  # Query Poincaré index with temporal causal bounds
  lthcm query "What is the primary constraint?" --as-of 1730000000 --top-k 5
  
  # Inspect index generation and geometric distortion metrics
  lthcm inspect --index-dir .stoma/lthcm_index
  ```
- **Test Suite**: `./LTHCM/.venv/bin/pytest LTHCM/tests -q` (**133 passed**).

### 2.5 Stoma (`stoma/`) — Memory Mouth & Multimodal Ingestion
- **Architectural Role**: Offline document parser and chunking pipeline handling PDFs, source code, images, and tables (VRDU). Enforces token bounds ($\le 254$), causal timestamps, generational commit-chains, and zero-hallucination deterministic span filtering (`DeterministicSpanFilter`).
- **Key Python Modules**: `stoma.ingest`, `stoma.chunking`, `stoma.filters`, `stoma.crawlers.openreview`.
- **CLI Commands**:
  ```bash
  # Ingest raw document into structured chunks
  stoma ingest document.pdf --output-dir .stoma/chunks
  
  # Start ambient filesystem watcher
  stoma watch /home/akarta/Documents --memory-root .stoma
  
  # Verify knowledge-graph span grounding against raw text
  stoma graph verify-grounding --graph graph.json --source source.txt
  ```
- **Test Suite**: `CUDA_VISIBLE_DEVICES="" ./stoma/.venv/bin/pytest stoma/tests -q` (**127 passed**).

### 2.6 Dianoia (`dianoia/`) — Deliberative MCTS & Constructive Thought Walks
- **Architectural Role**: Deliberative reasoning engine executing Monte Carlo Tree Search thought walks over the LTHCM memory graph (`EXPAND`, `BACKTRACK`, `ANSWER`). Implements Arm F two-pass constructive planning (chronological note generation $\to$ final synthesis) and Anadrome deterministic RFC 8785 replay logging.
- **Key Python Modules**: `dianoia.walk`, `dianoia.mcts`, `dianoia.anadrome.replay`, `dianoia.anadrome.canonical`.
- **CLI Commands**:
  ```bash
  # Execute bounded deliberation walk
  dianoia walk --query "Trace dependency chains" --max-steps 10 --budget 500
  
  # Replay recorded Anadrome continuation trees bit-for-bit
  dianoia replay --trace-file ~/.dianoia/replay_trees.jsonl --policy greedy
  ```
- **Test Suite**: `./.venv/bin/pytest dianoia/tests -q` (**97 passed**).

### 2.7 Kleroterion (`kleroterion/`) — Stochastic Decision Policy
- **Architectural Role**: Entropy-aware stochastic sampling engine preventing mode collapse in Dianoia's thought walks. Computes dimension-scaled ($1/\sqrt{d}$) masked-softmax distributions, adaptive entropy feedback, and epistemic shadow branching.
- **Key Python Modules**: `kleroterion.policy`, `kleroterion.entropy`, `kleroterion.sampling`.
- **Test Suite**: `./kleroterion/.venv/bin/pytest kleroterion/tests -q` (**43 passed**).

### 2.8 Krisis (`krisis/`) — Claim Grounding & Conformal Gates
- **Architectural Role**: Post-hoc verification and answer selection engine. Applies KnowNo finite-sample split conformal prediction uncertainty sets ($P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$), 6-decimal Half-Even scoring, contradiction penalty matrices, and ASPEST active prediction abstention logging (`krisis_active.jsonl`).
- **Key Python Modules**: `krisis.engine.selector`, `krisis.gates.conformal`, `krisis.calibration`, `krisis.aspest`.
- **CLI Commands**:
  ```bash
  # Evaluate candidate responses under conformal uncertainty sets
  krisis evaluate --candidates candidates.json --evidence evidence.json
  
  # Calibrate conformal quantiles over held-out scores
  krisis calibrate --calibration-data cal.jsonl --alpha 0.05
  ```
- **Test Suite**: `./.venv/bin/pytest krisis/tests -q` (**40 passed**).

### 2.9 Phylax (`phylax/`) — Capability Security Gateway & Sandbox
- **Architectural Role**: Zero-trust capability mediation boundary enforcing default-deny execution policies. Generates action-bound single-use cryptographic approval tokens, maps tools to risk tiers (T0 to T3), verifies argument digests against tampering, and executes untrusted code inside unshared Linux Bubblewrap containers (`--unshare-net`, `--ro-bind / /`).
- **Key Python Modules**: `phylax.gateway`, `phylax.approval`, `phylax.sandbox.bubblewrap`, `phylax.audit`.
- **CLI Commands**:
  ```bash
  # Verify audit hash-chain integrity
  phylax audit verify --audit-db ~/.phylax/audit.db
  
  # Check policy decision for an action
  phylax check --action write_file --target /etc/hosts
  ```
- **Test Suite**: `./.venv/bin/pytest phylax/tests -q` (**36 passed**).

### 2.10 Paideia (`paideia/`) — Active Boundary QLoRA & Continuous Retraining
- **Architectural Role**: Sovereign model fine-tuning and active policy distillation engine. Ingests boundary failures ($\sigma^2 > 0$) via `AttemptRolloutVarianceSampler`, executes second-order distributed `ShampooOptimizer` / AdamW QLoRA, and coordinates continuous autonomous retraining (`paideia loop`).
- **Key Python Modules**: `paideia.training.runner`, `paideia.training.shampoo`, `paideia.training.loop`, `paideia.data.boundary_sampler`.
- **CLI Commands**:
  ```bash
  # Run continuous retraining daemon loop
  paideia loop --log-file ~/.metis/krisis_active.jsonl --interval 120.0 --auto-swap
  
  # Train standalone QLoRA adapter
  paideia train --data boundary_data.jsonl --optimizer shampoo --rank 16
  ```
- **Test Suite**: `./paideia/.venv/bin/pytest paideia/tests -q` (**72 passed**).

### 2.11 Dokime (`dokime/`) — Statistical Evaluation & Preregistered Release Gates
- **Architectural Role**: Scientific evaluation gatekeeper enforcing 10,000 paired bootstrap confidence intervals, permutation null-hypothesis tests, and standardized task suites (IFEval, Symbolon, Executable Graph, Heuresis Discovery, SIS Token Minimization). Authorizes model promotion into Hestia serving.
- **Key Python Modules**: `dokime.tasks.ifeval`, `dokime.tasks.symbolon_benchmark`, `dokime.tasks.executable_graph`, `dokime.tasks.heuresis`, `dokime.explainers.sis`.
- **CLI Commands**:
  ```bash
  # Run IFEval programmatic instruction following suite
  dokime ifeval --output ifeval_report.json
  
  # Run Symbolon 4-arm discrete retrieval benchmark
  dokime symbolon-benchmark --top-k 5 --level 4
  
  # Execute repeated executable graph reasoning rollouts
  dokime graph-rollout --attempts 3 --output rollout_report.json
  ```
- **Test Suite**: `./.venv/bin/pytest dokime/tests -q` (**218 passed**).

---

## 3. Subsystem Interconnect & Dataflow Architecture

The diagram below maps the complete dataflow and security perimeter interconnecting all 11 modules:

```mermaid
flowchart TD
    subgraph ClientLayer ["Client & Interaction Boundary"]
        USER(["Operator / Browser / Voice / Mobile"])
        VSCODE["VS Code Void IDE Client<br/><i>Pyle Hub • Lochos Queue • Dashboard</i>"]
        PROSOPON["Prosopon Web SPA<br/><i>FastAPI • WebRTC Pyle • Whisper/Piper</i>"]
    end

    subgraph OrchestrationLayer ["Orchestration & Control Plane"]
        KYBERNETES["Kybernetes Engine<br/><i>7-Stage Pipeline • Coding Agent • Lochos Squad</i>"]
        PHYLAX["Phylax Security Gateway<br/><i>Action-Bound Tokens • Bubblewrap Sandbox</i>"]
        PINAX["Pinax Tool Catalogue<br/><i>FTS5 System Tools • T0-T3 Risk Tiers</i>"]
    end

    subgraph CognitionLayer ["Topological Memory & Deliberation"]
        STOMA["Stoma Ingestion<br/><i>VRDU • Span Grounding</i>"]
        LTHCM["LTHCM Memory<br/><i>Poincaré Disk • ScaNN Tree-AH</i>"]
        DIANOIA["Dianoia MCTS<br/><i>Arm F Constructive Planning</i>"]
        KLEROTERION["Kleroterion<br/><i>Adaptive Entropy Feedback</i>"]
        KRISIS["Krisis Verifier<br/><i>KnowNo Conformal Sets • ASPEST</i>"]
    end

    subgraph ServingTrainingLayer ["Serving, Training & Release Verification"]
        HESTIA["Hestia Inference Server<br/><i>GGUF llama.cpp • Speculative Cascades • CAS Swap</i>"]
        PAIDEIA["Paideia Training Engine<br/><i>Shampoo QLoRA • Continuous Retrainer</i>"]
        DOKIME["Dokime Release Gates<br/><i>10k Paired Bootstrap • IFEval • Symbolon</i>"]
    end

    %% Client Interactions
    USER <-->|WebAuthn / Push-to-Talk Voice| PROSOPON
    USER <-->|Desktop Editing & Multi-Agent| VSCODE
    VSCODE <-->|HTTP / SSE / Tokens| KYBERNETES
    PROSOPON <-->|SSE TurnStream / Approvals| KYBERNETES

    %% Ingestion & Memory Storage
    STOMA -->|Generational Chunks [1, 384]| LTHCM
    LTHCM <-->|Poincaré Light-Cone Retrieval| KYBERNETES
    LTHCM <-->|Topological Thought Graph| DIANOIA

    %% Deliberation & Stochastic Planning
    KYBERNETES <-->|Pass 1 & Pass 2 Constructive Planning| DIANOIA
    DIANOIA <-->|Logits & Entropy Scaling| KLEROTERION

    %% Inference Serving & Capability Security
    KYBERNETES <-->|Candidate Generation| HESTIA
    KYBERNETES <-->|Tool Execution & Approval Cards| PHYLAX
    PHYLAX <-->|Tool Discovery & Policy Classification| PINAX
    PHYLAX -->|Bubblewrap Isolation| KYBERNETES

    %% Answer Gating & Active Learning
    KYBERNETES <-->|Candidate Answers & Citations| KRISIS
    KRISIS -->|Selective Prediction Abstention Logs| PAIDEIA
    PAIDEIA -->|Candidate Adapter Binary| DOKIME
    DOKIME -->|Bootstrap Validation Sign-Off| HESTIA
    HESTIA -.->|Generational Hot-Swap (/manage/swap)| HESTIA
```

### 3.1 Key Interconnect Protocols & Invariants

1. **Ingestion & Causal Past-Light-Cone Contract (Stoma $\to$ LTHCM)**:
   - Chunks are bounded to $\le 254$ tokens and embedded as single-facet vectors of shape `[1, 384]`.
   - Ingestion timestamps must record UTC epoch seconds (`time.time()`). Documents with timestamps exceeding query evaluation time are excluded by Lorentzian past-light-cone filtering.
   - `DeterministicSpanFilter` in Stoma validates entity, relation, and timestamp spans against raw source text before insertion into LTHCM.

2. **Constructive Deliberation Contract (Kybernetes $\to$ Dianoia $\to$ Kleroterion)**:
   - When a multi-hop or counterfactual question is routed to deliberation, Kybernetes invokes Dianoia's **Arm F Two-Pass Constructive Planning**:
     - *Pass 1*: Traverses the graph, constructing chronological milestone notes.
     - *Pass 2*: Synthesizes the final answer using the structured timeline, eliminating output runway exhaustion (truncation ceiling).
   - Dianoia scales action logits by $1/\sqrt{d}$ and passes valid action masks to Kleroterion to ensure exploration entropy without visiting disconnected nodes.

3. **Inference & Serving Gating Contract (Kybernetes $\to$ Hestia)**:
   - Kybernetes communicates with Hestia via the OpenAI-compatible `/v1/chat/completions` REST API over loopback port `8080` with bearer token authentication.
   - When speculative decoding is configured, Hestia pairs lightweight draft models (e.g. Qwen 0.5B/1.5B) with primary weights, deferring verification through speculative cascades.

4. **Claim Grounding & Uncertainty Gating (Kybernetes $\to$ Krisis)**:
   - Responses generated by Hestia are decomposed into atomic claims by Krisis.
   - Krisis applies **KnowNo Finite-Sample Split Conformal Prediction Sets**:
     - If the prediction set $|\mathcal{C}(X)| == 1$: Winner is emitted with statistical confidence $\ge 1 - \alpha$.
     - If $|\mathcal{C}(X)| > 1$ or $|\mathcal{C}(X)| == 0$: Ambiguous/empty set triggers active selective abstention (`Reason.AMBIGUOUS_PREDICTION_SET`), logging the instance to `~/.metis/krisis_active.jsonl`.

5. **Capability Security & Sandboxing (Kybernetes $\to$ Phylax $\to$ Bubblewrap)**:
   - Any mutating operation (file write, block edit, shell execution, package install) requires an explicit capability authorization from Phylax.
   - Phylax classifies the action by Pinax risk tier:
     - `T0` (Read-only, passive): Auto-approved.
     - `T1` (Local workspace edit): Operator or owner approval required.
     - `T2` (Host-mutating command): Requires passkey biometric verification.
     - `T3` (Privileged execution): Requires host confirmation step-up.
   - Approved commands execute inside a restricted Linux Bubblewrap container (`--unshare-net`, `--ro-bind / /`, `--tmpfs /tmp`).

6. **Closed-Loop Retraining & CAS Serving Hot-Swap (Krisis $\to$ Paideia $\to$ Dokime $\to$ Hestia)**:
   - `metis-paideia-retrainer.service` continuously monitors `~/.metis/krisis_active.jsonl`.
   - Once $\ge 10$ new boundary cases accumulate, Paideia's `AttemptRolloutVarianceSampler` stratifies samples into learning frontier ($\sigma^2 > 0$), retention checks, and hard diagnostics.
   - Second-order distributed `ShampooOptimizer` trains a LoRA adapter producing `adapter_model.bin` with SHA-256 verification.
   - Dokime executes paired 10,000 bootstrap resamples on held-out suites ($p < 0.05$).
   - Upon promotion, Paideia invokes Hestia `/manage/swap` with a generation-checked compare-and-swap (CAS) ticket, atomically updating the live serving weights without restarting the process.

---

## 4. Operational Runbook & Systemd Ambient Daemons

Metis manages its runtime via systemd user services. All processes are hardened with `UMask=0077`, `NoNewPrivileges=true`, and `PrivateTmp=true`.

### 4.1 Ambient Background Daemons

To inspect or manage the 5 ambient services:

```bash
# Check status of all daemons
systemctl --user status metis-hestia metis-kybernetes metis-prosopon metis-stoma-watcher metis-paideia-retrainer

# Tail live journal logs
journalctl --user -u metis-kybernetes -f
journalctl --user -u metis-paideia-retrainer -f

# Restart services after code changes
systemctl --user restart metis-kybernetes metis-prosopon
```

### 4.2 Single-Command Stack Diagnostic (`kybernetes doctor`)

Verify dependencies, models, and subsystem readiness in one step:

```bash
./.venv/bin/kybernetes doctor
```

### 4.3 Emergency Remote Panic Lockdown

If an untrusted peer connection or rogue activity is detected, execute panic lockdown via CLI or REST:

```bash
# CLI command
prosopon device lockdown

# REST API equivalent (requires admin token)
curl -X POST http://127.0.0.1:8000/api/devices/lockdown \
  -H "Authorization: Bearer $PROSOPON_HOST_OWNER_TOKEN"
```
*Effect*: Sub-500ms instant closure of all WebRTC SCTP DataChannels, cancellation of all in-flight Kybernetes agent turns, and revocation of non-host device session tokens in `.metis/access.db`.

---

## 5. Verification & Test Suite Gate Summary

All 1,207 tests across the 11 subsystems pass deterministically:

```bash
# Core verification sequence across all modules:
./.venv/bin/pytest dokime/tests -q                                # 218 passed (Dokime)
./.venv/bin/pytest kybernetes/tests -q                            # 216 passed (Kybernetes)
./LTHCM/.venv/bin/pytest LTHCM/tests -q                           # 133 passed (LTHCM)
CUDA_VISIBLE_DEVICES="" ./stoma/.venv/bin/pytest stoma/tests -q   # 127 passed (Stoma)
./.venv/bin/pytest dianoia/tests -q                               # 97 passed (Dianoia)
./hestia/.venv/bin/pytest hestia/tests -q                         # 83 passed, 1 skipped (Hestia)
./prosopon/.venv/bin/pytest prosopon/tests -q                     # 76 passed (Prosopon)
./paideia/.venv/bin/pytest paideia/tests -q                       # 72 passed (Paideia)
./kleroterion/.venv/bin/pytest kleroterion/tests -q               # 43 passed (Kleroterion)
./.venv/bin/pytest krisis/tests -q                                # 40 passed (Krisis)
./.venv/bin/pytest phylax/tests -q                                # 36 passed (Phylax)
./.venv/bin/pytest experiments/text_vs_graph/test_*.py -q         # 27 passed (Experiments)
./.venv/bin/pytest tests -q                                       # 39 passed (Root Integration)
npm --prefix vscode run compile                                   # 0 compile errors (Void IDE)
# Monorepo Total: 1,207 passed, 1 skipped (100% Green)
```
