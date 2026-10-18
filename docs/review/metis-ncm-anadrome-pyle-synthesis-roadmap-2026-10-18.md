# Metis Ecosystem Architecture Review & Implementation Roadmap
**Documents Reviewed:**
1. `metis-ncm-schedion-integration-spec-2026-10-18-a3f41c7d-2aa95bb2.txt` (Schedion)
2. `lthcm-ncm-symbolon-integration-spec-2026-10-18-c51d8a62-8951392a.txt` (Symbolon)
3. `dioptra-ncm-anabathmos-integration-spec-2026-10-18-b7e29d04-13b8d9f4.txt` (Anabathmos)
4. `metis-anadrome-walk-replay-experiment-spec-aa69f396.txt` (Anadrome)
5. `mss-pyle-remote-access-devices-spec-c949cfb0.txt` (Pyle)

**Review Date:** 2026-10-18  
**Repository Baseline:** `gitlab.com/akarta-315h/metis` @ `eb6920e` (main)  
**Hardware Baseline:** Single GPU Node (NVIDIA RTX 5060 Ti 16GB, local Linux host)

---

## 1. Executive Summary & Cross-Spec Synthesis

The five reviewed specifications represent five distinct evolutionary vectors for the Metis ecosystem:
- **Pyle** upgrades the Metis Sovereign Studio (MSS) access perimeter from a fragile loopback/LAN prototype to a multi-route, device-scoped, cryptographically audited remote workbench.
- **Anadrome** introduces offline walk-trace replay policy optimization for Dianoia thought-graph walks, avoiding expensive RL/GRPO training while enforcing zero outcome imputation.
- **Schedion** tests whether coarse-to-fine discrete concept codes (Next-Concept Modeling, Hazy Research) can serve as a cheap multi-candidate drafting engine feeding Krisis answer selection.
- **Symbolon** tests whether NCM's multiscale discrete codes provide an auxiliary candidate retrieval signal for LTHCM memory.
- **Anabathmos** integrates NCM into Dioptra as a mechanistic interpretability testbed for hierarchical representations and level-forcing interventions.

### Cross-System Synthesis Matrix

| Spec / Project | Core Question | Owning Subsystem | Primary Risk / Challenge | Feasibility Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Pyle** | How to securely access MSS from phone/laptop with device identity and least privilege? | `prosopon` (UI/auth), `phylax` (gates), `kybernetes` (attribution) | Loopback exemption bypass via SSH/proxies (R4), unauthenticated WebSocket (R5), global admin pairing (R1). | **Production Ready** (Immediate implementation recommended). |
| **Anadrome** | Can Dianoia walk policy be improved via offline replay trees without model training? | `dianoia` (walks), `dokime` (bootstrap CI), `kleroterion` | Combinatorial blowup of intermediate answer-probes & shadow expansions; observation hashing drift. | **High Value Research** (Phase-gated bring-up). |
| **Schedion** | Can NCM parallel generation produce useful draft summaries for Krisis? | `schedion/` (new), `krisis` (evaluation) | Summaries lack claims/citations (fail closed in Krisis); fixed 512-token input window; GPU OOM. | **Conditional Experiment** (Default outcome is an archive note). |
| **Symbolon** | Can discrete codes act as matchable tokens for LTHCM candidate retrieval? | `symbolon/` (new), `LTHCM` (retrieval) | NCM context window is only 48 tokens; code collisions across multi-window passages; loss of contradictions. | **Conditional Experiment** (Offline study first; no retriever edits). |
| **Anabathmos** | Where do facts emerge in NCM's hierarchy, and can coarse levels steer generation? | `dioptra` (external repo only) | Non-HuggingFace custom PyTorch modules require manual hook mapping; Dioptra isolation. | **Standalone Lab Research** (Dioptra-internal only). |

---

## 2. Detailed Critical Review & Gap Analysis

### 2.1 Schedion (Metis x NCM Candidate Engine)
**Specification:** `metis-ncm-schedion-integration-spec-2026-10-18-a3f41c7d-2aa95bb2.txt`

#### Strengths
- **Rigorous Grounding**: Pinning upstream NCM (`c702742`) and requiring zero modifications to the NCM tree avoids upstream maintenance drag.
- **Honest Scope Boundary**: Restricts Schedion strictly to XSum-class summarization; explicitly forbids treating it as a general QA or chat engine.
- **Strict Measurement Baseline**: Uses Dokime paired bootstrap CIs against a matched Hestia baseline rather than reusing NCM paper figures.

#### Gaps & Vulnerabilities
1. **The Krisis Contract Deadlock (Critical Gap)**:
   - Krisis's source-verified contract (`krisis/krisis/contracts/candidate.py`) mandates structured `claims` with verdicts and `cited_passage_ids`.
   - A raw string from NCM's VQ-VAE decoder cannot satisfy this contract. If Schedion emits empty claims or fake citations, Krisis's eligibility gates fail closed (100% rejection). If Schedion uses an LLM to extract claims from the NCM summary, the compute savings of NCM are entirely eliminated.
   - *Improvement*: Define a deterministic sentence-boundary claim mapper for Schedion in Stage 1: each sentence in the decoded summary is treated as an unverified candidate claim anchored directly to the 512-token source span. In Stage 2, document the exact gate rejection distribution as a baseline characteristic.
2. **GPU VRAM Contention on RTX 5060 Ti**:
   - The RTX 5060 Ti has 16GB VRAM. If Hestia has a 26B/32B quantized model loaded (consuming 12–14GB), loading NCM + GPT-2-large perplexity scorer will trigger immediate CUDA OOM.
   - *Improvement*: Introduce an explicit Hestia swap gate: Schedion must invoke Hestia's `/v1/models/unload` endpoint or verify $\ge 8\text{ GB}$ free VRAM before allocating CUDA tensors.
3. **512-Token Input Window Truncation**:
   - NCM's XSum model truncates the conditioned document to exactly 512 raw GPT-2 tokens. Real Metis documents are much longer.
   - *Improvement*: Pre-filter documents using Stoma's relevance scoring so the conditioned 512-token span contains the highest-density information rather than an arbitrary file head.

---

### 2.2 Symbolon (LTHCM x NCM Auxiliary Retrieval)
**Specification:** `lthcm-ncm-symbolon-integration-spec-2026-10-18-c51d8a62-8951392a.txt`

#### Strengths
- **Retriever Preservation**: Explicitly respects that `LTHCMRetriever.search` currently has no candidate-restriction argument; conducts E2 via offline candidate capture and fusion without touching LTHCM core code.
- **Code Identity Discipline**: Requires discrete code integers to be scoped by `(checkpoint, tokenizer, level, pos)`.
- **Contradiction Protection (E2d)**: Specifically measures whether discrete code filtering drops contradictory evidence needed by cellular sheaf Laplacians.

#### Gaps & Vulnerabilities
1. **Window-to-Passage Granularity Mismatch**:
   - Stoma passages are bounded at $\le 254$ tokens. NCM's XSum context window is 48 tokens.
   - A single passage generates 5 to 6 discrete windows. If the position scoping rule only includes position within the 48-token window, code collisions between Window 1 and Window 4 will generate false similarity signals.
   - *Improvement*: Expand the code tuple identity to explicitly include the passage-relative window index: `(checkpoint_id, tokenizer_id, level, window_idx, pos, code_int)`.
2. **Dense Metric Incompatibility**:
   - Counting raw discrete code overlaps between queries and passages operates like a localized bag-of-words. LTHCM's primary power is topological/hyperbolic Poincaré taxonomy and Lorentzian light-cone attention. Code overlap may show near-zero correlation with hyperbolic distance.
   - *Improvement*: Add a preliminary diagnostic in E1 measuring rank correlation (Spearman's $\rho$) between NCM code overlap count and LTHCM POVM density operator inner products across 1,000 document pairs.

---

### 2.3 Anabathmos (Dioptra x NCM Mechanistic Interpretability)
**Specification:** `dioptra-ncm-anabathmos-integration-spec-2026-10-18-b7e29d04-13b8d9f4.txt`

#### Strengths
- **Clean Repo Boundary**: Confines all work to Dioptra (`gitlab.com/akarta-315h/dioptra`); strictly zero imports or tasks in Metis.
- **Natural Conceptual Fit**: NCM's coarse-to-fine hierarchy is natively aligned with Dioptra's multi-scale representation analysis.
- **Bit-Exactness Check for A4**: Demands that un-intervened runs of the patched generation loop match the released upstream path bit-for-bit.

#### Gaps & Vulnerabilities
1. **Non-Standard PyTorch Hook Ingestion**:
   - Dioptra assumes TransformerLens or canonical site-ID conventions. NCM uses standalone `Block` implementations in `models/common.py`.
   - *Improvement*: Construct a formal `NCMSiteMap` adapter in Dioptra's lab layer mapping `transformer.blocks.{i}.attn` and `transformer.blocks.{i}.mlp` to Dioptra's canonical site taxonomy before attempting automated probe runs.
2. **Single-Window Context Bound on Interpretability**:
   - `reconstruct_multiscales` operates on only 48 tokens. Evaluating complex reasoning or entity substitutions across multi-sentence paragraphs will require stitched multi-window decodes.
   - *Improvement*: Add a multi-window decode visualizer in A1 that stitches contiguous 48-token windows into unified paragraph comparisons.

---

### 2.4 Anadrome (Offline Walk-Trace Replay Policy Optimization)
**Specification:** `metis-anadrome-walk-replay-experiment-spec-aa69f396.txt`

#### Strengths
- **Training-Free Policy Improvement**: Exploits recorded decision trees instead of fragile, compute-heavy RL/GRPO loops.
- **Zero Imputation Invariant**: Unsupported transitions are strictly reported as unsupported, never fabricated or smoothed.
- **Baseline Sanity Check (4.3)**: Failing closed if replaying the baseline does not reproduce recorded traces with 100% fidelity guarantees harness correctness.
- **Cluster Bootstrap Statistical Discipline**: Evaluates policies via 10,000-resample paired bootstrap CIs clustered by question family in Dokime.

#### Gaps & Vulnerabilities
1. **Combinatorial Cost of Intermediate Probes & Shadows**:
   - Generating an answer probe at *every* visited node and spawning shadow branches on alternative EXPAND actions will increase logging runtimes by $5\times$ to $10\times$.
   - On a local 5060 Ti, logging 100 walks could require $>12$ hours of continuous GPU inference.
   - *Improvement*: 
     - Restrict answer probes to states with path depth $\ge 2$.
     - Bound shadow expansions to exactly 1 branch per walk with maximum depth $\le 2$.
     - Implement an in-process SQLite cache of answer-probe generations keyed by `(state_id_hash, model_id, prompt_hash)`.
2. **State Equivalence Representation Drift**:
   - Observation equivalence requires hashing visited nodes, remaining budget, and eligible actions. Floating-point logit variations or non-deterministic dictionary key orderings can cause identical graph states to fail hash matching.
   - *Improvement*: Use RFC 8785 Canonical JSON (JCS) with fixed float precision (`round(score, 4)`) to compute the immutable `observation_hash`.
3. **Policy Execution Sandboxing**:
   - Section 4.8 states candidate policy code is untrusted.
   - *Improvement*: Run policy selection functions in an in-process AST-restricted sandbox (reusing Kybernetes's `SafeCodeMode` AST validator) with a 50ms execution timeout per decision.

---

### 2.5 Pyle (MSS Remote Access, Device Identity & Management)
**Specification:** `mss-pyle-remote-access-devices-spec-c949cfb0.txt`

#### Strengths
- **Exhaustive Threat Modeling**: Addresses the 10 real vulnerabilities identified in existing Metis code (R1–R10).
- **Zero Cloud Reliance**: Local-first architecture requiring zero external relays or accounts; treats Tailscale as an optional, unbundled transport.
- **Strict Boundary Ownership**: Prosopon owns the perimeter; Kybernetes and Phylax remain on loopback and are protected by service tokens and device attribution.
- **Audience-Bound Multi-Route Enrolment**: Prevents session token reuse across different origins (LAN vs Tailnet vs SSH).

#### Gaps & Vulnerabilities
1. **Loopback Exemption Vulnerability (R4 & R5 Urgent Priority)**:
   - In current code (`prosopon/prosopon/server.py`), loopback callers are exempt from authentication when not in LAN mode, and `/api/auth/local-token` grants admin access to any loopback connection. An SSH tunnel (`ssh -L 8000:localhost:8000`) or reverse proxy delivers remote traffic with `request.client.host == "127.0.0.1"`, completely bypassing auth!
   - Furthermore, `/api/lochos/ws` accepts WebSocket connections without an auth handshake.
   - *Improvement*: Phase 0 must be implemented immediately: eliminate the loopback exemption, require a host-owner credential file (`mode 0600`) for local bootstrap via `prosopon open`, and enforce token verification before `websocket.accept()`.
2. **In-Flight Agent Execution During Revocation**:
   - Revoking a device drops its WebSocket connection, but if the device launched a multi-step `TerminalCodingAgent` run, the background execution might continue executing shell commands.
   - *Improvement*: Store the active `agent_run_id` in the device session table. When a device is revoked or a panic lockdown is triggered, Prosopon must issue a cancellation signal to `kybernetes` to terminate the active agent task before the next tool dispatch.
3. **Cross-Origin Browser Cookie Siloing**:
   - Browsers sandboxing cookies to origins mean that a phone switching from home Wi-Fi (LAN origin) to cellular data (Tailnet origin) loses its session.
   - *Improvement*: Provide a seamless "Enrol Additional Route" flow: an authenticated session on Route A can mint a short-lived (60s) single-use transfer ticket that automatically redeems into an audience-bound session when opening Route B.

---

## 3. Unified 5-Track Implementation Roadmap

To avoid GPU resource contention and maintain clean architectural boundaries, implementation is sequenced across 5 parallel/staged tracks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TRACK A: MSS Security & Remote Perimeter (Pyle)                             │
│ Phase 0: Loopback Closure ➔ Phase 1: SQLite Registry ➔ Phase 2: Routes      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ (CPU / Web only; zero GPU contention)
┌─────────────────────────────────────▼───────────────────────────────────────┐
│ TRACK B: Offline Cognitive Replay (Anadrome)                                │
│ Phase 0: Variance Floor ➔ Phase 1: Logging Trees ➔ Phase 2: Zoo Replay      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ (GPU sequential scheduling)
┌─────────────────────────────────────▼───────────────────────────────────────┐
│ TRACK C & D: NCM Auxiliary Exploration (Schedion & Symbolon)                │
│ Stage 0: 5060 Ti Bring-up ➔ Stage 1: Adapters ➔ Stage 2: Offline Eval       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ (External repository)
┌─────────────────────────────────────▼───────────────────────────────────────┐
│ TRACK E: Mechanistic Interpretability (Dioptra Anabathmos)                  │
│ Stage 0: Bring-up ➔ Stage 1: Inspector A1 ➔ Stage 2: Interventions A2-A4   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 0: Baseline Security Hardening & Hardware Gates

#### Milestone 0.1: Pyle Perimeter Lockdown (Track A)
- **Objective**: Fix critical vulnerabilities R1, R4, R5 before exposing any network listener.
- **Actions**:
  1. Remove loopback exemption in `prosopon/prosopon/server.py`. Remove `/api/auth/local-token`.
  2. Implement host-owner bootstrap credential file (`.metis/host_owner.token`, `mode 0600`) generated at server startup.
  3. Add `prosopon open` CLI command that reads the token, mints a 60-second single-use local session, and launches the default browser.
  4. Secure `/api/lochos/ws` by requiring a valid single-use WebSocket ticket prior to `websocket.accept()`.
  5. Default all new device pairings to `Companion` role (least privilege), not `Admin`.
- **Gate Test**: Tests T1, T2, T3, T13 pass. Unauthorized loopback/SSH requests return HTTP 401/403.

#### Milestone 0.2: NCM Hardware & Memory Validation (Tracks C, D, E)
- **Objective**: Validate whether NCM checkpoints execute on the RTX 5060 Ti 16GB without OOM.
- **Actions**:
  1. Clone pinned NCM checkout (`c702742`) to scratch directory.
  2. Fetch `XSum` tokenizer (VQ-VAE) and generator weights.
  3. Run `evaluate.py --n 5` on RTX 5060 Ti; record peak VRAM and wall-clock latency.
  4. Measure VQ-VAE decode overhead for 48-token context windows.
- **Gate Gate G-m0a / G-s0**: Peak VRAM must be $\le 6.5\text{ GB}$. If VRAM $> 10\text{ GB}$, NCM cannot run concurrently with Hestia; document sequential unloading requirement.

#### Milestone 0.3: Anadrome Noise Floor Calibration (Track B)
- **Objective**: Determine outcome variance and logging overhead of Dianoia thought walks.
- **Actions**:
  1. Run 10 identical baseline walks over 5 standard questions using Hestia local model.
  2. Measure step-to-step variance, failure rates, and wall-clock time per turn.
  3. Prototype intermediate answer-probes and measure token overhead. Set shadow expansion depth bounds ($\text{depth} \le 2$).
- **Gate**: Establish minimum detectable effect size for 95% paired bootstrap CIs.

---

### Phase 1: Core Storage Engines & Offline Harnesses

#### Milestone 1.1: Pyle Device Registry & Web Management (Track A)
- **Actions**:
  1. Create SQLite store `prosopon/data/access.db` (tables: `devices`, `device_sessions`, `pairing_links`, `routes`, `access_audit`).
  2. Implement SHA-256 hash-chained access audit logger.
  3. Build QR code and pairing link generator with 8-digit fallback code.
  4. Implement `Manage Devices` window in Prosopon frontend (`index.html`, `settings.js`) with tabs: *Devices*, *Pending Links*, *Routes*, *Activity*.
  5. Add CLI suite: `prosopon device list`, `prosopon device pair`, `prosopon device revoke`, `prosopon device lockdown`.
- **Gate Test**: Tests T4–T12, F1, F2a pass. Devices can be created, inspected, and revoked in real time.

#### Milestone 1.2: Anadrome Decision-Tree Logging Harness (Track B)
- **Actions**:
  1. Extend Dianoia walk logger to capture full continuation trees:
     - Environment identity (model SHA, prompt template, index hash).
     - Eligible action set (all candidate EXPAND nodes, BACKTRACK, ANSWER).
     - RFC 8785 canonical observation hash.
     - Answer-probe outcomes at visited nodes (depth $\ge 2$).
     - Single shadow branch continuation (depth $\le 2$).
  2. Store trees in append-only JSONL format (`~/.dianoia/replay_trees.jsonl`).
- **Gate Test**: Baseline Sanity Gate (4.3): Replaying the baseline policy over recorded trees reproduces original traces bit-for-bit with 100.0% agreement.

#### Milestone 1.3: Schedion & Symbolon Offline Adapters (Tracks C & D)
- **Actions**:
  1. Create isolated `schedion/` adapter wrapping NCM generation path. Implement deterministic sentence-boundary claim mapper for Krisis.
  2. Create isolated `symbolon/` adapter wrapping VQ-VAE tokenizer. Implement multi-window code extraction with passage-relative window indexes.
  3. Execute Symbolon E1: evaluate reconstruction exact-preservation rates across probe texts.
- **Gate**: Symbolon E1 diagnostic report recorded; Schedion emits validated `Candidate` records without throwing schema validation errors.

---

### Phase 2: Controlled Experiments & Multi-Route Transport

#### Milestone 2.1: Pyle Multi-Route Hardening & Audience Binding (Track A)
- **Actions**:
  1. Implement Route Manager supporting LAN (`0.0.0.0` or private IP), SSH tunnel instructions, and optional `tailscale serve`.
  2. Enforce strict session audience binding: session tokens presented on a different route origin than issued are rejected.
  3. Implement "Enrol Additional Route" flow with single-use 60s transfer tickets.
  4. Auto-manage `allowed_hosts` and CORS headers dynamically from enabled routes.
- **Gate Test**: Tests T14, T15, F4', F5 pass. Cross-origin session reuse is rejected.

#### Milestone 2.2: Anadrome Policy Zoo & Offline Replay (Track B)
- **Actions**:
  1. Implement candidate walk policy zoo in Dianoia:
     - Policy 1: Greedy similarity EXPAND (highest embedding similarity first).
     - Policy 2: High-entropy stochastic selection (Kleroterion scaled logits).
     - Policy 3: Early-exit answer threshold (trigger ANSWER if candidate support $> \tau$).
     - Policy 4: Sheaf contradiction backtrack (force BACKTRACK immediately on detected conflict).
  2. Execute replay harness over Phase 1 logged decision trees. Filter questions to mutually supported episodes (coverage $\ge 80\%$).
  3. Evaluate candidate policies using Dokime 10k paired bootstrap CIs.
- **Gate**: Only policies demonstrating statistically significant improvements over baseline without increasing hard errors advance to Phase 3.

#### Milestone 2.3: Schedion & Symbolon Controlled Studies (Tracks C & D)
- **Actions**:
  1. **Schedion Stage 3**: Controlled comparison between Hestia summary candidates vs Schedion candidates over identical 512-token spans. Record Krisis gate pass/fail distributions, factual grounding scores, and wall-clock times.
  2. **Symbolon E2**: Offline candidate capture and fusion over SciFact/HotpotQA benchmarks:
     - Arm (c1): Uncapped additive union.
     - Arm (c2): Capped fusion at matched top-$k$.
     - Arm (d): Contradiction evidence retention check.
- **Gate**: Dokime evaluation manifests generated for Schedion and Symbolon. Schedion and Symbolon remain disabled by default unless statistically superior.

---

### Phase 3: Production Integration & Promotion Decisions

#### Milestone 3.1: Pyle Remote Operations & Panic Lockdown (Track A)
- **Actions**:
  1. Enable Operator preset with remote approval rights for T1/T2 actions; enforce host confirmation step-up for T3 actions.
  2. Implement active agent task termination on device revocation.
  3. Implement one-click / one-command Panic Lockdown (`prosopon device lockdown`).
  4. Add doctor diagnostics: check TLS state, file permissions on `access.db`, unauthenticated endpoints, and legacy token migration.
- **Gate**: All 16 security tests (T1–T16) and functional tests (F1–F7) pass. Pyle promoted to default remote access architecture in MSS.

#### Milestone 3.2: Anadrome Frozen Live Walk Test (Track B)
- **Actions**:
  1. Freeze test dataset questions and pre-declare random seeds.
  2. Run fresh, live Dianoia walks using promoted policy candidate(s) versus baseline controller.
  3. Compute final paired bootstrap CIs via Dokime.
  4. Decision Gate: If promoted policy achieves higher terminal correctness at equal or lower cost, author candidate policy into `dianoia/walk.py` as an approved controller option.

#### Milestone 3.3: Schedion / Symbolon Archival or Admission (Tracks C & D)
- **Actions**:
  1. If Schedion shows measured quality/speed advantage under Krisis gating, register as an optional candidate source in Kybernetes. Otherwise, archive the benchmark report in `docs/review/` and delete temporary model weights.
  2. If Symbolon shows higher benchmark recall without degrading contradiction coverage, implement `FilteredRetrievalAdapter` in LTHCM. Otherwise, publish negative research note and archive.

---

### Phase 4: Dioptra Anabathmos Lab Research (Track E)

*Executed independently inside `dioptra` repository*:
1. **A1 Level-Trajectory Inspector**: CLI tool decoding multiscale discrete codes across context windows into unified experiment records.
2. **A2 Fact-Appearance Localization**: Measure at which discrete hierarchy level specific named entities, dates, and numbers first stabilize.
3. **A3 Coarse Level-0 Interventions**: Force coarse level-0 codes from Document A while conditioning on Document B; observe semantic transfer.
4. **A4 Mid-Hierarchy Resampling**: Patch NCM generation loop to freeze levels $0..k$ and resample finer details; verify bit-exactness on un-intervened runs.

---

## 4. Hardware Scheduling & Isolation Matrix

To ensure stability on the local development environment:

| Track / Job | Compute Device | VRAM Requirement | Scheduling Rule |
| :--- | :--- | :--- | :--- |
| **Track A (Pyle)** | Host CPU / SQLite | 0 MB VRAM | Runs concurrently at all times; no GPU lock. |
| **Track B (Anadrome)** | RTX 5060 Ti | 12–14 GB (Hestia LLM) | Runs during Dianoia deliberation; requires Hestia active. |
| **Track C (Schedion)** | RTX 5060 Ti | 4–6 GB (NCM VQ-VAE) | **Exclusive Run**: Hestia must be idle or unloaded. |
| **Track D (Symbolon)** | RTX 5060 Ti | 2–3 GB (NCM VQ-VAE) | Can run on CPU or when Hestia is idle. |
| **Track E (Anabathmos)**| RTX 5060 Ti | 4–6 GB (NCM PyTorch) | Runs in isolated Dioptra environment; Hestia unloaded. |

---

## 5. Decision Summary & Recommended Next Actions

1. **Immediate Execution: Pyle Phase 0 (Security Fixes)**
   - Eliminate the loopback exemption in `prosopon`.
   - Remove `/api/auth/local-token`.
   - Authenticate `/api/lochos/ws`.
   - Protect local access via `host_owner.token` and `prosopon open`.
2. **Author Anadrome Decision-Tree Logger**
   - Implement Phase 0 variance test and Phase 1 continuation tree recording in `dianoia`.
3. **NCM Standalone Sanity Gate**
   - Download NCM XSum checkpoint to an isolated scratch folder and run the 5060 Ti VRAM bring-up test before writing any Metis integration code.
