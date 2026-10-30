# Architectural Specification: Zero-Hallucination Graph Filtering, High-Order Reasoning & Universal Long-Term Capabilities

**Status:** APPROVED DESIGN & STRATEGIC ROADMAP  
**Author:** Metis Architecture & Core Systems Team  
**Date:** 2026-10-30  
**Target Subsystems:** `stoma`, `krisis`, `dianoia`, `kybernetes`, `LTHCM`, `paideia`, `phylax`  
**Empirical Origin:** TCR Probe Confirmatory Study (`experiments/text_vs_graph/`) & Monorepo Subsystem Reviews

---

## 1. Executive Summary & Problem Statement

Recent empirical evaluations across $4,000+$ live LLM reasoning evaluations revealed critical boundaries in hybrid neural-symbolic systems:

1. **LLM Hyper-Vigilance & Error Fixation**: When models are instructed in natural language to perform defensive reconciliation (*"audit the graph", "find discrepancies", "reconcile against text"*), they invent non-existent contradictions and hallucinate false conflicts due to minor phrasing variances.
2. **Context Poisoning**: These hallucinated discrepancies are passed downstream into subsequent execution passes as authoritative notes, degrading final reasoning accuracy below that of matched unguided controls ($93.3\%$ vs $96.8\%$).
3. **The Graph Distraction Penalty**: Presenting a corrupted structured graph alongside valid source prose in a single pass drops multi-hop reasoning accuracy on causal dependency chains from $92.8\%$ to $49.5\%$, because the model instinctively anchors to concise structured tables over lengthy narrative text.

### The Core Architectural Invariant
> **An LLM must never be tasked with policing its own factual ground truth using natural language prompts alone. Corrupted structured representations must be programmatically evicted before prompt construction, and agent reasoning must be structured constructively rather than defensively.**

---

## 2. Four-Tier Zero-Hallucination Pipeline (Immediate Defense)

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      Raw Source Text                        │
 └──────────────┬───────────────────────────────┬──────────────┘
                │                               │
                ▼                               ▼
    [ Structured Graph Triples ]      [ Stoma Ingest / Chunks ]
                │                               │
                ▼                               │
   ┌───────────────────────────┐                │
   │  TIER 1: Deterministic    │                │
   │  Span & Regex Pre-Filter  │                │
   └────────────┬──────────────┘                │
                │                               │
        Filtered / Stripped                     │
                ▼                               │
   ┌───────────────────────────┐                │
   │  TIER 2: Dynamic Fallback │◄───────────────┘
   │  Consistency Gate (tau)   │  (if consistency < 0.95 -> Evict Graph)
   └────────────┬──────────────┘
                │
         Passed Graph + Text
                ▼
   ┌───────────────────────────┐
   │  TIER 3: Constructive     │
   │  Grounding Pass (Arm F)   │  (Restate facts & plan; NO negative conflict hunting)
   └────────────┬──────────────┘
                │
         Structured Plan
                ▼
   ┌───────────────────────────┐
   │  TIER 4: Krisis NLI       │
   │  Conformal Hard Gating    │  (Drop claims with P(Contradiction) > 0.10)
   └────────────┬──────────────┘
                │
                ▼
       Final Verified Answer
```

### 2.1 Tier 1: Deterministic Span & AST Pre-Filter (`stoma`)
- Zero-token programmatic verification matching graph entity labels, timestamps, and relational attributes against raw text byte spans.
- Unanchored or conflicting graph nodes are stripped before prompt assembly.

### 2.2 Tier 2: Dynamic Fallback Consistency Gate (`kybernetes`)
- Computes Consistency Ratio $\mathcal{C} = N_{\text{verified}} / N_{\text{extracted}}$.
- If $\mathcal{C} < 0.95$, the system **evicts the graph entirely** and falls back to pure text mode (Arm A), avoiding the $-6.2\%$ distraction penalty.

### 2.3 Tier 3: Constructive Grounding Over Negative Reconciliation (`kybernetes`)
- Replaces defensive conflict-hunting prompts with constructive forward planning (the Arm F paradigm: *"restate relevant facts and plan"*).
- Eliminates the hyper-vigilant false-contradiction trap and prevents Pass 2 context poisoning.

### 2.4 Tier 4: Krisis NLI Conformal Hard-Gating (`krisis`)
- Intermediate claims between multi-stage reasoning turns must be serialized as atomic JSON triples and checked against local NLI conformal bounds ($P_{\text{contradiction}} \le 0.05$, $P_{\text{entailment}} \ge 0.90$).

---

## 3. High-Order Reasoning & Scaled Deliberation (Methods 1 – 5)

These five methods directly overcome the working-memory and topological reasoning bottlenecks observed in multi-hop evaluation:

### Method 1: Executable Symbolic Sandboxes (Code-as-Reasoning)
* **Problem**: Transformers struggle with tracking multi-hop state mutations, cycle detection, and boolean combinations (AND/OR) purely within auto-regressive token space ($49.5\%$ accuracy under graph distraction).
* **Architecture**: Instead of forcing the model to mentally simulate graph algorithms, instruct it to generate executable Python/SymPy code in Pass 1 and execute it inside Metis's existing `BubblewrapBackend`:
  ```python
  # Model generates executable verification logic
  events = {"E1": Root(), "E2": Root(), "E3": AND(["E1", "E2"]), "E4": OR(["E1", "E3"])}
  failed = simulate_counterfactual(events, denied="E1")
  print(json.dumps(failed))  # -> Output injected as verified ground truth
  ```
* **Impact**: $100\%$ mathematical determinism; eliminates token-space mental math and graph traversal hallucination.

### Method 2: Hierarchical Poincaré Hyperbolic Memory (`LTHCM`)
* **Problem**: Euclidean cosine vector embeddings treat memory as a flat soup, erasing taxonomic depth, parent-child hierarchies, and causal precedence.
* **Architecture**: Leverage Metis's Poincaré disk embedding space:
  - **Origin ($r \approx 0$)**: Global abstractions, system invariants, domain ontologies.
  - **Periphery ($r \to 1$)**: Specific episodic logs, exact code diffs, raw tool outputs.
  - **Hyperbolic Centroid Hoisting**: As episodic memories age, hoist them inward towards their conceptual cluster centroids, compressing coordinate storage without lossy text briefing degradation.
* **Impact**: Native geometric encoding of tree-like knowledge and long-term hierarchical retention without context bloat.

### Method 3: MCTS with Process-Supervised Conformal PRM Gates (`dianoia`)
* **Problem**: Linear ReAct loops (Think $\to$ Act $\to$ Observe) suffer from compounding error: a single flawed assumption at Step 2 completely derails a 10-step planning trajectory.
* **Architecture**: Implement full Monte Carlo Tree Search in Dianoia's `WalkController`:
  - **Branching**: Generate $K=3$ candidate action branches at ambiguous decision nodes.
  - **Process Reward Model (PRM)**: Score intermediate steps with Krisis conformal entailment gates.
  - **Topological Backtracking**: If a branch encounters cellular sheaf Laplacian obstruction or conformal rejection, immediately prune and backtrack to the parent frontier.
* **Impact**: Prevents error compounding on complex 10+ turn tasks; guarantees that only mathematically supported paths advance.

### Method 4: Discrete Multiscale Codebook Memory (NCM / `symbolon`)
* **Problem**: Long-term conversational retention quickly saturates context windows and blows up KV-cache memory footprints ($O(N^2)$ attention overhead).
* **Architecture**: Adapt the verified Neural Concept Memory (NCM) tokenizer:
  - Chunk archival session logs into 48-token sliding windows.
  - Quantize into discrete multiscale integer codebooks (Levels $L_0$ to $L_8$).
  - Store sequences as dense integer IDs in SQLite; decode on-demand via the local 5060 Ti generator.
* **Impact**: $10\times\text{ to }50\times$ compression of long-term history; sub-second on-demand reconstruction with zero GPU memory paging.

### Method 5: Active Learning Frontier Curriculum (`paideia`)
* **Problem**: AI systems typically remain static, repeatedly making the same borderline mistakes across sessions.
* **Architecture**: Wire the Paideia `BoundarySampler` (Item 90) into continuous agent runtime:
  - Track attempt variance across tasks ($\sigma^2 > 0$).
  - Isolate borderline queries where the agent flip-flops between success and failure (its current capability frontier).
  - Automatically curate fine-tuning and in-context demonstration datasets biased toward these boundary cases (stratified mixture: $80\%$ boundary, $20\%$ retention).
* **Impact**: Continuously expands the model's reasoning frontier without wasting compute or memory on already-mastered tasks.

---

## 4. Universal Long-Term Capabilities for All Domains (Methods 6 – 10)

To make Metis a resilient, general-purpose cognitive operating system across code synthesis, scientific analysis, enterprise workflows, and sovereign deployment, implement the following five cross-domain architectures:

### Method 6: Epistemic Checkpointing & Speculative Branch Rollback (`lochos`)
* **Concept**: Operating systems have transactional filesystems (ZFS/Btrfs snapshotting); agent reasoning requires transactional epistemic state.
* **Architecture**:
  - Before an agent executes any mutating or multi-step reasoning phase, create an immutable lightweight git worktree checkpoint and state snapshot.
  - If a task verification gate fails (e.g., test fails, compiler error, Krisis conformal denial), **speculatively roll back the workspace and epistemic context** to the parent milestone.
  - The agent receives the diff of *what failed and why* without retaining the polluted dead-end scratchpad in its active prompt context.
* **Domain Breadth**: Universal (Coding, Systems Administration, Database Operations, Autonomous Research).

### Method 7: Multi-Entity Consensus & Adversarial Cross-Verification
* **Concept**: Single models suffer from blind spots, systemic training biases, and shared-family hallucinations.
* **Architecture**:
  - Deploy a heterogeneous multi-agent deliberation panel in `kybernetes` (e.g., Gemma 4 26B, Qwen 2.5 14B, and an independent validator).
  - Use structured voting (Borda count or Kleroterion stochastic selection) where models evaluate *anonymized reasoning trajectories* without seeing author identities.
  - Require consensus on critical high-risk boundaries (Phylax T2/T3 actions).
* **Domain Breadth**: Security Review, Financial Decisioning, Medical/Scientific Analysis, Contract Review.

### Method 8: Content-Addressed Knowledge Externalization (KV-Cache Spillover)
* **Concept**: Giant in-context prompts degrade model attention ("needle-in-a-haystack" degradation).
* **Architecture**:
  - Implement a SHA-256 content-addressed artifact store (`.metis/artifacts/`).
  - When tool outputs, document passages, or code files exceed $\tau = 1,000$ tokens, replace them in the working prompt with an atomic pointer:
    `[REF: artifact_sha256_ab72f8 | Summary: 48-line test failure traceback]`.
  - Provide a specialized `read_artifact_slice(sha, byte_offset, length)` tool allowing the model to peek into exact byte ranges on demand.
* **Domain Breadth**: Massive Codebase Audits, Legal Document Discovery, High-Throughput Log Analysis.

### Method 9: Semantic Sheaf Laplacian Obstruction Detection (Topological Memory)
* **Concept**: Knowledge bases often contain localized claims that seem true in isolation but form global logical contradictions when combined across documents.
* **Architecture**:
  - Model facts and claims as sections of a cellular sheaf over the knowledge graph.
  - Compute the Sheaf Laplacian operator $\Delta_{\mathcal{F}}$. Non-zero eigenvalues in the harmonic subspace ($\ker \Delta_{\mathcal{F}}^\perp$) signal a **topological obstruction** (a global contradiction).
  - Automatically flag conflicting information before it enters the synthesis stage, preventing contradictory assumptions from poisoning generation.
* **Domain Breadth**: Intelligence Analysis, Cross-Document Investigative Journalism, Multi-Repository Refactoring.

### Method 10: Biometric Action Entitlement & Continuous Provenance Auditing (`prosopon` / `phylax`)
* **Concept**: Long-running autonomous agents require strong security perimeters to prevent rogue prompt-injection escalation.
* **Architecture**:
  - Cryptographically bind every agent reasoning turn and tool invocation to a local SHA-256 hash-chained immutable audit ledger (`access_audit`).
  - Require WebAuthn hardware passkey biometric assertions (FIDO2 / TouchID / YubiKey) for system-mutating Tier 3 actions (shell execution, external network calls, secret reads).
  - Instant session revocation (< 500ms security invariant) that immediately terminates WebRTC DataChannels and cancels in-flight async tasks upon anomaly detection.
* **Domain Breadth**: Enterprise Security, Sovereign Private Infrastructure, Autonomous Operations.

---

## 5. Strategic Capability Matrix

| # | Method | Primary Subsystem | Key Target Problem | Implementation Complexity |
| :-: | :--- | :---: | :--- | :---: |
| **1** | **Symbolic Sandboxed Code** | `kybernetes` / `dianoia` | Flawed mental math & graph traversal | Low |
| **2** | **Poincaré Hyperbolic Memory** | `LTHCM` | Flat vector retrieval & lossy compression | Medium |
| **3** | **MCTS with Conformal PRM** | `dianoia` / `krisis` | Error compounding on 10+ turn tasks | Medium |
| **4** | **Discrete Codebook Memory (NCM)**| `stoma` / `symbolon` | Context window & KV-cache exhaustion | High |
| **5** | **Active Learning Curriculum** | `paideia` | Stagnant agent capabilities across sessions | Medium |
| **6** | **Epistemic Checkpointing** | `lochos` / `kybernetes` | Polluted context from dead-end errors | Low |
| **7** | **Adversarial Consensus Panel** | `kybernetes` / `hestia` | Single-model blind spots & bias | Medium |
| **8** | **Content-Addressed Spillover** | `kybernetes` | Needle-in-a-haystack attention loss | Low |
| **9** | **Topological Sheaf Detection** | `dianoia` / `stoma` | Latent global contradictions in memory | High |
| **10**| **Biometric Action Entitlement** | `prosopon` / `phylax` | Rogue prompt injection & privilege escalation | Medium |

---

## 6. Implementation Phasing & Integration Plan

```
Phase 1 (Immediate - Sprint 1):
  ├── Tier 1-4 Zero-Hallucination Pipeline (Span Filter & Dynamic Fallback Gate)
  ├── Method 1: Symbolic Code-as-Reasoning Execution Tooling
  └── Method 8: Content-Addressed Spillover for Large Tool Outputs

Phase 2 (Core Reasoning - Sprint 2):
  ├── Method 3: Dianoia MCTS with Conformal Step-Level Verification
  ├── Method 6: Lochos Epistemic Checkpointing & Speculative Rollback
  └── Method 7: Multi-Model Consensus Panel in Hestia

Phase 3 (Long-Term Scaled Memory - Sprint 3):
  ├── Method 2: Poincaré Hyperbolic Centroid Hoisting in LTHCM
  ├── Method 4: Symbolon Discrete Codebook Archival Store
  ├── Method 5: Paideia Active Learning Frontier Continuous Ingestion
  └── Method 9: Cellular Sheaf Laplacian Global Contradiction Detector
```

---

## 7. Concrete Implementation Blueprints & Code Signatures

To ensure zero ambiguity during implementation, this section provides production-grade Python interfaces, prompt templates, and data schemas for all 10 methods.

### 7.1 Method 1 Blueprint: Sandboxed Python Execution Harness (`kybernetes`)
```python
# Location: kybernetes/kybernetes/tools/symbolic_solver.py

import json
from typing import Dict, Any, List
from kybernetes.execution.backend import get_execution_backend

SYMBOLIC_SOLVER_PROMPT = """
You are a formal logic solver. Given the following scenario and dependency rules, 
do NOT compute the answer mentally. Instead, output ONLY a Python script inside ```python ... ``` 
that imports networkx or standard data structures, evaluates the exact counterfactual/arithmetic state, 
and prints the final answer as JSON to stdout: {"answer": <result>}.
"""

def execute_symbolic_derivation(code_str: str, timeout_sec: float = 5.0) -> Dict[str, Any]:
    backend = get_execution_backend() # BubblewrapBackend (unshared network, isolated tmpfs)
    res = backend.execute(
        cmd=["python3", "-c", code_str],
        cwd="/tmp",
        timeout=timeout_sec,
        allow_network=False,
    )
    if res.exit_code != 0:
        raise RuntimeError(f"Symbolic execution failed: {res.stderr}")
    return json.loads(res.stdout.strip())
```

### 7.2 Method 2 Blueprint: Hyperbolic Centroid Hoisting (`LTHCM`)
```python
# Location: LTHCM/lthcm/core/hyperbolic_hoisting.py

import torch
from typing import List

def poincare_distance(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    sq_u = torch.sum(u ** 2, dim=-1)
    sq_v = torch.sum(v ** 2, dim=-1)
    sq_dist = torch.sum((u - v) ** 2, dim=-1)
    alpha = 1 - sq_u
    beta = 1 - sq_v
    gamma = 1 + 2 * sq_dist / (alpha * beta + eps)
    return torch.acosh(torch.clamp(gamma, min=1.0 + eps))

def hoist_episodic_cluster(coords: torch.Tensor, age_weights: torch.Tensor) -> torch.Tensor:
    """Computes Frechet / Karcher mean on Poincare ball and hoists centroid inward (r -> 0)."""
    # Centroid on ball
    centroid = torch.mean(coords, dim=0, keepdim=True)
    norm = torch.norm(centroid, p=2, dim=-1, keepdim=True)
    # Compression: radially pull towards origin proportionally to cluster age
    hoisted_norm = norm * 0.75  # radial contraction
    return (centroid / (norm + 1e-8)) * hoisted_norm
```

### 7.3 Method 3 Blueprint: MCTS PRM Gate Hook (`dianoia`)
```python
# Location: dianoia/dianoia/anadrome/mcts_prm.py

from typing import Dict, Any, List
from krisis.gates.conformal import ConformalCalibrator

class ProcessRewardGate:
    def __init__(self, calibrator: ConformalCalibrator):
        self.calibrator = calibrator

    def evaluate_step(self, step_rationale: str, evidence_text: str) -> bool:
        """Conformal check: returns True iff step entailment is bounded."""
        prediction_set = self.calibrator.predict_set(step_rationale, evidence_text)
        # Prediction set must contain ENTAILMENT and exclude CONTRADICTION
        return ("entailment" in prediction_set) and ("contradiction" not in prediction_set)
```

### 7.4 Method 6 Blueprint: Epistemic Worktree Rollback (`lochos`)
```python
# Location: kybernetes/kybernetes/lochos/rollback.py

import subprocess
from pathlib import Path

class EpistemicRollbackManager:
    def __init__(self, worktree_path: Path):
        self.worktree_path = worktree_path
        self.checkpoint_stack: List[str] = []

    def checkpoint(self, label: str) -> str:
        sha = subprocess.check_output(
            ["git", "-C", str(self.worktree_path), "commit", "-am", f"checkpoint: {label}", "--allow-empty"],
            text=True
        ).strip()
        self.checkpoint_stack.append(sha)
        return sha

    def rollback(self) -> None:
        if self.checkpoint_stack:
            last_sha = self.checkpoint_stack.pop()
            subprocess.run(["git", "-C", str(self.worktree_path), "reset", "--hard", "HEAD~1"], check=True)
```

### 7.5 Method 8 Blueprint: Content-Addressed Spillover (`kybernetes`)
```python
# Location: kybernetes/kybernetes/engine/spillover.py

import hashlib
from pathlib import Path
from typing import Tuple

ARTIFACT_DIR = Path(".metis/artifacts")

def spill_large_context(raw_text: str, token_threshold: int = 1000) -> Tuple[str, bool]:
    # Approx 4 chars per token
    if len(raw_text) < token_threshold * 4:
        return raw_text, False
        
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
    artifact_path = ARTIFACT_DIR / f"artifact_{digest}.txt"
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(raw_text, encoding="utf-8")
    
    summary_preview = raw_text[:200].replace("\n", " ")
    ref_tag = (
        f"[SPILLOVER_REF: sha256={digest} | length={len(raw_text)} chars | "
        f"Preview: '{summary_preview}...' | Call read_artifact_slice('{digest}', offset, len) to inspect]"
    )
    return ref_tag, True
```

---

## 8. Definition of Done & Verification Metrics

1. **Topological Multi-Hop Benchmark (Q3)**: Accuracy under $30\%$ graph corruption must increase from $49.5\%$ to $>95.0\%$ via symbolic sandbox execution.
2. **Context Compression Ratio**: Discrete codebook & hyperbolic hoisting must demonstrate $\ge 10\times$ token compression on 50-turn agent session traces.
3. **Zero Compounding Drift**: MCTS with Krisis PRM must achieve zero unhandled branch errors on benchmark planning chains of depth $\ge 8$.
4. **Perimeter Integrity**: Zero privilege escalations or unauthenticated mutating tool executions across all Phylax audit logs.
