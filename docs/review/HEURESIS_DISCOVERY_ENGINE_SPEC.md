# Architectural Specification: Heuresis Autonomous Scientific Discovery Engine

**Status:** APPROVED ARCHITECTURAL SPECIFICATION  
**Author:** Metis Architecture & Core Science Team  
**Date:** 2026-10-30  
**Target Subsystems:** `heuresis` (discovery engine), `stoma` (peer-review ingestion), `LTHCM` (concept geometry), `dianoia` (sheaf obstruction), `krisis` (conformal gating), `dokime` (temporal evaluation)  
**Origin:** Inspired by "The Alien Space of Science" (arXiv:2603.01092v3) & Metis Topological Research Synthesis

---

## 1. Executive Summary & Objective

Standard language models generate research ideas by sampling from their pretraining distribution. This creates a severe **conceptual mode collapse**: models persistently propose incremental, obvious pairings already prevalent in the literature (e.g., *"LLM + RAG"*, *"Transformers + Graphs"*).

The **Heuresis** subsystem transforms scientific discovery into an **asymmetric, corpus-grounded combinatorial search over discrete conceptual atoms**, combined with **abductive synthesis for genuinely Out-of-Vocabulary (OOV) concept invention**.

### Key Architectural Invariants
1. **Separation of Search and Generation**: Combinatorial exploration over millions of candidate concept combinations is executed using fast, lightweight scoring models ($<15$ minutes on local GPU). LLMs are invoked *only* to write up the top-ranked candidates.
2. **True Out-of-Vocabulary Emergence**: The engine does not merely recombine known words; it uses **Cellular Sheaf Laplacian Obstructions** and **HDBSCAN Outlier Bridging** to abductively synthesize entirely new conceptual primitives.
3. **Open Peer-Review Grounding**: Replaces naive author co-occurrence with rich social-topological data from peer-review forums (OpenReview, ICLR/NeurIPS discussions), capturing real-world community boundaries, reviewer objections, and rejected paper frontiers.

---

## 2. High-Level System Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                         INGESTION & TOPOLOGY ENGINE                         │
 └──────────────────────┬───────────────────────────────┬──────────────────────┘
                        │                               │
                        ▼                               ▼
            [ Open Peer-Review Feeds ]        [ Primary Research Corpus ]
            (OpenReview API, Discussions)     (ArXiv, BioRxiv, Internal Docs)
                        │                               │
                        ▼                               ▼
       ┌────────────────────────────────┐  ┌──────────────────────────────────┐
       │ Peer-Review Boundary Crawler   │  │ Stoma Conceptual Unit Distiller  │
       │ (Disciplinary boundaries,      │  │ (Transforms papers into atomic   │
       │  reviewer objections, debate)  │  │  method statements & invariants) │
       └────────────────┬───────────────┘  └────────────────┬─────────────────┘
                        │                                   │
                        │                                   ▼
                        │                      ┌─────────────────────────┐
                        │                      │ HDBSCAN Density Cluster │
                        │                      └────┬───────────────┬────┘
                        │                           │               │
                        │                    (Centroids)       (Outliers)
                        │                           │               │
                        ▼                           ▼               ▼
        ┌────────────────────────────────┐    ┌───────────┐   ┌───────────────┐
        │ Community Availability Scorer  │    │ Concept   │   │ Outlier       │
        │ (Identifies unbridged fields)  │    │ Atoms     │   │ "Islands"     │
        └────────────────┬───────────────┘    └─────┬─────┘   └───────┬───────┘
                         │                          │                 │
                         └──────────────┬───────────┴─────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Combinatorial Triple Search │
                         │ (Millions scored in <15 min)│
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │ Sheaf Obstruction Evaluator │
                         │ (Laplacian Delta_F != 0)    │
                         └──────────────┬──────────────┘
                                        │
                                        ├─► [ No Obstruction ] ──► Krisis Conformal Gate
                                        │
                                        └─► [ Topological Paradox Detected ]
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │ Abductive Concept Synthesizer│
                                     │ (Coins novel OOV Primitive) │
                                     └──────────────┬──────────────┘
                                                    │
                                                    ▼
                                     ┌─────────────────────────────┐
                                     │ Dokime Forward-in-Time Eval │
                                     └─────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Peer-Review Boundary Crawler (`stoma/crawlers/openreview.py`)
Instead of using static author profiles, Heuresis models community boundaries by scraping peer-review forums:
* **Source**: OpenReview API (`api2.openreview.net/notes`), NeurIPS/ICLR rebuttal tracks.
* **Extraction Schema**:
  ```json
  {
    "forum_id": "openreview_2026_xyz",
    "paper_title": "...",
    "primary_area": "cs.LG",
    "reviewer_profiles": ["quant_finance", "topology", "neuroscience"],
    "objections": [
      "Authors failed to connect with the Riemannian metric literature in differential geometry."
    ],
    "boundary_pair": ["graph_neural_networks", "riemannian_geometry"],
    "decision": "Reject - lack of theoretical bridge"
  }
  ```
* **Availability Score Metric**: A candidate combination $(A, B)$ has **Low Community Availability** if:
  1. No published papers currently contain both $A$ and $B$.
  2. Reviewers have explicitly cited the absence of a bridge between $A$'s field and $B$'s field.
  3. The respective communities publish in disjoint conference venues.

---

### 3.2 Discovering Out-of-Vocabulary (OOV) Concepts

Heuresis implements three distinct mechanisms to escape the vocabulary boundary of its source corpus:

#### A. Cellular Sheaf Laplacian Obstruction Detection (`dianoia/sheaf.py`)
* **Mathematical Basis**: Locally consistent claims can produce a global topological contradiction when glued across restriction maps.
* **Detection**:
  Let $\mathcal{F}$ be the cellular sheaf over the concept graph. Compute the Sheaf Laplacian operator $\Delta_{\mathcal{F}}$.
  $$\text{Obstruction Index } \mathcal{O} = \dim(\ker \Delta_{\mathcal{F}}^\perp)$$
* **Abductive Synthesis Trigger**:
  If $\mathcal{O} > 0$, an irreconcilable paradox exists between Concept $A$ and Concept $B$. The system triggers an abductive inference prompt to the primary reasoning model:
  > *"Concepts $A$ and $B$ are both experimentally validated, but mathematically contradict each other under current assumptions $P_1$ and $P_2$. What latent, unobserved property, dimension, or symmetry $X$ would resolve this contradiction and preserve both invariants?"*
* **Outcome**: The generated entity $X$ is added to LTHCM as a **brand-new conceptual atom**.

#### B. Cross-Domain Operator Isomorphism (Structural Analogy Hoisting)
* **Mechanism**:
  1. Strip domain-specific nouns from method statements (e.g., replace *"electron bandgap"* with *"bipartite potential barrier"*).
  2. Index operator subgraphs into Poincaré hyperbolic space (`LTHCM`).
  3. When an operator in Domain 1 (e.g. non-equilibrium thermodynamics) is topologically isomorphic to an unsolved bottleneck in Domain 2 (e.g. generative diffusion models), the operator is hoisted across the boundary, coining a novel domain concept.

#### C. Outlier "Island" Unit Cross-Breeding
* Standard clustering algorithms (HDBSCAN) drop unclustered units as noise ($5\%\text{–}10\%$ of all papers).
* In Heuresis, these outlier units are preserved in `LTHCM` as **Unclassified Fringe Units**.
* The candidate generator enforces an exploration quota: **$15\%$ of all candidate triples must combine one established high-coherence atom with an unclassified fringe unit**. This systematically injects non-standard ideas into mainstream paradigms.

---

### 3.3 Fast Dual-Scorer Architecture (`heuresis/scorers/`)

The search space over $N = 300$ atoms yields $\binom{300}{3} \approx 4.45\times 10^6$ triples. Scoring must run without per-candidate LLM calls:

1. **Coherence Scorer ($S_{\text{coh}}$)**:
   - Lightweight $43\text{M}$-parameter transformer trained via contrastive loss on co-occurring literature concepts.
   - Evaluates: *Can these 3 concepts plausibly function in a single experimental design?*
2. **Community Availability Scorer ($S_{\text{avail}}$)**:
   - Two-tower contrastive model over the peer-review community boundary graph.
   - Evaluates: *Is any existing research group currently working on this intersection?*
3. **The Discovery Objective**:
   $$\text{Score}(A, B, C) = z(S_{\text{coh}}) - \beta \cdot z(S_{\text{avail}})$$
   High coherence combined with low availability isolates **under-explored, high-potential research frontiers**.

---

### 3.4 Conformal Selection & Proposal Write-Up (`krisis` & `hestia`)

1. **Krisis Conformal Gate**:
   - Candidates must exceed calibrated thresholds $\tau_{\text{coh}}$ and $\tau_{\text{novelty}}$ determined via conformal prediction ($1 - \alpha = 0.95$).
   - If no candidates in the corpus meet the bar, Krisis emits `ABSTAIN` (*"No valid discoveries in current corpus"*), preventing low-grade ideation.
2. **Hestia Write-Up Protocol**:
   Only the top $K \le 10$ vetted candidates are passed to `hestia` for structured generation:
   - Title & Core Research Question
   - Provenance Trail (Source papers, reviewer comments, atom IDs)
   - Proposed Falsifiable Test & Verification Criteria
   - Hardware / Compute Budget Estimate

---

## 4. Evaluation & Validation Protocol (`dokime`)

Heuresis proposals must earn their place through two rigorous, objective gates:

### Gate 1: Dokime Forward-in-Time Ranking Gate
* **Corpus Split**: Freeze all concept atomization and scorer weights on pre-2024 literature.
* **Test Set**: Evaluate against all real published papers from 2025–2026.
* **Metric**:
  - **Hit Rate**: Top $1\%$ scored candidates must recover actual future breakthrough combinations at $\ge 15\times$ random chance.
  - **Diversity**: Effective unique idea count must exceed independent LLM prompting by $\ge 2\times$.

### Gate 2: Blind Triple-Check Quality Gate
* Generate 10 Heuresis proposals, 10 raw frontier LLM proposals (Gemini 3.8 / Claude 3.5), and 10 random combinations.
* Blind human expert audit evaluating:
  1. **Scientific Feasibility** (1–5)
  2. **Non-Trivial Novelty** (1–5)
  3. **Empirical Falsifiability** (Pass/Fail)

---

## 5. Implementation Roadmap & Subsystem Ownership

| Phase | Milestone | Subsystem | Deliverables | Gate Condition |
| :---: | :--- | :---: | :--- | :--- |
| **H1** | **OpenReview Crawler** | `stoma` | `stoma/src/stoma/crawlers/openreview.py` | Ingests 1,000 public reviews with boundary tags. |
| **H2** | **Concept Atom Distiller** | `stoma` / `LTHCM` | `heuresis/distiller.py`, `heuresis/atoms.py` | Distills papers into discrete method atoms + outlier islands. |
| **H3** | **Dual Fast Scorers** | `paideia` / `heuresis` | `heuresis/scorers/coherence.py`, `availability.py` | 4.5M triples scored on RTX 5060 Ti in $<15$ minutes. |
| **H4** | **Sheaf OOV Synthesizer** | `dianoia` | `dianoia/sheaf_oov.py` | Triggers abductive concept generation on Laplacian obstruction. |
| **H5** | **Dokime Benchmark** | `dokime` | `dokime/src/dokime/tasks/heuresis_eval.py` | Reusable temporal ranking benchmark. |

---

## 6. Definition of Done

1. Heuresis operates entirely on local hardware (RTX 5060 Ti 16GB) without external API dependencies.
2. Candidate ranking over $\ge 3\text{M}$ combinations executes in $< 15\text{ minutes}$.
3. Discovered candidates carry complete provenance trails back to raw source chunks.
4. All existing $1,110$ tests in the Metis monorepo remain $100\%$ green.

