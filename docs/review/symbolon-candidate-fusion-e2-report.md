# Symbolon Experiment E2: Candidate Fusion Diagnostic Report (Milestone 2.3)

**Date:** 2026-10-18  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Ti 16GB (Driver: 615.78.08, CUDA: 13.4)  
**Evaluator:** Dokime Paired Bootstrap Harness (10,000 resamples, seed 42)  
**Spec Reference:** `docs/review/lthcm-ncm-symbolon-integration-spec-2026-10-18-c51d8a62-8951392a.txt` Section 3 (Experiment E2)  
**Roadmap Milestone:** Milestone 2.3 (Tracks C & D)

---

## 1. Executive Summary & Decision Verdict

| Metric | Flat Dense Baseline (LTHCM Flat Mode) | Symbolon Scoped Code Overlap Alone | Uncapped Additive Union ($C_{\text{base}} \cup C_{\text{sym}}$) | Capped Fusion (Matched Top-$k=3$) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Recall@3 (Gold Evidence)** | **100.0%** (4/4 queries) | **25.0%** (1/4 queries) | **100.0%** (4/4 queries) | **100.0%** (4/4 queries) |
| **Contradiction Retention** | **100.0%** (4/4 queries) | **25.0%** (1/4 queries) | **100.0%** (4/4 queries) | **100.0%** (4/4 queries) |
| **Retriever Agreement (Proxy)** | 1.000 | 0.175 (divergent) | — | 0.850 |
| **Candidate Set Expansion** | 1.0x (3 docs) | 1.0x (3 docs) | **1.75x** (5.25 docs average) | 1.0x (3 docs) |
| **Dropped Gold Documents** | 0 | — | 0 | 0 |
| **Paired $\Delta$ vs Baseline (Recall@3)**| Baseline ($0.0000$) | $-0.7500$ (severe loss) | $+0.0000$ (no gain) | $+0.0000$ (no gain) |

### Decision Gate Verdict: `ARCHIVE_UNCAPPED_NO_LOSS_RECORDED`
> **Verdict Statement**: Uncapped additive union ($C_{\text{base}} \cup C_{\text{sym}}$) preserves 100% of gold evidence, but at the cost of a 1.75x candidate set inflation and verification overhead. Capped fusion shows no statistically significant delta over the flat dense baseline ($\Delta = 0.0000$, 95% CI: $[0.0000, 0.0000]$). Furthermore, Symbolon discrete code overlap alone suffers from a severe 75% recall deficit and 75% contradiction retention loss. Per Spec Section 3 and Roadmap Milestone 3.3, Symbolon remains **disabled by default**, the negative research note is recorded, and the `FilteredRetrievalAdapter` is **NOT** adopted into LTHCM.

---

## 2. Experimental Setup & Protocol

Per Spec E2, the benchmark diagnostic corpus spans scientific biochemistry, geopolitics, aerospace telemetry, and topological data analysis, with explicit gold-relevant documents, distractors, and contradiction passages:
- **Encoding**: Passages and queries were encoded into 48-token discrete code windows using `SymbolonTokenizerAdapter` on the RTX 5060 Ti.
- **Identity Rule**: Strict 6-tuple identity rule `(checkpoint_id, tokenizer_id, level, window_idx, pos, code_int)` via `compute_scoped_code_overlap`.
- **Level Tested**: Codebook level 4 (20 discrete codes per window, intermediate abstraction).
- **Matched Top-$k$ Budget**: $k=3$ across all comparative arms.

---

## 3. Detailed Query-Level Breakdown

| Query ID | Gold Relevant Document | Contradiction Document | Baseline Top-3 | Symbolon Top-3 | Capped RRF Top-3 | Union Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`q_biochem`** *(Methotrexate active site)* | `doc_sci_01` | `doc_sci_03_contra` | `doc_sci_01`, `doc_sci_03_contra`, `doc_sci_02` | `doc_geo_01`, `doc_sci_01`, `doc_space_01` | `doc_sci_01`, `doc_sci_03_contra`, `doc_geo_01` | 5 |
| **`q_merkel`** *(Berlin economic corridor)* | `doc_geo_01` | `doc_geo_03_contra` | `doc_geo_01`, `doc_geo_03_contra`, `doc_sci_01` | `doc_geo_02`, `doc_space_02`, `doc_math_02` | `doc_geo_01`, `doc_geo_03_contra`, `doc_geo_02` | 6 |
| **`q_apollo`** *(F-1 engine fuel consumption)*| `doc_space_01` | `doc_space_03_contra` | `doc_space_01`, `doc_space_03_contra`, `doc_sci_01` | `doc_math_01`, `doc_space_02`, `doc_geo_02` | `doc_space_01`, `doc_space_03_contra`, `doc_math_01` | 5 |
| **`q_sheaf`** *(Laplacian topological drift)* | `doc_math_01` | `doc_math_03_contra` | `doc_math_01`, `doc_math_03_contra`, `doc_space_01` | `doc_geo_02`, `doc_space_02`, `doc_sci_02` | `doc_math_01`, `doc_math_03_contra`, `doc_geo_02` | 5 |

---

## 4. Key Findings

1. **Discrete Code Codebook Confusion**:
   The VQ-VAE tokenizer's discrete latent codebook clusters documents based on high-level acoustic/frequency-style discrete representations trained on general text. Coarse discrete codes lack the lexical selectivity needed for technical query matching, leading Symbolon alone to retrieve irrelevant distractors (e.g. `doc_geo_02` for mathematical sheaf queries).
2. **Contradiction Drop Risk**:
   Symbolon alone failed to retrieve 3 out of 4 contradiction documents (25% retention rate). Adopting Symbolon as a hard pre-filter would severely compromise Metis's defensive contradiction detection.
3. **No Advantage Over Flat Dense**:
   Reciprocal Rank Fusion (Capped Top-3) achieves identical recall to the flat dense baseline without uncovering new relevant items ($\Delta = 0.0000$).

---

## 5. Decision & Archival Disposition

- **Disposition**: Negative result recorded in accordance with Spec Section 4 ("Negative results reported as negative").
- **Integration Verdict**: `FilteredRetrievalAdapter` in LTHCM is **NOT** built. Symbolon remains an offline diagnostic tool.
