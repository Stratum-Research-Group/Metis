# Schedion Stage 3 Controlled Comparison Report (Milestone 2.3)

**Date:** 2026-10-18  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Ti 16GB (Driver: 615.78.08, CUDA: 13.4)  
**Evaluator:** Dokime Paired Bootstrap Harness (10,000 resamples, seed 42) & KrisisSelector (`krisis-policy-0.1.json`)  
**Spec Reference:** `docs/review/metis-ncm-schedion-integration-spec-2026-10-18-a3f41c7d-2aa95bb2.txt` Section 5 (Stage 3)  
**Roadmap Milestone:** Milestone 2.3 (Tracks C & D)

---

## 1. Executive Summary & Decision Verdict

| Metric | Arm A: Baseline (Hestia Extractive/Autoregressive) | Arm B: Candidate (Schedion NCM XSum Generator) | Comparison / Delta |
| :--- | :--- | :--- | :--- |
| **Krisis Pass Rate** | 0.0% (5/5 `NO_AUTHORIZED_CANDIDATE`) | 0.0% (5/5 `NO_VALID_CANDIDATE`) | $\Delta = 0.0000$ (No difference) |
| **Factual Grounding** | **100.0%** (mean token overlap) | **26.3%** (mean token overlap) | $\mathbf{-73.7\%}$ severe regression |
| **Generation Latency** | **< 1.0 ms** | **813.0 ms** | 800x latency penalty |
| **Peak VRAM** | 0 MB (CPU extractive) | ~780 MB (NCM VQ-VAE + Generator) | +780 MB allocation |
| **Domain Portability** | Universal (retains technical terms) | Degrades to BBC news/sports tropes | Severe out-of-domain drift |

### Decision Gate Verdict: `ARCHIVE_NO_ADOPTION`
> **Verdict Statement**: Schedion demonstrates no measured benefit over baseline in quality, factual grounding, latency, or gate pass rate under identical 512-token conditioned spans ($\Delta = 0.0000$, grounding 26.3% vs 100.0%, latency 813.0 ms vs < 1.0 ms). Per Spec Section 5 and Roadmap Milestone 3.3, Schedion remains disabled by default, the experiment record is archived, and Schedion is **NOT** adopted as a candidate source in Kybernetes.

---

## 2. Experimental Controls & Discipline

Per Spec Section 5 (Stage 3), both arms were evaluated under strict experimental controls:
1. **Identical Conditioned Span**: Both arms were fed the exact same 512-token prefix extracted via `SchedionGenerator.prepare_conditioned_span()`.
2. **Comparable Summary Target**: 1–2 sentence summary length targets (~25–50 tokens).
3. **Identical Gate**: All candidate objects were evaluated through `KrisisSelector.evaluate()` against the active normative policy `krisis-policy-0.1.json`.
4. **Claim Attribution**: Sentence boundaries were mapped to structured `Claim` objects citing the conditioned passage span.

---

## 3. Detailed Results Across Probes

| Document ID & Category | Baseline Output (Hestia Arm A) | Schedion Output (NCM Arm B) | Factual Grounding (A vs B) | Krisis Outcome (A vs B) |
| :--- | :--- | :--- | :--- | :--- |
| **`doc_01_geopolitics`**<br>*(International Relations)* | *"Chancellor Angela Merkel met with European Union finance ministers in Berlin on Monday to finalize agreements..."* | *"The EU has agreed to form all customs, transport transport t..."* | 100.0% vs 46.2% | `NO_AUTHORIZED_CANDIDATE` vs `NO_VALID_CANDIDATE` |
| **`doc_02_space_propulsion`**<br>*(Aerospace & Telemetry)* | *"During the Apollo 11 launch on July 16, 1969, the Saturn V S-IC first stage burned 20,000 pounds of propellant per second..."* | *"A company that operates its 11-year private space telescope,..."* | 100.0% vs 14.3% | `NO_AUTHORIZED_CANDIDATE` vs `NO_VALID_CANDIDATE` |
| **`doc_03_biochem`**<br>*(Pharmacology)* | *"Methotrexate is a classical antifolate that acts as a tight-binding competitive inhibitor of human dihydrofolate reductase..."* | *"A giant temple has moved a closer to finding a site with a p..."* | 100.0% vs 23.1% | `NO_AUTHORIZED_CANDIDATE` vs `NO_VALID_CANDIDATE` |
| **`doc_04_topological_data`**<br>*(Math & Sheaves)* | *"A cellular sheaf over an abstract simplicial complex assigns local vector spaces to vertices and edges..."* | *"\"intons is a large lining of shrapnel, fish and chocolate ch..."* | 100.0% vs 20.0% | `NO_AUTHORIZED_CANDIDATE` vs `NO_VALID_CANDIDATE` |
| **`doc_05_consensus`**<br>*(Distributed Systems)* | *"The Raft consensus algorithm decomposes state machine replication into leader election, log replication, and safety invariants..."* | *"It's one of the great tragedies of global sporting event it..."* | 100.0% vs 27.8% | `NO_AUTHORIZED_CANDIDATE` vs `NO_VALID_CANDIDATE` |

---

## 4. Root Cause Analysis of Out-of-Domain Failure

1. **Pretraining Bias on BBC XSum**:
   The released NCM checkpoint (`artifacts/xsum/generator.pt`) was trained exclusively on the BBC Extreme Summarization corpus (general news, British politics, sports, entertainment). When prompted with specialized technical passages (biochemistry active sites, cellular sheaf Laplacians, Raft quorum consensus), the cross-attention conditioning fails to steer the discrete sequence generator into appropriate semantic domains.
2. **Hallucinatory Trope Substitution**:
   Instead of preserving key entities or numerical measurements, the NCM generator substitutes familiar BBC training distribution tokens—producing hallucinated phrases such as *"giant temple"*, *"shrapnel, fish and chocolate"*, and *"tragedies of global sporting event"*.
3. **Krisis Gate Rejection**:
   KrisisSelector correctly rejected all Schedion candidates as `NO_VALID_CANDIDATE` due to ungrounded claims and hallucinated terms, validating Krisis's defensive integrity.

---

## 5. Archival Verdict & Disposition

- **Decision**: Schedion is archived in `docs/review/` as an offline research exploration.
- **Production Status**: Schedion remains **disabled by default**. It will not be integrated into Kybernetes' active candidate generator pipeline.
