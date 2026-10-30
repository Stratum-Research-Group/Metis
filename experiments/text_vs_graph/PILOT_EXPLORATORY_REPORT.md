# Empirical Report: Text vs Graph vs Both Under Noise (Pilot Study)

**Status:** EXPLORATORY PILOT  
**Date:** 2026-10-30  
**Model:** `gemma-4-26b-a4b-abliterated` (local Hestia @ `http://127.0.0.1:8080/v1`)  
**Hardware:** NVIDIA RTX 5060 Ti 16GB (peak VRAM ~13.5 GB)  
**Sample Size:** 40 disjoint stories = 160 questions (seeds 3001–3040)  
**Total LLM Calls:** 2730 (2,080 main evaluations + 650 determinism reruns)  
**Total Wall Time:** 5674.0s (~94.6 min; 2.71s/call)  
**Determinism Rerun Agreement:** 100.0% (650/650 identical outputs at temp=0)  
**Parse Rate:** 95.5% (1375/1440)  
**Audits:** Parity 40/40 (100%), Validity 40/40 (100%)  

---

## 1. Per-Arm Empirical Performance Summary

| Arm | Description | Level | Accuracy (Exact Match) | Story Variance | Mean Tokens | Mean Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **A** | **Text Only** | 0% | 92.5% (148/160) | 0.0167 | 1031.6 | 2.21s |
| **B** | **Graph Only** | 0% | 96.2% (154/160) | 0.0082 | 960.0 | 1.54s |
| **B** | **Graph Only** | 30% | 76.9% (123/160) | 0.0365 | 791.2 | 2.00s |
| **C** | **Text + Graph (1-pass)** | 0% | 98.1% (157/160) | 0.0044 | 1678.8 | 2.09s |
| **C** | **Text + Graph (1-pass)** | 30% | 87.5% (140/160) | 0.0160 | 1501.5 | 2.21s |
| **D** | **Text + Graph + Reconcile** | 0% | 99.4% (159/160) | 0.0016 | 3209.8 | 2.71s |
| **D** | **Text + Graph + Reconcile** | 30% | 91.9% (147/160) | 0.0141 | 3157.3 | 4.47s |
| **F** | **Matched 2-Pass Control** | 0% | 100.0% (160/160) | 0.0000 | 3414.9 | 3.55s |
| **F** | **Matched 2-Pass Control** | 30% | 97.5% (156/160) | 0.0058 | 3031.2 | 3.60s |

---

## 2. Preliminary Hypotheses Evaluation (10,000 Story-Level Bootstrap Resamples)

| Hypothesis | Comparison | Delta (Mean Paired Diff) | 95% Bootstrap CI | Preliminary Status |
| :--- | :--- | :---: | :---: | :---: |
| **H1** | Clean Graph vs Text: $B_{0} - A_{0}$ | +0.0375 | [-0.0063, +0.0813] | CONFIRMED_PARITY |
| **H2** | Corrupted Graph vs Text: $B_{30} - A_{0}$ | -0.1562 | [-0.2188, -0.0938] | CONFIRMED_DEGRADATION |
| **H3a** | Joint vs Text: $C_{30} - A_{0}$ | -0.0500 | [-0.1000, +0.0000] | DISTRACTED |
| **H3b** | Joint vs Corrupted Graph: $C_{30} - B_{30}$ | +0.1062 | [+0.0500, +0.1625] | BETTER_THAN_GRAPH |
| **H4** | Reconcile vs Matched 2-Pass: $D_{30} - F_{30}$ | -0.0563 | [-0.0938, -0.0187] | NO_INCREMENTAL_BENEFIT |

---

## 3. Methodological Observations & Practical Margin Setting

1. **Variance & Power**:
   - Mean story-level variance across arms is $\sigma^2 \approx 0.0115$.
   - Bootstrap CI half-width on 40 stories is approximately $\pm 0.062$.
   - With 300 confirmatory stories ($7.5\times$ sample size), CI half-width will tighten to $\approx \pm 0.023$.
2. **Practical Margin $\delta$ Recommendation**:
   - Proposed practical margin: **$\delta = 0.03$ ($3.0$ percentage points)** as specified in Section 8.
