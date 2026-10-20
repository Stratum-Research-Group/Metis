# Empirical Report: SwiLA-Style Streaming Session State (Stage 1, Revised)

**Date:** 2026-10-20  
**Pinned Metis Commit:** `509d70cceb770b8b227ec99ec1ba0d2de7c8cd67`  
**Pinned SwiLA Commit:** `4b43bb5c9d3e76659e5f07740c15b046cddb581b` (`github.com/lindermanlab/switching-linear-attention`)  
**Experiment Protocol:** [`docs/review/metis-swila-session-state-experiment-plan-v3-509d70c-d2ce8894.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-swila-session-state-experiment-plan-v3-509d70c-d2ce8894.txt)  
**Audit Reference:** [`docs/review/metis-swila-stage1-report-audit-20261020-bbf93822.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-swila-stage1-report-audit-20261020-bbf93822.txt)  
**Execution Runtime:** 33.2s (CPython 3.12, PyTorch CPU, 20 seeds $\times$ 3 stream mixes $\times$ 1,200 writes)  
**Artifact JSON:** [`experiments/swila_session_state/results/tier_a_results.json`](file:///home/akarta/Projects/Metis/experiments/swila_session_state/results/tier_a_results.json)

---

## 1. Executive Verdict: Gate 1 FAIL (Stop Rule Triggered)

Per Section 9 (G1) and Section 10 of the preregistered plan:
> *"G1 Tier A: S1 vs B1, B2 and R1 on the primary metric; the result must be PASS or STRONG PASS against ALL three comparators, majority recall drop <= 0.02, and not confined to balanced mixes. NO EFFECT or BELOW TARGET against any of them stops the experiment and Tier B is not run."*

**Protocol Verdict:** **BELOW TARGET / NO EFFECT**.
- **$S1$ vs $R1$ (Error-Driven Responsibilities vs Random Partitioning Control):**
  - Skewed mix: $est = -0.0044$, $95\%\text{ CI } [-0.0124, 0.0016] \implies$ **BELOW TARGET**
  - Balanced mix: $est = -0.0012$, $95\%\text{ CI } [-0.0075, 0.0050] \implies$ **BELOW TARGET**
  - Bursty mix: $est = +0.0044$, $95\%\text{ CI } [0.0000, 0.0112] \implies$ **BELOW TARGET**
  - *Per G4 protocol language:* No positive gain of error-driven responsibilities over arbitrary random partitioning is supported.
- **$S1$ vs $B2$ (Wide Single-Expert Delta Rule):**
  - Skewed mix: $est = -0.0521$, $95\%\text{ CI } [-0.0757, -0.0300] \implies$ **NO EFFECT** ($B2$ wins: $0.154$ vs $0.102$)
  - Bursty mix: $est = -0.0863$, $95\%\text{ CI } [-0.1129, -0.0603] \implies$ **NO EFFECT** ($B2$ wins: $0.141$ vs $0.055$)
  - The single wide expert outperforms SwiLA under skewed and bursty regime switching.
- **$S1$ vs $B4$ (Equal-Byte Recency Window):**
  - Skewed mix: $est = -0.4616$, $95\%\text{ CI } [-0.4955, -0.4282] \implies$ **NO EFFECT** ($B4$ dominates: $0.564$ vs $0.102$)
  - Bursty mix: $est = -0.7928$, $95\%\text{ CI } [-0.8486, -0.7375] \implies$ **NO EFFECT** ($B4$ dominates: $0.848$ vs $0.055$)
  - Balanced mix: $est = +0.2050$, $95\%\text{ CI } [+0.1625, +0.2425] \implies$ **STRONG PASS** ($S1$ wins: $0.748$ vs $0.542$)
- **Gate 1 Action:** Because $S1$ is `BELOW TARGET` against $R1$ across all mixes and `NO EFFECT` against $B2$ on skewed/bursty mixes, **Tier B text streams are halted and not run**.

---

## 2. Gate 0 (G0) Validity Checks: 6/6 PASS

All 6 preregistered validity checks (8a–8f) passed before experimental evaluations:

| Check | Specification Requirement | Result | Verified Metric / Invariant |
| :--- | :--- | :---: | :--- |
| **8a Causality** | Mutating future writes ($t \ge 10$) leaves past outputs ($t < 10$) unchanged; queries do not mutate state $S$. | **PASS** | Bit-identical outputs; $S$ unchanged post-query |
| **8b $K=1$ Reductions** | Plain $K=1$ matches NumPy delta rule within $1\times 10^{-5}$; gated $K=1$ matches gated NumPy rule; $R1$ (coef $0.5$) matches $S1$. | **PASS** | Max abs difference $< 1.2 \times 10^{-7}$ across all steps |
| **8c Symmetry** | Zero init + flat $\pi_k$ maintains identical expert states; seeded breaking achieves effective experts $> 1.5$. | **PASS** | Effective experts $= 3.98 > 1.5$; relative prediction diff $= 0.185 \gg 10^{-4}$ |
| **8d Determinism** | Re-executing identical stream seed yields bit-identical writes and candidate sets. | **PASS** | Bit-identical across repeated runs |
| **8e Controls** | Permuted candidate labels and independent random association stream both bounded near chance floor. | **PASS** | Permuted recall $= 0.018 \le 0.08$; independent random assoc recall $= 0.021 \le 0.08$ |
| **8f Candidate Rules** | Target present exactly once; written before query; delay feasible; candidate target position randomized across $0 \dots 49$. | **PASS** | 100% valid schedules; position-agnostic scoring |

---

## 3. Tier A Experimental Results (20 Seeds $\times$ 1,200 Writes)

### Primary Metric: Minority-Topic Recall@1 (Macro over Topics 2 & 3 at Delays 100 & 1000)

| Arm | Description | Skewed (70/20/7/3) | Balanced (25/25/25/25) | Bursty | State Floats | Total Resident Bytes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1** | SwiLA Recurrence ($H=1, K=4$) | **0.102** | **0.748** | **0.055** | 16,384 (64 KB) | 372,736 B |
| **R1** | Fixed Random Routing (Control) | **0.106** | **0.749** | **0.050** | 16,384 (64 KB) | 372,736 B |
| **B1** | $K=4$ Narrow Single-Expert Heads | **0.100** | **0.534** | **0.089** | 16,384 (64 KB) | 372,736 B |
| **B2** | One Wide Single-Expert Head | **0.154** | **0.571** | **0.141** | 16,384 (64 KB) | 372,736 B |
| **B3** | Running Mean Vector | 0.001 | 0.010 | 0.004 | 64 (256 B) | 307,456 B |
| **B4** | Recency Window (FIFO Nearest-Key) | **0.564** | **0.542** | **0.848** | 16,384 (64 KB) | 372,736 B |
| **B5** | Exact KV Store (All writes) | 1.000 | 1.000 | 1.000 | Unbounded | Unbounded |
| **O1** | Oracle Target Vector | 1.000 | 1.000 | 1.000 | Ceiling | Ceiling |
| **C0** | Uniform Random Chance ($1/50$) | 0.018 | 0.016 | 0.018 | Floor | Floor |

---

## 4. Paired Bootstrap Statistical Comparisons ($S1$ vs Comparators)

95% Confidence Intervals computed over 10,000 paired bootstrap resamples across 20 stream seeds with $\delta_0 = 0.05$.

### Skewed Mix (70% Topic 0, 20% Topic 1, 7% Topic 2, 3% Topic 3)
- **$S1 - B1$:** Point Estimate $= +0.0018$, $95\%\text{ CI } [-0.0193, +0.0260] \implies$ **BELOW TARGET**
- **$S1 - B2$:** Point Estimate $= -0.0521$, $95\%\text{ CI } [-0.0757, -0.0300] \implies$ **NO EFFECT** ($B2$ wins)
- **$S1 - R1$:** Point Estimate $= -0.0044$, $95\%\text{ CI } [-0.0124, +0.0016] \implies$ **BELOW TARGET**
- **$S1 - B4$:** Point Estimate $= -0.4616$, $95\%\text{ CI } [-0.4955, -0.4282] \implies$ **NO EFFECT** ($B4$ wins)

### Balanced Mix (Equal Topic Probability 25%)
- **$S1 - B1$:** Point Estimate $= +0.2138$, $95\%\text{ CI } [+0.1763, +0.2525] \implies$ **STRONG PASS**
- **$S1 - B2$:** Point Estimate $= +0.1762$, $95\%\text{ CI } [+0.1412, +0.2112] \implies$ **STRONG PASS**
- **$S1 - R1$:** Point Estimate $= -0.0012$, $95\%\text{ CI } [-0.0075, +0.0050] \implies$ **BELOW TARGET**
- **$S1 - B4$:** Point Estimate $= +0.2050$, $95\%\text{ CI } [+0.1625, +0.2425] \implies$ **STRONG PASS** ($S1$ wins)

### Bursty Mix (Long majority runs with interleaved minority bursts)
- **$S1 - B1$:** Point Estimate $= -0.0338$, $95\%\text{ CI } [-0.0747, -0.0012] \implies$ **NO EFFECT** ($B1$ wins)
- **$S1 - B2$:** Point Estimate $= -0.0863$, $95\%\text{ CI } [-0.1129, -0.0603] \implies$ **NO EFFECT** ($B2$ wins)
- **$S1 - R1$:** Point Estimate $= +0.0044$, $95\%\text{ CI } [0.0000, +0.0112] \implies$ **BELOW TARGET**
- **$S1 - B4$:** Point Estimate $= -0.7928$, $95\%\text{ CI } [-0.8486, -0.7375] \implies$ **NO EFFECT** ($B4$ wins)

---

## 5. Mechanism Diagnostics & Computational Footprint

- **Effective Number of Experts:** $\exp(H(\text{mean usage})) = 4.00 \pm 0.00$ across all mixes. All 4 experts were utilized with near-perfect uniformity.
- **Dead-Expert Fraction:** $0.0\%$. No expert starvation occurred.
- **Expert-Topic Alignment (Normalized Mutual Information):**
  - Skewed mix NMI: $0.0028 \pm 0.0015$
  - Balanced mix NMI: $0.0025 \pm 0.0012$
  - Bursty mix NMI: $0.0031 \pm 0.0018$
  - *Interpretation:* The NMI is indistinguishable from zero ($< 0.005$). Without trained router projections, the error-driven update does not cause experts to align with distinct topic regimes; routing is dominated by the initial Gaussian projection partition.
- **Oracle Distractor Collisions:** $0$ collisions observed across all seeds (no distractor within cosine $> 0.999$ of target).
- **Latency & Throughput:**
  - Online write update: $441.7\,\mu\text{s}$ per write
  - State snapshot query: $4.81\,\text{ms}$ per query
  - Peak RAM: $732.1\,\text{MB}$

---

## 6. Preregistered Protocol Takeaways

1. **Protocol Language Adherence (Hypothesis H1):**
   - At the tested configuration ($K=4, \beta=0.5, \tau=1.0, D_k=64, D_v=64$), **no positive gain of error-driven responsibilities over random partitioning is supported** ($S1$ vs $R1$).
   - Per Plan Section 2, this evaluates the recurrence with fixed, untrained parameters. This result does **not** refute the published paper (which trains router parameters end-to-end), nor does it invalidate separate chunk aggregation architectures such as Sylloge/PULSAR.
2. **Context-Dependent Baseline Performance ($B4$ vs $S1$):**
   - Recency window $B4$ substantially outperforms recurrent memory on skewed streams ($0.564$ vs $0.102$) and bursty streams ($0.848$ vs $0.055$), where recent writes carry the bulk of relevant context.
   - Conversely, on balanced streams with frequent interleaved topics, $S1$ outperforms $B4$ ($0.748$ vs $0.542$, `STRONG PASS`), demonstrating that recurrent state integration provides superior long-term retention when topics are uniformly distributed over time.
3. **Stage Approvals and Scope:**
   - **Tier B (text streams with offline embedder):** Halted per protocol stop rule.
   - **Stage 3 (learned projections on GPU):** Outside the scope of the Stage 1 plan and remains a separate open decision.
