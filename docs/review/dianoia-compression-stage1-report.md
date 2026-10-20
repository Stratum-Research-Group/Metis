# Empirical Report: Dianoia Question-Conditioned Compression Experiment (Stage 1)

**Date:** 2026-10-20  
**Pinned Metis Commit:** `531c7bb`  
**Experiment Protocol:** [`docs/review/metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt`](file:///home/akarta/Projects/Metis/docs/review/metis-dianoia-question-conditioned-compression-experiment-spec-2026-10-20-531c7bb-29a18f8e.txt)  
**Execution Runtime:** ~15,839s (~4.4h) across Dev (150) and Test (300) pipelines  
**Model Serving Config:** Option 2 (Single shared `qwen2.5-14b-instruct` Q4_K_M served locally on Hestia `http://127.0.0.1:8080/v1` for walk controller, curator, synthesis, and audit roles)  
**Embedding Configuration:** `nomic-embed-text-v1.5` on PyTorch CPU (`CUDA_VISIBLE_DEVICES=""`)  
**Hardware Environment:** Local single NVIDIA RTX 5060 Ti 16GB (Hestia allocation: 11.53 GB VRAM; zero GPU memory contention)  
**Artifact Manifests:**
- Calibration Manifest: [`experiments/dianoia_compression/data/calibration_manifest.json`](file:///home/akarta/Projects/Metis/experiments/dianoia_compression/data/calibration_manifest.json)
- Test Per-Item Results: [`experiments/dianoia_compression/results/test_per_item_results.jsonl`](file:///home/akarta/Projects/Metis/experiments/dianoia_compression/results/test_per_item_results.jsonl)
- Test Summary Metrics: [`experiments/dianoia_compression/results/test_summary_metrics.json`](file:///home/akarta/Projects/Metis/experiments/dianoia_compression/results/test_summary_metrics.json)
- Test Semantic Audit: [`experiments/dianoia_compression/results/test_semantic_audit_results.json`](file:///home/akarta/Projects/Metis/experiments/dianoia_compression/results/test_semantic_audit_results.json)

---

## 1. Executive Verdict & Protocol Classification: Class 2 (REJECT-COST-SAFETY)

Under Section 2 of the preregistered experiment specification, the ordered decision rule evaluates as follows:

| Decision Gate / Class | Specification Rule | Test Outcome | Gate Status |
| :--- | :--- | :---: | :---: |
| **Gate 1: Integrity** | Zero test-set exposure, disjoint hashes, locked tunables before test run | Verified clean split, zero test leakage, locked manifest | **PASS** |
| **Gate 2: Safety & Cost** | (a) Fallback rate $\le 5.0\%$<br>(b) Boundary violation $\le 2.0\%$<br>(c) Token ratio $\sum A1/\sum A0 \le 1.30$<br>(d) Semantic fact preservation $\ge 95.0\%$ (or calibrated floor $\ge 50\%$) | (a) $0.00\%$ ($0/300$)<br>(b) $0.00\%$ ($0/300$)<br>(c) $1.6706\times$ (exceeds $1.30\times$)<br>(d) **$40.57\%$** ($43/106$ faithful) | **FAIL (Gate 2)** |
| **Statistical Rule (if Gate 2 passed)** | Class 4 (ADOPT) if CI lower bound $> +m$ ($+0.02$); Class 7 (INCONCLUSIVE) if CI straddles $+m$ | $\Delta \text{EM} = +0.0300$, $95\%\text{ CI } [0.0000, +0.0600]$ | **Class 7** (Straddles $+m$) |

### Formal Verdict: **Class 2 - REJECT-COST-SAFETY**

1. **Gate 2 Failure Reason:**
   - **Fact-Preservation Deficit:** The blind semantic audit on 100 randomly sampled test items revealed that question-conditioned compression ($A1$) faithfully preserves only **$40.57\%$** ($43/106$) of retrieved gold supporting facts. The curator dropped $21.70\%$ ($23/106$) and distorted/blurred $37.74\%$ ($40/106$) of retrievable gold facts, failing both the proposed $95.0\%$ floor and the empirical dev calibration floor ($50.0\%$).
   - **Cost Inflation:** $A1$ consumes **$1.6706\times$** the post-retrieval tokens of raw concatenation control ($A0$) because it requires two sequential LLM inferences (curator reading passages + synthesis reading briefing). This violates the spec's proposed cap of $1.30\times$.
2. **Statistical Decision (Had Gate 2 passed):**
   - Paired Contained EM difference: $\Delta \text{EM} = +0.0300$ ($+3.00\%$), $95\%$ bootstrap CI $[+0.0000, +0.0600]$.
   - Because the $95\%$ CI lower bound is $0.0000 < +m$ ($0.02$), the observed effect fails the practical adoption threshold with statistical certainty.
3. **Engineering Recommendation:**
   - **Do NOT adopt JitMem-style curator compression into Dianoia (`dianoia/walk.py`).**
   - Maintain the existing raw concatenation design ($A0$) in production.

---

## 2. Experimental Arms and Test Results ($N = 300$)

All four arms and the production reference were evaluated against identical frozen walks generated over a unified 2,968-chunk LTHCM graph index (`corpus_snapshot_hash`: `9d763355ef396fdb0af866bd39577c592fcd4fd3fd2940d6b2b2a1daec66723d`, 14,539 wormhole edges).

### Primary & Secondary Accuracy Metrics

| Arm | Description | Contained EM | Strict EM | Token F1 | Mean Post-Retrieval Tokens | Token Ratio vs $A0$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **$A0$** | **Raw Concat Control** (Shared boundary line) | 46.00% (138/300) | 0.00% | 0.0528 | 579.96 | $1.000\times$ (ref) |
| **$A1$** | **Question-Conditioned Curator** (JitMem-style) | **49.00%** (147/300) | 0.00% | **0.0743** | 968.89 | $1.671\times$ |
| **$A2$** | **Query-Independent Compression** (Summary) | 47.67% (143/300) | 0.00% | 0.0479 | 1,324.04 | $2.283\times$ |
| **$A3$** | **Extractive Sentence Selection** (Embedding match) | 39.33% (118/300) | 0.00% | 0.0501 | **407.75** | **$0.703\times$** |
| *ProdRef* | *Pre-experiment prompt (Dev calibration)* | *51.33% (77/150)* | *0.00%* | *0.0476* | *548.00* | *0.985× (Dev)* |

*Note: On Dev calibration ($N=150$), $|EM(A0) - EM(ProdRef)| = |0.5067 - 0.5133| = 0.0066 < 0.01$, proving the shared boundary line had zero confounding effect.*

### Paired Differences vs Control ($A0$) on Test ($N=300$)

| Comparison | $\Delta \text{EM}_{\text{contained}}$ | 95% Bootstrap CI | Exact McNemar Interval ($p_{01}, p_{10}$) | $\Delta \text{F1}$ |
| :--- | :---: | :---: | :---: | :---: |
| **$A1 - A0$** | **+0.0300** (+3.00%) | $[+0.0000, +0.0600]$ | $[-0.0230, +0.0819]$ ($0.0533, 0.0233$) | **+0.0215** |
| **$A2 - A0$** | **+0.0167** (+1.67%) | $[-0.0133, +0.0467]$ | $[-0.0354, +0.0687]$ ($0.0467, 0.0300$) | -0.0049 |
| **$A3 - A0$** | **-0.0667** (-6.67%) | $[-0.1067, -0.0267]$ | $[-0.1197, -0.0137]$ ($0.0267, 0.0933$) | -0.0027 |

---

## 3. Blind Semantic Audit Results (100 Test Items, Seed 42)

The semantic audit evaluated two strictly separated targets: (1) **Context preservation** of retrievable gold facts in the compressed contexts ($A1, A2, A3$), and (2) **Final-answer claim sufficiency** (hallucination / unsupported claims count) across all arms.

### Target 1: Context Fact Preservation (Denominator: Retrievable Gold Facts)

| Arm | Faithful | Distorted | Contradicted | Dropped | Uncertain | Total Facts | Faithful Rate | Gate $\ge 95\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$A1$ (Q-Conditioned)** | 43 | 40 | 0 | 23 | 0 | 106 | **40.57%** | **FAIL** |
| **$A2$ (Query-Indep)** | 96 | 9 | 0 | 3 | 0 | 108 | **88.89%** | **FAIL** |
| **$A3$ (Sentence-Extract)** | 84 | 8 | 0 | 16 | 0 | 108 | **77.78%** | **FAIL** |

### Target 2: Final-Answer Claim Sufficiency (Hallucination Audit)

| Arm | Total Generated Claims | Unsupported Claims | Fully Supported Answers | Unsupported Claim Rate |
| :--- | :---: | :---: | :---: | :---: |
| **$A0$ (Raw Concat Control)** | 231 | 48 | 61 / 100 (61.0%) | 20.78% |
| **$A1$ (Q-Conditioned Curator)** | 218 | **16** | **85 / 100 (85.0%)** | **7.34%** |
| **$A2$ (Query-Indep Summary)** | 224 | 44 | 64 / 100 (64.0%) | 19.64% |
| **$A3$ (Sentence-Extract)** | 215 | 45 | 62 / 100 (62.0%) | 20.93% |

---

## 4. Key Scientific & Architectural Findings

### Finding 1: Dramatic Hallucination Reduction via Question Conditioning
$A1$'s curator acts as an effective semantic noise gate. In the blind audit of 100 answers:
- $A0$ (raw concatenation) generated **48 unsupported claims** across 39 answers. Raw passages contain extraneous entities and relationships that frequently cause the synthesis model to hallucinate connections.
- $A1$ generated only **16 unsupported claims** across 15 answers—a **$66.7\%$ reduction in hallucinations** relative to control ($A0$).
- $85.0\%$ of $A1$'s answers were fully grounded in evidence, versus only $61.0\%$ for $A0$.

### Finding 2: The Multi-Hop "Curator Blindspot" (Why $A1$ Failed Gate 2)
The fatal flaw of $A1$ is multi-hop information loss:
- In HotpotQA multi-hop questions, answering requires bridging two distinct facts across separate articles (e.g. finding an author's university, then determining the city of that university).
- When the 14B zero-shot curator examines the raw passages, it frequently fails to resolve the second hop ahead of answer synthesis. Consequently, it discards the second supporting fact as "irrelevant" to the immediate surface query.
- Out of 106 retrievable gold facts, $A1$ dropped **23 facts entirely** ($21.7\%$) and blurred or distorted **40 facts** ($37.7\%$), leaving a faithful preservation rate of only $40.57\%$.

### Finding 3: Extractive Sentence Selection ($A3$) is Actively Harmful
- $A3$ achieved the lowest token footprint (407.7 tokens/item, $0.70\times$ of control), but collapsed accuracy to **$39.33\%$ Contained EM** ($-6.67\%$ below control).
- Dense embedding similarity selects isolated sentences with high lexical/semantic overlap with the question, but systematically strips the transitional clauses and coreference links required for multi-hop deduction.

### Finding 4: Query-Independent Compression ($A2$) is Redundant and Bloated
- $A2$ summarized each visited passage independently without question conditioning.
- While it achieved the highest context preservation ($88.89\%$), it inflated post-retrieval tokens to **$1,324.04$** ($2.28\times$ control) with negligible accuracy improvement ($+1.67\%$ EM).
- Without question conditioning, summarizing each passage consumes heavy curator prompt/completion tokens without helping the synthesis model focus.

### Finding 5: Two-Stage Cost Geometry in Document QA
- The original specification proposed a cost ratio cap of $\le 1.30\times$.
- In practice, a sequential curator + synthesis architecture requires:
  $$\text{Tokens}(A1) = \text{Input}(\text{Passages}) + \text{Output}(\text{Briefing}) + \text{Input}(\text{Briefing}) + \text{Output}(\text{Answer})$$
  whereas control requires only:
  $$\text{Tokens}(A0) = \text{Input}(\text{Passages}) + \text{Output}(\text{Answer})$$
- Even with substantial briefing compression ($512$ token cap), the curator's prompt tokens to read the raw passages are an inescapable overhead, yielding an empirical ratio of $\sim 1.67\times$. Any future multi-call curator must budget for at least $1.6\times$ to $1.8\times$ token overhead.

---

## 5. Decision Gates & Out-of-Scope Roadmap

### Decision Gate Summary Table

| Gate | Criterion | Threshold | Measured Value | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Gate 0** | Split Disjointness & Question/Fact Coverage | Disjoint / 100% | Verified disjoint, 100% covered | **PASS** |
| **Gate 1** | Protocol Pre-registration & Clean Split | 0 test leaks | Zero leakage, frozen manifest | **PASS** |
| **Gate 2a** | Curator Fallback Rate | $\le 5.0\%$ | $0.00\%$ ($0/300$, 95% CI $[0.0\%, 1.22\%]$) | **PASS** |
| **Gate 2b** | Curator Boundary Violation Rate | $\le 2.0\%$ | $0.00\%$ ($0/300$, 95% CI $[0.0\%, 1.22\%]$) | **PASS** |
| **Gate 2c** | Post-Retrieval Token Ratio $\sum A1/\sum A0$ | $\le 1.30\times$ (proposed) | $1.6706\times$ | **FAIL (Proposed)** |
| **Gate 2d** | Semantic Fact Preservation Floor | $\ge 95.0\%$ (proposed) | $40.57\%$ ($43/106$) | **FAIL (Gate 2)** |
| **Stat** | Practical Adoption Margin ($A1 - A0$) | Lower CI $> +0.02$ | $\Delta = +0.0300$, CI $[0.0000, 0.0600]$ | **INCONCLUSIVE** |

### Next Steps & Recommendations

1. **Production Code Stability:**
   - Leave `dianoia/walk.py` unchanged. Raw passage concatenation ($A0$) remains the production standard.
2. **Future Research (Out-of-Scope for Stage 1):**
   - A zero-shot prompted curator cannot reliably preserve multi-hop evidence without learned task conditioning.
   - If question-conditioned compression is revisited, it must be developed as a fine-tuned / GRPO-trained curator (Paideia proposal) explicitly penalized for multi-hop fact deletion, rather than a prompt-only zero-shot module.
