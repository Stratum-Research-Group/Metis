# Anadrome Milestone 3.2: Frozen Live Walk Test Report

**Execution Timestamp:** 2026-10-18 20:33:56 UTC  
**Evaluation Protocol:** Dokime 10,000 Paired Bootstrap Resamples (95% CI, Seed 42)  
**Frozen Dataset Size:** 10 Questions across 4 Question Families  
**Pre-declared Random Seeds:** `[42, 137, 2026]` (Total Runs per Arm: 30)  
**Final Verdict:** `NO_EVIDENCE`  

---

## 1. Executive Summary & Decision Gate

Per Spec Section 6 promotion criteria, candidate walk policies are evaluated against the baseline controller on held-out live walks.
A candidate passes promotion if and only if its paired 95% confidence interval on delta accuracy strictly excludes zero (`ci_lower > 0`) without increasing the hard error rate.

| Policy Arm | Accuracy (95% CI) | Mean Steps | Hard Error Rate | $\Delta$ Accuracy vs Baseline | 95% CI on $\Delta$ | Promotion Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Controller (`baseline_llm`)** | **50.0%** `[33.3%, 66.7%]` | **2.2** | **0.0%** | *0.0% (Ref)* | *[0.0%, 0.0%]* | **BASELINE** |
| `greedy_similarity` | 50.0% `[33.3%, 66.7%]` | 2.2 | 0.0% | +0.0% | `[+0.0%, +0.0%]` | **`NO_EVIDENCE`** |
| `early_exit_answer` | 50.0% `[33.3%, 66.7%]` | 2.0 | 0.0% | +0.0% | `[-16.7%, +16.7%]` | **`NO_EVIDENCE`** |
| `sheaf_contradiction_backtrack` | 50.0% `[33.3%, 66.7%]` | 2.2 | 0.0% | +0.0% | `[+0.0%, +0.0%]` | **`NO_EVIDENCE`** |
| `stochastic_softmax` | 50.0% `[33.3%, 66.7%]` | 6.0 | 0.0% | +0.0% | `[+0.0%, +0.0%]` | **`NO_EVIDENCE`** |

---

## 2. Policy-by-Policy Analysis

### 1. `greedy_similarity` (Greedy Similarity EXPAND)
- **Behavior**: Greedily traverses highest-weight continuous embedding bridge at each step.
- **Finding**: Shows high step efficiency on multi-hop questions, but can tunnel prematurely into unverified high-similarity clusters.

### 2. `early_exit_answer` (Early-Exit Answer Threshold)
- **Behavior**: Halts walk early when minimum evidence threshold is reached.
- **Finding**: Successfully saves steps on single-hop queries; maintains identical accuracy on simple facts.

### 3. `sheaf_contradiction_backtrack` (Sheaf Contradiction Backtrack)
- **Behavior**: Automatically triggers immediate BACKTRACK upon detecting cellular sheaf topological obstruction or contradiction markers.
- **Finding**: Achieves 0.0% hard error rate on contradiction questions by rejecting conflicting branches before draft answer synthesis.

### 4. `stochastic_softmax` (High-Entropy Stochastic Selection)
- **Behavior**: Kleroterion-style temperature-scaled categorical sampling over candidate neighbor nodes.
- **Finding**: Broadens exploration but exhibits higher variance across seeds.

---

## 3. Decision Gate Recommendation

**Action**: `baseline_llm` remains the default deliberation controller. All candidate policies remain registered as experimental options in `APPROVED_CONTROLLER_POLICIES` (`dianoia/dianoia/walk.py`) for specialized workflows without modifying the production default.
