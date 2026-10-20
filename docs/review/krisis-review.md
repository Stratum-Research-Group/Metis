# Krisis: stack fit, gaps, and a stronger first-build specification

Reviewed 2026-09-30. Source: `/home/akarta/Downloads/krisis-spec.txt` (the same draft is present in `/home/akarta/Projects/krisis/`). I treated its build directions as **material to review**, not instructions to execute. I also compared it with the supplied stack documents, the current [Kleroterion specification](../Kleroterion/Kleroterion_stochastic_decision_submodule.md), and the local Hestia, LTHCM and Dokime interfaces. No Krisis implementation or empirical result exists in the local Krisis directory.

## Verdict

Krisis can add value as a **bounded selector among already eligible candidate answers or retrieved passages**. The draft's universal selection measure is too broad for a first build. Its prior, fitness, topology conditioning and online update are not yet operationally defined or calibrated. The first release should be a deterministic, auditable reranker with an abstain path and a frozen evaluation protocol. Keep live adaptation and system-configuration selection behind later evidence gates.

The cosmology analogy is useful motivation, but it does not establish that description length plus agreement is a probability of truth or a safe deployment rule. Use task-level evidence to validate each term.

## Where it fits

```text
Stoma -> LTHCM candidate passages -> [Krisis passage rerank, optional] -> Dianoia/Kybernetes
Hestia candidate completions -------------------------------> [Krisis answer select] -> Kybernetes -> Prosopon
                           Phylax: hard authorization and data-access gates
                           Dokime: perturbation sets and independent evaluation
                           Paideia: offline training/adapter promotion, separate from Krisis
```

**Kybernetes owns when to invoke Krisis and the final response state.** Krisis returns a ranked selection with features, scores, rejected candidates and abstention reason. It does not send an answer directly to Prosopon, authorize a tool, modify LTHCM, change Hestia weights or declare itself correct. Phylax restrictions and authoritative source/version checks apply **before and after** selection. Dianoia continues to own walk execution and answer verification. Dokime owns frozen perturbation datasets and verdicts.

Kleroterion's current specification gives it bounded stochastic **Dianoia action selection** from a host-supplied eligible action set. Krisis would select **answer candidates or passages**, never walk actions. Dianoia revalidates a Kleroterion action before execution; Kybernetes revalidates a Krisis answer or passage before publication. Keep their policy state, randomness, metrics and approvals separate. Neither component may expand eligibility, override authoritative checks, or treat a scored vector as a source identity. Kleroterion's proposed retrieval Recall@10 benchmark also does not establish that Krisis improves answer quality; the two need distinct held-out tests.

Build Krisis as an installable library and thin adapter first. A standalone repo is reasonable for ownership and testing; a separate network service adds latency, authentication and version skew without a demonstrated need. Add a service only if multiple processes truly require it.

## Priority gaps and repairs

| Priority | Draft gap | Why it matters | Required improvement / proof |
| --- | --- | --- | --- |
| P0 | The draft uses model negative log-likelihood as `L(m)` for a system configuration. A completion likelihood describes text under a specific model and prompt, not the code length of weights, memory and sampler configuration. | The proposed `2^-L(m)` mixes different objects and can favor fluent, confidently wrong text. Length and tokenization also change the value. | Rename this a calibrated **model-likelihood feature** for answer candidates; never call it a universal configuration prior. Record model/tokenizer/template/version, logprob availability and normalization. Compare with a no-logprob selector. |
| P0 | The “prior is free” claim is unverified. Local Hestia currently forwards chat request bodies and backend responses; it does not guarantee token-level logprobs or a stable cross-backend format. | A core feature may be absent, expensive, or incomparable across models. | Add a Hestia capability probe and schema contract. If logprobs are unavailable, use a documented feature-free baseline or disable the feature; do not invent scores. Test on the pinned llama.cpp build/model. |
| P0 | Agreement can reward the same wrong answer across paraphrases. Seed or time shifts may legitimately change the answer; model-generated paraphrases can leak the answer or alter meaning. | Stability alone is not correctness or grounding. | Define task-specific equivalence and invariance labels. Check that perturbations preserve the answer when intended; score citation support, factual correctness and calibrated abstention separately from agreement. Include stable-wrong and valid-answer-change tests. |
| P0 | `1[F>=tau]`, `argmax(-L+lambda F)` and exponential-weights updating are three distinct selection policies with no stated precedence. `L` may be in bits, nats or bytes; `F` mixes agreement and reward without scale. | The same candidates may receive contradictory decisions and results will not be reproducible. | Choose one v0 decision rule with feature units, normalization, thresholds, tie policy and abstention. Freeze coefficients on development data. Keep Hedge/config selection as a separate later experiment. |
| P0 | “No single-sample fitness anywhere in the stack” is too sweeping, while perturbing every live query/candidate may multiply model calls beyond the 16 GB and latency budget. | Either cost explodes or the rule is quietly violated. | Require multiple validated perturbations for a **reported consistency estimate**, but permit a safe baseline/abstention when none are affordable. Set per-turn candidate, perturbation, token, time and GPU budgets. Record which fitness features were measured versus unavailable. |
| P0 | When no candidate passes `tau`, behavior is unspecified. | A selector may silently relax a safety gate or return a known-bad answer. | Return a typed `NO_ELIGIBLE_CANDIDATE` with reasons. Kybernetes may retrieve more evidence, ask a clarifying question or abstain under a bounded policy. Never relax Phylax, provenance or contradiction gates. |
| P1 | `S` is described as an “algorithmic sufficient statistic” without a definition or evidence. LTHCM paths and topology are versioned state, not a guaranteed sufficient summary of the workload. | Scores may drift when the index changes; online updates can invalidate cached comparisons. | Replace the assertion with explicit, measured features such as path length, edge confidence, source diversity and index generation. Ablate each feature against a flat reranker; freeze snapshot per decision. |
| P1 | Retrieval reweighting and output selection have different objects, metrics and owners. | A single opaque score risks double-counting LTHCM's ranking and Dianoia verification. | Expose two adapters with separate schemas: `rank_passages` returns stable passage IDs and citations; `select_answers` returns candidate IDs and final status. Do not mutate LTHCM graph weights in v0. |
| P1 | Online `mu_{t+1}` assumes immediate trustworthy loss for each arm. Live feedback is delayed, sparse, biased toward the selected output, and may be absent. | Self-reinforcing selection can degrade without a visible signal. | Keep v0 read-only/frozen. Later run shadow updates with explicit reward provenance, logged selection propensity/exploration, bounded weights, holdout checks, rollback and Dokime promotion. No unsupervised agreement-only online promotion. |
| P1 | Workload-derived `G` is not enough to guarantee semantic preservation. A semigroup requirement is unnecessary for a finite test suite unless composition is actually used and validated. | Perturbation quality can dominate the measured “fitness.” | Dokime owns versioned perturbation generators and human/audited validation slices. Record transformation type, source query, expected invariant, seed and generator version. Keep private live queries out of reusable caches unless retention/consent allows it. |
| P1 | The draft logs prior, fitness, `tau` and `S` version but omits candidate lineage, safety status, budget and model/index identity. | Decisions cannot be replayed or audited. | Log candidate IDs, source/citation IDs, feature values and missingness, policy version, eligibility mask/reasons, Hestia model digest, LTHCM generation, perturbation-set hash, latency/tokens and final Kybernetes trace ID. Redact sensitive content. |

## Recommended v0 decision contract

Input: one request/trace ID; a fixed set of candidate answers **or** passages; their stable IDs and provenance; authoritative eligibility result; model/index versions; optional measured likelihood, consistency and support features; and a hard inference budget. Candidates must all answer the same frozen query against the same evidence snapshot. Do not compare raw logprobs from different models as though they share a scale.

Output: selected candidate ID or `NO_ELIGIBLE_CANDIDATE`; ranked alternatives; rejection reasons; feature values with missingness; policy and snapshot versions; cost; and a status Kybernetes can display. Recheck candidate/source validity before publication.

For eligible candidates, begin with a simple, predeclared rule such as:

`score(c) = w_support * grounded_support(c) + w_consistency * validated_consistency(c) + w_model * calibrated_model_feature(c) - w_cost * normalized_cost(c)`

Hard access, provenance, contradiction and version checks are **outside** this score and cannot be compensated by a high likelihood. Tune weights and thresholds on development data, freeze them for held-out evaluation, and allow missing optional features without replacing them with fabricated values. Start with answer selection; enable passage reranking only after its separate retrieval benchmark shows a gain.

## Evaluation required before release

1. **Datasets:** held-out grounded-answer and contradiction cases, multi-hop cases where Dianoia may help, exact-fact questions, and cases where paraphrase preserves meaning versus time/context genuinely changes the answer. Keep source, model and perturbation revisions fixed.
2. **Baselines:** Kybernetes/Dianoia without Krisis; highest model likelihood when available; majority/self-consistency vote; citation/verification-only rerank; and Krisis ablations without each feature. For passage reranking, compare LTHCM's original ranking and a strong flat/hybrid reranker at equal candidate/evidence budget.
3. **Metrics:** answer correctness and grounded claim/citation support, contradiction miss rate, abstention risk/coverage, stable-wrong rate, selection regret versus an oracle **upper bound**, p50/p95 latency, tokens/GPU time, extra model calls and failure rate. Report agreement as diagnostic, not the primary success metric.
4. **Protocol:** paired per-question predictions and intervals, multiple seeds where stochastic, all failures in denominators, frozen decision rule, independent final test, and per-slice results. Dokime should render the report from a replayable bundle; its current synthetic 25-item smoke task cannot establish this gain.
5. **Gate:** promote only if a predeclared primary quality metric beats the strongest equal-budget baseline with a meaningful paired interval, while contradiction, abstention, latency and resource guardrails pass. Otherwise keep Krisis opt-in and report the negative result.

## Direct answers to the draft's five open questions

1. **`tau`:** use a fixed, dev-calibrated threshold for v0. Do not use a rolling quantile: it forces some candidates to fail even when all are good, and some to pass when all are bad. Detect drift separately; require Dokime review before changing the threshold.
2. **Agreement metric:** task-specific answer/claim equivalence with citation support. Exact match is suitable only for normalized closed-form answers. Embedding similarity is a screening feature, not a correctness verdict; model judging requires calibration and an independent audit set.
3. **`S`:** versioned, immutable LTHCM/index snapshot per request. Recompute or invalidate cached features when generation changes. Online structure updates are a later, separately evaluated policy.
4. **Harness ownership:** Dokime versions/generates and validates perturbation sets, scores held-out outcomes and issues comparisons. Krisis consumes a bounded set and selects; Kybernetes orchestrates.
5. **Nothing passes:** return `NO_ELIGIBLE_CANDIDATE` with reasons; Kybernetes follows a bounded retry/ask/abstain policy. Do not silently lower a hard gate.

## Suggested first-build sequence

1. Freeze the `select_answers` schema, authoritative eligibility order and Hestia capability probe. Implement a deterministic selector with no logprob requirement.
2. Add Dokime perturbation fixtures and independent correctness/grounding labels. Establish no-Krisis and simple reranker baselines.
3. Add optional consistency and calibrated likelihood features one at a time, with cost and error logging. Evaluate against stable-wrong cases and held-out tasks.
4. Integrate as a Kybernetes library adapter with Phylax and citation rechecks. Measure end-to-end cost on the target hardware.
5. Only after demonstrated answer-selection gain, test separate LTHCM passage reranking. Consider online Hedge or configuration-space selection after reliable reward, propensity, rollback and shadow-evaluation infrastructure exists.

This scope makes the first version production-oriented: bounded decisions, explicit abstention, clear ownership, reproducible evidence and no hidden learning from unverified feedback.
