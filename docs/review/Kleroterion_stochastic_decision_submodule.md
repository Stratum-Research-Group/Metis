# Kleroterion

A stochastic decision submodule for the stack, with Dianoia as its first consumer.

Status: revised proposed specification, 2026-09-29. No implementation, repository creation, benchmark result, or paper evidence is implied by this document.

## 1. Purpose, scope, and architectural decision

Build Kleroterion as a **standalone repository/package in the stack**, with **Dianoia's routing/walk policy as its first consumer contract**. Packaging and host semantics are separate decisions: the module is independently maintained, while its initial interface is grounded in eligible graph actions and their scores. A repository scaffold can be created before evaluation; empirical retrieval or topology claims cannot.

The reusable mechanism combines continuous Gaussian proposals, masked softmax over action logits, reproducible selection, and optional entropy feedback. Its first use is to choose a Dianoia walk action, not to implement diffusion over LTHCM's memory graph.

This revision supersedes the earlier recommendation to keep Kleroterion inside an existing subproject and promote it only later. It also replaces the arbitrary per-feature noise default and always-on conditional-entropy controller.

The user-supplied architecture establishes LTHCM as a pinned, read-only submodule and identifies an existing stochastic-resonance retrieval baseline. These are integration requirements here; current repository APIs and baseline behavior have not been independently inspected for this document. The interfaces below are proposed contracts, not claims that matching functions already exist.

### In scope

- A standalone stochastic decision package and a Dianoia adapter.
- Explicit action IDs, host-owned action logits, and immutable memory references.
- Dimension-aware proposal budgets and authoritative eligibility filters.
- Separate conditional and marginal entropy diagnostics.
- A frozen-snapshot comparative retrieval evaluation contract.

### Out of scope

- Graph heat diffusion, personalized PageRank, persistent-homology retrieval, or claims of topological denoising.
- Modifying LTHCM, its index, graph, embeddings, filters, or pinned revision as a side effect.
- Learning the scorer, defining rewards, or interpreting sampling noise as calibrated uncertainty.
- Publishing retrieval gains based only on a mathematical specification or an unrun evaluation harness.

## 2. Architecture and data flow

```text
Dianoia: query + walk state + immutable memory snapshot
                          |
Host enumerates actions and applies authoritative eligibility
                          |
Kleroterion proposes bounded changes to permitted continuous policy features
                          |
Host rechecks proposal feasibility and scores each action
                          |
Fixed eligible action set → masked softmax per proposal
                          |
Average proposal probabilities → marginal action distribution
                          |
Argmax or categorical sample → ActionDecision (never a document)
                          |
Dianoia revalidates action and snapshot before executing the walk
                          |
Existing host retrieval/output checks → real document IDs and citations
```

The first-consumer action space is an ordered list of legal next-hop edge/node actions and, when appropriate, a host-authorized stop action. The adapter must map these abstract action types to actual Dianoia types before implementation acceptance. Kleroterion cannot invent actions or expand the supplied action set.

**Dianoia owns:** candidate enumeration, causal/contradiction/provenance filters, scoring, walk budgets, execution, returned documents, and citation construction.

**Kleroterion owns:** proposal generation, stable masked softmax, probability aggregation, sampling, diagnostics, and optional temperature-state updates. It never commits a walk step or mutates memory.

**LTHCM boundary:** all reads use the pinned revision and a declared immutable snapshot. Dependency upgrades, graph writes, embedding changes, and index rebuilds require separate work outside this contract.

## 3. Mathematical formulation

### 3.1 Continuous Brownian increments and Gaussian proposals

Let X be a d-dimensional vector of normalized, perturbable **policy features**. X is not a graph node, document identity, or memory topology. For a general drift–diffusion process,

\[
 dX_t=\mu(X_t,t)\,dt+L(X_t,t)\,dW_t,
\]

where W is an r-dimensional Wiener process and L is d-by-r. Euler–Maruyama gives

\[
 \widetilde X=X+\mu(X,t)\Delta t+L(X,t)\sqrt{\Delta t}\,\epsilon,
 \quad \epsilon\sim\mathcal N(0,I_r).
\]

The stochastic increment has covariance

\[
 \operatorname{Cov}(\widetilde X-X\mid X)=\Delta t LL^\top.
\]

For constant coefficients this is an exact unconstrained increment. State-dependent coefficients require a discretization study. The first Dianoia adapter uses **zero drift and independent proposals around the supplied features**, with no persistent Brownian trajectory. Repeated independent perturbations around a fixed state are Gaussian proposals; they are not diffusion on the graph.

A future trajectory mode would require explicit state ownership, time semantics, and accumulation/reset rules. It is not part of the initial adapter.

### 3.2 Dimension-aware noise budget

Use an RMS total displacement budget b per decision proposal, rather than a fixed per-coordinate sigma:

\[
 \mathbb E\|L\sqrt{\Delta t}\epsilon\|_2^2
 =\Delta t\operatorname{tr}(LL^\top)\le b^2.
\]

For isotropic proposals over d active normalized coordinates,

\[
 L=\sigma I_d,\qquad \sigma=\frac{b}{\sqrt{d\Delta t}}.
\]

Here b is a per-proposal budget, so changing dt while holding b fixed rescales L. A physical continuous-time experiment must instead hold L fixed and study variance proportional to dt. Do not conflate these parameterizations.

The prior sigma=0.05 at d=768 and dt=1 implied RMS displacement approximately 1.39. That default is removed. Perturbations are disabled by default (`noise_rms_budget=0`) until a task-specific budget passes stability evaluation.

An RMS budget is not a hard bound: Gaussian tails remain unbounded. When exploration is enabled, require a host-approved hard radius R >= b > 0 and clip the raw increment to that radius before scoring, or use another explicitly specified bounded proposal policy. Clipping changes its distribution; Gaussian covariance tests apply before clipping. Enforce the hard radius and immutable-coordinate constraints again after any domain projection. Invalid final proposals fail the decision without a hidden fallback.

For a supplied anisotropic matrix, validate its shape, finiteness, and trace budget. Record feature normalization and active coordinates. Never perturb document IDs, graph edges, provenance fields, causal timestamps, or contradiction labels.

A noise budget is accepted only if a held-out stability sweep meets declared bounds on top-k overlap and Recall@k degradation against the same unperturbed scorer. Also report rank correlation on a fixed candidate universe. Norm bounds alone do not establish rank stability.

### 3.3 Host action logits and stable masked softmax

For fixed eligible actions a_1,...,a_K, Dianoia supplies

\[
 z_i^{(m)}=f_\theta(q,s,\widetilde X^{(m)},a_i,\text{snapshot}),
\]

where q is the query and s is the authoritative walk state. The scorer version and feature schema are part of the adapter contract. Larger logits mean better actions; negate costs explicitly if needed.

For the nonempty eligible set V and temperature tau > 0, compute

\[
 p_i^{(m)}=
 \begin{cases}
 \dfrac{\exp((z_i^{(m)}-c_m)/\tau)}
 {\sum_{j\in V}\exp((z_j^{(m)}-c_m)/\tau)},&i\in V,\\
 0,&i\notin V,
 \end{cases}
 \quad c_m=\max_{j\in V}z_j^{(m)}.
\]

Reject nonfinite eligible logits and unsupported numerical ranges. Subtract the maximum before temperature scaling. Equal logits give uniform probabilities at every positive temperature.

For M independent proposals at the **same query, walk state, snapshot, eligible set, and temperature**, estimate the marginal distribution by

\[
 \widehat q_i=\frac1M\sum_{m=1}^{M}p_i^{(m)}.
\]

Average probabilities, not logits. In general, expected softmax is not softmax of expected logits. Sample an action from q-hat in exploratory mode; choose its stable-order argmax in deterministic mode. With nonzero proposal noise, argmax alone does not make a run deterministic.

### 3.4 Why this is not graph-diffusion retrieval

No graph transition matrix or Laplacian is defined by Gaussian feature perturbation. This spec implements neither a heat operator nor a restart walk. A separate future LTHCM retrieval proposal must define:

- Graph construction, edge direction/weights, normalization, and snapshot semantics.
- The exact transition matrix or Laplacian and row/column convention.
- Seed/query distribution, diffusion time, and either a heat operator such as exp(-tL) or an explicitly distinct restart/PPR operator.
- Dangling nodes, disconnected components, unreachable relevant nodes, and restart behavior.
- How graph mass becomes document rankings while preserving host filters.
- Comparisons against current retrieval and stochastic resonance at matched budgets.

Long-time heat diffusion emphasizes stationary/low-frequency structure under the chosen operator; it does not automatically identify persistent homology classes. Any persistence claim requires a defined filtration, persistence computation, and task-specific evidence. No “topological denoising for free” claim is supported here.

## 4. Entropy-controlled exploration

### 4.1 Measure the intended distribution

For H(p) = -sum p_i log(p_i), with zero terms omitted, report separately:

\[
 \widehat H_{\rm conditional}=\frac1M\sum_m H(p^{(m)}),
 \qquad \widehat H_{\rm marginal}=H(\widehat q),
\]

\[
 \widehat J=H(\widehat q)-\frac1M\sum_m H(p^{(m)})\ge0.
\]

J measures disagreement among proposal distributions. It is not calibrated epistemic uncertainty. Finite-M marginal entropy is an estimate, with sampling error and nonlinear-estimator bias; M=1 cannot reveal between-proposal exploration.

Counterexample: noise flips a huge two-action logit gap with equal probability. Each proposal has nearly zero entropy, while marginal choice entropy approaches log(2). A conditional-entropy controller would incorrectly raise temperature if its stated goal were aggregate exploration.

Normalize marginal entropy by log(|V|) when |V| > 1. For one eligible action, return probability one, define normalized entropy as zero, and freeze feedback. For none, return `NO_ELIGIBLE_ACTION`; only the host may decide to stop or abstain.

### 4.2 Optional controller, disabled by default

Feedback uses **estimated marginal entropy**, never per-proposal conditional entropy as a proxy. It requires M >= 2 plus an evaluation-validated proposal count; two samples alone do not guarantee a reliable estimate. Log M and proposal variability, and freeze adaptation when a configured uncertainty check rejects the estimate. The uncertainty estimator and threshold must be declared before controller evaluation.

For an eligible estimate h_t, use bounded log-temperature feedback:

\[
 \bar h_t=(1-\alpha)\bar h_{t-1}+\alpha h_t,
 \quad e_t=h_t^*-\bar h_t,
\]

\[
 \log\tau_{t+1}=\operatorname{clip}
 (\log\tau_t+\eta e_t,\log\tau_{\min},\log\tau_{\max}).
\]

Initialize the EMA to the first eligible observation. Apply a deadband |e_t| <= delta. The returned decision uses tau_t; updates affect the next accepted decision only. Scope/reset controller state by query episode and policy profile; do not aggregate unrelated queries or candidate sets as if they were noise draws from one decision. Reset smoothing when action support changes.

There is **no universal entropy target**. The host supplies a validated query-profile policy. Exact-fact queries default to zero proposal noise, argmax, and no feedback; Kleroterion never forces such queries toward an arbitrary entropy level. Exploratory profiles must justify their targets using retrieval metrics.

For fixed logits, conditional entropy is nondecreasing in temperature. Entropy of a mixture across different proposal logits need not share that monotonicity; this feedback law is experimental, not a guaranteed convergent controller. Acceptance requires an empirical temperature-response check for the intended profile. Freeze or disable it for unstable or adverse response, and report saturation. Keep noise budget fixed while tuning temperature; simultaneous adaptation is outside this version.

## 5. Safety and identity invariants

1. **Authoritative filters cannot be bypassed.** Causal, contradiction, provenance, access, and snapshot constraints are evaluated by the host using original authoritative data. Perturbed features cannot alter those facts.
2. **Eligibility can only narrow.** Proposal-dependent checks may remove actions, never restore a host-rejected action. For one probability batch, use the intersection of the original eligibility and every proposal's eligibility, then score/normalize all proposals on that same set. An empty intersection fails closed.
3. **Revalidate before execution.** Dianoia checks the selected action against current authoritative constraints and the expected snapshot/version. A stale or newly invalid action aborts the transaction; rebuild the request instead of executing it or silently selecting another action.
4. **Actions are not evidence.** Kleroterion returns action IDs and probabilities. Only the host resolves actual retrieved documents and validates citations after the walk. Existing output-level checks remain mandatory even for individually legal steps.
5. **Weighted values are a distinct type.** An optional reusable weighted-feature utility may compute sum(q_i v_i), but the Dianoia action adapter does not expose it as a selection mode. A `WeightedFeatureVector` must never be accepted as a document, node identity, citation, or executable action.
6. **No implicit writes.** Proposals, failures, retries, and evaluation runs cannot modify LTHCM or advance its pinned revision. Persisted controller/RNG updates commit only with a successfully accepted host decision.

## 6. Proposed module interfaces

```text
propose_action(request, config, controller_state, rng_state) -> ActionDecision

RoutingRequest:
  request_id, episode_id
  query_context                         # read-only
  walk_state                            # authoritative, read-only
  snapshot_id, lthcm_revision
  feature_schema_version
  policy_features: finite vector[d]
  actions: ordered ActionID[K]           # unique, host-enumerated
  authoritative_mask: bool[K]
  policy_profile                        # exact_fact or validated exploration profile

Host callbacks:
  proposal_features(x, bounded_increment) -> x_proposed
  proposal_mask(request, x_proposed) -> bool[K]
  score_actions(request, x_proposed, actions) -> finite logits[K]
  revalidate_before_execute(action, snapshot_id) -> accept or typed rejection

ControllerState:
  log_temperature
  smoothed_marginal_entropy: optional float
  support_fingerprint, profile_version, episode_id
  eligible_update_count

ActionDecision:
  request_id, snapshot_id, lthcm_revision
  selected_action_id: ActionID
  action_ids, marginal_probabilities
  conditional_entropy, marginal_entropy, proposal_disagreement
  proposal_count, entropy_estimate_status
  temperature_used, temperature_next
  controller_state_next, rng_state_next
  diagnostics: displacement, clipping_count, eligible_count,
               saturation, scorer_calls, elapsed_time
```

Use separate random streams for proposals and action selection. Return typed errors for invalid configuration, malformed inputs, empty eligibility, invalid proposals, scorer failures, and numeric failures. Work on local copies of RNG/controller state; callbacks must be side-effect free. The host uses request IDs and version checks to prevent duplicate execution. An explicit adapter transaction coordinates action acceptance with state commitment; Kleroterion does not implement an external transaction itself.

## 7. Configurable parameters

Defaults are conservative baselines, not empirically tuned settings.

| Parameter | Default / requirement | Meaning |
|---|---|---|
| `proposal_mode` | independent | Fresh proposals from host policy features |
| `dt` | 1.0 | Positive; unit declared in proposal configuration |
| `noise_rms_budget` | 0 | Total normalized displacement budget; no per-feature default |
| `noise_hard_radius` | required if noise > 0 | Finite radius R >= b |
| `diffusion_matrix` | isotropic budget-derived | Optional matrix satisfying trace budget |
| `proposal_count` | 1 | M; controller requires validated M >= 2 |
| `selection_mode` | argmax | Adapter supports argmax or sample |
| `temperature_initial` | 1.0 | Positive, inside explicit bounds |
| `temperature_bounds` | required for adaptation | Finite 0 < min <= max |
| `adapt_temperature` | false | Enabled only for validated exploration profiles |
| `entropy_target` | unset | Required profile policy when adaptation is enabled |
| `entropy_ema_alpha` | required for adaptation | In (0,1] |
| `controller_gain` | required for adaptation | Finite, nonnegative, response-tested |
| `entropy_deadband` | required for adaptation | In [0,1] |
| `entropy_quality_gate` | required for adaptation | Estimator, sampling budget, and freeze criterion |
| `seed` | caller supplied | Record RNG algorithm and version |
| `stability_limits` | required before enabling noise | Minimum top-k overlap and maximum recall loss |

Exact-fact profiles enforce noise=0, argmax, and adaptation=false regardless of exploratory defaults. Record scorer version, configuration hash, feature schema, action ordering, and snapshot identifiers for replay. Cross-platform bitwise equality is not assumed.

## 8. Pseudocode

```text
function propose_action(req, cfg, ctrl, rng):
    validate(req, cfg, ctrl)
    effective = resolve_profile(req.policy_profile, cfg)
    enforce_exact_fact_overrides(effective)
    local_rng, next_ctrl = clone(rng), clone(ctrl)
    tau = effective_temperature(ctrl, effective)
    proposals = []
    common_mask = copy(req.authoritative_mask)

    repeat effective.proposal_count times:
        raw_increment = budgeted_gaussian(effective, local_rng.proposals)
        increment = clip_to_radius_if_enabled(raw_increment, effective)
        x = proposal_features(req.policy_features, increment)
        verify_finite_shape_radius_and_immutable_fields(req, x, effective)
        common_mask &= proposal_mask(req, x)
        proposals.append(x)

    if no_true(common_mask):
        return error(NO_ELIGIBLE_ACTION)    # no persisted state change

    distributions = []
    for x in proposals:
        logits = score_actions(req, x, req.actions)
        validate_logits(logits, common_mask)
        distributions.append(stable_masked_softmax(logits, common_mask, tau))

    q = mean(distributions, axis=proposals)
    H_cond = mean(entropy(p) for p in distributions)
    H_marg = entropy(q)
    disagreement = H_marg - H_cond
    selected = sample(q, local_rng.selection) if effective.mode == SAMPLE
               else stable_argmax(q, common_mask)

    reset_smoothing_if_episode_profile_or_support_changed(next_ctrl, req, common_mask)
    quality = assess_entropy_estimate(distributions, effective)
    if effective.adapt and count_true(common_mask) > 1 and quality.accepted:
        h = H_marg / log(count_true(common_mask))
        next_ctrl = bounded_marginal_entropy_update(next_ctrl, h, effective)

    return ActionDecision(selected, q, H_cond, H_marg, disagreement,
                          temperature_used=tau, next_ctrl, local_rng,
                          request_and_snapshot_ids, diagnostics)

function host_accept(decision):
    check_request_not_already_executed(decision.request_id)
    revalidate_before_execute(decision.action, decision.snapshot_id)
    execute_and_commit_with_host_transaction(decision)
    # A rejection discards proposed RNG/controller state and requires a new request.
```

The intersection policy makes support conservative and can depend on M; log support attrition and hold this policy constant across experiments. With proposal-dependent filtering, report the distribution as the batch policy's distribution, not an unbiased estimate of an unconstrained Gaussian-mixture policy. Empty-intersection rates are an evaluation metric.

## 9. Repository and integration guidance

Proposed layout; no repository is created by this document update:

```text
kleroterion/                         # standalone repository in the stack
  stochastic_decision_submodule.md
  src/
    proposals.*
    masked_softmax.*
    marginal_policy.*
    entropy_feedback.*
    interfaces.*
  adapters/dianoia/
  tests/
  eval/
    snapshot_manifest.*
    baseline_adapters/
    metrics.*
    experiment_profiles/
```

Dianoia consumes a pinned Kleroterion version through the stack's chosen dependency mechanism. Whether that mechanism is a Git submodule or a package dependency is an implementation choice, distinct from the agreed standalone-repository boundary.

Before connecting the adapter, map real Dianoia action types, scorer inputs, filter order, stop/abstention behavior, and transaction semantics to section 6. Inspect the existing LTHCM stochastic-resonance path and document overlap; reuse compatible interfaces instead of silently duplicating it. Keep the LTHCM revision and snapshot fixed throughout the comparison.

Start with scorer-only behavior, then fixed-temperature/noise experiments, then optional marginal feedback. Log probabilities, proposal budgets, conditional and marginal entropy, proposal count, eligible-set attrition, failures, latency, and outcomes. Do not log sensitive raw query/state data by default.

## 10. Evaluation contract and acceptance criteria

### 10.1 Frozen experiment manifest

Freeze corpus/graph/index snapshots, LTHCM and Dianoia revisions, scorer/embedding versions, query set, relevance judgments, contradiction annotations/rules, provenance rules, and exact-fact versus exploratory query labels. Separate tuning and held-out test queries. Record hardware, warm/cold-cache policy, walk depth, returned k, seeds, proposal count, scorer calls, and compute budgets.

### 10.2 Primary metric and baselines

**Primary retrieval metric: Recall@10**, macro-averaged per query over judged relevant eligible documents. Queries with no relevant eligible documents are evaluated separately for correct abstention. If the product requires another k, change and freeze it before evaluation; do not choose k after seeing results.

Secondary metrics: MRR@10, nDCG@10, top-10 overlap with the unperturbed scorer, contradiction rate, provenance/causal violations, abstention performance, eligible-set attrition, empty-intersection rate, and end-to-end p50/p95 latency. Report exact-fact and exploratory strata separately as well as aggregate results.

Required arms:

| Arm | Proposal noise | Selection | Feedback |
|---|---|---|---|
| Current LTHCM retriever | Existing behavior | Existing behavior | Existing behavior |
| Existing LTHCM stochastic resonance | Existing behavior | Existing behavior | Existing behavior |
| Dianoia scorer-only | Off | Argmax | Off |
| Softmax-only | Off | Sample, fixed temperature | Off |
| Gaussian-only | On | Argmax of proposal scorer; M=1 | Off |
| Gaussian + softmax | On | Sample marginal policy | Off |
| Full Kleroterion controller | On | Sample marginal policy | Marginal entropy |

The existing-retriever arms may use different orchestration; document that difference. Add Dianoia's current end-to-end configuration if not identical to an arm above. Compare both matched-compute budgets and operational configurations, including the full cost of M scorer calls. To isolate feedback, hold proposal count, noise budget, initialization, masks, and walk budgets constant between Gaussian + softmax and the full controller. Use multiple seeds and paired query-level comparisons; predeclare confidence interval methodology and seed/query aggregation.

### 10.3 Explicit acceptance gates

Before enabling noise or running the held-out acceptance evaluation, the manifest must supply numeric values for:

- Minimum top-10 overlap `overlap_min` and maximum tolerated Recall@10 loss `recall_loss_max` during noise-budget calibration.
- Minimum worthwhile Recall@10 improvement `gain_min` over each designated retrieval comparator.
- Exact-fact noninferiority margin `exact_fact_loss_max`.
- Maximum acceptable contradiction rate `contradiction_max` and its permitted regression margin.
- Absolute p95 latency budget and allowed latency regression.
- Maximum failed/empty-intersection rate and entropy-estimation error tolerance.

These values are intentionally unset pending workload and product requirements. Missing values mean **evaluation acceptance is blocked**, not automatically passed. Repository scaffolding and correctness implementation can proceed.

For a retrieval-gain claim, require the predeclared confidence lower bound on paired Recall@10 improvement to exceed `gain_min` against the designated current and stochastic-resonance baselines. Report all comparisons, including losses. Exact-fact performance must meet its noninferiority gate. Contradiction and latency gates must pass. Hard causal/provenance/eligibility violations must be zero in the evaluated outputs; observed zero is not a universal correctness proof.

Define contradiction rate as the fraction of returned documents rejected by the frozen contradiction adjudication procedure; additionally report query-level incidence and coverage so empty outputs cannot hide failures. Unjudged cases and adjudicator uncertainty must be reported rather than counted as clean.

### 10.4 Correctness and regression tests

- Softmax: finite nonnegative probabilities, unit sum within declared numeric tolerance, zero masked mass, shift invariance, equal-logit uniformity, stable ties, extreme-logit handling.
- Diffusion: unbounded raw increments match mean zero and covariance dt * L * L-transpose; fixed-L variance scales with dt. Budget parameterization has dimension-independent RMS across tested d. Final bounded proposals satisfy radius and feature invariants.
- Masks: a rejected action never becomes eligible after perturbation; masks intersect before normalization; all-invalid and stale-snapshot cases fail without execution.
- Identity: weighted feature vectors are rejected wherever action/document/citation types are required. Selected routes cannot bypass output-level contradiction/provenance checks.
- Entropy: reproduce the two-action random-flip example; conditional entropy stays near zero while estimated marginal entropy approaches log(2) with adequate M. Verify the controller consumes the latter. Test finite-M quality gates, singleton support, support resets, saturation, and nonmonotone-response disablement.
- Profiles: exact-fact overrides always disable noise, sampling, and adaptation. No default entropy target is injected.
- Replay: pinned environments reproduce outputs from saved states; independent episodes do not share controller state; errors and rejected actions do not commit RNG/controller updates; retries cannot execute twice.
- Read-only boundary: integration tests verify no LTHCM working-tree, pin, graph, embedding, or index changes after successful calls, failures, and evaluation.
- Statistical tests: declare sample sizes, confidence levels, and multiple-test handling. Test Gaussian moments before clipping and bounded-policy properties after clipping.

## 11. Evidence and publication boundary

A scaffold establishes packaging. Correctness tests establish bounded implementation properties. An evaluation harness establishes a way to measure performance. Only completed comparative runs establish results for the frozen workload and budgets evaluated.

This Kleroterion specification supports none of the following claims by itself: improved retrieval, improved persistence, heat-kernel retrieval, topological denoising, calibrated uncertainty, or production readiness. The future graph-diffusion proposal needs its own operators, implementation, and evidence. The standalone Kleroterion repository may proceed independently of those claims.
