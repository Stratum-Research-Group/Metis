# Shared production build contract for the local AI stack

Use this file together with each component build prompt. It is a **build specification for proposed components**, not a claim that they are implemented. The existing Stoma, LTHCM, Dianoia, Hestia and Dokime README/improvement files are design inputs. The [Kleroterion specification](../Kleroterion/Kleroterion_stochastic_decision_submodule.md) and [Krisis review](krisis-review.md) add two proposed selection components; neither document is implementation or benchmark evidence. Before implementing an adapter, inspect the pinned dependency's real API and record any mismatch. Do not infer an API from a README alone.

## Goal and boundaries

Build a local, single-owner assistant stack that can ingest documents, retrieve evidence, deliberate when useful, select among eligible walk actions and candidate answers, use approved tools, serve local models, evaluate changes, learn from valid outcomes, and provide text/voice access. Runtime must work without paid APIs or cloud services. Model downloads and dependency installation can be a separate, explicit setup phase; runtime must fail clearly when required local assets are missing.

The ownership boundaries are:

| Owner | Responsibility |
| --- | --- |
| Stoma | Ingestion, chunking, provenance, index lifecycle |
| LTHCM | Retrieval and graph/verification signals |
| Dianoia | Bounded evidence walk and answer/verification outcome |
| Kleroterion | Bounded stochastic selection among Dianoia-supplied eligible walk actions; probabilities and diagnostics, no action execution |
| Hestia | Local model serving, model identity and GPU admission |
| Krisis | Optional selection among already eligible answer candidates; passage reranking only after separate evaluation |
| Dokime | Reproducible evaluation and release comparisons |
| Phylax | Policy decisions, approvals and auditable guarded actions |
| Kybernetes | Planning, orchestration, budgets, trace and client API |
| Prosopon | Authenticated text/voice interface and approval UI |
| Paideia | Validated datasets, baselines, training and adapter registry entries |

**Selection boundaries:** Dianoia enumerates and scores legal actions, applies authoritative causal/provenance/contradiction and snapshot constraints, and revalidates a Kleroterion-selected action before executing it. Kleroterion cannot invent an action, widen eligibility, return a document as an action, mutate LTHCM, or commit a walk step. Kybernetes decides whether to invoke Krisis and remains responsible for the final response. Krisis may rank eligible answer candidates but cannot authorize tools, change model weights or memory, bypass Phylax, or turn agreement into a claim of correctness. Kybernetes revalidates the selected answer, citations, policy and snapshot before publication. Neither selector replaces Dianoia verification or Dokime evaluation.

## Dependency and version rules

- The v2 prompts require read-only LTHCM and Dianoia submodules at `8a71605379ee32059e714feb3648cf74f4b42dc6` and `da7ce83cb05f3b44764fc59ab5c325995c6fd07c`. Preserve those exact pins in the new projects unless the owner changes the requirement. Never edit, copy into project-owned source, or push submodule contents.
- Stoma's supplied README documents a different LTHCM pin: `336f4af9f624aca5b4978a5e3ce222ce9638da35`. Treat this as a compatibility question to **test and resolve explicitly**, not a reason to silently change either pin.
- Record the tested combination of component commits, schema versions, model digests and index format in `stack.lock.json` or an equivalent machine-readable manifest. A startup/doctor check must reject incompatible combinations with an actionable error.
- When Kleroterion and Krisis have implementation revisions, pin and record them separately. Dianoia consumes a tested Kleroterion package or submodule version; Kybernetes consumes a tested Krisis adapter version. Do not invent commit pins from the design documents or require either selector in the default path before its release gate passes.
- Preserve a stable library or HTTP adapter boundary. A submodule is source provenance, not permission to reach into arbitrary internal files or assume every project must run in one process.

## Minimum versioned contracts

Publish JSON Schema/OpenAPI or equivalent typed schemas with examples and compatibility tests. An owner may choose the exact package location; each schema must have an explicit version.

1. **Identity and trace:** `request_id`, `turn_id`, `trace_id`, `session_id`, component revision, UTC timestamp, and monotonic duration. Do not confuse epoch seconds with relative time. Propagate trace IDs across all calls.
2. **Evidence:** stable source and passage IDs, source hash and location, index generation, encoder identity, document timestamp, query `as_of`, rank/score, graph availability and causal fallback. Citation IDs must resolve to the exact passage version used.
3. **Answer outcome:** distinguish `verified`, `conflicted`, `insufficient_evidence`, `verification_unavailable`, `failed` and `cancelled`. Record verifier mode and evidence; a completed model response is not a correctness label.
4. **Model identity:** server/backend version, model file digest, quantization, tokenizer/chat template, generation settings and actual served model ID. A friendly alias alone is insufficient for evaluation or adapter compatibility.
5. **Authorization:** authenticated subject, verb, canonical resource, argument digest, policy version, matched rule, decision, one-use approval handle and execution receipt. Never reuse an approval for changed arguments.
6. **Streaming events:** ordered, versioned events with turn ID and sequence number; include progress, approval, response delta, completion/error/cancellation and final outcome. Reconnect must not duplicate side effects.
7. **Evaluation/training:** immutable item IDs, dataset/split/revision, all candidate routes and outcomes, scoring version, failures, timing, privacy label and train/dev/test membership.
8. **Dianoia ↔ Kleroterion action decision:** ordered action IDs/types, host-computed logits or feature schema, authoritative eligibility mask and reasons, query/walk state ID, immutable LTHCM snapshot, scorer/policy version, noise/temperature budget, RNG seed or replay state, and deadline. Return selected **action ID** or `NO_ELIGIBLE_ACTION`, probabilities, proposal/entropy diagnostics, cost and typed error. Dianoia checks eligibility and snapshot again before action execution; rejected or failed actions must not commit Kleroterion controller/RNG state. Actions are not passage IDs, citations or evidence.
9. **Kybernetes ↔ Krisis answer selection:** fixed query/evidence snapshot, candidate answer IDs and cited passages, authoritative eligibility/rejection reasons, resolved model identity, policy version, optional calibrated likelihood/consistency/support features with missingness, perturbation-set version and hard candidate/token/time budget. Return selected **answer ID** or `NO_ELIGIBLE_CANDIDATE`, ranked alternatives, score components, abstention reason, cost and trace ID. Phylax/access, provenance, contradiction and snapshot gates cannot be compensated by a high score. Kybernetes rechecks the selected result before release. A future passage-rerank contract must use stable passage IDs and be evaluated separately.

## Existing integration issues to resolve at the correct boundary

- Dianoia checks `GET /v1/models`; the supplied Hestia improvement file says this endpoint is missing. Hestia owns the endpoint; clients need a contract test and a clear unsupported-server error until it exists.
- Stoma's supplied README says its current index has zero graph edges, and Dianoia's direct CLI rejects it. Kybernetes must use an honest flat retrieval fallback. A graph builder must be added and evaluated before claiming that a Dianoia walk ran over Stoma's index.
- Stoma uses epoch-second document timestamps and warns that LTHCM's default query timestamp is `100.0`. Require explicit `as_of` in shared retrieval calls and test past/future behavior.
- Dianoia and Stoma improvement files identify pickle-based index loading. Never deserialize an untrusted index. Define a versioned, validated data-only interchange before sharing index files across trust boundaries.
- Dokime's current 25-item synthetic task is a smoke fixture, not a production quality gate. Add held-out stack tasks and resolved artifact provenance before using it for release decisions.
- Kleroterion's specification describes masked-softmax/Gaussian action selection with optional entropy feedback, but does not prove retrieval improvement. Start with its scorer-only/argmax baseline, zero proposal noise for exact-fact profiles, and feedback disabled until matched held-out comparisons justify it. A stochastic action cannot bypass Dianoia's authoritative filters or create a citation.
- Krisis's draft mixes answer selection, retrieval reweighting, configuration priors and online updates. Its model log-probability feature is not a prior over complete stack configurations; Hestia does not yet guarantee token-level logprobs. First test a frozen, bounded answer selector with an explicit abstain path. Keep passage reranking, online Hedge and config-space selection optional until separately specified and evaluated.
- Dokime owns versioned perturbation sets and independent held-out outcomes for both selectors. Agreement across perturbations is diagnostic; a stable but wrong answer remains wrong. Avoid counting the same perturbations or labels as both tuning and final evidence.

## Production release standard for each new project

“Production level” here means a dependable first release within its stated scope, not every future feature. Each component must provide:

- A clear architecture and documented public interface; strict input/config validation and version compatibility checks.
- Structured error codes and predictable failure behavior; bounded time, memory, retries, concurrency and queues where relevant; cancellation and cleanup.
- A defined trust boundary: local runtime, least privilege, prompt-injection handling for untrusted content, safe file paths, privacy/redaction and retention controls.
- Unit tests for pure logic, integration tests against real public interfaces or realistic local fakes, and failure tests for unavailable dependencies, malformed data, cancellation and restart. A mock-only green suite does not prove live compatibility.
- One reproducible demo/acceptance command, exact dependency/model requirements, a setup/doctor check, machine-readable logs, and a release manifest with source commit and dirty-tree state.
- Performance and quality measurements on target hardware where the feature depends on it. If unavailable, report **unverified** rather than inventing a result.
- No silent fallback that changes semantics. Degraded paths must be explicit in the response and trace.
- Selection components require separate release gates: Kleroterion must beat the current Dianoia action policy on its preregistered retrieval/answer-relevant metric at matched budgets without causal, provenance, contradiction or eligibility regressions; Krisis must improve held-out grounded answer quality or selective risk over the strongest equal-budget selector without worsening latency, failure or abstention guardrails. Log per-query decisions and negative results. A spec, passing unit suite or synthetic smoke fixture is not a gain claim.

## First stack-wide gate

On a pinned manifest, ingest a small file with Stoma; retrieve a passage through LTHCM with an explicit `as_of`; answer via Hestia and Kybernetes; display a citation in Prosopon; and capture the case in Dokime with actual model identity. Repeat with a zero-edge index, future-dated passage, denied tool action, unavailable model and cancelled client. Each path must end with its declared status within a bounded time.

After the baseline gate passes, add **separate opt-in integration gates**. For Kleroterion, Dianoia supplies a frozen eligible action set, Kleroterion selects one, and Dianoia rejects a stale/invalid action before execution; compare the full walk to Dianoia's current policy at equal budget. For Krisis, Kybernetes supplies eligible answer candidates and a frozen evidence snapshot, Krisis selects or abstains, and Kybernetes rechecks citations and policy before streaming; compare to the no-Krisis answer path. Keep both selectors disabled by default until their own Dokime release evidence passes.
