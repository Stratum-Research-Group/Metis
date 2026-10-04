# Local AI stack: missing contracts and integration plan

Reviewed 2026-09-29 from the eight prompt `.txt` files and the ten supplied README/improvement Markdown files. This plan connects the existing Stoma, LTHCM, Dianoia, Hestia, and Dokime work to proposed Paideia, Phylax, Kybernetes, and Prosopon. It is a document review, not a new code or GPU benchmark. Existing component-specific findings remain in the five `*impro.md` files; this file covers gaps **between** components.

## The biggest architectural gap

The nine names currently describe responsibilities, but not a single executable contract. Fixing individual components without a versioned data and API boundary will leave a stack that works only through ad hoc checkout paths, incompatible index snapshots, model aliases, and undocumented fallbacks. The first stack milestone should be **one cited answer from a Stoma-ingested document through LTHCM, Hestia, Kybernetes, and Prosopon**, with Dokime recording the result and Phylax mediating file/tool access.

## Integration blockers visible in the supplied files

| Priority | Evidence | Decision and acceptance check |
| --- | --- | --- |
| P0 | The v2 prompts pin LTHCM at `8a71605379ee32059e714feb3648cf74f4b42dc6`; `stomaREADME.md` pins `336f4af9f624aca5b4978a5e3ce222ce9638da35`. Dianoia's README also uses `8a716...`. | Choose and test one compatible release set. Record commit and API/schema versions in a stack lock manifest. An integration check rejects incompatible pins before serving. Do not update read-only submodules merely to make versions match; upgrade in their owning repositories. |
| P0 | `stomaREADME.md` says Stoma v1 emits zero graph edges and Dianoia's CLI refuses a zero-edge index. | Define a supported flat retrieval route now. To enable walking, add a versioned graph-building step in the owning component and test quality; never report a flat answer as a Dianoia walk. |
| P0 | `hestiaimpro.md` records that Hestia has `/models` but Dianoia checks `/v1/models`. | Implement and contract-test `/v1/models` with Dianoia; freeze model IDs and error shape. Test primary and fallback discovery on a real Hestia instance. |
| P0 | Stoma records epoch-second timestamps while its README warns LTHCM's default query timestamp is `100.0`. | Make query time mandatory in the shared retrieval API, with UTC/epoch units specified; test past/future document behavior and record causal fallback. |
| P0 | Paideia, Phylax, Kybernetes, and Prosopon v2 prompts all demand LTHCM and Dianoia git submodules, even when a component only needs a remote API or stable client. | Draw a dependency graph and use versioned packages or API clients at runtime. Keep submodules only for source-level dependencies or integration fixtures. Every repo should pin only what it imports or tests. |
| P0 | The prompts call Kybernetes “built last” yet place Prosopon and Paideia before it in numbered prompts; Prosopon requires Kybernetes API and Paideia needs its routing outcomes. | Build a minimal Kybernetes API and event schema before Prosopon and Paideia. Delay learned policy until sufficient labeled outcomes exist. |
| P1 | Dianoia and Stoma currently use pickle-compatible indexes; their improvement files identify untrusted-load and schema risks. | Define a versioned, data-only `IndexSnapshot` with encoder digest, dimensions, chunker, graph status, provenance, timestamp units and generation ID. Reject incompatible snapshots before loading. |
| P1 | Dokime's improvement file says its only built-in task is a synthetic 25-question smoke task and model provenance is incomplete. | Use Dokime as the stack release gate only after adding real, held-out tasks, endpoint/model digests, complete failure accounting and replayable run bundles. Keep smoke scores out of capability claims. |

## Minimal ownership map

| Component | Owns | Must not silently own |
| --- | --- | --- |
| Stoma | Source extraction, chunking, provenance, index mutation and deletion | Answer synthesis or hidden policy decisions |
| LTHCM | Versioned retrieval, graph candidate generation, optional verification signals | Certifying answer truth from a retrieval score |
| Dianoia | Bounded walk, evidence selection, synthesis request and explicit verifier/abstention state | Ingestion, model process management or unbounded retries |
| Hestia | Model identity, inference/embedding endpoints, GPU admission and lifecycle | Authorization of user files or evaluation verdicts |
| Dokime | Immutable evaluation runs, comparisons and release evidence | Treating a smoke score or model alias as model identity |
| Phylax | Policy decisions, approval binding and audit; guarded capability adapters | Assuming a preflight check alone is an OS security boundary |
| Kybernetes | Request planning, orchestration, budgets, outcome events and client API | Inventing correctness labels from completed turns |
| Prosopon | Authenticated text/voice UI, sessions, approvals and accessible status | Direct model/memory/tool calls |
| Paideia | Audited datasets, baseline comparisons, optional adapters and training provenance | Training from raw trajectories as if they were gold labels |

## Contracts to write before another component scaffold

Store schemas under a small versioned `contracts/` package or a repository-neutral spec. Each schema needs a version and examples; JSON Schema/OpenAPI is sufficient initially. The repository location is an implementation choice, but one owner must approve changes and publish compatibility checks.

1. **Identity and provenance:** `request_id`, `turn_id`, `trace_id`, `session_id`, source/passage IDs, source hash and location, index generation, corpus/encoder/model digests, component version, and UTC timestamp units. Keep user identifiers pseudonymous in benchmark and training exports.
2. **Retrieval:** query text or embedding identity, `as_of` time, domain/filter, `top_k`, generation consistency, ranked passages with stable IDs and scores, citation locations, `causal_fallback`, graph availability, and typed unavailable/incompatible errors.
3. **Dianoia result:** final text, cited passage IDs per claim, walk steps, verifier mode (`active`, `inactive`, `unavailable`), outcome (`verified`, `conflicted`, `insufficient_evidence`, `verification_unavailable`, `failed`), retries and phase timings. A verifier signal is evidence, not a truth certificate.
4. **Hestia:** `/v1/models`, chat/embedding requests, resolved model digest/quantization/template, load state, stream event/error format, cancellation, rate limits and overload responses.
5. **Capability decision:** authenticated subject, verb, canonical resource, argument digest, policy version, effect, matched rule, approval ID/expiry and execution receipt. Approval grants one exact action.
6. **Turn events:** `started`, `plan_selected`, `component_started/completed`, `approval_pending/resolved`, `evidence`, `response_delta`, `completed/failed/cancelled`, monotonic sequence number and final status. Prosopon consumes only this API.
7. **Evaluation/training:** immutable item ID, dataset/split/revision, all route candidates attempted, per-route outcome, quality scorer/version, latency/cost, missing/failure flags, consent/retention label and train/dev/test membership. A route label must be derived from paired outcomes, not completion.

## Missing cross-cutting work

These do not need new branded services yet. Assign clear owners and make them acceptance gates:

- **Data lifecycle:** classify private sources, propagate access labels into passages and citations, define retention for Stoma index, Dianoia/Kybernetes logs, Prosopon sessions, Dokime artifacts and Paideia datasets. Document deletion propagation and backup/restore. A delete should not leave retrievable content or training examples unnoticed.
- **Operations:** one local stack manifest for component revisions, model files/digests, index schema and ports; setup/doctor checks; startup order; health/readiness; bounded queues; GPU scheduling between Hestia inference and Paideia training; upgrade/rollback procedure.
- **Observability:** propagate one trace ID through every step. Record model/index identity, branch taken, latency, tokens, VRAM, failures, abstentions and approval decisions with configurable redaction. Set p50/p95 and reliability targets only after a real baseline.
- **Trust boundary:** retrieved documents, web/file search results and tool outputs are untrusted data; plugins and network clients may be less trusted than the owner. Phylax enforcement must sit at capability execution. Prosopon phone access needs pairing and transport security.
- **Release evidence:** Dokime should run frozen quality, citation, safety, failure and performance suites through the public Kybernetes API. Compare a change to the strongest simpler baseline at equal evidence and cost budgets. Publish negative results and rollback on regressions.

## Recommended sequence

1. **Lock the existing five:** resolve LTHCM/Dianoia/Stoma pin compatibility, Hestia model-list compatibility, the retrieval timestamp contract and the zero-edge Dianoia fallback. Add a stack contract smoke test.
2. **Create the minimum safe orchestrator:** Phylax guarded capability gateway plus Kybernetes adapters and bounded plan executor. Keep the initial route rules fixed and observable.
3. **Prove the user flow:** Stoma ingest -> LTHCM retrieve -> Hestia synthesis -> Kybernetes cited result -> Prosopon text UI; then add Dianoia only where a graph is available. Test denial, timeout and citation failure.
4. **Make measurement real:** extend Dokime with held-out end-to-end tasks, real model provenance, complete failure accounting and a replayable run bundle. Establish quality, latency, memory and reliability baselines.
5. **Add voice and learning:** implement Prosopon voice after text and approvals are stable. Start Paideia with dataset audits and cheap baselines; release an adapter only if held-out end-to-end benefit survives Dokime gates.

## First integrated acceptance test

On one pinned stack manifest, ingest a small document with a unique fact. Ask from an authenticated Prosopon session. The response must cite the exact Stoma passage and index generation, carry the same trace ID through Kybernetes and Hestia, and be stored in a Dokime run with resolved model identity. Repeat with a future-dated passage, a zero-edge index, a denied file action, and Hestia unavailable. Each case must return its declared status within a bounded time; none may be silently reported as a verified answer.
