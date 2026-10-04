# Kybernetes: orchestration review and improved build brief

Reviewed 2026-09-29. Sources: both `8 kybernetes*.txt` prompts and the supplied stack documents. This is a design review; the Kybernetes implementation was not available here.

## Verdict

The proposed four exclusive routes are too narrow for the stack's actual work. A question may need Stoma/LTHCM retrieval **and** Dianoia deliberation **and** a tool. Define a typed execution plan and outcome model first; keep the initial router simple and measurable.

## Highest-impact gaps and changes

| Priority | Gap in the prompt | Concrete change | Acceptance gate |
| --- | --- | --- | --- |
| P0 | `DIRECT/MEMORY/DELIBERATE/TOOL` implies one route, while the demo asks for LTHCM + Dianoia together. Stoma, Dokime, Phylax and Hestia are not represented in the flow. | Use an ordered plan: `authorize -> retrieve -> deliberate? -> tool? -> synthesize -> verify -> respond`, with bounded branches and explicit skips. Stoma owns ingestion, LTHCM serves retrieval, Hestia serves models, Phylax guards capabilities. | A multi-hop query executes retrieval plus deliberation and returns cited evidence; trace shows each step. |
| P0 | No versioned API or error contract exists for any dependency. | Define request/response schemas for Hestia, LTHCM/Stoma, Dianoia, tools, and Prosopon, including timeout, cancellation, idempotency, provenance, and error codes. | Contract tests run against pinned revisions; incompatible components fail at startup. |
| P0 | The prompt logs an “outcome” per turn but provides no ground truth or feedback mechanism. | Separate observed events from labels. Store route, candidate alternatives, evidence IDs, user feedback when explicit, verifier state, answer status, latency/cost, and failure reason. Dokime creates evaluation labels from controlled tasks. | Logs never mark an answer correct just because it completed. |
| P0 | No policy or trust boundary appears in the build sequence although Phylax requires consultation before reads and actions. | Integrate a Phylax guarded tool gateway before adding file search; enforce at execution and bind approvals to exact arguments. Treat retrieved text and tool output as untrusted. | Unauthorized file/tool access fails in integration tests. |
| P1 | The demo's “direct factual question” may demand current or grounded information that a small model cannot verify. | Define direct-answer eligibility, uncertainty thresholds, and an abstain/ask/retrieve fallback. Require claim-to-source citations for retrieved facts. | Unsupported factual claims are flagged in Dokime's grounded-answer suite. |
| P1 | No limits or recovery behavior are specified. | Add per-turn deadline, token/step budgets, retries, circuit breakers, cancellation propagation, and degraded modes when Hestia/LTHCM/Dianoia is unavailable. | Fault injection produces bounded failure with clear status, no hung session or duplicate action. |
| P1 | “Streams final response” can expose content before verification or Phylax approval finishes. | Decide response status before streaming, or label speculative drafts distinctly; never stream an unresolved contradicted answer as verified. | Client disconnect cancels downstream work; conflict/approval outcomes are correctly represented. |
| P1 | Session state has no ownership, retention, or replay rules. | Persist versioned turns and references, not unlimited raw evidence; define deletion and redaction. Keep trace IDs across services. | A restarted session preserves authorized context and deleting a session removes its logs according to policy. |

## Revised build order

1. Freeze stack contracts, pins and a minimal trace schema. Implement adapters with contract tests against Hestia, LTHCM/Stoma, Dianoia, and Phylax.
2. Build a deterministic plan executor with budgets, cancellation, structured outcomes, and CLI. Start with direct, retrieval-only, tool-only, and retrieval-plus-deliberation cases.
3. Add a rule router using measurable features: intent, freshness, evidence need, tool need, and uncertainty. Compare to always-retrieve and always-deliberate policies in Dokime.
4. Add verified, redacted event logging suitable for Paideia only after labeling and retention rules exist.
5. Expose a stable streaming API to Prosopon; test backpressure, reconnect and pending approvals.

## First milestone

Run an end-to-end four-case fixture: direct answer, Stoma-ingested source retrieval with citation, Dianoia walk over a graph-enabled index, and Phylax-approved calculator/file action. Include a failure case for a Dianoia index with zero graph edges: use the declared flat fallback rather than pretending the walk ran.

See [stack integration plan](stack-integration.md) for blockers and release gates.
