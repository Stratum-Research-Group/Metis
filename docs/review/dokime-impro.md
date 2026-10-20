# Dokime improvement review

Reviewed: 2026-09-29. Scope: Dokime's tracked source, tests, documentation, and the two local example runs. This is a review and implementation plan, not a new model benchmark.

## Executive assessment

Dokime already has useful foundations: task plugins, a model/task runner, per-query JSONL results, dataset and result checksums, explicit failure records, paired comparisons, and a passing test suite. Its largest gap is **benchmark validity**. The only built-in task is 25 synthetic, easy science questions. The example results cannot establish a real model's capability because the manifest records a model name but neither the serving endpoint nor the resolved model artifact; the included mock server advertises the same model name. The harness is suitable for development smoke tests, but it is not yet a trustworthy decision system for model or pipeline changes.

The fastest route to a step change is to (1) define a real, versioned evaluation suite aligned with the local AI stack, (2) make every run independently auditable, and (3) make comparison rules and failure handling reflect the intended decision.

## Evidence and present benchmark

- `src/dokime/tasks/registry.py` registers only `arc_easy_25`; `src/dokime/tasks/arc.py` calls it a synthetic 25-item ARC-style dataset. It is not the official ARC-Easy benchmark and should be labelled `synthetic_arc_smoke_25` in reports or metadata.
- The two local runs in `runs/` show 9/25 (36%) and 12/25 (48%) for the advertised `llama-3.1-8b-instruct` model. Current comparison code reproduces a paired delta of +0.12, 95% bootstrap interval [-0.12, +0.36], and `NO_CHANGE` verdict. There are 6 improved, 3 regressed, and 16 unchanged items; 25 complete pairs and no logged failures. **Do not present +12 points as a demonstrated gain.** These runs are ignored by Git, and their manifests have no resolved model identity or endpoint. Their original server cannot be verified from the artifacts.
- `examples/mock_server.py` answers deterministically from prompt hashes, returns responses under the same model ID, and intentionally changes answers for a prompt variant. This is useful for harness testing, but its scores must never be mixed with real-model results.
- `.venv/bin/python -m pytest -q` passed: 127 tests in 0.87 seconds. This verifies current behavior, not benchmark validity or live Hestia/model operation.

## Ranked gaps and changes

| Priority | Gap and evidence | Change | Done when |
| --- | --- | --- | --- |
| P0 | One tiny synthetic task gives high uncertainty and poor coverage. | Add a versioned benchmark registry with at least one licensed, established public suite and private, held-out tasks representing Dokime's actual jobs. Record source, revision, license, split, item IDs, contamination risk, scorer version, and exclusions. Keep the 25-item task as smoke only. | A benchmark card and immutable dataset artifact exist for every scored task; smoke results are clearly separated from decision reports. |
| P0 | Model provenance is weak: `src/dokime/manifest.py` stores model IDs and host details, but no endpoint, resolved model file/digest, quantization, serving backend/version, or generation response metadata. A seed alone does not guarantee deterministic generation. | Capture the actual server URL (redacted if needed), backend build, resolved model digest/config, tokenizer/chat template, generation parameters, seed support, and Dokime commit with dirty-tree status. Mark unknown fields explicitly and reject a release-grade run if required fields are missing. | A reviewer can identify and rerun the exact model and serving setup from the artifact. |
| P0 | `src/dokime/runner.py` excludes failed requests from accuracy and `src/dokime/compare.py` pairs only successful rows. An unreliable candidate can therefore look better on the surviving subset. | Report both success-only accuracy and end-to-end accuracy with failures counted as incorrect; always show attempt count, failure rate, failure types, and complete-pair coverage. Define which metric gates releases before running. | A candidate with more failures cannot improve the release verdict solely by dropping hard queries. |
| P0 | The run loop opens `results.jsonl` with `"w"`, writes directly, and has no resume or completed-run state. An interruption leaves partial artifacts and rerunning a chosen ID overwrites evidence. | Reserve run IDs atomically, write to a temporary run directory, checkpoint each query, support verified resume, and atomically finalize manifest and a `COMPLETED` marker. Comparison should refuse partial runs by default. | Kill/resume and duplicate-ID tests prove no silent overwrite, missing rows, or apparently complete partial run. |
| P1 | Ordinary run comparison checks dataset hashes and answers, but does not enforce equal seed, temperature, token limit, scorer version, or model artifact. Prompt changes may be intentional; other changes can confound a claimed prompt gain. | Add a comparison contract: classify intended variable (`prompt`, `model`, `server`, `pipeline`), assert all other controlled fields equal, and emit an explicit comparability report. Require an override with reason for exploratory comparisons. | A prompt comparison refuses a changed model or decoding setup; a model comparison requires identical prompt and dataset. |
| P1 | The paired bootstrap code in `src/dokime/stats.py` estimates a percentile interval and labels a tail fraction of the uncentered bootstrap distribution a `p_value`. This is not a separately calibrated null test; ties and small discrete samples need special care. | Choose a predeclared primary inference method. For binary paired accuracy, use an exact paired test (McNemar/binomial on discordant pairs) alongside a paired interval; for general bounded scores, use a validated paired permutation test and interval. Keep the 10k bootstrap for descriptive intervals, with a method label. Ensure verdict, interval, and test cannot contradict silently. | Known small-sample and all-ties cases match reference calculations; statistical method and limitations appear in each report. |
| P1 | No practical cost/performance benchmark: only per-call wall latency is kept; token usage, time to first token, throughput, warm/cold state, and memory are absent. `src/dokime/client.py` discards API usage and finish reason. | Capture response metadata and benchmark warm-up, randomized request order, p50/p95 latency, tokens/s, total tokens, peak VRAM/RAM, and failures under specified concurrency. Keep quality and performance reports separate, then show their trade-off. | Same model/config can be compared for quality and measured resource cost under a documented load profile. |
| P1 | Task input validation is weak: loaded query IDs/answers and plugin scorer semantics are not checked centrally; a prompt template can omit the question and still produce a score. | Validate nonempty unique IDs, valid choice/answer structure, finite scores in the documented range, stable task/scorer versions, and prompt inclusion policy. Add a `dokime validate-task` command and report invalid-response rate separately from wrong answers. | Malformed or duplicate items fail before model calls; extraction/scoring fixtures cover common response formats and abstentions. |
| P1 | Local run outputs are ignored (`.gitignore`), while the README gives no reproducible real-model benchmark recipe or published baseline artifact. | Add a small committed benchmark specification and a documented artifact retention path (release attachments/object storage), plus `dokime verify <run>` for checksums/schema/completeness and `dokime report` for a compact human summary. Never commit secrets or giant raw outputs by default. | A fresh checkout can locate, verify, and interpret a named reference run. |
| P2 | The source is compact, but `cli.py` and `compare.py` are large, many tests are organized as fix rounds, and plugin discovery suppresses import errors (`src/dokime/tasks/registry.py`). | Split parsing/reporting/comparison validation into focused modules; consolidate tests by feature; surface plugin load failures with actionable diagnostics. Add formatting, type checking, package build, and a small integration smoke run in CI. | A broken task plugin is visible, and CI checks the installed package plus core workflows. |

## Benchmark redesign

### Three tiers

1. **Smoke, every commit:** retain the synthetic 25-item task for wiring, schema, parser, retry, and report checks. It is never a model leaderboard.
2. **Decision suite, every candidate:** use a frozen held-out set covering the stack's real use cases. At minimum, include factual QA, instruction following, structured output, retrieval-grounded answers with citation checks, abstention on unanswerable questions, and robustness to paraphrases. Define an objective scorer or blind human rubric for each task. Keep development and decision splits separate; never tune prompts on the decision split.
3. **Deep audit, before release:** run established external benchmarks where licenses and task relevance permit, plus adversarial slices, repeated seeds for stochastic settings, and performance/load profiles. Publish task-level outcomes, not just one blended score.

### Run protocol

- Freeze dataset and scorer versions, model artifact, server build, prompt/chat template, decoding settings, hardware profile, and run order before testing. Use explicit model digest rather than a display name.
- Predeclare primary metric and minimum meaningful improvement. For each task report numerator/denominator, 95% interval, failure rate, invalid-output rate, and paired delta against a frozen baseline. For multiple tasks or many prompt candidates, declare a multiplicity policy and reserve a final held-out confirmation run.
- Randomize item order with a recorded seed, run each candidate under comparable warm/cold conditions, and repeat stochastic configurations across several generation seeds. An evaluation seed is useful for replay but is not a substitute for repeated trials.
- Keep per-query records and an immutable manifest; include generated response or a privacy-safe reference, usage, finish reason, elapsed time, error category, and scorer version so failures can be audited. Treat private examples with access controls and redaction.
- Set release gates on quality **and** reliability. Example policy to customize before use: no material regression on a critical task, end-to-end success at least as good as baseline, failure rate below a chosen cap, and p95 latency/VRAM within the deployment budget. A nonsignificant result means inconclusive, not equivalence.

### First useful benchmark milestone

Build one genuinely representative 200-500 item frozen decision set, stratified by task and difficulty, then compare one real baseline and one candidate under identical serving conditions. Size is a planning range, not a power claim: calculate needed sample size from the observed discordant-pair rate and the smallest improvement that matters. Publish a benchmark card, run manifests, per-task metrics, paired deltas, failure/invalid-output counts, and a small qualitative error review. This gives Dokime a credible decision loop before adding more tasks or dashboards.

## Suggested delivery order

1. **Integrity first:** run-state/atomic writing, resume, `verify`, task validation, and failure-aware report metrics. These prevent misleading or lost evidence.
2. **Real benchmark:** benchmark cards, held-out representative data, scorer fixtures, a reproducible Hestia run recipe, and one verified real-model baseline.
3. **Sound comparisons:** comparison contract, calibrated statistical tests, repeated stochastic trials, and predeclared gates.
4. **Operational performance:** token/finish metadata, latency distribution, throughput/load profile, resource measurements, and quality-cost reports.
5. **Maintainability:** clearer modules, CI packaging/integration checks, plugin diagnostics, and docs that distinguish smoke demonstrations from release benchmarks.

## Verification checklist for the next implementation pass

- Unit tests: duplicate/empty query IDs, invalid answer keys, scorer errors, all-failure runs, partial runs, resume, duplicate run IDs, mismatched manifests, all-ties/small-discordance statistics.
- Integration test: start the mock server, run the smoke task, verify artifacts, then compare two controlled prompt variants. Label all output `mock` and keep it out of benchmark results.
- Real-model test: record actual Hestia/model digests and perform two repeat runs; audit several stored responses against raw server outputs and independently recompute metrics.
- Benchmark acceptance: every published decision result names its dataset revision, scorer, model digest, endpoint/backend version, prompt hash, settings, completion status, failure counts, and exact comparison contract.
