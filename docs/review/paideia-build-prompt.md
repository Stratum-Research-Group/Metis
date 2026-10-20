# Build Paideia — production-grade local training and policy evaluation

You are building **Paideia from scratch** as a new Python project. Read `STACK_BUILD_CONTRACT.md` and the supplied Paideia v2 `.txt` brief, plus LTHCM, Dianoia, Hestia and Dokime documentation before coding. Deliver a complete, runnable first release within the scope below. Do not treat a passing smoke test or a fabricated adapter as a trained improvement. Make routine design choices independently; document deviations and unresolved external incompatibilities.

## Product outcome

Paideia converts **validated, labeled local outcomes** into deployable routing or walk-policy artifacts. The first release has two independent pipelines:

1. **Router:** given a request and permitted context features, choose a route/action from a versioned route set. Distinguish a bridge/comparison *question-type classifier* from a *best-route policy*: the former is not automatically a misrouting label.
2. **Walk policy:** given a bounded Dianoia state, predict `EXPAND(node_id)`, `BACKTRACK`, `SWITCH_SEED` where supported, or `ANSWER`, with structured reason metadata. Train only from trajectories with defensible action labels; raw model choices are not gold labels.

Start with cheap deterministic and classical baselines. Add QLoRA only when the available sample size and held-out results justify it. Preserve the user's QLoRA target: Unsloth, 4-bit, 7–8B base models, and measured operation under 16 GB VRAM on the RTX 5060 Ti 16 GB and Tesla V100 SXM2 16 GB where each machine is available. If a named GPU is unavailable, mark that hardware gate unverified.

## Non-negotiable constraints

- Local training and inference; no paid API or uploads of prompts, logs, weights or private datasets.
- Add read-only LTHCM and Dianoia submodules at the exact v2 prompt commits. Verify the gitlinks and checked-out revisions; never modify or copy their files into Paideia-owned source.
- Every run has immutable ID, before/after manifest, dataset and split checksums, source commits, exact base/model/tokenizer/template digests, hyperparameters, seed, package/CUDA versions, hardware line, adapter digest, status and failure reason. Preserve partial runs as partial; never silently overwrite.
- Separate train/dev/final test. Deduplicate before splitting, group related questions/sources, and prevent time leakage where later outcomes are used. Final test is touched once per release candidate.
- No synthetic labels in v1. Synthetic records may test the loader but may never enter a release dataset or claim.
- Negative results and failures must appear in reports. An adapter is called an improvement only if a predeclared primary held-out metric improves over the strongest relevant baseline, its paired 95% interval excludes zero, and latency/memory/failure guardrails pass. Use a valid paired null test when reporting a p-value; do not derive one naively from an ordinary bootstrap tail.

## Build in this order

### 1. Scaffold and contracts

Create an installable Python package, CLI, typed config, schemas, test suite and docs. Pin a tested dependency profile; provide CPU-only dataset/evaluation smoke tests and a separate GPU training profile. Specify schema versions for Dianoia logs, LTHCM outcomes, Kybernetes routes and Hestia adapter registry. Inspect actual dependency formats and implement adapters with fixtures, not guessed field names.

### 2. Dataset and label pipeline

- Import LTHCM benchmark routing outcomes and Dianoia JSONL walks with strict schema validation. Preserve source IDs, `walk_id` linkage, attempt/re-walk status, verifier availability, final outcome and privacy label.
- For best-route training, require paired outcomes for the same question across eligible routes under a declared utility function (quality, failure and cost). Mark ties and missing counterfactuals explicitly; never use “completed” as “correct.”
- For walk training, exclude unresolved conflicts, unverifiable answers and ambiguous actions by default. Include provenance explaining why each included action is a target. Bound state/context size and validate node IDs.
- Produce a dataset card: task, licenses, source revisions, time range, splits, class counts, excluded counts/reasons, duplicate/leakage checks, consent/retention status and known bias.

### 3. Baseline ladder

Implement majority, existing regex/rule policy, a simple lexical or logistic model, and optionally a small embedding classifier. Evaluate each with identical folds and route definitions. Report absolute end-to-end quality, misrouting under the declared utility, failures, p50/p95 latency and memory. Decide whether QLoRA is warranted from this evidence.

### 4. QLoRA runner

Expose learning rate, rank, epochs, batch/accumulation, held-out split, seed, max sequence length, target modules and early stopping. Refuse incompatible base/tokenizer/VRAM configurations before expensive work. Coordinate exclusive GPU use or admission with Hestia. Write manifests atomically, checkpoint safely, support verified resume, and handle OOM as a recorded failed run. Do not silently switch models or precision.

### 5. Evaluation and promotion

Use the same held-out IDs for baseline and candidate; save per-item predictions, scores, route decisions, inference timings and failures. Report paired bootstrap CI with 10,000 resamples/seed 42 for continuity with LTHCM, plus an appropriate null test where needed. Predeclare one primary metric and guardrails. Have Dokime consume or independently recompute the report from the run bundle. Expose calibration and abstention if probabilities are used.

### 6. Export and operations

Export adapter plus a versioned Hestia registry entry containing base/adapter/tokenizer digests, compatible backend, task/schema, training data fingerprint, evaluation run ID, expected resource footprint and rollback target. Validate the pair before registration. Provide `doctor`, `dataset validate`, `train`, `evaluate`, `verify-run`, `export` and a human-readable report. Define retention/deletion of sensitive logs and derived datasets.

## Required tests and demo

- Unit tests: invalid schema, duplicate IDs, mixed attempt linkage, leakage, ties/missing route outcomes, incompatible node IDs, checksum mismatch and metric edge cases.
- Integration tests: read-only submodule adapters, Dokime report consumption, Hestia registry validation, interrupted/resumed run, OOM/failure manifest, clean package install.
- Real demonstration: use the cited N=500 LTHCM outcome data **only if available and correctly labeled**. Report question-type classification separately from best-route misrouting. Compare with regex and cheap baselines on frozen held-out IDs. If data lacks paired route outcomes, deliver the dataset audit and classifier result, explicitly say best-route improvement is not yet measurable.

## Definition of done

A clean checkout can validate a local dataset, reproduce a baseline report, train where compatible hardware/data exist, verify every artifact hash, and load or reject an adapter deterministically. The final handoff includes architecture, exact commands, schemas, test results, measured GPU use, a real/negative/unverified result statement, and remaining risks. No release claim is based solely on training loss or an easy smoke set.
