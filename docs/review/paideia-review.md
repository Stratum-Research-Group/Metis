# Paideia: training harness review and improved build brief

Reviewed 2026-09-29. Sources: both `6 paideia*.txt` prompts and the supplied stack README/improvement files. This is a design review; Paideia implementation or trained-model results were not inspected.

## Verdict

The prompt starts with a training method before proving that the available logs contain valid labels, that a small classifier cannot solve the task, or that deployment will improve an end-to-end outcome. Make Paideia an **evidence-to-policy pipeline**. A QLoRA adapter is one candidate output, not the default definition of success.

## Highest-impact gaps and changes

| Priority | Gap in the prompt | Concrete change | Acceptance gate |
| --- | --- | --- | --- |
| P0 | “Question text -> bridge/comparison label” is a task-type classifier, while “misrouting rate” requires a label for which route actually wins under a defined cost/quality objective. These are different targets. | Define separate datasets for task type and best-action routing. Produce best-action labels from paired outcomes for **all** allowed routes on the same query; allow ties/unknown. | Dataset card states label source, route set, utility formula, tie policy, and per-class counts. |
| P0 | Dianoia walk records are trajectories and reasons, not proof that each action was good. Failed and rewalk records can reinforce bad decisions. | Join attempts by `walk_id`; attach final correctness, evidence coverage, latency, verifier state, and action-level credit only where it is defensible. Exclude or explicitly label unresolved conflicts and missing verification. | Loader rejects unlabeled/ambiguous records for supervised action training and reports excluded counts. |
| P0 | Random held-out fraction can leak near-duplicate questions, documents, or later outcomes into training. N=500 is small for training and a convincing final test. | Split by question family, source/corpus and time where relevant; deduplicate before splitting; freeze a final test before tuning; use nested/dev tuning for model selection. | Leakage audit and immutable test IDs are saved in the manifest. |
| P0 | QLoRA on a 7–8B model may be unnecessary for a two-label router and may crowd 16 GB when Hestia is serving models. | Compare regex, majority, lexical/logistic, small embedding classifier, prompt-only LLM, then QLoRA. Schedule training with Hestia GPU admission or exclusive lease. | Adapter ships only if it beats the strongest cheap baseline on preregistered utility and fits a measured memory/latency budget. |
| P1 | “Confidence interval excludes zero” alone omits multiplicity, failures, and practical value. | Predeclare one primary metric: end-to-end quality minus latency/compute penalty, with misrouting, answer quality, abstention, and failure rate as guardrails. Use paired per-query intervals and a suitable paired null test; report absolute values and slices. | Dokime can reproduce the release verdict from per-query predictions and complete pairs. |
| P1 | Dataset checksums and seeds are insufficient to replay a run. | Record exact source revisions, selection query, tokenizer/chat template, library and CUDA versions, quantization, split IDs, prompt formatting, adapter target modules, optimizer, early stopping, base license, hardware use, and dirty tree. Hash finalized artifacts. | Fresh environment can validate the bundle and rerun evaluation without guessing inputs. |
| P1 | The export says only “base model + adapter pair.” | Define a signed or hash-verified registry record with base digest, adapter digest, tokenizer/template digest, task, input/output schema, calibration, compatible Hestia backend, rollback target, and evaluation run ID. | Hestia rejects incompatible pairs; Kybernetes can canary and revert them. |
| P1 | No privacy boundary exists for raw questions, reasons, and walk logs. | Add consent/retention policy, redaction mode, deletion propagation, data lineage, and an option to train on local sanitized features only. | Removing a source invalidates derived datasets/adapters or records the retraining obligation. |

## Revised build order

1. Define versioned event, label, dataset, and adapter contracts with Kybernetes, Dianoia, Hestia, and Dokime. Start with synthetic fixtures and no model training.
2. Create deterministic loaders and a dataset audit: schema validation, paired route outcomes, duplicates, leakage, class balance, exclusions, privacy status.
3. Run cheap baselines and a frozen held-out evaluation. Keep the regex router as a named baseline, not assumed ground truth.
4. Add QLoRA only when the data volume and baseline gap justify it. Measure peak VRAM on each target device; the V100 and RTX 5060 Ti need separate compatibility runs.
5. Register an adapter only after Dokime validates the bundle and a Kybernetes integration run shows a useful end-to-end gain with no guardrail regression.

## First milestone

Produce a **router dataset card and baseline report** for the cited N=500 outcomes, including exact label definitions and leakage audit. If best-route outcomes are absent, stop at task-type classification and do not call its error rate “misrouting.”

See [stack integration plan](stack-integration.md) for shared contracts, dependency pins, and release order.
