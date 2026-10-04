# Krisis policy 0.1 review

Reviewed 2026-09-30 against `krisis-spec-v4.txt`. The supplied `krisis-policy-0.1.txt` parses as JSON; it was copied to `krisis-policy-0.1.json` before the fixes. This file addresses manifest values and governance. Schema-enforcement defects are in `krisis-policy.schema-review.md`; fixture defects are in `krisis-fixtures-0.1-review.md`.

**Update 2026-09-30:** The listed policy fixes have been applied to `krisis-policy-0.1.json`. The original `.txt` remains the untouched supplied input. The new affine likelihood map is explicitly synthetic for fixture replay; the policy remains `draft` and cannot be frozen until real model calibration, sealed-test, and independent audit evidence exist.

## Concrete defects

1. **Unknown grounded class has no executable route — `/class_routing/rules` and `/class_routing/unclassified`.** An unknown class other than the literal `unclassified`, with `requires_grounding=true` and `verifier_active=true`, matches no rule for either logprobs state. The prose description of the default does not normalize input classes. Add explicit normalization to `unclassified` or a matching catch-all rule, then enumerate the full route product to verify exactly one outcome per combination.

2. **Aggregate-reason contract is incomplete — `/reason_enum` and `/reason_precedence`.** `FEATURE_INVALID` and `CANCELLED` appear in the enum but not precedence. No mapping converts raw access or contradiction failures into an aggregate reason. The table does not state whether partial upstream failures participate when a valid candidate survives. Add a stage-specific failure-to-reason table, define candidate-only versus aggregate reasons, and define precedence only over terminal abstention causes if survivor selection is allowed.

3. **Likelihood calibration is not executable — `/profiles/grounded_v1/score/cal_ll_map/ll_min` and `/ll_max`.** Both bounds are `null`. The weighted score can be checked when fixtures supply `cal_ll`, but the manifest cannot reproduce that feature from raw log probabilities. Freeze the calibrated transformation and its version, or keep `grounded_v1` inactive until they are known.

4. **The seal assertion is unsupported — `/governance/sealed_test/hash_recorded_before_tuning` and `/sha256`.** The former is `true` while the hash is `null`. Record the actual sealed artifact hash before tuning, including a timestamped commitment to the evaluation protocol and metric; do not represent the seal as complete while the hash is absent.

5. **One-person governance is not independent — `/governance/roles/tuning`, `/roles/approval`, `/roles/labeling`, and `/approval_lag_hours`.** The same `stack_owner` tunes and approves after 24 hours. A different model family can supply labels, but those labels are still model judgments and may share systematic errors with the verifier. For an independent release claim, use an external test custodian and blinded independent adjudication, with one final test access and an immutable access log. If all work remains solo, call the evidence internal and use prospective post-freeze cases without reusing failed sealed tests.

6. **Audit gates cannot be calculated per critical slice — `/audit_card`.** `min_audit_cases=400` applies globally, despite six named critical slices. The card lacks per-slice eligible denominators, a confidence level, a specified upper-bound calculation for false support, and a lower-bound calculation for extraction recall. Define these before sealing so `support_false_positive_ucb_target=0.05` and `extraction_recall_min=0.98` are reproducible pass/fail gates.

## Acceptance check

Freeze only after the manifest has a committed test hash, complete and unique routing, deterministic aggregate-reason mappings, a reproducible calibration map, and a preregistered audit method. Independently adjudicated results are required for an independence claim.
