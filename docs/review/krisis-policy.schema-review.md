# Krisis policy schema review

Reviewed 2026-09-30 against `krisis-spec-v4.txt`. The supplied `krisis-policy.schema.txt` parses as JSON; it was copied to `krisis-policy.schema.json` before the fixes. This file addresses defects in the schema, not defects in the current policy values or fixtures.

**Update 2026-09-30:** The listed schema fixes have been applied to `krisis-policy.schema.json`. The original `.txt` remains the untouched supplied input. Frozen policies now require sealed-test and calibration evidence; policy 0.1 remains a draft.

## Concrete defects

1. **A frozen policy can lack its sealed-test commitment — `/properties/status` and `/properties/governance`.** Validation accepts `status="frozen"` while `/governance/sealed_test/sha256` is `null`. Add a conditional rule requiring a valid 64-character SHA-256 value when status is frozen. Require a recorded seal timestamp and test artifact identifier if the governance claim is intended to be auditable.

2. **Frozen grounded profiles can violate their own hard gates — `/properties/profiles`.** Validation accepts `grounded_support_only.tau="0.700000"` alongside `per_claim_min="0.800000"`, and accepts `grounded_v1.tau="0.500000"` even though its required lower bound is `0.7 × 0.800000 = 0.560000`. The current policy values are coherent; the schema fails to keep future frozen values coherent. Enforce `tau > per_claim_min` for support-only and `tau > w_conf × per_claim_min` for grounded-v1 with exact decimal comparison. A companion semantic validator may be necessary because JSON Schema alone cannot express this arithmetic cleanly.

3. **Routing coverage can disappear while validation passes — `/properties/class_routing/properties/rules`.** Replacing the rules with an empty array still validates. Require a nonempty table and validate that every supported class × grounding × verifier × logprobs combination resolves exactly once, including unknown classes. Reject ambiguous overlaps as well as uncovered combinations.

4. **Reason precedence can duplicate or omit reasons — `/properties/reason_precedence` and `/properties/reason_enum`.** A duplicate precedence list validates. Add `uniqueItems` and a semantic check that every aggregate-eligible reason appears exactly once, with an explicit exclusion list for candidate-only reasons. Require a defined mapping from raw gate failures to aggregate reasons.

5. **Numeric strings permit nonsensical thresholds — `/properties/profiles` and `/properties/audit_card`.** The schema accepts negative tau values and does not enforce a six-decimal, finite, bounded representation for confidence, score weights, or audit targets. Apply a decimal-string pattern and field-specific ranges; validate profile weights and their sum with exact decimal arithmetic.

## Acceptance check

Validation must reject each mutated policy above when marked frozen. A second independent validator should enumerate routes, check precedence completeness, and evaluate both grounded-profile gate/tau inequalities using decimal arithmetic.
