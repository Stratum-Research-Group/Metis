# Krisis fixtures 0.1 review

Reviewed 2026-09-30 against `krisis-spec-v4.txt` and `krisis-policy-0.1.json`. The supplied `krisis-fixtures-0.1.txt` parses as JSON; it was copied to `krisis-fixtures-0.1.json` before the fixes. Calculations below use the manifest's decimal-string values, six-place half-even rounding, and stated profile thresholds.

**Update 2026-09-30:** The listed fixture fixes have been applied to `krisis-fixtures-0.1.json`. The original `.txt` remains the untouched supplied input. The table below describes the original defects; the edited JSON now includes routing, gate, snapshot, and synthetic raw-likelihood inputs.

## Concrete defects

1. **Route-dependent expected results lack inputs — fixtures `A`, `A2`, `B`, `F`, `J`, `K`, `L`.** `A`, `B`, `F`, and `K` omit both capability flags; `A2` omits `verifier_active`; `J` and `L` also omit question class and grounding requirement. Each fixture must name the selected profile or supply the complete routing tuple. `B` is a direct contradiction: with active verifier and `logprobs=false`, its `0.900000` confidence passes support-only tau `0.850000` and would select, not abstain as expected.

2. **Grounding gates cannot be replayed — fixtures `A`, `A2`, `F`, `J`, `K`, `L`.** Their selected or scored grounded candidates lack complete cited-passage IDs, resolution results, and snapshot identity. `J` includes passage `p9` without a candidate citation to `p9`. Supply these inputs or explicit deterministic gate stubs. Fixture `H` needs snapshot identity to establish that its `c1`/`c2` merge is within one snapshot.

3. **Wrong threshold in expected explanation — fixture `A` `/expected/note`.** `0.7 × 0.900000 + 0.3 × 0.800000 = 0.870000`, but the note tests against `0.550000`. The manifest's grounded-v1 tau is `0.700000`. Replace the stale threshold.

4. **Aggregate reason is not derivable from the declared precedence — fixtures `B`, `D`, `L`.** `B` assigns `FEATURE_INVALID` to `c1` but expects `NO_VALID_CANDIDATE` without a policy mapping. `D` expects `NO_AUTHORIZED_CANDIDATE` from raw `access` and `contradiction` labels that have no enum mapping. `L` expects selection after one upstream failure although `UPSTREAM_GENERATION_FAILED` is second in global precedence. Add typed per-stage failure mappings and an explicit survivor rule, then assert exact aggregate status and reason.

5. **Stale-snapshot fixture lacks rebuild state — fixture `C`.** The expected text says one full rebuild is permitted under budget, yet the request gives no remaining rebuild count or budget state. A full-decision fixture cannot distinguish rebuild from final `STALE` abstention. Add the state or scope this fixture explicitly to the first stale-detection step.

## Arithmetic recomputation

| Fixture | Manifest calculation | Expected-output defect |
| --- | --- | --- |
| `A` | Grounded-v1 `0.7 × 0.900000 + 0.3 × 0.800000 = 0.870000`; passes tau `0.700000`. | Note uses `0.550000`. |
| `A2` | Support-only `0.900000`; passes tau `0.850000`. | Route still lacks verifier capability. |
| `B` | Support-only, if routed there: `0.900000`; passes tau `0.850000`. Grounded-v1 cannot score missing `cal_ll`. | Expected abstention changes with omitted capability. |
| `F` | Grounded-v1 `0.7 × 0.810000 + 0.3 × 0.200000 = 0.627000`; below tau `0.700000`. | Profile input omitted. |
| `J` | Grounded-v1 `0.7 × 0.910000 + 0.3 × 0.700000 = 0.847000`; passes tau `0.700000`. | Route and grounding inputs omitted. |
| `K` | Grounded-v1 `0.7 × 0.850000 + 0.3 × 0.350000 = 0.700000`; equality passes tau `0.700000`. Support-only would produce `0.850000`, also passing its tau. | Score is not unique without capabilities. |
| `L` | Grounded-v1 `0.7 × 0.900000 + 0.3 × 0.800000 = 0.870000`; passes tau `0.700000`. | Route omitted; partial-failure precedence unresolved. |

Fixtures `C`, `D`, `E`, `G`, `H`, and `I` contain no candidate score arithmetic to recompute. `G`'s lexicographic choice is `01JAAA`, as expected. The arithmetic table checks supplied feature values; the manifest's null likelihood calibration bounds prevent recomputing `cal_ll` from raw log probabilities.

## Acceptance check

Every fixture should be decidable from its JSON request and the frozen policy alone: one route, explicit hard-gate inputs or stubs, exact score when applicable, one status, and one aggregate reason for abstentions.
