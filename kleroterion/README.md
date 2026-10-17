# Kleroterion (Κληρωτήριον)

> **Stochastic decision submodule for the Metis ecosystem with Dianoia as first consumer.**

**Kleroterion** (Greek: *κληρωτήριον* — the randomized allotment device used in Athenian democracy to select officials fairly without bias) is a reproducible stochastic decision engine. Within Metis, it provides entropy-aware action selection over candidate decision sets, grounding deliberation in mathematically disciplined exploration rather than uncontrolled randomness.

---

## 1. Purpose & Design Principles

Kleroterion separates **policy proposal and randomized action selection** from **host-owned memory, eligibility, and execution boundaries**:

1. **Host-Owned Eligibility & Immutability**:
   - The host (Dianoia) enumerates legal actions, applies causal, provenance, and contradiction filters, and computes action logits.
   - Kleroterion cannot invent actions, widen eligibility, return an arbitrary document, or mutate LTHCM memory.
2. **Dimension-Aware Continuous Gaussian Proposals**:
   - Perturbations are applied to continuous policy features using dimension-scaled spherical or diagonal Gaussian proposals ($1/\sqrt{d}$ factor), preventing variance blowup in high dimensions.
3. **Numerically Stable Masked Softmax**:
   - Ineligible actions are strictly masked ($-\infty$) before softmax normalization with double-precision exponentiation, preventing numerical overflow and leakage.
4. **Marginal Policy Aggregation**:
   - Aggregates action distributions over multiple proposal draws to compute robust marginal probabilities:
     $$p(a) = \frac{1}{M} \sum_{m=1}^M \text{softmax}(\mathbf{z}^{(m)})_a$$
5. **Entropy Diagnostics & Optional Feedback**:
   - Computes Shannon entropy, conditional entropy, and support fingerprints.
   - Supports optional adaptive temperature smoothing based on entropy estimates without destabilizing host policies.
6. **Strict Reproducibility**:
   - Deterministic Pseudo-Random Number Generation (PRNG) parameterized by explicit integer seeds and step indices.

---

## 2. Architecture & Data Flow

```text
       +----------------------------------------------------------------+
       |             Dianoia (Host Deliberation Engine)                 |
       |  Query + Graph Walk State + Pinned Immutable LTHCM Snapshot    |
       +-------------------------------+--------------------------------+
                                       |
       1. Enumerate legal actions & apply strict eligibility gates
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Kleroterion Action Selector                        |
       |                                                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 1. Continuous Gaussian Proposals (Dimension-Aware)       │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 2. Host Proposal Feasibility Re-check & Action Scoring   │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 3. Numerically Stable Masked Softmax                     │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 4. Marginal Policy Averaging & Entropy Diagnostics       │  |
       |  └────────────────────────────┬─────────────────────────────┘  |
       |                               │                                |
       |                               v                                |
       |  ┌──────────────────────────────────────────────────────────┐  |
       |  │ 5. Sample Action or Argmax -> ActionDecision             │  |
       |  └──────────────────────────────────────────────────────────┘  |
       +-------------------------------+--------------------------------+
                                       |
       6. Return ActionDecision (Selected ActionID + Diagnostic Entropy)
                                       |
                                       v
       +----------------------------------------------------------------+
       |             Dianoia Walk Execution & Revalidation              |
       |  Re-verify action feasibility & snapshot before graph step     |
       +----------------------------------------------------------------+
```

---

## 3. Core Components & Module Structure

| Module | Description |
| :--- | :--- |
| `kleroterion.core` | Top-level entrypoints: `propose_action`, `host_accept`, and `resolve_effective_config`. |
| `kleroterion.masked_softmax` | Numerically stable masked softmax over eligible action logits, rejecting empty or invalid candidate sets. |
| `kleroterion.proposals` | Dimension-aware continuous Gaussian perturbation generator. |
| `kleroterion.marginal_policy` | Probability distribution aggregation, Shannon entropy computation, and categorical sampling. |
| `kleroterion.entropy_feedback` | Controller state updates, support fingerprint tracking, and adaptive temperature feedback. |
| `kleroterion.interfaces` | Pydantic and dataclass contracts: `RoutingRequest`, `ActionDecision`, `ActionID`, `HostCallbacks`, `KleroterionConfig`, `ControllerState`. |
| `kleroterion.errors` | Typed exception hierarchy: `NoEligibleActionError`, `InvalidProposalError`, `ScorerError`, `RevalidationError`. |

---

## 4. Installation & Environment

Kleroterion requires Python `>=3.10` and depends only on `numpy`.

```bash
# Within the Metis workspace
uv sync

# Or standalone installation
pip install -e .
```

---

## 5. Usage Example

```python
import numpy as np
from kleroterion.core import propose_action
from kleroterion.interfaces import (
    RoutingRequest,
    ActionID,
    KleroterionConfig,
    PolicyProfile,
    PolicyProfileType,
    HostCallbacks,
    ControllerState,
    RngState,
)

# 1. Host defines legal actions and initial eligibility
actions = [ActionID("expand_node_A"), ActionID("expand_node_B"), ActionID("backtrack")]
eligibility_mask = [True, True, False]

# 2. Host callbacks for scoring and proposal evaluation
callbacks = HostCallbacks(
    proposal_features=lambda x, inc: x + inc,
    proposal_mask=lambda r, x: list(r.authoritative_mask),
    score_actions=lambda r, x, acts: np.array([2.5, 3.1, 0.8]),
    revalidate_before_execute=lambda act, snap: (True, None),
)

# 3. Create routing request
request = RoutingRequest(
    request_id="req_001",
    episode_id="ep_001",
    query_context="How does Metis route actions?",
    walk_state="node_A",
    snapshot_id="snap_001",
    lthcm_revision="rev_001",
    feature_schema_version="v1.0",
    policy_features=np.array([1.0, 0.5, -0.2, 0.8], dtype=np.float64),
    actions=actions,
    authoritative_mask=eligibility_mask,
    policy_profile=PolicyProfile(
        profile_name="exploratory",
        profile_type=PolicyProfileType.EXPLORATORY,
    ),
)

config = KleroterionConfig(
    proposal_count=4,
    noise_rms_budget=0.1,
    noise_hard_radius=0.2,
    temperature_initial=1.0,
)

ctrl_state = ControllerState(log_temperature=0.0)
rng = RngState(
    proposal_rng=np.random.default_rng(42),
    selection_rng=np.random.default_rng(100),
)

# 4. Propose action
decision = propose_action(request, config, ctrl_state, rng, callbacks)

print(f"Selected Action: {decision.selected_action_id}")
print(f"Marginal Probabilities: {decision.marginal_probabilities}")
print(f"Marginal Entropy: {decision.marginal_entropy:.4f}")
```

---

## 6. Testing & Quality Verification

Run the complete test suite:

```bash
uv run pytest kleroterion/tests -v
```

The test suite validates:
- Ineligible action masking and strict zero-probability bounds
- Determinism and repeatability given identical PRNG seeds
- Dimension-aware perturbation scaling invariance
- Shannon entropy diagnostics across uniform, skewed, and single-choice distributions
- Fail-closed behavior on empty candidate sets or NaN/Inf inputs
- Adaptive temperature feedback stabilization
