"""Tests for hard eligibility gates."""

import pytest

from krisis.contracts.candidate import Candidate, Claim, GateResults
from krisis.contracts.policy import ProfileConfig
from krisis.gates.evaluator import GateEvaluator


@pytest.fixture
def grounded_profile():
    return ProfileConfig(
        kind="grounded",
        requires_grounding=True,
        requires_capability=["verifier_active"],
        hard_gates=["access", "provenance", "contradiction", "citation_resolution", "per_claim_support"],
        per_claim_min="0.800000",
        tau="0.700000",
    )


def test_access_gate_failure(grounded_profile):
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c1",
        gate_results=GateResults(access=False),
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.900000")],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand)
    assert passed is False
    assert gate == "access"


def test_unsupported_claim_rejection(grounded_profile):
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c1",
        gate_results=GateResults(),
        claims=[
            Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.950000"),
            Claim(claim_id="cl2", verdict="unsupported", calibrated_confidence="0.900000"),
        ],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand)
    assert passed is False
    assert gate == "per_claim_support"
    assert "unsupported claim cl2" in detail


def test_low_confidence_claim_rejection(grounded_profile):
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c1",
        gate_results=GateResults(),
        claims=[
            Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.790000"),  # Below 0.800000
        ],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand)
    assert passed is False
    assert gate == "per_claim_support"
    assert "confidence 0.790000 < per_claim_min" in detail


def test_valid_grounded_candidate_passes(grounded_profile):
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c1",
        gate_results=GateResults(),
        claims=[
            Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.850000"),
            Claim(claim_id="cl2", verdict="supported", calibrated_confidence="0.920000"),
        ],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand)
    assert passed is True
    assert gate is None
