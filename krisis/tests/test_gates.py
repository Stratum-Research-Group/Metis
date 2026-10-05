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
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
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
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
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
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
        claims=[
            Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.850000"),
            Claim(claim_id="cl2", verdict="supported", calibrated_confidence="0.920000"),
        ],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand)
    assert passed is True
    assert gate is None


def test_omitted_or_partial_gate_results_fail_closed(grounded_profile):
    """Item 8: Omitted, null, or partial gate results fail closed."""
    evaluator = GateEvaluator(grounded_profile)
    # Completely empty GateResults
    cand_empty = Candidate(
        id="c_empty",
        gate_results=GateResults(),
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.950000")],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand_empty)
    assert passed is False
    assert gate == "access"

    # Partial GateResults (access=True, but provenance is None)
    cand_partial = Candidate(
        id="c_partial",
        gate_results=GateResults(access=True),
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.950000")],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand_partial)
    assert passed is False
    assert gate == "provenance"


def test_nonexistent_citation_resolution_fails_closed(grounded_profile):
    """Item 8: Candidates citing non-existent passages fail citation_resolution gate."""
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c_bad_cite",
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
        citations=["nonexistent_passage_999"],
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.950000")],
    )
    # Provided valid evidence only contains "doc1_p1"
    passed, gate, detail = evaluator.evaluate_candidate(cand, valid_passage_ids={"doc1_p1"})
    assert passed is False
    assert gate == "citation_resolution"
    assert "does not exist in verified evidence" in detail


def test_non_finite_confidence_fails_closed(grounded_profile):
    """Item 8: NaN, Inf, or out-of-range confidence values fail closed."""
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c_nan",
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="NaN")],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand)
    assert passed is False
    assert gate == "feature_valid"


def test_claim_citing_nothing_fails_citation_resolution(grounded_profile):
    """Item 10: Supported claim citing nothing fails citation_resolution when evidence is present."""
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c_uncited",
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.950000", cited_passage_ids=[])],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand, valid_passage_ids={"doc1_p1"})
    assert passed is False
    assert gate == "citation_resolution"
    assert "cites no evidence passages" in detail


def test_untrusted_caller_gate_producer_rejected(grounded_profile):
    """Item 10: Gate results with producer 'caller' are rejected."""
    evaluator = GateEvaluator(grounded_profile)
    cand = Candidate(
        id="c_caller",
        gate_results={"access": True, "provenance": True, "contradiction": True, "citation_resolution": True, "producer": "caller"},
        claims=[Claim(claim_id="cl1", verdict="supported", calibrated_confidence="0.950000", cited_passage_ids=["doc1_p1"])],
    )
    passed, gate, detail = evaluator.evaluate_candidate(cand, valid_passage_ids={"doc1_p1"})
    assert passed is False
    assert gate == "provenance"

