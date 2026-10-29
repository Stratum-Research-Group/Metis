"""Unit tests for Krisis active learning contract extraction (ASPEST)."""

import pytest
from krisis.contracts.candidate import Candidate, Claim, EvidencePassage
from krisis.contracts.decision import KrisisDecision
from krisis.contracts.enums import DecisionStatus, Reason
from krisis.contracts.request import Capabilities, KrisisRequest
from krisis.contracts.active_learning import derive_active_record, KrisisActiveRecord


def test_derive_active_record_conformal_singleton():
    """Verify that singleton conformal predictions receive rejection score 0.0 (no need for active labeling)."""
    cand = Candidate(id="c1", text="42", claims=[Claim(claim_id="clm_1", fact="42", verdict="supported", calibrated_confidence="0.98")])
    req = KrisisRequest(query_id="q1", question="Answer to life?", candidates=[cand])
    dec = KrisisDecision(
        status=DecisionStatus.SELECTED.value,
        selected_id="c1",
        conformal={"status": "singleton", "prediction_set_ids": ["c1"], "set_size": 1},
    )

    rec = derive_active_record(dec, req)
    assert isinstance(rec, KrisisActiveRecord)
    assert rec.query_id == "q1"
    assert rec.rejection_score == 0.0
    assert rec.prediction_set_size == 1
    assert rec.decision_status == "selected"


def test_derive_active_record_conformal_ambiguous():
    """Verify that ambiguous conformal sets receive rejection score proportional to ambiguity 1 - 1/|C|."""
    cand1 = Candidate(id="c1", text="A", claims=[Claim(claim_id="clm_1", fact="A", verdict="supported", calibrated_confidence="0.85")])
    cand2 = Candidate(id="c2", text="B", claims=[Claim(claim_id="clm_2", fact="B", verdict="supported", calibrated_confidence="0.83")])
    cand3 = Candidate(id="c3", text="C", claims=[Claim(claim_id="clm_3", fact="C", verdict="supported", calibrated_confidence="0.82")])
    req = KrisisRequest(query_id="q_amb", question="Which letter?", candidates=[cand1, cand2, cand3])

    dec = KrisisDecision(
        status=DecisionStatus.ABSTAINED.value,
        reason=Reason.AMBIGUOUS_PREDICTION_SET.value,
        conformal={"status": "ambiguous", "prediction_set_ids": ["c1", "c2", "c3"], "set_size": 3},
    )

    rec = dec.to_active_record(req)
    assert rec.decision_status == "abstained"
    assert rec.prediction_set_size == 3
    # 1 - 1/3 = 0.6667
    assert rec.rejection_score == pytest.approx(0.6667, abs=1e-3)
    assert len(rec.candidates) == 3


def test_derive_active_record_conformal_empty():
    """Verify that empty conformal prediction sets receive maximum rejection score 1.0."""
    req = KrisisRequest(query_id="q_empty", question="Unknown territory", candidates=[])
    dec = KrisisDecision(
        status=DecisionStatus.ABSTAINED.value,
        reason=Reason.CONFORMAL_EMPTY_SET.value,
        conformal={"status": "empty", "prediction_set_ids": [], "set_size": 0},
    )

    rec = derive_active_record(dec, req)
    assert rec.rejection_score == 1.0
    assert rec.prediction_set_size == 0


def test_derive_active_record_gate_failure_and_scores():
    """Verify heuristic scores fallback when conformal prediction is not invoked."""
    cand = Candidate(id="c1", text="Unverified answer", claims=[])
    req = KrisisRequest(query_id="q_gate", question="Hard problem", candidates=[cand])
    dec = KrisisDecision(
        status=DecisionStatus.ABSTAINED.value,
        reason=Reason.NO_VALID_CANDIDATE.value,
        scores={"c1": "0.45"},
    )

    rec = dec.to_active_record(req)
    # 1.0 - 0.45 = 0.55
    assert rec.rejection_score == pytest.approx(0.55, abs=1e-3)
    assert rec.candidates[0]["score"] == "0.45"
