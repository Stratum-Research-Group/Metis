"""Tests for KnowNo Conformal Uncertainty in Krisis."""

from decimal import Decimal
import pytest

from krisis.contracts.candidate import Candidate, Claim, EvidencePassage, GateResults
from krisis.contracts.enums import DecisionStatus, Reason
from krisis.contracts.request import Capabilities, KrisisRequest
from krisis.engine.selector import KrisisSelector
from krisis.gates.conformal import ConformalCalibrator, ConformalPredictionSet


def test_conformal_calibrator_q_hat_computation():
    """Verify finite-sample quantile calculation according to KnowNo / Split Conformal formula."""
    # 10 calibration scores: [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    # nonconformities (1 - score): [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    calib = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    
    # alpha = 0.20 -> 1 - alpha = 0.80.
    # rank = ceil((10 + 1) * 0.80) = ceil(8.8) = 9
    # 0-indexed rank 9 is index 8 -> nonconformities[8] = 0.45
    q_hat = ConformalCalibrator.compute_q_hat(calib, alpha=0.20)
    assert pytest.approx(q_hat, abs=1e-5) == 0.45

    # alpha = 0.05 -> 1 - alpha = 0.95.
    # rank = ceil(11 * 0.95) = ceil(10.45) = 11 -> clamped to n = 10 -> index 9 -> 0.50
    q_hat_conservative = ConformalCalibrator.compute_q_hat(calib, alpha=0.05)
    assert pytest.approx(q_hat_conservative, abs=1e-5) == 0.50


def test_conformal_calibrator_invalid_inputs():
    with pytest.raises(ValueError, match="Calibration set cannot be empty"):
        ConformalCalibrator.compute_q_hat([], alpha=0.05)

    with pytest.raises(ValueError, match="Alpha must be in"):
        ConformalCalibrator.compute_q_hat([0.9], alpha=1.5)


def test_conformal_prediction_set_statuses():
    # Singleton
    scores = {"c1": 0.92, "c2": 0.60, "c3": 0.40}
    # q_hat = 0.15 -> threshold = 1.0 - 0.15 = 0.85
    # Only c1 >= 0.85
    pset = ConformalCalibrator.evaluate(scores, q_hat=0.15, alpha=0.10)
    assert pset.is_singleton is True
    assert pset.status == "singleton"
    assert pset.selected_id == "c1"
    assert pset.candidate_ids == ["c1"]

    # Ambiguous (|C| > 1)
    # q_hat = 0.35 -> threshold = 0.65
    # c1 (0.92) and c2 (0.70) are >= 0.65
    scores_amb = {"c1": 0.92, "c2": 0.70, "c3": 0.40}
    pset_amb = ConformalCalibrator.evaluate(scores_amb, q_hat=0.35, alpha=0.10)
    assert pset_amb.is_singleton is False
    assert pset_amb.status == "ambiguous"
    assert pset_amb.selected_id is None
    assert pset_amb.set_size == 2
    assert pset_amb.candidate_ids == ["c1", "c2"]

    # Empty (|C| == 0)
    # q_hat = 0.05 -> threshold = 0.95
    # No candidate >= 0.95
    pset_empty = ConformalCalibrator.evaluate(scores_amb, q_hat=0.05, alpha=0.10)
    assert pset_empty.is_empty is True
    assert pset_empty.status == "empty"
    assert pset_empty.candidate_ids == []
    assert pset_empty.set_size == 0


@pytest.fixture
def default_selector():
    return KrisisSelector("krisis-policy-0.1.json")


def create_candidate(cid: str, conf: str) -> Candidate:
    return Candidate(
        id=cid,
        text=f"Answer {cid}",
        cited_passage_ids=["pass_1"],
        gate_results=GateResults(access=True, provenance=True, contradiction=True, citation_resolution=True),
        claims=[
            Claim(
                claim_id=f"{cid}_cl1",
                verdict="supported",
                calibrated_confidence=conf,
                cited_passage_ids=["pass_1"],
            )
        ],
    )


def test_krisis_selector_conformal_singleton_selection(default_selector):
    c1 = create_candidate("cand_1", "0.950000")
    c2 = create_candidate("cand_2", "0.850000")
    ev = [EvidencePassage(passage_id="pass_1", text="Verified context text")]

    # With q_hat = 0.10, threshold = 0.90 -> only cand_1 survives
    req = KrisisRequest(
        query="What is the answer?",
        question_class="closed_form_number",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True),
        candidates=[c1, c2],
        evidence=ev,
        conformal_q_hat=Decimal("0.100000"),
        conformal_alpha=Decimal("0.050000"),
    )

    decision = default_selector.evaluate(req)
    assert decision.status == DecisionStatus.SELECTED.value
    assert decision.selected_id == "cand_1"
    assert decision.conformal is not None
    assert decision.conformal["status"] == "singleton"
    assert decision.conformal["selected_id"] == "cand_1"
    assert decision.conformal["set_size"] == 1


def test_krisis_selector_conformal_ambiguity_abstention(default_selector):
    c1 = create_candidate("cand_1", "0.950000")
    c2 = create_candidate("cand_2", "0.920000")
    ev = [EvidencePassage(passage_id="pass_1", text="Verified context text")]

    # With q_hat = 0.10, threshold = 0.90 -> both cand_1 and cand_2 survive (|C| = 2)
    req = KrisisRequest(
        query="Ambiguous question?",
        question_class="closed_form_number",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True),
        candidates=[c1, c2],
        evidence=ev,
        conformal_q_hat=Decimal("0.100000"),
        conformal_alpha=Decimal("0.050000"),
    )

    decision = default_selector.evaluate(req)
    assert decision.status == DecisionStatus.ABSTAINED.value
    assert decision.reason == Reason.AMBIGUOUS_PREDICTION_SET.value
    assert decision.conformal is not None
    assert decision.conformal["status"] == "ambiguous"
    assert decision.conformal["set_size"] == 2
    assert decision.conformal["candidate_ids"] == ["cand_1", "cand_2"]


def test_krisis_selector_conformal_empty_set_abstention(default_selector):
    c1 = create_candidate("cand_1", "0.850000")
    c2 = create_candidate("cand_2", "0.820000")
    ev = [EvidencePassage(passage_id="pass_1", text="Verified context text")]

    # With q_hat = 0.05, threshold = 0.95 -> none meet threshold
    req = KrisisRequest(
        query="Difficult question?",
        question_class="closed_form_number",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True),
        candidates=[c1, c2],
        evidence=ev,
        conformal_q_hat=Decimal("0.050000"),
        conformal_alpha=Decimal("0.050000"),
    )

    decision = default_selector.evaluate(req)
    assert decision.status == DecisionStatus.ABSTAINED.value
    assert decision.reason == Reason.CONFORMAL_EMPTY_SET.value
    assert decision.conformal is not None
    assert decision.conformal["status"] == "empty"
    assert decision.conformal["set_size"] == 0


def test_krisis_selector_dynamic_calibration_scores(default_selector):
    c1 = create_candidate("cand_1", "0.950000")
    c2 = create_candidate("cand_2", "0.850000")
    ev = [EvidencePassage(passage_id="pass_1", text="Verified context text")]

    # Calibration scores from held-out correct predictions
    calib = [Decimal("0.98"), Decimal("0.95"), Decimal("0.90"), Decimal("0.85"), Decimal("0.80")]
    req = KrisisRequest(
        query="Question with calibration scores",
        question_class="closed_form_number",
        requires_grounding=True,
        capabilities=Capabilities(verifier_active=True),
        candidates=[c1, c2],
        evidence=ev,
        calibration_scores=calib,
        conformal_alpha=Decimal("0.20"),
    )

    decision = default_selector.evaluate(req)
    assert decision.status in (DecisionStatus.SELECTED.value, DecisionStatus.ABSTAINED.value)
    assert decision.conformal is not None
    assert "q_hat" in decision.conformal
    assert decision.conformal["alpha"] == 0.20
