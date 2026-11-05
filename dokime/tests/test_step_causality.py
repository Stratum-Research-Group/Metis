"""Tests for Causal Deliberation Step Audit (Item 112 / arXiv:2610.21182)."""

import pytest
from dokime.explainers.step_causality import (
    CausalDeliberationAuditor,
    DeliberationStep,
    StepCausalRole,
)


def test_purely_causal_reasoning_trace():
    """Verify that steps whose semantic contents mediate intermediate results are marked CAUSALLY_GROUNDED."""
    # A multi-step arithmetic / logic pipeline where each step provides an intermediate result
    steps = [
        "x = 5",
        "y = x * 2",  # y = 10
        "z = y + 7",  # z = 17
    ]

    def evaluate_steps(trace: list[str]) -> int:
        local_scope = {}
        for s in trace:
            try:
                exec(s, {}, local_scope)
            except Exception:
                return -999  # failure
        return local_scope.get("z", -999)

    report = CausalDeliberationAuditor.audit_trace(
        steps=steps,
        evaluate_fn=evaluate_steps,
        scorer_fn=lambda outcome: 1.0 if outcome == 17 else 0.0,
    )

    assert report.total_steps == 3
    assert report.causally_grounded_steps == 3
    assert report.causal_grounding_ratio == 1.0
    assert report.ritual_placeholder_steps == 0
    assert report.redundant_inert_steps == 0
    assert report.is_deliberation_valid is True
    assert report.deliberation_integrity_score == 1.0

    for sa in report.step_audits:
        assert sa.role == StepCausalRole.CAUSALLY_GROUNDED
        assert sa.corruption_delta == 1.0


def test_ritual_placeholder_detection():
    """Verify that steps that merely satisfy count/format requirements but whose content is unused are detected."""
    steps = [
        "Premise A: Socrates is human.",
        "Buffer filler: Let me think deeply about various philosophical concepts.",
        "Buffer filler: Contemplating the nature of mortality and truth.",
        "Conclusion: Therefore Socrates is mortal.",
    ]

    def ritual_evaluator(trace: list[str]) -> str:
        # System checks:
        # 1. Must have >= 4 steps (deliberation length check)
        # 2. Extract conclusion from first and last steps only, completely ignoring middle steps
        if len(trace) < 4:
            return "ERROR: trace too short"
        
        has_premise = any("Socrates is human" in s for s in trace)
        has_conclusion = any("Socrates is mortal" in s for s in trace)
        if has_premise and has_conclusion:
            return "CORRECT: Socrates is mortal"
        return "INCORRECT"

    report = CausalDeliberationAuditor.audit_trace(
        steps=steps,
        evaluate_fn=ritual_evaluator,
        scorer_fn=lambda outcome: 1.0 if "CORRECT" in str(outcome) else 0.0,
    )

    assert report.total_steps == 4
    # Step 0 and Step 3 are causal (their contents are checked)
    assert report.step_audits[0].role == StepCausalRole.CAUSALLY_GROUNDED
    assert report.step_audits[3].role == StepCausalRole.CAUSALLY_GROUNDED

    # Step 1 and Step 2 are ritual placeholders:
    # Deleting them drops trace length below 4 -> outcome becomes ERROR (deletion delta = 1.0)
    # But corrupting their content still keeps trace length at 4 -> outcome remains CORRECT (corruption delta = 0.0)!
    assert report.step_audits[1].role == StepCausalRole.RITUAL_PLACEHOLDER
    assert report.step_audits[2].role == StepCausalRole.RITUAL_PLACEHOLDER

    assert report.ritual_placeholder_steps == 2
    assert report.causally_grounded_steps == 2
    assert report.causal_grounding_ratio == 0.5
    assert report.ritual_placeholder_ratio == 0.5
    # Since ritual ratio >= 0.4, validity is rejected
    assert report.is_deliberation_valid is False


def test_redundant_inert_step():
    """Verify that a step whose deletion and corruption both have zero effect is marked REDUNDANT_INERT."""
    steps = [
        "fact_1 = 10",
        "unused_comment = 'this does nothing at all'",
        "fact_2 = 20",
    ]

    def evaluator(trace: list[str]) -> int:
        scope = {}
        for s in trace:
            try:
                exec(s, {}, scope)
            except Exception:
                pass
        return scope.get("fact_1", 0) + scope.get("fact_2", 0)

    report = CausalDeliberationAuditor.audit_trace(
        steps=steps,
        evaluate_fn=evaluator,
        scorer_fn=lambda outcome: 1.0 if outcome == 30 else 0.0,
    )

    assert report.total_steps == 3
    assert report.step_audits[1].role == StepCausalRole.REDUNDANT_INERT
    assert report.redundant_inert_steps == 1


def test_report_serialization():
    """Verify report dictionary export adheres to typed schema."""
    steps = [DeliberationStep(index=0, content="step 0"), DeliberationStep(index=1, content="step 1")]
    report = CausalDeliberationAuditor.audit_trace(
        steps=steps,
        evaluate_fn=lambda t: len(t),
    )
    d = report.to_dict()
    assert d["total_steps"] == 2
    assert "causal_grounding_ratio" in d
    assert "step_audits" in d
    assert len(d["step_audits"]) == 2
