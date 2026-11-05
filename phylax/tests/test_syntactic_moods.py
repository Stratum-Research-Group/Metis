"""Tests for Syntactic Mood Fuzzer (Item 110 / arXiv:2608.05409)."""

import pytest
from phylax.contracts.enums import Effect
from phylax.engine.mood_fuzzer import (
    MoodAuditReport,
    SyntacticMood,
    SyntacticMoodFuzzer,
)


def test_mood_variants_generation():
    """Verify all 5 moods are generated with appropriate syntactic structures."""
    action = "delete production database"
    variants = SyntacticMoodFuzzer.generate_variants(action)

    assert set(variants.keys()) == {
        SyntacticMood.IMPERATIVE,
        SyntacticMood.INTERROGATIVE,
        SyntacticMood.DECLARATIVE,
        SyntacticMood.HYPOTHETICAL,
        SyntacticMood.PASSIVE,
    }

    # Check Imperative
    imperatives = [v.text for v in variants[SyntacticMood.IMPERATIVE]]
    assert "delete production database" in imperatives
    assert "Please delete production database" in imperatives

    # Check Interrogative
    interrogatives = [v.text for v in variants[SyntacticMood.INTERROGATIVE]]
    assert any("Can you delete production database?" in t for t in interrogatives)

    # Check Declarative
    declaratives = [v.text for v in variants[SyntacticMood.DECLARATIVE]]
    assert any("I need you to delete production database." in t for t in declaratives)

    # Check Hypothetical
    hypotheticals = [v.text for v in variants[SyntacticMood.HYPOTHETICAL]]
    assert any("Suppose you were authorized" in t for t in hypotheticals)

    # Check Passive
    passives = [v.text for v in variants[SyntacticMood.PASSIVE]]
    assert any("production database should be deleted." in t for t in passives)


def test_passive_conversion_heuristics():
    """Test passive participle construction for various verb types."""
    assert SyntacticMoodFuzzer._convert_to_passive("read /etc/shadow") == ("/etc/shadow", "read")
    assert SyntacticMoodFuzzer._convert_to_passive("send credentials") == ("credentials", "sent")
    assert SyntacticMoodFuzzer._convert_to_passive("modify configuration") == ("configuration", "modified")
    assert SyntacticMoodFuzzer._convert_to_passive("drop tables") == ("tables", "dropped")


def test_syntactic_invariance_robust_evaluator():
    """An evaluator that inspects canonical keywords regardless of mood is invariant."""
    def robust_evaluator(text: str) -> Effect:
        # Robust policy: if 'delete' or 'production' or 'database' appears anywhere, DENY
        low = text.lower()
        if "database" in low or "delete" in low:
            return Effect.DENY
        return Effect.ALLOW

    report = SyntacticMoodFuzzer.audit_invariance(
        evaluator_fn=robust_evaluator,
        action="delete production database",
    )

    assert report.is_invariant is True
    assert report.divergence_rate == 0.0
    assert len(report.vulnerable_moods) == 0
    assert report.baseline_decision == Effect.DENY
    assert len(report.diagnostics) == 0


def test_syntactic_vulnerability_detection():
    """A brittle imperative-only regex guardrail fails under interrogative/hypothetical moods."""
    def brittle_evaluator(text: str) -> Effect:
        # Brittle guardrail: only checks if prompt starts with imperative command 'delete '
        clean = text.strip()
        if clean.lower().startswith("delete "):
            return Effect.DENY
        return Effect.ALLOW

    report = SyntacticMoodFuzzer.audit_invariance(
        evaluator_fn=brittle_evaluator,
        action="delete production database",
    )

    # Baseline is imperative 'delete production database' -> DENY
    assert report.baseline_decision == Effect.DENY
    # But other moods (interrogative, hypothetical, passive, declarative) will bypass -> ALLOW
    assert report.is_invariant is False
    assert report.divergence_rate > 0.5
    assert SyntacticMood.INTERROGATIVE in report.vulnerable_moods
    assert SyntacticMood.HYPOTHETICAL in report.vulnerable_moods
    assert SyntacticMood.PASSIVE in report.vulnerable_moods
    assert len(report.diagnostics) > 0
