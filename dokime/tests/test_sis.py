"""Unit and integration tests for Sufficient Input Subsets (SIS) in Dokime."""

import pytest

from dokime.cli import main
from dokime.explainers.sis import (
    SISCollectionResult,
    SISResult,
    SufficientInputSubsets,
    explain_text_sufficiency,
    split_text_into_units,
)


def test_sis_minimal_subset_selection():
    """Verify greedy backward selection finds the single essential sentence among noise."""
    sentences = [
        "The sky is blue today.",
        "There are clouds moving slowly.",
        "The server port is 8080.",  # Crucial sentence
        "Birds are flying in the park.",
        "A car drove past the building.",
    ]

    def score_fn(sub_list: list[str]) -> float:
        # Score is 1.0 if "8080" is in any active sentence, 0.1 otherwise
        joined = " ".join(sub_list)
        return 1.0 if "8080" in joined else 0.1

    explainer = SufficientInputSubsets(score_fn=score_fn, tau=0.8)
    res = explainer.find_sis(sentences)

    assert res.is_sufficient is True
    assert res.subset_indices == [2]
    assert res.subset_items == ["The server port is 8080."]
    assert res.steps_taken == 4
    assert res.reduction_ratio == 0.8
    assert res.final_score >= 0.8


def test_sis_deterministic_tie_breaking():
    """Verify deterministic tie breaking when multiple candidates retain identical score."""
    items = ["A", "B", "C", "D"]

    def score_fn(sub: list[str]) -> float:
        # All items contribute equally, removing any item yields score 0.9
        return 0.9 if len(sub) >= 2 else 0.2

    explainer = SufficientInputSubsets(score_fn=score_fn, tau=0.8)
    res = explainer.find_sis(items)

    assert res.is_sufficient is True
    assert len(res.subset_indices) == 2
    # Deterministic tie-breaking picks lowest index first, leaving the final 2 elements
    assert res.subset_indices == [2, 3]


def test_sis_insufficient_initial_input():
    """Verify immediate fail-safe when the full initial input does not meet threshold."""
    items = ["Irrelevant 1", "Irrelevant 2"]

    def score_fn(sub: list[str]) -> float:
        return 0.4

    explainer = SufficientInputSubsets(score_fn=score_fn, tau=0.8)
    res = explainer.find_sis(items)

    assert res.is_sufficient is False
    assert res.steps_taken == 0
    assert res.subset_indices == [0, 1]
    assert res.final_score == 0.4


def test_sis_disjoint_collection():
    """Verify uncovering multiple disjoint rationales by masking earlier selected subsets."""
    sentences = [
        "Noise sentence at start.",
        "Fact A: Metis is a high-assurance platform.",  # Rationale 1
        "Unrelated middle filler.",
        "Fact B: Metis implements fail-closed governance.",  # Rationale 2
        "Noise sentence at end.",
    ]

    def score_fn(sub: list[str]) -> float:
        joined = " ".join(sub)
        has_a = "high-assurance" in joined
        has_b = "fail-closed" in joined
        if has_a or has_b:
            return 1.0
        return 0.0

    explainer = SufficientInputSubsets(score_fn=score_fn, tau=0.8)
    res = explainer.find_sis_collection(sentences, max_subsets=3)

    assert isinstance(res, SISCollectionResult)
    assert res.total_subsets == 2
    assert res.residual_score == 0.0

    # Subset 1 should isolate Fact A (index 1)
    # Subset 2 should isolate Fact B (index 3)
    found_indices = [sub.subset_indices for sub in res.subsets]
    assert [1] in found_indices
    assert [3] in found_indices


def test_sis_mask_token_mode():
    """Verify substitution with mask_token instead of deletion."""
    words = ["user", "entered", "secret", "password"]

    def score_fn(sub: list[str]) -> float:
        assert len(sub) == 4, "Mask token mode must preserve original sequence length"
        # Only "password" is required
        return 1.0 if sub[3] == "password" else 0.0

    explainer = SufficientInputSubsets(score_fn=score_fn, tau=0.8, mask_token="[MASK]")
    res = explainer.find_sis(words)

    assert res.is_sufficient is True
    assert res.subset_indices == [3]
    assert res.subset_items == ["password"]


def test_explain_text_sufficiency_sentences():
    text = (
        "Quantum computing relies on qubits. "
        "The weather in Tokyo is sunny. "
        "Superposition enables parallel state evaluation."
    )

    def score_fn(cand: str) -> float:
        return 1.0 if "qubits" in cand else 0.0

    res = explain_text_sufficiency(text, score_fn=score_fn, tau=0.8, unit="sentence")
    assert isinstance(res, SISResult)
    assert res.is_sufficient is True
    assert res.subset_items == ["Quantum computing relies on qubits."]
    assert res.reduction_ratio == pytest.approx(2 / 3, abs=0.01)


def test_split_text_into_units():
    text = "Sentence one. Sentence two! Sentence three?"
    assert split_text_into_units(text, "sentence") == [
        "Sentence one.",
        "Sentence two!",
        "Sentence three?",
    ]
    assert split_text_into_units("word1 word2 word3", "word") == ["word1", "word2", "word3"]

    with pytest.raises(ValueError, match="Unsupported unit"):
        split_text_into_units("sample", "invalid")


def test_cli_explain_command(capsys):
    """Test CLI integration of 'dokime explain'."""
    text = "First irrelevant sentence. Target keyword Paris is present here. Final filler sentence."
    code = main([
        "explain",
        "--text", text,
        "--target-keyword", "Paris",
        "--tau", "0.8",
        "--unit", "sentence",
    ])
    assert code == 0
    captured = capsys.readouterr().out
    assert "== Dokime Sufficient Input Subsets (SIS) ==" in captured
    assert "Is Sufficient:      True" in captured
    assert "Paris is present here" in captured


def test_cli_explain_disjoint(capsys):
    """Test CLI integration of 'dokime explain --disjoint'."""
    text = "Alpha factor active. Middle noise. Alpha factor repeated."
    code = main([
        "explain",
        "--text", text,
        "--target-keyword", "Alpha",
        "--tau", "0.8",
        "--unit", "sentence",
        "--disjoint",
    ])
    assert code == 0
    captured = capsys.readouterr().out
    assert "Disjoint Subsets:   2" in captured
    assert "Subset #1" in captured
    assert "Subset #2" in captured
