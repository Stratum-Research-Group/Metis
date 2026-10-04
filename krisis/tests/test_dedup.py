"""Tests for candidate deduplication."""

from krisis.contracts.candidate import Candidate
from krisis.dedup.exact import ExactDeduplicator


def test_exact_dedup_merges_identical_text_and_citations():
    dedup = ExactDeduplicator()
    cands = [
        Candidate(id="c1", normalized_text="the answer is 42", citations=["p1"], snapshot_id="snap1"),
        Candidate(id="c2", normalized_text="the answer is 42", citations=["p1"], snapshot_id="snap1"),
        Candidate(id="c3", normalized_text="the answer is NOT 42", citations=["p1"], snapshot_id="snap1"),
    ]

    unique, merged, kept_separate = dedup.deduplicate(cands)

    assert len(unique) == 2
    assert unique[0].id == "c1"
    assert unique[1].id == "c3"

    assert len(merged) == 1
    assert merged[0]["into"] == "c1"
    assert merged[0]["absorbed"] == ["c2"]
    assert merged[0]["lineage_logged"] is True

    assert "c3" in kept_separate


def test_candidates_without_text_are_not_merged():
    dedup = ExactDeduplicator()
    cands = [
        Candidate(id="c1", citations=["p1"]),
        Candidate(id="c2", citations=["p1"]),
    ]

    unique, merged, kept_separate = dedup.deduplicate(cands)
    assert len(unique) == 2
    assert len(merged) == 0
