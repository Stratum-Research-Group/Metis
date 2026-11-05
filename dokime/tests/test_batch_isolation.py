"""Tests for In-Context Batch Isolation Guard (Item 115 / arXiv:2610.40241)."""

import pytest
from dokime.explainers.batch_isolation import (
    BatchIsolationAuditor,
    BatchIsolationReport,
    BatchItem,
)


def test_perfectly_isolated_batch():
    """Verify metrics when batch evaluation is completely identical to isolated evaluation."""
    items = [
        BatchItem(item_id="c1", content="Paris is the capital of France."),
        BatchItem(item_id="c2", content="The Earth is flat."),
        BatchItem(item_id="c3", content="Water freezes at 0 degrees Celsius."),
    ]

    def isolated_eval(item: BatchItem) -> str:
        return "FALSE" if "flat" in item.content else "TRUE"

    def batch_eval(batch: list[BatchItem]) -> dict[str, str]:
        # Perfectly isolated batch processor
        return {it.item_id: isolated_eval(it) for it in batch}

    report = BatchIsolationAuditor.audit_batch(
        items=items,
        isolated_eval_fn=isolated_eval,
        batch_eval_fn=batch_eval,
    )

    assert report.total_items == 3
    assert report.isolated_agreement_ratio == 1.0
    assert report.permutation_invariance_ratio == 1.0
    assert report.is_batching_safe is True
    assert len(report.contaminated_item_ids) == 0


def test_neighbor_contaminated_batch():
    """Verify detection when items are influenced by preceding items in a batch."""
    items = [
        BatchItem(item_id="c1", content="Extreme violation content."),
        BatchItem(item_id="c2", content="Borderline neutral text."),
    ]

    def isolated_eval(item: BatchItem) -> str:
        return "FAIL" if "violation" in item.content else "PASS"

    def contaminated_batch_eval(batch: list[BatchItem]) -> dict[str, str]:
        # Simulated contamination: if the first item failed, judge becomes overly harsh on subsequent items
        results = {}
        harsh_mode = False
        for it in batch:
            if "violation" in it.content:
                harsh_mode = True
                results[it.item_id] = "FAIL"
            elif harsh_mode:
                results[it.item_id] = "FAIL"  # Contaminated!
            else:
                results[it.item_id] = "PASS"
        return results

    report = BatchIsolationAuditor.audit_batch(
        items=items,
        isolated_eval_fn=isolated_eval,
        batch_eval_fn=contaminated_batch_eval,
    )

    assert report.total_items == 2
    # c2 is contaminated because in forward batch it is preceded by c1 (fails), but in reverse batch it precedes c1 (passes)
    assert report.is_batching_safe is False
    assert "c2" in report.contaminated_item_ids
    assert report.isolated_agreement_ratio < 1.0


def test_serialization():
    """Verify report serialization."""
    items = [BatchItem(item_id="c1", content="text")]
    report = BatchIsolationAuditor.audit_batch(
        items=items,
        isolated_eval_fn=lambda it: "OK",
        batch_eval_fn=lambda b: {it.item_id: "OK" for it in b},
    )
    d = report.to_dict()
    assert d["total_items"] == 1
    assert d["is_batching_safe"] is True
    assert len(d["item_results"]) == 1
