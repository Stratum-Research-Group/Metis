"""
Answer-time verification for Dianoia.

After the walk produces an ANSWER, re-runs LTHCM's sheaf/NLI contradiction
check between the drafted answer and the retrieved evidence (answer-vs-evidence,
not passage-vs-passage).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore

try:
    from lthcm.core.sheaf import SheafContradictionFilter
except ImportError:
    SheafContradictionFilter = Any  # type: ignore


def verify_answer_against_evidence(
    answer_text: str,
    evidence_passages: List[Dict[str, Any]],
    answer_embedding: torch.Tensor,
    evidence_embeddings: torch.Tensor,
    sheaf_filter: SheafContradictionFilter,
    use_nli: bool = True,
) -> Dict[str, Any]:
    """
    Check the drafted answer for contradictions against the retrieved evidence.

    This is answer-vs-evidence verification: we build a set of passages that
    includes the answer as the first element, then run sheaf evaluation on
    the complete set. Conflicts involving the answer node indicate the answer
    contradicts the evidence.

    Args:
        answer_text: The drafted answer text.
        evidence_passages: List of evidence passage dicts (must have 'content'
            and optionally 'doc_id' keys).
        answer_embedding: [d_model] tensor for the answer.
        evidence_embeddings: [N, d_model] tensor for evidence passages.
        sheaf_filter: LTHCM's SheafContradictionFilter instance.
        use_nli: Whether to use NLI-based semantic verification.

    Returns:
        Dict with:
            - conflicts_found: bool
            - conflict_detail: str | None (human-readable conflict report)
            - report: str (summary of what was checked and the outcome)
            - sheaf_result: raw sheaf evaluation result
    """
    # Build combined passage list: answer first, then evidence
    answer_passage = {
        "doc_id": "__dianoia_answer__",
        "content": answer_text,
    }
    all_passages = [answer_passage] + list(evidence_passages)

    # Stack embeddings: answer first, then evidence
    all_embeddings = torch.cat(
        [answer_embedding.unsqueeze(0), evidence_embeddings], dim=0
    )

    sheaf_result = sheaf_filter.evaluate_passages(
        passage_embeddings=all_embeddings,
        passages=all_passages,
        use_nli=use_nli,
    )

    # Check if any conflicting pair involves the answer node
    answer_conflicts = []
    for pair in sheaf_result.get("conflicting_pairs", []):
        if pair["doc_a"] == "__dianoia_answer__" or pair["doc_b"] == "__dianoia_answer__":
            answer_conflicts.append(pair)

    conflicts_found = len(answer_conflicts) > 0

    # Build human-readable conflict detail
    conflict_detail = None
    if conflicts_found:
        lines = []
        for pair in answer_conflicts:
            other_id = pair["doc_b"] if pair["doc_a"] == "__dianoia_answer__" else pair["doc_a"]
            other_content = pair["content_b"] if pair["doc_a"] == "__dianoia_answer__" else pair["content_a"]
            lines.append(
                f"Answer conflicts with [{other_id}]: "
                f"'{other_content[:200]}...' "
                f"(obstruction energy: {pair['obstruction_energy']:.4f})"
            )
        conflict_detail = "\n".join(lines)

    # Report actual NLI usage, never mere intent or flag
    actual_nli_active = bool(use_nli and getattr(sheaf_filter, "nli_model", None) is not None)
    if actual_nli_active:
        nli_name = getattr(sheaf_filter, "nli_model_name", "cross-encoder")
        method_str = f"sheaf cohomological contradiction filter (NLI active: {nli_name})"
    else:
        method_str = "sheaf cohomological contradiction filter (NLI inactive: vector sheaf Laplacian only)"

    # Build report: state what was checked and the outcome
    report_lines = [
        f"Verification: answer checked against {len(evidence_passages)} evidence passage(s).",
        f"Method: {method_str}.",
        f"Obstruction energy: {sheaf_result.get('total_obstruction', 0.0):.4f}.",
        f"Consistent: {sheaf_result.get('is_consistent', True)}.",
    ]
    if conflicts_found:
        report_lines.append(
            f"Answer-evidence conflicts found: {len(answer_conflicts)}."
        )
    else:
        report_lines.append("No answer-evidence conflicts detected.")
    report = "\n".join(report_lines)

    return {
        "verified": True,
        "verifier_mode": "sheaf_nli" if actual_nli_active else "sheaf_laplacian",
        "provenance": "dianoia_sheaf_verification",
        "conflicts_found": conflicts_found,
        "conflict_detail": conflict_detail,
        "report": report,
        "sheaf_result": sheaf_result,
    }
