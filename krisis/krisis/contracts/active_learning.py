"""Active learning contracts and record extraction for Krisis selective prediction (ASPEST).

Enables exporting abstained and uncertain query evaluations as training signals
for model fine-tuning and DPO alignment in Paideia (arXiv:2304.03870).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

from krisis.contracts.decision import KrisisDecision
from krisis.contracts.enums import DecisionStatus
from krisis.contracts.request import KrisisRequest


class KrisisActiveRecord(BaseModel):
    """An active learning sample extracted from a Krisis evaluation."""
    model_config = ConfigDict(extra="allow")

    query_id: str
    question: str
    decision_status: str
    decision_reason: Optional[str] = None
    rejection_score: float = Field(ge=0.0, le=1.0)
    prediction_set_size: Optional[int] = None
    selected_id: Optional[str] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def derive_active_record(
    decision: KrisisDecision,
    request: KrisisRequest,
    embedding: Optional[List[float]] = None,
) -> KrisisActiveRecord:
    """Derives an active selective prediction record from a Krisis evaluation.
    
    Computes rejection score U(x) in [0, 1] quantifying abstention urgency:
    - For KnowNo conformal sets:
      * singleton (C=1): 0.0 (high confidence single answer)
      * ambiguous (C>1): 1 - 1/|C| (larger sets indicate higher epistemic ambiguity)
      * empty (C=0): 1.0 (complete out-of-distribution or under-confident prediction)
    - For heuristic or gate abstentions:
      * selected: 0.0
      * abstained with candidate scores: 1.0 - max(candidate_scores)
      * gate failure / zero survivors: 1.0
    """
    prediction_set_size: Optional[int] = None

    if decision.conformal:
        conf = decision.conformal
        status = conf.get("status")
        set_ids = conf.get("prediction_set_ids", [])
        prediction_set_size = conf.get("set_size", len(set_ids))

        if status == "singleton":
            rejection_score = 0.0
        elif status == "ambiguous":
            if prediction_set_size and prediction_set_size >= 2:
                rejection_score = 1.0 - (1.0 / float(prediction_set_size))
            else:
                rejection_score = 0.5
        elif status == "empty":
            rejection_score = 1.0
        else:
            rejection_score = 0.5
    else:
        if decision.status == DecisionStatus.SELECTED.value:
            rejection_score = 0.0
        else:
            if decision.scores:
                try:
                    num_scores = [float(v) for v in decision.scores.values()]
                    max_score = max(num_scores) if num_scores else 0.0
                    rejection_score = max(0.0, min(1.0, 1.0 - max_score))
                except (ValueError, TypeError):
                    rejection_score = 1.0
            else:
                rejection_score = 1.0

    # Format candidates metadata
    candidate_summaries = []
    for cand in request.candidates:
        candidate_summaries.append({
            "id": cand.id,
            "text": cand.text,
            "claims_count": len(cand.claims),
            "score": decision.scores.get(cand.id),
        })

    return KrisisActiveRecord(
        query_id=request.query_id or f"query_{abs(hash(request.question or ''))}",
        question=request.question or "",
        decision_status=decision.status,
        decision_reason=decision.reason,
        rejection_score=round(rejection_score, 4),
        prediction_set_size=prediction_set_size,
        selected_id=decision.selected_id,
        candidates=candidate_summaries,
        embedding=embedding,
        metadata={
            "profile": decision.profile,
            "requires_grounding": request.requires_grounding,
            "conformal_status": decision.conformal.get("status") if decision.conformal else None,
        },
    )
