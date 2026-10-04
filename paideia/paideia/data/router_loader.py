"""LTHCM router dataset loader and counterfactual label processor."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import ValidationError

from paideia.contracts.lthcm_schema import (
    LTHCMRoutingRecord,
    RouteOutcome,
    RouteUtility,
)


class RouterDatasetAudit:
    """Audit report for routing dataset ingestion."""
    def __init__(self):
        self.total_records: int = 0
        self.valid_records: int = 0
        self.schema_violations: int = 0
        self.duplicates_removed: int = 0
        self.missing_counterfactuals: int = 0
        self.ties_excluded: int = 0
        self.class_counts_type: Dict[str, int] = {}
        self.class_counts_best_route: Dict[str, int] = {}
        self.exclusion_reasons: Dict[str, int] = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "schema_violations": self.schema_violations,
            "duplicates_removed": self.duplicates_removed,
            "missing_counterfactuals": self.missing_counterfactuals,
            "ties_excluded": self.ties_excluded,
            "class_counts_type": self.class_counts_type,
            "class_counts_best_route": self.class_counts_best_route,
            "exclusion_reasons": self.exclusion_reasons,
        }


def load_router_dataset(
    path: Path,
    utility: Optional[RouteUtility] = None,
    eligible_routes: Optional[List[str]] = None,
) -> Tuple[List[LTHCMRoutingRecord], RouterDatasetAudit]:
    """
    Load and strictly validate LTHCM routing outcomes.
    
    Args:
        path: Path to JSON or JSONL file containing LTHCM routing records.
        utility: Declared utility function for deriving best-route counterfactuals.
        eligible_routes: Declared list of route IDs that must be present in every pair.
        
    Returns:
        Tuple of (clean records, audit summary).
    """
    if utility is None:
        utility = RouteUtility()
        
    audit = RouterDatasetAudit()
    clean_records: List[LTHCMRoutingRecord] = []
    seen_queries: Set[str] = set()

    with open(path, "r", encoding="utf-8") as f:
        # Check if JSON array or JSONL
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == "[":
            raw_items = json.load(f)
        else:
            raw_items = [json.loads(line) for line in f if line.strip()]

    audit.total_records = len(raw_items)

    for raw in raw_items:
        try:
            record = LTHCMRoutingRecord.model_validate(raw)
        except (ValidationError, Exception) as exc:
            audit.schema_violations += 1
            reason = f"Schema error: {type(exc).__name__}"
            audit.exclusion_reasons[reason] = audit.exclusion_reasons.get(reason, 0) + 1
            continue

        # Deduplication on normalized question/query_id
        q_norm = record.question.strip().lower()
        if q_norm in seen_queries or record.query_id in seen_queries:
            audit.duplicates_removed += 1
            audit.exclusion_reasons["Duplicate question/query_id"] = (
                audit.exclusion_reasons.get("Duplicate question/query_id", 0) + 1
            )
            continue
        seen_queries.add(q_norm)
        seen_queries.add(record.query_id)

        # Track type counts
        if record.gold_type:
            audit.class_counts_type[record.gold_type] = (
                audit.class_counts_type.get(record.gold_type, 0) + 1
            )

        # Counterfactual best-route evaluation
        if eligible_routes:
            best_route = record.get_best_route(utility, eligible_routes)
            if best_route is None:
                # Distinguish missing counterfactual from tie
                has_all = all(r in record.route_outcomes for r in eligible_routes)
                if not has_all:
                    audit.missing_counterfactuals += 1
                    audit.exclusion_reasons["Missing counterfactual route"] = (
                        audit.exclusion_reasons.get("Missing counterfactual route", 0) + 1
                    )
                else:
                    audit.ties_excluded += 1
                    audit.exclusion_reasons["Tie under utility tolerance"] = (
                        audit.exclusion_reasons.get("Tie under utility tolerance", 0) + 1
                    )
            else:
                audit.class_counts_best_route[best_route] = (
                    audit.class_counts_best_route.get(best_route, 0) + 1
                )

        clean_records.append(record)
        audit.valid_records += 1

    return clean_records, audit
