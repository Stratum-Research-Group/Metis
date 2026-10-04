"""Exact text and citation deduplication engine with lineage tracking."""

from typing import Any
from krisis.contracts.candidate import Candidate


class ExactDeduplicator:
    """Deduplicates candidates based on exact normalized text and citation spans."""

    def deduplicate(
        self,
        candidates: list[Candidate],
    ) -> tuple[list[Candidate], list[dict[str, Any]], list[str]]:
        """Deduplicate candidates.
        
        Returns:
            unique_candidates: list[Candidate]
            merged_lineage: list[dict[str, Any]]  # [{"into": ..., "absorbed": [...], "lineage_logged": True}]
            kept_separate: list[str]  # IDs of candidates kept distinct
        """
        if not candidates:
            return [], [], []

        merged_into: dict[str, list[str]] = {}  # primary_id -> list of absorbed_ids
        seen_keys: dict[tuple[str, tuple[str, ...], str | None], Candidate] = {}
        unique_list: list[Candidate] = []
        all_ids = [c.id for c in candidates]

        for cand in candidates:
            text_norm = (cand.normalized_text or cand.text or "").strip().lower()
            if not text_norm:
                # Without text, cannot be declared an exact text duplicate
                unique_list.append(cand)
                continue

            citations = tuple(sorted(cand.citations or cand.cited_passage_ids or []))
            snapshot = cand.snapshot_id

            key = (text_norm, citations, snapshot)

            if key in seen_keys:
                primary = seen_keys[key]
                if primary.id not in merged_into:
                    merged_into[primary.id] = []
                merged_into[primary.id].append(cand.id)
            else:
                seen_keys[key] = cand
                unique_list.append(cand)

        merged_records: list[dict[str, Any]] = []
        absorbed_ids: set[str] = set()
        for into_id, absorbed in merged_into.items():
            merged_records.append(
                {
                    "into": into_id,
                    "absorbed": absorbed,
                    "lineage_logged": True,
                }
            )
            absorbed_ids.update(absorbed)

        # Kept separate are candidates that were not absorbed into anything else
        kept_separate = [cid for cid in all_ids if cid not in absorbed_ids and cid not in merged_into]

        return unique_list, merged_records, kept_separate
