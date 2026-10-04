"""LTHCM / Stoma memory and evidence retrieval adapter with explicit temporal bounds."""

import time
from typing import Any, Dict, List, Optional
from kybernetes.contracts.turn import Citation


class LTHCMAdapter:
    """
    Adapter for Long-Term Hierarchical Context Memory (LTHCM) and Stoma index.
    Always specifies an explicit `as_of` epoch timestamp to avoid temporal leakage.
    """

    def __init__(self, memory_root: Optional[str] = None):
        self.memory_root = memory_root
        self._local_corpus = self._init_local_corpus()

    def _init_local_corpus(self) -> List[Dict[str, Any]]:
        return [
            {
                "passage_id": "pass_metis_arch_01",
                "source_id": "doc_metis_overview",
                "source_title": "Metis Ecosystem Architecture",
                "text": "Kybernetes serves as the orchestrator and steersman of the Metis local AI stack, coordinating Hestia, LTHCM, Dianoia, Phylax, and Krisis behind a unified Prosopon interface.",
                "keywords": ["kybernetes", "metis", "orchestrator", "prosopon", "hestia"],
                "created_epoch": 1700000000.0,
            },
            {
                "passage_id": "pass_phylax_sec_01",
                "source_id": "doc_phylax_specs",
                "source_title": "Phylax Security Gateway Specification",
                "text": "Phylax enforces capability authorization at the point of action with immutable SHA-256 hash-chained audit trails and one-use approval digests.",
                "keywords": ["phylax", "security", "approval", "audit", "capability"],
                "created_epoch": 1705000000.0,
            },
            {
                "passage_id": "pass_dianoia_walk_01",
                "source_id": "doc_dianoia_deliberation",
                "source_title": "Dianoia Graph Walk Deliberation",
                "text": "Dianoia conducts multi-step graph deliberation across connected evidence nodes, falling back cleanly to flat passage reranking when 0 edges are traversed.",
                "keywords": ["dianoia", "graph", "deliberation", "multi-hop", "walk", "edges"],
                "created_epoch": 1710000000.0,
            },
            {
                "passage_id": "pass_krisis_gate_01",
                "source_id": "doc_krisis_evaluation",
                "source_title": "Krisis Answer Selection & Scoring",
                "text": "Krisis scores answer candidates using 6-decimal-place half-even rounding, exact candidate deduplication, and strict eligibility gates.",
                "keywords": ["krisis", "scoring", "gates", "selection", "evaluation"],
                "created_epoch": 1715000000.0,
            },
        ]

    def retrieve(
        self,
        query: str,
        as_of: Optional[float] = None,
        top_k: int = 3,
    ) -> List[Citation]:
        """
        Retrieve relevant memory passages subject to explicit temporal cut-off `as_of`.
        as_of: POSIX epoch timestamp in seconds. Passages created after as_of are excluded.
        """
        current_time = as_of if as_of is not None else time.time()
        q_tokens = set(query.lower().split())

        scored_passages = []
        for item in self._local_corpus:
            if item["created_epoch"] > current_time:
                continue

            matches = sum(1 for kw in item["keywords"] if kw in query.lower())
            token_overlap = sum(1 for tok in q_tokens if tok in item["text"].lower())
            total_score = matches * 0.4 + token_overlap * 0.1

            if total_score > 0:
                scored_passages.append((total_score, item))

        scored_passages.sort(key=lambda x: x[0], reverse=True)
        results: List[Citation] = []
        for idx, (score, item) in enumerate(scored_passages[:top_k], 1):
            results.append(
                Citation(
                    citation_id=f"[{idx}]",
                    passage_id=item["passage_id"],
                    source_id=item["source_id"],
                    source_title=item["source_title"],
                    text=item["text"],
                    score=round(min(1.0, 0.5 + score * 0.2), 3),
                )
            )

        return results
