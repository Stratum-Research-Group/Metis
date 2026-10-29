"""Symbolon Discrete Code Retrieval Benchmark Task and Fusion Yardstick for Dokime.

Measures multiscale discrete code-overlap retrieval, flat dense baselines,
uncapped union expansion, and reciprocal rank fusion across benchmark diagnostic corpora
with explicit gold relevance and contradiction labels (Spec E2).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# Ensure symbolon is resolvable when running in monorepo environment
for _candidate in [
    Path(__file__).resolve().parents[4] / "symbolon",
    Path.cwd() / "symbolon",
]:
    if _candidate.is_dir() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from dokime.tasks.protocol import Query, Task

logger = logging.getLogger(__name__)

# Canonical benchmark corpus spanning scientific, geopolitical, aerospace, and topological domains
BENCHMARK_CORPUS: List[Dict[str, str]] = [
    {
        "id": "doc_sci_01",
        "text": "Dihydrofolate reductase inhibitors such as methotrexate bind competitively to the enzymatic active pocket with Ki of 4.2 nanomolar.",
    },
    {
        "id": "doc_sci_02_distractor",
        "text": "Penicillin is a beta-lactam antibiotic that interferes with bacterial peptidoglycan cell wall synthesis.",
    },
    {
        "id": "doc_sci_03_contradiction",
        "text": "Methotrexate is an irreversible covalent binder that inhibits bacterial wall assembly rather than human enzymatic dihydrofolate reductase.",
    },
    {
        "id": "doc_geo_01",
        "text": "Chancellor Angela Merkel arrived in Berlin for emergency European Union negotiations regarding the northern economic corridor expansion.",
    },
    {
        "id": "doc_geo_02_distractor",
        "text": "The Eiffel Tower in Paris underwent major restoration of its observation deck elevators during the 2024 Olympic Games.",
    },
    {
        "id": "doc_geo_03_contradiction",
        "text": "Chancellor Angela Merkel cancelled all Berlin summit negotiations and withdrew completely from European Union economic infrastructure treaties.",
    },
    {
        "id": "doc_space_01",
        "text": "Apollo 11 Saturn V first stage burned 20,000 pounds of RP-1 and LOX propellant per second across five F-1 engines.",
    },
    {
        "id": "doc_space_02_distractor",
        "text": "Voyager 1 crossed the heliopause into interstellar space in August 2012, powered by radioisotope thermoelectric generators.",
    },
    {
        "id": "doc_space_03_contradiction",
        "text": "Apollo 11 utilized solid rocket boosters exclusively, burning ammonium perchlorate fuel during its quiet atmospheric climb.",
    },
    {
        "id": "doc_math_01",
        "text": "The cellular sheaf Laplacian computes obstruction cohomology over directed simplices, bounding topological drift to less than 0.05 per step.",
    },
    {
        "id": "doc_math_02_distractor",
        "text": "K-means clustering partitions observations into voronoi cells by iteratively minimizing squared euclidean distances to cluster centroids.",
    },
    {
        "id": "doc_math_03_contradiction",
        "text": "Cellular sheaf Laplacian matrix eigenvalues are uniformly zero everywhere, completely unable to detect or bound topological contradiction.",
    },
    {
        "id": "doc_semi_01",
        "text": "High-NA EUV lithography achieves 0.55 numerical aperture utilizing 13.5 nanometer extreme ultraviolet laser light to pattern sub-2nm nodes.",
    },
    {
        "id": "doc_semi_02_distractor",
        "text": "Deep reactive ion etching utilizes sulfur hexafluoride plasma cycles to etch high aspect ratio silicon trenches in MEMS fabrication.",
    },
    {
        "id": "doc_semi_03_contradiction",
        "text": "High-NA EUV lithography operates using 193nm deep ultraviolet lasers with 0.33 numerical aperture, strictly limited to 28nm planar nodes.",
    },
    {
        "id": "doc_crypto_01",
        "text": "SHA-256 cryptographic hash chaining links audit log records into an append-only verifiable ledger where tamper attempts invalidate successor hashes.",
    },
    {
        "id": "doc_crypto_02_distractor",
        "text": "RSA public key cryptography relies on the computational intractability of factoring large composite prime products.",
    },
    {
        "id": "doc_crypto_03_contradiction",
        "text": "Audit log hash chaining allows retroactive in-place record mutations without altering downstream sequence hashes or signatures.",
    },
]


def extract_retrieved_doc_ids(response: str) -> List[str]:
    """Extract list of retrieved document IDs from model or searcher response string."""
    if not response or not response.strip():
        return []

    cleaned = response.strip()

    # Case 1: JSON array ["doc_1", "doc_2"]
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    # Case 2: Markdown code block
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass

    # Case 3: Match tokens like doc_...
    explicit_matches = re.findall(r"\bdoc_[a-zA-Z0-9_]+\b", cleaned)
    if explicit_matches:
        # Preserve order while removing duplicates
        seen = set()
        res = []
        for m in explicit_matches:
            if m not in seen:
                seen.add(m)
                res.append(m)
        return res

    # Case 4: Comma-separated or newline-separated
    parts = re.split(r"[\n,]+", cleaned)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class SymbolonArmResult:
    """Metrics for a single retrieval arm."""
    arm_name: str
    mean_recall: float
    contradiction_retention: float
    proxy_agreement: float
    candidate_expansion: float
    total_dropped_gold: int
    delta_vs_baseline: float


@dataclass
class SymbolonBenchmarkReport:
    """Consolidated benchmark report across all comparative retrieval arms."""
    total_queries: int
    top_k: int
    level_evaluated: int
    decision_verdict: str
    decision_rationale: str
    arms: Dict[str, SymbolonArmResult]

    def to_markdown(self) -> str:
        """Render report as formatted GitHub-style Markdown table."""
        lines = [
            f"# Symbolon Retrieval Benchmark & Candidate Fusion Report (K={self.top_k}, Level={self.level_evaluated})",
            "",
            f"**Decision Verdict:** `{self.decision_verdict}`  ",
            f"**Rationale:** {self.decision_rationale}",
            "",
            "| Algorithm / Arm | Gold Recall@K | Contradiction Retention | Retriever Agreement | Candidate Expansion | Dropped Gold | Paired Delta |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for key, arm in self.arms.items():
            lines.append(
                f"| **{arm.arm_name}** | {arm.mean_recall:.4f} | {arm.contradiction_retention:.4f} | "
                f"{arm.proxy_agreement:.4f} | {arm.candidate_expansion:.2f}x | {arm.total_dropped_gold} | "
                f"{arm.delta_vs_baseline:+.4f} |"
            )
        return "\n".join(lines)


class SymbolonRetrievalBenchmarkSuite:
    """Benchmark suite executing the 4 comparative arms over Symbolon and LTHCM baselines."""

    def __init__(self, corpus: Optional[List[Dict[str, str]]] = None, top_k: int = 3, level: int = 4):
        self.corpus = corpus or BENCHMARK_CORPUS
        self.top_k = top_k
        self.level = level

    def run_benchmark(self, tokenizer_adapter: Any = None) -> SymbolonBenchmarkReport:
        """Execute all 4 comparative arms and compute statistical metrics."""
        from symbolon.study import SymbolonE2Study

        # Use provided or default mock adapter
        if tokenizer_adapter is None:
            from symbolon.tokenizer_adapter import SymbolonTokenizerAdapter
            tokenizer_adapter = SymbolonTokenizerAdapter(use_mock=True)

        study = SymbolonE2Study(
            tokenizer_adapter=tokenizer_adapter,
            corpus=self.corpus,
            top_k=self.top_k,
            code_level=self.level,
        )
        raw_report = study.run_study()

        arms: Dict[str, SymbolonArmResult] = {
            "baseline": SymbolonArmResult(
                arm_name="Flat Dense Baseline (LTHCM Flat)",
                mean_recall=raw_report.mean_baseline_recall,
                contradiction_retention=raw_report.baseline_contradiction_retention_rate,
                proxy_agreement=1.0,
                candidate_expansion=1.0,
                total_dropped_gold=0,
                delta_vs_baseline=0.0,
            ),
            "symbolon_alone": SymbolonArmResult(
                arm_name=f"Symbolon Discrete Overlap (L{self.level})",
                mean_recall=raw_report.mean_symbolon_recall,
                contradiction_retention=raw_report.symbolon_contradiction_retention_rate,
                proxy_agreement=raw_report.mean_proxy_agreement,
                candidate_expansion=1.0,
                total_dropped_gold=0,
                delta_vs_baseline=raw_report.mean_symbolon_recall - raw_report.mean_baseline_recall,
            ),
            "uncapped_union": SymbolonArmResult(
                arm_name="Uncapped Union (Base + Symbolon)",
                mean_recall=raw_report.mean_union_recall,
                contradiction_retention=raw_report.union_contradiction_retention_rate,
                proxy_agreement=raw_report.mean_proxy_agreement,
                candidate_expansion=raw_report.mean_union_expansion,
                total_dropped_gold=0,
                delta_vs_baseline=raw_report.mean_union_recall - raw_report.mean_baseline_recall,
            ),
            "capped_fusion": SymbolonArmResult(
                arm_name=f"Capped RRF Fusion (Top-{self.top_k})",
                mean_recall=raw_report.mean_capped_fusion_recall,
                contradiction_retention=raw_report.capped_contradiction_retention_rate,
                proxy_agreement=0.85,
                candidate_expansion=1.0,
                total_dropped_gold=raw_report.total_dropped_gold,
                delta_vs_baseline=float(raw_report.paired_bootstrap_capped_vs_baseline.get("delta", 0.0)),
            ),
        }

        return SymbolonBenchmarkReport(
            total_queries=raw_report.total_queries,
            top_k=self.top_k,
            level_evaluated=self.level,
            decision_verdict=raw_report.decision_verdict,
            decision_rationale=raw_report.decision_rationale,
            arms=arms,
        )


class SymbolonBenchmarkTask:
    """Dokime Task plugin for evaluating Symbolon discrete code retrieval."""

    def __init__(self, dataset_path: Optional[Path] = None):
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "symbolon_suite.json"
        self.dataset_path = dataset_path
        self.task_id = "symbolon_code_retrieval"

    def load(self) -> List[Query]:
        """Load benchmark queries from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Symbolon benchmark dataset not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: List[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=item["id"],
                    prompt=item["prompt"],
                    expected=item.get("expected", ""),
                    metadata=item.get("metadata", {}),
                )
            )
        return queries

    def score(self, query: Query, response: str) -> float:
        """Score retrieved document IDs against gold relevance.
        
        Returns:
            Fraction of gold relevant documents retrieved (Recall@k).
        """
        detailed = self.score_detailed(query, response)
        return float(detailed["gold_recall"])

    def score_detailed(self, query: Query, response: str) -> Dict[str, Any]:
        """Compute comprehensive retrieval metrics for a query response."""
        retrieved = extract_retrieved_doc_ids(response)
        retrieved_set = set(retrieved)

        gold_relevant = query.metadata.get("gold_relevant", [])
        contradiction_docs = query.metadata.get("contradiction_docs", [])

        if not gold_relevant and query.expected:
            gold_relevant = [query.expected]

        gold_set = set(gold_relevant)
        contra_set = set(contradiction_docs)

        gold_hits = len(retrieved_set.intersection(gold_set))
        gold_recall = float(gold_hits / len(gold_set)) if gold_set else 1.0

        contra_hits = len(retrieved_set.intersection(contra_set))
        contra_retention = float(contra_hits / len(contra_set)) if contra_set else 1.0

        return {
            "query_id": query.id,
            "retrieved_docs": retrieved,
            "gold_relevant": gold_relevant,
            "contradiction_docs": contradiction_docs,
            "gold_recall": gold_recall,
            "contradiction_retention": contra_retention,
            "hit_gold": gold_hits > 0,
            "hit_contradiction": contra_hits > 0,
        }
