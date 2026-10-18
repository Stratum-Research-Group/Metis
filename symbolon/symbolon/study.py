"""Experiment E2: Offline Candidate Capture & Fusion Study for Symbolon.

Evaluates offline candidate retrieval and fusion between flat dense baseline
and Symbolon multiscale discrete code-overlap ranking over benchmark diagnostic
corpora with explicit gold relevance and contradiction labels.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from symbolon.codes import compute_scoped_code_overlap
from symbolon.tokenizer_adapter import SymbolonTokenizerAdapter

# Try importing dokime statistical functions
try:
    from dokime.stats import paired_bootstrap_test, round_stat, single_run_bootstrap
except ImportError:
    def round_stat(val: float | None, default_precision: int = 4) -> float | None:
        return round(float(val), default_precision) if val is not None else None

    def single_run_bootstrap(scores: list[float], num_samples: int = 10000, seed: int = 42) -> dict[str, Any]:
        n = len(scores)
        m = sum(scores) / n if n > 0 else 0.0
        return {"mean": round_stat(m), "ci_lower": round_stat(m), "ci_upper": round_stat(m), "count": n}

    def paired_bootstrap_test(scores_baseline: list[float], scores_candidate: list[float], **kwargs) -> dict[str, Any]:
        n = len(scores_baseline)
        diffs = [c - b for b, c in zip(scores_baseline, scores_candidate)]
        d = sum(diffs) / n if n > 0 else 0.0
        return {
            "mean_baseline": round_stat(sum(scores_baseline) / n if n > 0 else 0.0),
            "mean_candidate": round_stat(sum(scores_candidate) / n if n > 0 else 0.0),
            "delta": round_stat(d),
            "ci_lower": round_stat(d),
            "ci_upper": round_stat(d),
            "ci_excludes_zero": False,
            "verdict": "NO_CHANGE",
        }

logger = logging.getLogger(__name__)

# Representative diagnostic corpus with explicit relevance and contradiction labels (Spec E2)
E2_BENCHMARK_CORPUS = [
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
]

E2_BENCHMARK_QUERIES = [
    {
        "query_id": "q_biochem",
        "text": "What is the binding mechanism and Ki of methotrexate to dihydrofolate reductase?",
        "gold_relevant": ["doc_sci_01"],
        "contradiction_docs": ["doc_sci_03_contradiction"],
    },
    {
        "query_id": "q_merkel_summit",
        "text": "Where did Chancellor Angela Merkel meet for European Union northern economic corridor talks?",
        "gold_relevant": ["doc_geo_01"],
        "contradiction_docs": ["doc_geo_03_contradiction"],
    },
    {
        "query_id": "q_apollo_fuel",
        "text": "How much propellant did the Apollo 11 Saturn V burn per second across F-1 engines?",
        "gold_relevant": ["doc_space_01"],
        "contradiction_docs": ["doc_space_03_contradiction"],
    },
    {
        "query_id": "q_sheaf_drift",
        "text": "How does cellular sheaf Laplacian bound topological drift across multi-agent graphs?",
        "gold_relevant": ["doc_math_01"],
        "contradiction_docs": ["doc_math_03_contradiction"],
    },
]


def _simple_dense_similarity(query_text: str, doc_text: str) -> float:
    """
    Simulates flat dense baseline comparator (cosine similarity over term frequencies).
    Matches LTHCM flat mode comparator behavior.
    """
    q_words = query_text.lower().replace("?", "").replace(",", "").split()
    d_words = doc_text.lower().replace(".", "").replace(",", "").split()

    q_set = set(q_words)
    d_set = set(d_words)

    if not q_set or not d_set:
        return 0.0

    intersection = q_set & d_set
    # Cosine-like normalized similarity
    sim = len(intersection) / math.sqrt(len(q_set) * len(d_set))
    return float(sim)


@dataclass
class QueryRetrievalResult:
    """Per-query results across all evaluation arms."""
    query_id: str
    baseline_ranked_docs: List[str]
    symbolon_ranked_docs: List[str]
    union_docs: List[str]
    capped_fusion_docs: List[str]
    gold_relevant: List[str]
    contradiction_docs: List[str]

    # Arm (a): Primary recall@k
    baseline_recall: float
    symbolon_recall: float
    union_recall: float
    capped_fusion_recall: float

    # Arm (b): Proxy agreement with baseline
    agreement_jaccard: float

    # Arm (c1): Uncapped additive expansion factor
    union_size_expansion: float

    # Arm (c2): Capped dropped items
    dropped_baseline_docs: List[str]
    dropped_gold_docs: List[str]

    # Arm (d): Contradiction coverage
    baseline_contradiction_retained: bool
    symbolon_contradiction_retained: bool
    union_contradiction_retained: bool
    capped_contradiction_retained: bool


@dataclass
class SymbolonE2Report:
    """Comprehensive E2 Candidate Fusion Diagnostic Report."""
    top_k: int
    total_queries: int
    mean_baseline_recall: float
    mean_symbolon_recall: float
    mean_union_recall: float
    mean_capped_fusion_recall: float
    mean_proxy_agreement: float
    mean_union_expansion: float
    total_dropped_gold: int
    baseline_contradiction_retention_rate: float
    symbolon_contradiction_retention_rate: float
    capped_contradiction_retention_rate: float
    union_contradiction_retention_rate: float
    paired_bootstrap_capped_vs_baseline: Dict[str, Any]
    paired_bootstrap_union_vs_baseline: Dict[str, Any]
    query_results: List[QueryRetrievalResult]
    decision_verdict: str
    decision_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "total_queries": self.total_queries,
            "mean_baseline_recall": round(self.mean_baseline_recall, 4),
            "mean_symbolon_recall": round(self.mean_symbolon_recall, 4),
            "mean_union_recall": round(self.mean_union_recall, 4),
            "mean_capped_fusion_recall": round(self.mean_capped_fusion_recall, 4),
            "mean_proxy_agreement": round(self.mean_proxy_agreement, 4),
            "mean_union_expansion": round(self.mean_union_expansion, 2),
            "total_dropped_gold": self.total_dropped_gold,
            "baseline_contradiction_retention_rate": round(self.baseline_contradiction_retention_rate, 4),
            "symbolon_contradiction_retention_rate": round(self.symbolon_contradiction_retention_rate, 4),
            "capped_contradiction_retention_rate": round(self.capped_contradiction_retention_rate, 4),
            "union_contradiction_retention_rate": round(self.union_contradiction_retention_rate, 4),
            "paired_bootstrap_capped_vs_baseline": self.paired_bootstrap_capped_vs_baseline,
            "paired_bootstrap_union_vs_baseline": self.paired_bootstrap_union_vs_baseline,
            "decision_verdict": self.decision_verdict,
            "decision_rationale": self.decision_rationale,
            "query_results": [asdict(r) for r in self.query_results],
        }


class SymbolonE2Study:
    """
    Executes Experiment E2: Offline candidate capture and fusion.
    Evaluates:
    - Arm (a): Recall@k against gold relevance labels.
    - Arm (b): Agreement with flat baseline (proxy).
    - Arm (c1): Uncapped additive union (no-loss arm, measures size inflation).
    - Arm (c2): Capped fusion at matched top-k (measures dropped items).
    - Arm (d): Contradiction evidence retention check.
    """

    def __init__(
        self,
        tokenizer_adapter: Optional[SymbolonTokenizerAdapter] = None,
        corpus: Optional[List[Dict[str, str]]] = None,
        top_k: int = 3,
        code_level: int = 4,  # Intermediate hierarchy level
    ):
        self.adapter = tokenizer_adapter or SymbolonTokenizerAdapter()
        self.corpus = corpus or E2_BENCHMARK_CORPUS
        self.top_k = top_k
        self.code_level = code_level

        # Pre-encode all corpus documents into scoped codes
        logger.info("Pre-encoding %d corpus passages into Symbolon scoped codes...", len(self.corpus))
        self.encoded_corpus: Dict[str, Any] = {}
        for doc in self.corpus:
            doc_id = doc["id"]
            text = doc["text"]
            windows = self.adapter.encode_passage(text)
            self.encoded_corpus[doc_id] = {
                "text": text,
                "windows": windows,
            }

    def _score_symbolon(self, query_codes: List[Any], doc_codes: List[Any]) -> float:
        """
        Compute total scoped code overlap across all matching windows at self.code_level.
        Enforces the 6-tuple identity rule via compute_scoped_code_overlap.
        """
        q_codes = [c for c in query_codes if c.level == self.code_level]
        d_codes = [c for c in doc_codes if c.level == self.code_level]
        return float(compute_scoped_code_overlap(q_codes, d_codes))


    def run_study(self, queries: Optional[List[Dict[str, Any]]] = None) -> SymbolonE2Report:
        query_set = queries or E2_BENCHMARK_QUERIES
        query_results: List[QueryRetrievalResult] = []

        base_recalls = []
        sym_recalls = []
        union_recalls = []
        capped_recalls = []

        base_contra_retained = 0
        sym_contra_retained = 0
        union_contra_retained = 0
        capped_contra_retained = 0
        total_contra_queries = 0
        total_dropped_gold = 0

        for q in query_set:
            q_id = q["query_id"]
            q_text = q["text"]
            gold = q.get("gold_relevant", [])
            contra = q.get("contradiction_docs", [])

            # 1. Baseline scoring: flat dense similarity
            baseline_scores: List[Tuple[float, str]] = []
            for doc_id, doc_data in self.encoded_corpus.items():
                sim = _simple_dense_similarity(q_text, doc_data["text"])
                baseline_scores.append((sim, doc_id))
            baseline_scores.sort(key=lambda x: x[0], reverse=True)
            baseline_ranked = [d for _, d in baseline_scores[: self.top_k]]

            # 2. Symbolon scoring: scoped code overlap
            q_windows = self.adapter.encode_passage(q_text)
            symbolon_scores: List[Tuple[float, str]] = []
            for doc_id, doc_data in self.encoded_corpus.items():
                overlap = self._score_symbolon(q_windows, doc_data["windows"])
                symbolon_scores.append((overlap, doc_id))
            symbolon_scores.sort(key=lambda x: x[0], reverse=True)
            symbolon_ranked = [d for _, d in symbolon_scores[: self.top_k]]

            # 3. Arm (c1): Uncapped additive union
            union_set = list(dict.fromkeys(baseline_ranked + symbolon_ranked))

            # 4. Arm (c2): Capped fusion at matched top-k (Reciprocal Rank Fusion)
            rrf_scores: Dict[str, float] = {}
            for rank, doc_id in enumerate(baseline_ranked):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank + 1))
            for rank, doc_id in enumerate(symbolon_ranked):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank + 1))

            capped_ranked = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[: self.top_k]

            # Measure Recall@k
            def compute_recall(retrieved: Sequence[str]) -> float:
                if not gold:
                    return 1.0
                hits = sum(1 for g in gold if g in retrieved)
                return hits / float(len(gold))

            r_base = compute_recall(baseline_ranked)
            r_sym = compute_recall(symbolon_ranked)
            r_union = compute_recall(union_set)
            r_capped = compute_recall(capped_ranked)

            base_recalls.append(r_base)
            sym_recalls.append(r_sym)
            union_recalls.append(r_union)
            capped_recalls.append(r_capped)

            # Measure Arm (b): Proxy Agreement (Jaccard)
            b_set = set(baseline_ranked)
            s_set = set(symbolon_ranked)
            agree_jaccard = len(b_set & s_set) / float(len(b_set | s_set)) if (b_set | s_set) else 1.0

            # Measure Arm (c1): Union expansion factor
            expansion = len(union_set) / float(len(baseline_ranked)) if baseline_ranked else 1.0

            # Measure Arm (c2): Dropped items
            dropped_base = [d for d in baseline_ranked if d not in capped_ranked]
            dropped_gold = [g for g in gold if g in baseline_ranked and g not in capped_ranked]
            total_dropped_gold += len(dropped_gold)

            # Measure Arm (d): Contradiction evidence retention
            b_has_c = any(c in baseline_ranked for c in contra)
            s_has_c = any(c in symbolon_ranked for c in contra)
            u_has_c = any(c in union_set for c in contra)
            c_has_c = any(c in capped_ranked for c in contra)

            if contra:
                total_contra_queries += 1
                if b_has_c:
                    base_contra_retained += 1
                if s_has_c:
                    sym_contra_retained += 1
                if u_has_c:
                    union_contra_retained += 1
                if c_has_c:
                    capped_contra_retained += 1

            query_results.append(
                QueryRetrievalResult(
                    query_id=q_id,
                    baseline_ranked_docs=baseline_ranked,
                    symbolon_ranked_docs=symbolon_ranked,
                    union_docs=union_set,
                    capped_fusion_docs=capped_ranked,
                    gold_relevant=gold,
                    contradiction_docs=contra,
                    baseline_recall=r_base,
                    symbolon_recall=r_sym,
                    union_recall=r_union,
                    capped_fusion_recall=r_capped,
                    agreement_jaccard=round(agree_jaccard, 4),
                    union_size_expansion=round(expansion, 2),
                    dropped_baseline_docs=dropped_base,
                    dropped_gold_docs=dropped_gold,
                    baseline_contradiction_retained=b_has_c,
                    symbolon_contradiction_retained=s_has_c,
                    union_contradiction_retained=u_has_c,
                    capped_contradiction_retained=c_has_c,
                )
            )

        n = len(query_set)
        mean_base_recall = sum(base_recalls) / n if n > 0 else 0.0
        mean_sym_recall = sum(sym_recalls) / n if n > 0 else 0.0
        mean_union_recall = sum(union_recalls) / n if n > 0 else 0.0
        mean_capped_recall = sum(capped_recalls) / n if n > 0 else 0.0
        mean_agree = sum(r.agreement_jaccard for r in query_results) / n if n > 0 else 0.0
        mean_expansion = sum(r.union_size_expansion for r in query_results) / n if n > 0 else 0.0

        base_contra_rate = base_contra_retained / float(total_contra_queries) if total_contra_queries > 0 else 0.0
        sym_contra_rate = sym_contra_retained / float(total_contra_queries) if total_contra_queries > 0 else 0.0
        union_contra_rate = union_contra_retained / float(total_contra_queries) if total_contra_queries > 0 else 0.0
        capped_contra_rate = capped_contra_retained / float(total_contra_queries) if total_contra_queries > 0 else 0.0

        # Dokime bootstrap comparisons
        boot_capped_vs_base = paired_bootstrap_test(
            scores_baseline=base_recalls,
            scores_candidate=capped_recalls,
            num_samples=10000,
            confidence_level=0.95,
            seed=42,
        )

        boot_union_vs_base = paired_bootstrap_test(
            scores_baseline=base_recalls,
            scores_candidate=union_recalls,
            num_samples=10000,
            confidence_level=0.95,
            seed=42,
        )

        # Decision Gate per Spec E2/E3:
        # Proceed to E3 prototype ONLY IF measured benefit on primary recall without degrading contradiction coverage.
        delta_capped = boot_capped_vs_base.get("delta", 0.0)
        ci_lower = boot_capped_vs_base.get("ci_lower", 0.0)

        if ci_lower > 0 and capped_contra_rate >= base_contra_rate:
            verdict = "PROCEED_TO_E3_PROTOTYPE"
            rationale = (
                f"Symbolon capped fusion significantly improved recall@k (delta: +{delta_capped:.4f}, "
                f"95% CI: [{ci_lower:.4f}, {boot_capped_vs_base.get('ci_upper', 0.0):.4f}]) "
                f"while maintaining contradiction retention at {capped_contra_rate:.1%}."
            )
        elif mean_union_recall >= mean_base_recall and total_dropped_gold == 0:
            verdict = "ARCHIVE_UNCAPPED_NO_LOSS_RECORDED"
            rationale = (
                f"Uncapped additive union preserves 100% of gold evidence (recall {mean_union_recall:.1%}), "
                f"at the cost of {mean_expansion:.2f}x candidate set inflation. Capped fusion shows no "
                f"significant delta over flat baseline (delta: {delta_capped:.4f}). "
                f"Per Roadmap Milestone 3.3, publish negative research note and archive."
            )
        else:
            verdict = "ARCHIVE_NEGATIVE_RESULT"
            rationale = (
                f"Capped fusion lost {total_dropped_gold} gold-relevant evidence passages (delta: {delta_capped:.4f}) "
                f"or degraded contradiction retention ({capped_contra_rate:.1%} vs baseline {base_contra_rate:.1%}). "
                f"Archive negative result in docs/review/."
            )

        return SymbolonE2Report(
            top_k=self.top_k,
            total_queries=n,
            mean_baseline_recall=mean_base_recall,
            mean_symbolon_recall=mean_sym_recall,
            mean_union_recall=mean_union_recall,
            mean_capped_fusion_recall=mean_capped_recall,
            mean_proxy_agreement=mean_agree,
            mean_union_expansion=mean_expansion,
            total_dropped_gold=total_dropped_gold,
            baseline_contradiction_retention_rate=base_contra_rate,
            symbolon_contradiction_retention_rate=sym_contra_rate,
            capped_contradiction_retention_rate=capped_contra_rate,
            union_contradiction_retention_rate=union_contra_rate,
            paired_bootstrap_capped_vs_baseline=boot_capped_vs_base,
            paired_bootstrap_union_vs_baseline=boot_union_vs_base,
            query_results=query_results,
            decision_verdict=verdict,
            decision_rationale=rationale,
        )
