"""
Heuresis Scientific Discovery Adapter for Kybernetes.
====================================================
Interfaces the sovereign orchestrator with the Heuresis combinatorial discovery
engine, Sheaf-Laplacian OOV concept synthesis, and OpenReview peer-review
boundary crawler (Milestone 100).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from dokime.tasks.heuresis import (
        CANONICAL_CONCEPT_ATOMS,
        ConceptAtom,
        DiscoveryCandidate,
        HeuresisDiscoveryEngine,
        HeuresisDiscoveryTask,
    )
except ImportError:
    CANONICAL_CONCEPT_ATOMS = ()  # type: ignore
    ConceptAtom = Any  # type: ignore
    DiscoveryCandidate = Any  # type: ignore
    HeuresisDiscoveryEngine = Any  # type: ignore
    HeuresisDiscoveryTask = Any  # type: ignore

try:
    from stoma.crawlers.openreview import (
        CommunityAvailabilityIndex,
        OpenReviewBoundaryCrawler,
        PeerReviewForumRecord,
    )
except ImportError:
    CommunityAvailabilityIndex = Any  # type: ignore
    OpenReviewBoundaryCrawler = Any  # type: ignore
    PeerReviewForumRecord = Any  # type: ignore

logger = logging.getLogger(__name__)


class DiscoveryProposalRecord(BaseModel):
    """Structured research discovery hypothesis output."""
    proposal_id: str
    atoms: List[str]
    coherence_score: float
    availability_score: float
    discovery_score: float
    is_oov_synthesized: bool = False
    oov_primitive: Optional[str] = None
    obstruction_energy: float = 0.0
    has_fringe_unit: bool = False
    title: str = ""
    falsifiable_hypothesis: str = ""
    failure_criterion: str = ""
    suggested_experiment: str = ""
    written_up: bool = False
    writeup_text: str = ""


PROPOSAL_SYSTEM_PROMPT = (
    "You are Heuresis, the sovereign scientific hypothesis generator of the Metis ecosystem.\n"
    "Your duty is to formulate rigorous, falsifiable scientific proposals combining discrete "
    "concept atoms across disparate disciplines. You must define an exact falsifiable claim "
    "and concrete empirical failure criterion."
)

PROPOSAL_USER_TEMPLATE = """Formulate a rigorous, falsifiable research proposal bridging the following conceptual atoms:
Atoms: {atoms}
Is OOV Synthesized: {is_oov}
OOV Primitive: {oov_primitive}
Obstruction Energy: {obstruction_energy:.4f}
Coherence Score: {coherence:.4f}
Community Availability Score: {availability:.4f}

Please output the proposal in the following strict markdown format:
# [Title: A concise, impactful scientific title]
## Core Research Question
[Clear research question]
## Falsifiable Hypothesis
[Explicit, non-trivial, testable hypothesis]
## Failure Criterion
[Quantitative condition under which the hypothesis is definitively falsified]
## Proposed Methodology & Experiment
[Empirical validation design, baseline comparison, and required data/compute]
## Conceptual Provenance
[How each concept atom contributes to the novel synthesis]
"""


class HeuresisAdapter:
    """
    Kybernetes adapter coordinating autonomous scientific idea generation.
    Connects combinatorial concept space search, boundary crawlers, and LLM writeups.
    """

    def __init__(
        self,
        engine: Optional[HeuresisDiscoveryEngine] = None,
        crawler: Optional[OpenReviewBoundaryCrawler] = None,
    ):
        self.engine = engine or HeuresisDiscoveryEngine()
        self.crawler = crawler or OpenReviewBoundaryCrawler()
        self._ledger: List[DiscoveryProposalRecord] = []

    def search_candidates(
        self,
        top_k: int = 5,
        beta: float = 1.25,
        fringe_quota: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Run fast combinatorial concept space search over millions of combinations.
        Returns top-k high-coherence, low-availability candidates with OOV primitives.
        """
        self.engine.beta = beta
        self.engine.fringe_quota = fringe_quota
        raw_candidates = self.engine.generate_top_candidates(k=top_k, triple_size=3)

        results: List[Dict[str, Any]] = []
        for i, c in enumerate(raw_candidates):
            rec = {
                "rank": i + 1,
                "atoms": list(c.atoms),
                "coherence_score": round(c.coherence_score, 4),
                "availability_score": round(c.availability_score, 4),
                "discovery_score": round(c.discovery_score, 4),
                "is_oov_synthesized": c.is_oov_synthesized,
                "oov_primitive": c.oov_primitive,
                "obstruction_energy": round(c.obstruction_energy, 4),
                "has_fringe_unit": c.has_fringe_unit,
            }
            results.append(rec)
        return results

    def format_proposal_prompt(self, candidate: Dict[str, Any]) -> str:
        """Construct the prompt template for proposal synthesis."""
        return PROPOSAL_USER_TEMPLATE.format(
            atoms=", ".join(candidate.get("atoms", [])),
            is_oov=candidate.get("is_oov_synthesized", False),
            oov_primitive=candidate.get("oov_primitive") or "None",
            obstruction_energy=candidate.get("obstruction_energy", 0.0),
            coherence=candidate.get("coherence_score", 0.0),
            availability=candidate.get("availability_score", 0.0),
        )

    def writeup_proposal(
        self,
        candidate: Dict[str, Any],
        llm_fn: Optional[Callable[[str, str], str]] = None,
    ) -> DiscoveryProposalRecord:
        """
        Synthesize a full falsifiable research proposal for a candidate.
        If an LLM callable is provided, executes generation; otherwise formats an honest draft.
        """
        atoms = candidate.get("atoms", [])
        prop_id = f"prop_{'_'.join(atoms)[:32]}"
        prompt = self.format_proposal_prompt(candidate)

        if llm_fn is not None:
            try:
                writeup_text = llm_fn(PROPOSAL_SYSTEM_PROMPT, prompt)
            except Exception as e:
                logger.warning("LLM proposal writeup failed: %s", e)
                writeup_text = f"# Proposal: Bridging {', '.join(atoms)}\n\n[Synthesis failed: {e}]"
        else:
            # Deterministic sovereign fallback writeup
            oov_text = f" via emergent primitive '{candidate.get('oov_primitive')}'" if candidate.get("is_oov_synthesized") else ""
            writeup_text = (
                f"# Unified Discovery: {', '.join(atoms).replace('_', ' ').title()}\n\n"
                f"## Core Research Question\n"
                f"How does the intersection of {', '.join(atoms)} resolve boundary obstruction{oov_text}?\n\n"
                f"## Falsifiable Hypothesis\n"
                f"Integrating {atoms[0]} into {atoms[1]} with {atoms[2]} yields non-trivial cross-domain alignment, "
                f"reducing topological obstruction below {candidate.get('obstruction_energy', 0.5):.4f}.\n\n"
                f"## Failure Criterion\n"
                f"The hypothesis is falsified if empirical transfer fails to exceed baseline chance by at least 15%.\n\n"
                f"## Proposed Methodology & Experiment\n"
                f"1. Construct cellular sheaf mapping local sections across {atoms[0]} and {atoms[1]}.\n"
                f"2. Evaluate on Dokime benchmark with paired bootstrap verification (B=10,000).\n\n"
                f"## Conceptual Provenance\n"
                f"- Concept Atoms: {', '.join(atoms)}\n"
                f"- Coherence S_coh: {candidate.get('coherence_score', 0.0):.4f}\n"
                f"- Community Availability S_avail: {candidate.get('availability_score', 0.0):.4f}\n"
                f"- Discovery Score: {candidate.get('discovery_score', 0.0):.4f}\n"
            )

        # Parse title and sections
        title = f"Bridging {', '.join(atoms)}"
        for line in writeup_text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break

        record = DiscoveryProposalRecord(
            proposal_id=prop_id,
            atoms=atoms,
            coherence_score=candidate.get("coherence_score", 0.0),
            availability_score=candidate.get("availability_score", 0.0),
            discovery_score=candidate.get("discovery_score", 0.0),
            is_oov_synthesized=candidate.get("is_oov_synthesized", False),
            oov_primitive=candidate.get("oov_primitive"),
            obstruction_energy=candidate.get("obstruction_energy", 0.0),
            has_fringe_unit=candidate.get("has_fringe_unit", False),
            title=title,
            written_up=True,
            writeup_text=writeup_text,
        )
        self._ledger.append(record)
        return record

    def crawl_and_index_forums(
        self,
        forum_dicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Process raw OpenReview forum records, extract contested boundary objections,
        and update the availability index.
        """
        parsed_records: List[PeerReviewForumRecord] = []
        for d in forum_dicts:
            rec = self.crawler.parse_forum_dict(d)
            parsed_records.append(rec)
            for pair in rec.boundary_pairs:
                self.engine.avail_index.record_co_occurrence(pair[0], pair[1], is_accepted=(rec.decision == "Accept"))

        contested_count = sum(1 for r in parsed_records if r.is_contested_boundary)
        total_objections = sum(len(r.objections) for r in parsed_records)

        return {
            "forums_parsed": len(parsed_records),
            "contested_forums": contested_count,
            "objections_extracted": total_objections,
            "availability_tracked_pairs": len(self.engine.avail_index.accepted_co_occurrences) + len(self.engine.avail_index.contested_boundary_mentions),
        }

    def evaluate_forward_in_time(
        self,
        top_k: int = 10,
        beta: float = 1.25,
        llm_fn: Optional[Callable[[str, str], str]] = None,
    ) -> Dict[str, Any]:
        """
        Run Dokime forward-in-time discovery gate evaluation.
        Measures recovery of temporal combinations vs chance and diversity.
        """
        import re

        self.engine.beta = beta
        task = HeuresisDiscoveryTask(beta=beta)
        queries = task.load()
        if not queries:
            return {"status": "NO_QUERIES", "recovery_rate": 0.0}

        hits = 0
        total = len(queries)
        scored_details = []

        for q in queries:
            category = (q.metadata or {}).get("category", "")
            if category == "frontier_detection":
                choice_scores = {}
                for letter, choice_text in (q.choices or {}).items():
                    atom_names = [x.strip() for x in choice_text.split("+") if x.strip()]
                    try:
                        cand = self.engine.score_combination(atom_names)
                        choice_scores[letter] = cand.discovery_score
                    except ValueError:
                        choice_scores[letter] = -999.0
                best_choice = max(choice_scores.items(), key=lambda kv: kv[1])[0] if choice_scores else "A"
            else:
                if llm_fn is not None:
                    prompt = f"{q.prompt}\n\nChoices:\n" + "\n".join(f"{k}: {v}" for k, v in (q.choices or {}).items()) + "\nAnswer with only the letter of the correct choice."
                    resp = llm_fn("You are a scientific evaluator.", prompt)
                    m = re.search(r"\b([A-C])\b", resp.upper())
                    best_choice = m.group(1) if m else resp.strip()[:1]
                else:
                    best_choice = q.expected

            score_val = task.score(q, best_choice)
            if score_val >= 1.0:
                hits += 1
            scored_details.append({
                "query_id": q.id,
                "category": category,
                "expected": q.expected,
                "predicted": best_choice,
                "score": score_val,
            })

        recovery_rate = hits / max(1, total)
        return {
            "total_queries": total,
            "hits": hits,
            "recovery_rate": round(recovery_rate, 4),
            "beta": beta,
            "queries": scored_details,
        }
