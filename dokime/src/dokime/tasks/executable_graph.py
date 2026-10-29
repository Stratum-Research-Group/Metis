"""
Dokime Executable Multi-Hop Graph Reasoning Task & Deterministic Executor.
========================================================================
Reference: Evolve on Graph (arXiv:2610.35598), adapted for Metis Dokime.

Provides inspectable multi-hop reasoning test cases whose expected answers
come from deterministic execution over a frozen, hashed synthetic fact fixture
rather than an LLM-written answer key.

Exposes:
  - Missed constraints
  - Incomplete answer sets
  - Unsupported answers (guessing without supporting evidence)
  - Broken reasoning / relation chains
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dokime.tasks.protocol import Query, Task

logger = logging.getLogger(__name__)

# Frozen Schema Version
EXECUTABLE_GRAPH_SCHEMA_VERSION = "1.0.0"
EXECUTOR_VERSION = "1.0.0"


@dataclass(frozen=True)
class Fact:
    """Atomic directed knowledge graph fact tuple."""
    fact_id: str
    subject: str
    relation: str
    object: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
        }


# Canonical synthetic fact fixture spanning Computer Science, Biochemistry,
# Aerospace, and Distributed Systems.
SYNTHETIC_KNOWLEDGE_FIXTURE: List[Fact] = [
    # Computer Science & Languages
    Fact("f_cs_01", "python", "invented_by", "guido_van_rossum"),
    Fact("f_cs_02", "c_lang", "invented_by", "dennis_ritchie"),
    Fact("f_cs_03", "unix", "invented_by", "dennis_ritchie"),
    Fact("f_cs_04", "unix", "invented_by", "ken_thompson"),
    Fact("f_cs_05", "b_lang", "invented_by", "ken_thompson"),
    Fact("f_cs_06", "c_lang", "predecessor_of", "c_plus_plus"),
    Fact("f_cs_07", "c_plus_plus", "invented_by", "bjarne_stroustrup"),
    Fact("f_cs_08", "python", "paradigms", "object_oriented"),
    Fact("f_cs_09", "python", "paradigms", "functional"),
    Fact("f_cs_10", "c_plus_plus", "paradigms", "object_oriented"),
    Fact("f_cs_11", "bell_labs", "located_in", "murray_hill"),
    Fact("f_cs_12", "dennis_ritchie", "employed_at", "bell_labs"),
    Fact("f_cs_13", "ken_thompson", "employed_at", "bell_labs"),
    Fact("f_cs_14", "bjarne_stroustrup", "employed_at", "bell_labs"),
    Fact("f_cs_15", "guido_van_rossum", "employed_at", "cwi_amsterdam"),
    Fact("f_cs_16", "cwi_amsterdam", "located_in", "amsterdam"),

    # Biochemistry & Pharmacology
    Fact("f_bio_01", "methotrexate", "inhibits", "dhfr"),
    Fact("f_bio_02", "aminopterin", "inhibits", "dhfr"),
    Fact("f_bio_03", "dhfr", "catalyzes", "folate_pathway"),
    Fact("f_bio_04", "folate_pathway", "synthesizes", "thymidylate"),
    Fact("f_bio_05", "thymidylate", "required_for", "dna_synthesis"),
    Fact("f_bio_06", "penicillin", "inhibits", "pbp"),
    Fact("f_bio_07", "pbp", "catalyzes", "peptidoglycan_synthesis"),
    Fact("f_bio_08", "methotrexate", "drug_class", "antifolate"),
    Fact("f_bio_09", "aminopterin", "drug_class", "antifolate"),
    Fact("f_bio_10", "penicillin", "drug_class", "beta_lactam"),

    # Aerospace & Rocketry
    Fact("f_aero_01", "apollo_11", "launch_vehicle", "saturn_v"),
    Fact("f_aero_02", "apollo_12", "launch_vehicle", "saturn_v"),
    Fact("f_aero_03", "saturn_v", "engine_stage_1", "f1_engine"),
    Fact("f_aero_04", "saturn_v", "engine_stage_2", "j2_engine"),
    Fact("f_aero_05", "f1_engine", "propellant", "rp1_kerosene"),
    Fact("f_aero_06", "f1_engine", "oxidizer", "liquid_oxygen"),
    Fact("f_aero_07", "j2_engine", "propellant", "liquid_hydrogen"),
    Fact("f_aero_08", "j2_engine", "oxidizer", "liquid_oxygen"),
    Fact("f_aero_09", "f1_engine", "manufacturer", "rocketdyne"),
    Fact("f_aero_10", "rocketdyne", "located_in", "canoga_park"),

    # Distributed Consensus & Ledgers
    Fact("f_dist_01", "raft", "consensus_family", "leader_based"),
    Fact("f_dist_02", "paxos", "consensus_family", "leader_based"),
    Fact("f_dist_03", "pbft", "consensus_family", "byzantine"),
    Fact("f_dist_04", "raft", "fault_model", "crash_fault"),
    Fact("f_dist_05", "paxos", "fault_model", "crash_fault"),
    Fact("f_dist_06", "pbft", "fault_model", "byzantine_fault"),
    Fact("f_dist_07", "raft", "designed_by", "diego_ongaro"),
    Fact("f_dist_08", "paxos", "designed_by", "leslie_lamport"),
    Fact("f_dist_09", "diego_ongaro", "institution", "stanford_university"),
    Fact("f_dist_10", "stanford_university", "located_in", "stanford"),
]


def compute_fixture_hash(facts: Optional[List[Fact]] = None) -> str:
    """Compute deterministic SHA-256 hash of canonical sorted fixture."""
    fixture = facts or SYNTHETIC_KNOWLEDGE_FIXTURE
    serialized = json.dumps(
        sorted([f.to_dict() for f in fixture], key=lambda x: x["fact_id"]),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


FIXTURE_HASH = compute_fixture_hash()


class GraphQueryExecutor:
    """Deterministic graph query executor over synthetic fact fixture."""

    def __init__(self, facts: Optional[List[Fact]] = None):
        self.facts = facts or SYNTHETIC_KNOWLEDGE_FIXTURE
        # Index edges
        self.forward: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}  # (subj, rel) -> [(obj, fact_id)]
        self.reverse: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}  # (obj, rel) -> [(subj, fact_id)]

        for f in self.facts:
            self.forward.setdefault((f.subject, f.relation), []).append((f.object, f.fact_id))
            self.reverse.setdefault((f.object, f.relation), []).append((f.subject, f.fact_id))

    def execute_chain(self, start_entity: str, relations: Sequence[str]) -> Tuple[Set[str], List[str]]:
        """Execute a forward multi-hop chain starting from an entity.
        
        Returns:
            Tuple of (Set of terminal answer entity IDs, List of fact IDs used).
        """
        paths: List[Tuple[str, List[str]]] = [(start_entity, [])]

        for rel in relations:
            next_paths: List[Tuple[str, List[str]]] = []
            for ent, path_facts in paths:
                for obj, f_id in self.forward.get((ent, rel), []):
                    next_paths.append((obj, path_facts + [f_id]))
            paths = next_paths
            if not paths:
                break

        terminal_entities = {ent for ent, _ in paths}
        all_facts: Set[str] = set()
        for _, path_facts in paths:
            all_facts.update(path_facts)

        return terminal_entities, sorted(all_facts)

    def execute_intersection(self, conditions: Sequence[Tuple[str, str]]) -> Tuple[Set[str], List[str]]:
        """Execute an intersection query: find entities that satisfy all (entity, relation) pairs.
        
        Example: Find engines where (saturn_v, engine_stage_1) = ?e AND (?e, manufacturer) = rocketdyne.
        Returns:
            Tuple of (Set of intersecting answer entity IDs, List of fact IDs used).
        """
        candidate_entity_facts: List[Dict[str, Set[str]]] = []

        for anchor, rel in conditions:
            ent_to_facts: Dict[str, Set[str]] = {}
            # Check forward: anchor -> rel -> ?target
            fwd = self.forward.get((anchor, rel), [])
            for obj, fid in fwd:
                ent_to_facts.setdefault(obj, set()).add(fid)

            # Check reverse: ?target -> rel -> anchor
            rev = self.reverse.get((anchor, rel), [])
            for subj, fid in rev:
                ent_to_facts.setdefault(subj, set()).add(fid)

            candidate_entity_facts.append(ent_to_facts)

        if not candidate_entity_facts:
            return set(), []

        intersection = set(candidate_entity_facts[0].keys())
        for d in candidate_entity_facts[1:]:
            intersection = intersection.intersection(d.keys())

        all_facts: Set[str] = set()
        for ent in intersection:
            for d in candidate_entity_facts:
                all_facts.update(d[ent])

        return intersection, sorted(all_facts)

    def execute_reverse_chain(self, end_entity: str, relations: Sequence[str]) -> Tuple[Set[str], List[str]]:
        """Execute a backward multi-hop chain ending at an entity.
        
        Args:
            end_entity: Terminal entity id.
            relations: Forward relation sequence from target to end entity.
            
        Returns:
            Tuple of (Set of initial start entity IDs, List of fact IDs used).
        """
        paths: List[Tuple[str, List[str]]] = [(end_entity, [])]

        for rel in reversed(relations):
            prev_paths: List[Tuple[str, List[str]]] = []
            for ent, path_facts in paths:
                for subj, f_id in self.reverse.get((ent, rel), []):
                    prev_paths.append((subj, path_facts + [f_id]))
            paths = prev_paths
            if not paths:
                break

        start_entities = {ent for ent, _ in paths}
        all_facts: Set[str] = set()
        for _, path_facts in paths:
            all_facts.update(path_facts)

        return start_entities, sorted(all_facts)


@dataclass
class ExecutableCaseRecord:
    """Formal test case record conforming to Evolve on Graph Section 78-81."""
    case_id: str
    fixture_hash: str
    schema_version: str
    query_text: str
    query_executor_version: str
    query_pattern: str
    anchor_ids: List[str]
    question_text: str
    gold_answer_ids: List[str]
    evidence_fact_ids: List[str]
    wording_review_status: str  # "APPROVED"
    split: str  # "test" or "practice"
    generator_version: str
    seed: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_entities_and_citations(response: str) -> Tuple[List[str], List[str]]:
    """Parse predicted answer entity IDs and cited fact IDs from response."""
    if not response or not response.strip():
        return [], []

    cleaned = response.strip()

    # Case 1: Structured JSON payload {"answers": [...], "evidence": [...]}
    if "{" in cleaned and "}" in cleaned:
        json_match = re.search(r"(\{.*?\})", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                if isinstance(parsed, dict):
                    answers = parsed.get("answers", [])
                    evidence = parsed.get("evidence", []) or parsed.get("citations", [])
                    if isinstance(answers, list) and isinstance(evidence, list):
                        return (
                            [str(a).strip().lower() for a in answers if str(a).strip()],
                            [str(e).strip().lower() for e in evidence if str(e).strip()],
                        )
            except Exception:
                pass

    # Case 2: JSON array of answers
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return [str(a).strip().lower() for a in parsed if str(a).strip()], []
        except Exception:
            pass

    # Case 3: Token extraction (entity tokens and fact tokens like f_cs_01)
    fact_citations = re.findall(r"\bf_[a-z0-9_]+\b", cleaned.lower())
    stopwords = {"the", "a", "an", "is", "are", "and", "or", "in", "to", "of", "for", "based", "on", "with", "evidence", "facts"}
    
    # Answers: match tokens or words
    answer_tokens: List[str] = []
    # If lines formatted as "Answer: ..."
    for line in cleaned.splitlines():
        line_clean = line.strip()
        if re.match(r"^(?:answer|answers|result|entities)[:\s]+", line_clean, re.IGNORECASE):
            sub = re.sub(r"^(?:answer|answers|result|entities)[:\s]+", "", line_clean, flags=re.IGNORECASE)
            clause = re.split(r"(?:\.|\bbased\b|\bevidence\b|\bcitations?\b)", sub, flags=re.IGNORECASE)[0]
            items = re.split(r"[\s,;]+", clause)
            for it in items:
                norm = it.strip("[]\"'.,;:() \t\n").lower()
                if norm and norm not in fact_citations and norm not in stopwords:
                    answer_tokens.append(norm)

    if not answer_tokens:
        # Fallback: extract lowercase identifiers that look like entities
        tokens = re.split(r"[\n,;]+", cleaned)
        for t in tokens:
            cand = t.strip("[]\"'.,;:() \t\n").lower()
            if cand and not cand.startswith("f_") and cand not in stopwords:
                answer_tokens.append(cand)

    # De-duplicate preserving order
    seen_ans = set()
    dedup_ans = []
    for a in answer_tokens:
        if a not in seen_ans:
            seen_ans.add(a)
            dedup_ans.append(a)

    seen_facts = set()
    dedup_facts = []
    for f in fact_citations:
        if f not in seen_facts:
            seen_facts.add(f)
            dedup_facts.append(f)

    return dedup_ans, dedup_facts


def compute_f1_score(predicted: Sequence[str], gold: Sequence[str]) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and F1 score between predicted and gold tokens.
    
    Returns:
        Tuple of (precision, recall, f1).
    """
    pred_set = set(p.lower().strip() for p in predicted if p.strip())
    gold_set = set(g.lower().strip() for g in gold if g.strip())

    if not gold_set:
        # Expected empty set
        if not pred_set:
            return 1.0, 1.0, 1.0
        return 0.0, 1.0, 0.0

    if not pred_set:
        return 0.0, 0.0, 0.0

    intersection = pred_set.intersection(gold_set)
    precision = len(intersection) / float(len(pred_set))
    recall = len(intersection) / float(len(gold_set))
    if precision + recall == 0.0:
        return 0.0, 0.0, 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return precision, recall, f1


class ExecutableGraphTask:
    """Dokime Task plugin evaluating verifiable multi-hop graph reasoning."""

    def __init__(self, dataset_path: Optional[Path] = None):
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "executable_graph_suite.json"
        self.dataset_path = dataset_path
        self.task_id = "executable_graph_qa"

    def load(self) -> List[Query]:
        """Load benchmark queries from JSON suite file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Executable graph benchmark suite not found at {self.dataset_path}")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: List[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=item["case_id"],
                    prompt=item["question_text"],
                    expected="; ".join(sorted(item.get("gold_answer_ids", []))),
                    metadata=item,
                )
            )
        return queries

    def score(self, query: Query, response: str) -> float:
        """Compute answer F1 score against deterministically executed answer key.
        
        Returns:
            answer_F1 score in [0.0, 1.0].
        """
        detailed = self.score_detailed(query, response)
        return float(detailed["answer_F1"])

    def score_detailed(self, query: Query, response: str) -> Dict[str, Any]:
        """Compute detailed multi-metric evaluation record.
        
        Separates answer correctness from evidence support to ensure
        unsupported guesses are not credited as evidence-grounded successes.
        """
        pred_answers, pred_facts = parse_entities_and_citations(response)
        gold_answers = query.metadata.get("gold_answer_ids", [])
        gold_facts = query.metadata.get("evidence_fact_ids", [])

        # Answer metrics
        ans_p, ans_r, ans_f1 = compute_f1_score(pred_answers, gold_answers)
        exact_match = set(p.lower() for p in pred_answers) == set(g.lower() for g in gold_answers)

        # Evidence support metrics
        ev_p, ev_r, ev_f1 = compute_f1_score(pred_facts, gold_facts)
        evidence_grounded = bool(exact_match and (ev_r >= 1.0 or not gold_facts))

        return {
            "case_id": query.id,
            "predicted_answers": pred_answers,
            "gold_answers": gold_answers,
            "answer_precision": round(ans_p, 4),
            "answer_recall": round(ans_r, 4),
            "answer_F1": round(ans_f1, 4),
            "exact_set_match": exact_match,
            "predicted_evidence": pred_facts,
            "gold_evidence": gold_facts,
            "evidence_precision": round(ev_p, 4),
            "evidence_recall": round(ev_r, 4),
            "evidence_F1": round(ev_f1, 4),
            "evidence_grounded_success": evidence_grounded,
            "query_pattern": query.metadata.get("query_pattern", "unknown"),
        }


@dataclass
class RolloutCaseSummary:
    """Aggregated statistics for a single case across repeated rollouts."""
    case_id: str
    query_pattern: str
    num_attempts: int
    mean_score: float
    sample_variance: float
    category: str
    evidence_grounded_rate: float
    failure_distribution: Dict[str, int]


@dataclass
class ExecutableGraphRolloutReport:
    """Audit report for an executable graph multi-hop rollout study."""
    policy: str
    num_attempts: int
    total_cases: int
    total_attempts: int
    boundary_count: int
    retention_count: int
    hard_count: int
    consistent_partial_count: int
    mean_overall_score: float
    case_summaries: List[RolloutCaseSummary]

    def to_markdown(self) -> str:
        lines = [
            "# Dokime Executable Graph Repeated Rollout Diagnostic",
            "",
            f"**Policy:** `{self.policy}` | **Attempts per Case ($N$):** {self.num_attempts} | **Total Cases:** {self.total_cases}",
            f"**Taxonomy Breakdown:** {self.boundary_count} Boundary, {self.retention_count} Retention Checks, {self.hard_count} Hard Diagnostics, {self.consistent_partial_count} Consistent Partials",
            f"**Mean Aggregate F1:** {self.mean_overall_score:.4f}",
            "",
            "| Case ID | Pattern | Mean F1 | Sample Var | Grounded Rate | Category | Failures |",
            "| :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
        ]
        for s in self.case_summaries:
            fail_str = ", ".join(f"{k}:{v}" for k, v in sorted(s.failure_distribution.items())) if s.failure_distribution else "none"
            lines.append(
                f"| `{s.case_id}` | `{s.query_pattern}` | {s.mean_score:.4f} | {s.sample_variance:.4f} | {s.evidence_grounded_rate:.2f} | `{s.category}` | {fail_str} |"
            )
        return "\n".join(lines)


class ExecutableGraphRolloutSuite:
    """Orchestrates repeated rollout evaluations over ExecutableGraphTask."""

    def __init__(self, task: Optional[ExecutableGraphTask] = None):
        self.task = task or ExecutableGraphTask()

    def run_rollouts(
        self,
        num_attempts: int = 5,
        policy: str = "retrieval_baseline",
        seeds: Optional[Sequence[int]] = None,
        client: Optional[Any] = None,
        model_name: str = "qwen2.5-coder-7b",
        tested_revision: str = "bc10774",
    ) -> Tuple[List[Dict[str, Any]], ExecutableGraphRolloutReport]:
        r"""Run repeated evaluation rollouts across benchmark cases.

        Args:
            num_attempts: Number of stochastic rollouts per case ($N \ge 2$, e.g. 5).
            policy: 'retrieval_baseline', 'oracle', or 'live_hestia'.
            seeds: Optional explicit list of RNG seeds.
            client: Optional HestiaClient for live model execution.
            model_name: Model identifier string.
            tested_revision: Repository commit identifier.

        Returns:
            Tuple of (List of attempt record dicts, ExecutableGraphRolloutReport).
        """
        queries = self.task.load()
        if not seeds:
            seeds = [42 + i for i in range(num_attempts)]

        attempt_records: List[Dict[str, Any]] = []
        case_scores: Dict[str, List[float]] = {}
        case_grounded: Dict[str, List[bool]] = {}
        case_failures: Dict[str, Dict[str, int]] = {}
        case_patterns: Dict[str, str] = {}

        for q in queries:
            case_scores[q.id] = []
            case_grounded[q.id] = []
            case_failures[q.id] = {}
            pattern = q.metadata.get("query_pattern", "unknown")
            case_patterns[q.id] = pattern
            gold_ans = q.metadata.get("gold_answer_ids", [])
            gold_ev = q.metadata.get("evidence_fact_ids", [])

            for attempt_idx, seed in enumerate(seeds):
                latency = 0.05
                token_use = 45

                if policy == "live_hestia" and client is not None:
                    # Live model generation
                    resp_obj = client.complete(
                        prompt=q.prompt,
                        model=model_name,
                        temperature=0.7,
                        max_tokens=128,
                    )
                    raw_text = getattr(resp_obj, "text", str(resp_obj))
                    latency = getattr(resp_obj, "latency", 0.15)
                    token_use = getattr(resp_obj, "tokens", 60)
                    detailed = self.task.score_detailed(q, raw_text)
                    fl = None
                    if detailed["answer_F1"] < 1.0:
                        fl = "incorrect_answer" if detailed["answer_F1"] == 0.0 else "partial_answer"
                elif policy == "oracle":
                    payload = {"answers": list(gold_ans), "evidence": list(gold_ev)}
                    detailed = self.task.score_detailed(q, json.dumps(payload))
                    fl = None
                else:
                    # Simulated empirical retrieval baseline with realistic pattern-conditioned noise
                    rng = random.Random(seed * 31 + hash(q.id) % 10000)
                    p = rng.random()

                    if pattern == "single_hop":
                        if p < 0.8:
                            ans, ev, fl = gold_ans, gold_ev, None
                        else:
                            ans, ev, fl = gold_ans, [], "unsupported_guess"
                    elif pattern in ("multi_answer", "multi_answer_single_hop"):
                        if p < 0.4:
                            ans, ev, fl = gold_ans, gold_ev, None
                        elif p < 0.8:
                            ans, ev, fl = [gold_ans[0]], [gold_ev[0]], "incomplete_answer_set"
                        else:
                            ans, ev, fl = ["distractor_entity"], [], "unsupported_guess"
                    elif pattern == "chain_2hop":
                        if p < 0.4:
                            ans, ev, fl = gold_ans, gold_ev, None
                        elif p < 0.8:
                            ans, ev, fl = ["wrong_hop_entity"], [gold_ev[0]], "broken_chain"
                        else:
                            ans, ev, fl = gold_ans, [gold_ev[0]], "incomplete_evidence"
                    elif pattern == "chain_3hop":
                        if p < 0.2:
                            ans, ev, fl = gold_ans, gold_ev, None
                        elif p < 0.6:
                            ans, ev, fl = ["wrong_hop_entity"], gold_ev[:1], "broken_chain"
                        else:
                            ans, ev, fl = ["distractor_entity"], [], "unsupported_guess"
                    elif pattern == "intersection":
                        if p < 0.4:
                            ans, ev, fl = gold_ans, gold_ev, None
                        elif p < 0.8:
                            anchor = q.metadata.get("anchor_ids", ["distractor"])[0]
                            ans, ev, fl = [anchor], [gold_ev[0]] if gold_ev else [], "missed_constraint"
                        else:
                            ans, ev, fl = ["distractor_entity"], [], "unsupported_guess"
                    elif pattern == "negative_empty":
                        if p < 0.6:
                            ans, ev, fl = [], [], None
                        else:
                            ans, ev, fl = ["distractor_entity"], [], "unsupported_guess"
                    else:
                        ans, ev, fl = gold_ans, gold_ev, None

                    payload = {"answers": ans, "evidence": ev}
                    detailed = self.task.score_detailed(q, json.dumps(payload))

                ans_f1 = float(detailed["answer_F1"])
                is_grounded = bool(detailed["evidence_grounded_success"])

                case_scores[q.id].append(ans_f1)
                case_grounded[q.id].append(is_grounded)
                if fl:
                    case_failures[q.id][fl] = case_failures[q.id].get(fl, 0) + 1

                record = {
                    "case_id": q.id,
                    "attempt_index": attempt_idx,
                    "seed": seed,
                    "score": round(ans_f1, 4),
                    "evidence_grounded": is_grounded,
                    "latency": latency,
                    "token_use": token_use,
                    "failure_label": fl,
                    "model_and_config": f"{policy}:{model_name}",
                    "tested_revision": tested_revision,
                    "metadata": {
                        "query_pattern": pattern,
                        "gold_answers": gold_ans,
                        "gold_evidence": gold_ev,
                        "predicted_answers": detailed["predicted_answers"],
                        "predicted_evidence": detailed["predicted_evidence"],
                    },
                }
                attempt_records.append(record)

        # Summaries
        summaries: List[RolloutCaseSummary] = []
        b_cnt = 0
        r_cnt = 0
        h_cnt = 0
        cp_cnt = 0

        for q in queries:
            scores = case_scores[q.id]
            n = len(scores)
            mean_s = sum(scores) / float(n)
            var = sum((s - mean_s) ** 2 for s in scores) / float(n - 1) if n > 1 else 0.0
            grounded_rate = sum(1 for g in case_grounded[q.id] if g) / float(n)

            if var > 1e-4:
                cat = "BOUNDARY"
                b_cnt += 1
            elif abs(mean_s - 1.0) < 1e-6:
                cat = "RETENTION_CHECK"
                r_cnt += 1
            elif abs(mean_s - 0.0) < 1e-6:
                cat = "HARD_DIAGNOSTIC"
                h_cnt += 1
            else:
                cat = "CONSISTENT_PARTIAL"
                cp_cnt += 1

            summaries.append(
                RolloutCaseSummary(
                    case_id=q.id,
                    query_pattern=case_patterns[q.id],
                    num_attempts=n,
                    mean_score=round(mean_s, 4),
                    sample_variance=round(var, 4),
                    category=cat,
                    evidence_grounded_rate=round(grounded_rate, 4),
                    failure_distribution=case_failures[q.id],
                )
            )

        total_scores = [r["score"] for r in attempt_records]
        overall_mean = sum(total_scores) / float(len(total_scores)) if total_scores else 0.0

        report = ExecutableGraphRolloutReport(
            policy=policy,
            num_attempts=len(seeds),
            total_cases=len(queries),
            total_attempts=len(attempt_records),
            boundary_count=b_cnt,
            retention_count=r_cnt,
            hard_count=h_cnt,
            consistent_partial_count=cp_cnt,
            mean_overall_score=round(overall_mean, 4),
            case_summaries=summaries,
        )

        return attempt_records, report
