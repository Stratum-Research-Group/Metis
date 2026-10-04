"""
Universal Corpus Edge Builder for LTHCM Wormhole Graphs
======================================================
Constructs relational topological wormhole edges exclusively from unlabelled corpus text,
with zero access to queries, qrels, supporting facts, case IDs, or ground truth labels.

Follows the 4-Arm Honest Evaluation Protocol:
  Arm 1: No Edges (Flat LTHCM)
  Arm 2: Random Edges (Matching empirical degree distribution)
  Arm 3: Mined Edges (Corpus text-mined title mentions & IDF-weighted entity bridges)
  Arm 4: Oracle Edges (Gold pairs labeled strictly as an theoretical upper bound)
"""

import re
import math
import json
import random
from typing import List, Dict, Any, Tuple, Optional, Set, Union
from collections import defaultdict
import numpy as np
import torch


def _normalize_title(title: str) -> str:
    """Normalizes titles for mention matching."""
    return title.strip().lower()


_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "of",
    "not", "low", "high", "relies", "rely", "provides", "provide", "based", "which",
    "using", "used", "across", "between", "both", "all", "each", "under", "per"
}


def _extract_candidate_entities(text: str) -> Set[str]:
    """
    Extracts salient candidate entity n-grams and technical terms from text without external heavy models:
      1. Capitalized multi-word noun phrases: e.g., 'Turing Gold Medal', 'Zurich Quantum Institute'
      2. Technical alphanumeric codes/identifiers: e.g., 'LC-882', 'TwinScan EXE:5200', 'T790M'
      3. Salient content bigrams: e.g., 'surface codes', 'superconducting qubits'
    """
    entities = set()

    # Multi-word proper nouns / capitalized sequences
    multi_caps = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[a-zA-Z0-9]+)+\b", text)
    for ent in multi_caps:
        ent_clean = ent.strip()
        if len(ent_clean) > 3:
            entities.add(ent_clean.lower())

    # Technical identifiers with hyphens or numbers (e.g. EXE:5200, LC-882, T790M, G12C)
    tech_ids = re.findall(r"\b[A-Za-z0-9]+(?:[-:][A-Za-z0-9]+)+\b", text)
    for tid in tech_ids:
        tid_clean = tid.strip()
        if len(tid_clean) >= 3 and not tid_clean.isdigit():
            entities.add(tid_clean.lower())

    # Mixed alphanumeric tokens with uppercase and numbers (e.g., T790M, EXE5200, CD19)
    mixed_ids = re.findall(r"\b[A-Z]+[0-9]+[A-Za-z0-9]*\b", text)
    for mid in mixed_ids:
        mid_clean = mid.strip()
        if len(mid_clean) >= 3:
            entities.add(mid_clean.lower())

    # Content bigrams of adjacent non-stopwords
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        if w1 not in _STOPWORDS and w2 not in _STOPWORDS:
            entities.add(f"{w1} {w2}")

    return entities



def build_edges(
    docs: List[Dict[str, Any]],
    method: str = "title_mention+ner",
    max_out: int = 5,
    min_permeability: float = 0.40,
    default_permeability: float = 0.85,
    doc_embeddings: Optional[Union[Dict[str, Any], torch.Tensor, np.ndarray, List]] = None,
    scorer: Optional[Any] = None,
) -> List[Tuple[str, str, str, float]]:
    """
    Constructs relational wormhole edges strictly from corpus text.

    Args:
        docs: List of dicts, each with {"id": str, "text": str, optional "title": str}.
        method: "learned", "title_mention+ner", "title_mention_only", or "ner_only".
        max_out: Maximum outgoing degree per node to prevent hub explosion.
        min_permeability: Minimum edge permeability threshold [0.0, 1.0].
        default_permeability: Base permeability scaling factor for heuristic methods.
        doc_embeddings: Optional document dense embeddings for semantic feature scoring.
        scorer: Optional calibrated LearnedEdgeScorer instance for method='learned'.

    Returns:
        List of tuples: (src_id, dst_id, relation_type, permeability)
    """
    n_docs = len(docs)
    if n_docs < 2:
        return []

    # Map docs by ID and normalize content
    doc_map: Dict[str, Dict[str, Any]] = {}
    titles: Dict[str, str] = {}
    texts: Dict[str, str] = {}

    for d in docs:
        did = str(d["id"])
        doc_map[did] = d
        title = d.get("title", "")
        text = d.get("text", d.get("content", ""))
        titles[did] = title.strip()
        texts[did] = text.strip()

    candidate_edges: Dict[str, Dict[str, Tuple[str, float]]] = defaultdict(dict)

    # -------------------------------------------------------------
    # 0. Learned Edge Scoring (Supervised Checkpoint or Fixed-Weight Heuristic)
    # -------------------------------------------------------------
    if method in {"learned", "learned_edges", "fixed_weights", "fixed_weight_edges"}:
        from lthcm.graph.learned_edge_scorer import LearnedEdgeScorer, FixedWeightEdgeScorer
        if scorer is None:
            if method in {"fixed_weights", "fixed_weight_edges"}:
                scorer = FixedWeightEdgeScorer()
            else:
                scorer = LearnedEdgeScorer.load_default_checkpoint()

        # Build embedding dictionary
        emb_dict: Dict[str, Any] = {}
        if doc_embeddings is not None:
            if isinstance(doc_embeddings, dict):
                emb_dict = doc_embeddings
            elif isinstance(doc_embeddings, (torch.Tensor, np.ndarray, list)):
                for idx, d in enumerate(docs):
                    emb_dict[str(d["id"])] = doc_embeddings[idx]

        # Extract title mentions
        title_lookup = {}
        for did, title in titles.items():
            norm_t = _normalize_title(title)
            if len(norm_t) >= 3 and norm_t not in {"the", "and", "introduction", "overview", "methods"}:
                title_lookup[norm_t] = did

        compiled_titles = []
        for norm_title, target_id in title_lookup.items():
            compiled_titles.append((norm_title, target_id, re.compile(r"\b" + re.escape(norm_title) + r"\b")))

        title_mentions: Dict[str, Set[str]] = defaultdict(set)
        for src_id, text in texts.items():
            norm_text = text.lower()
            for norm_title, target_id, pattern in compiled_titles:
                if target_id == src_id:
                    continue
                if norm_title in norm_text and pattern.search(norm_text):
                    title_mentions[src_id].add(target_id)

        # Extract entities and compute IDFs
        doc_entities: Dict[str, Set[str]] = {}
        entity_df: Dict[str, int] = defaultdict(int)
        for did, text in texts.items():
            full_text = f"{titles.get(did, '')} {text}"
            ents = _extract_candidate_entities(full_text)
            doc_entities[did] = ents
            for e in sorted(ents):
                entity_df[e] += 1

        entity_idf: Dict[str, float] = {}
        max_df = max(3, int(0.25 * n_docs))
        for e, df in sorted(entity_df.items()):
            if 2 <= df <= max_df:
                entity_idf[e] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        inverted_index: Dict[str, List[str]] = defaultdict(list)
        for did, ents in doc_entities.items():
            for e in sorted(ents):
                if e in entity_idf:
                    inverted_index[e].append(did)

        # Propose candidate pairs (i, j)
        proposed_pairs: Dict[str, Set[str]] = defaultdict(set)
        for src in sorted(title_mentions):
            dsts = sorted(title_mentions[src])
            proposed_pairs[src].update(dsts)

        for e in sorted(inverted_index):
            posting = inverted_index[e]
            for i in range(len(posting)):
                u = posting[i]
                for j in range(i + 1, len(posting)):
                    v = posting[j]
                    proposed_pairs[u].add(v)
                    proposed_pairs[v].add(u)

        # Count in-degrees in proposal graph
        in_degrees: Dict[str, int] = defaultdict(int)
        for u in sorted(proposed_pairs):
            targets = proposed_pairs[u]
            for v in sorted(targets):
                in_degrees[v] += 1

        # Supervised scoring of proposed edges
        for u in sorted(proposed_pairs):
            targets = proposed_pairs[u]
            doc_u = doc_map[u]
            emb_u = emb_dict.get(u)
            for v in sorted(targets):
                doc_v = doc_map[v]
                emb_v = emb_dict.get(v)
                shared_ents = doc_entities.get(u, set()) & doc_entities.get(v, set())

                prob, _ = scorer.score_candidate(
                    doc_i=doc_u,
                    doc_j=doc_v,
                    shared_entities=shared_ents,
                    idf_map=entity_idf,
                    in_degree_j=in_degrees.get(v, 0),
                    emb_i=emb_u,
                    emb_j=emb_v,
                )

                if prob >= min_permeability:
                    rel_prefix = "fixed" if isinstance(scorer, FixedWeightEdgeScorer) else "learned"
                    rel = f"{rel_prefix}_title" if v in title_mentions[u] else f"{rel_prefix}_entity"
                    candidate_edges[u][v] = (rel, round(float(prob), 4))

    # -------------------------------------------------------------
    # 1. Title Mention Matching (A -> B if B's title appears in A)
    # -------------------------------------------------------------
    elif "title_mention" in method:
        # Precompute title lookup table for valid non-trivial titles
        title_lookup = {}
        for did, title in titles.items():
            norm_t = _normalize_title(title)
            # Avoid single letters or extremely generic titles
            if len(norm_t) >= 3 and norm_t not in {"the", "and", "introduction", "overview", "methods"}:
                title_lookup[norm_t] = did

        # Precompile regex patterns once outside document loop
        compiled_titles = []
        for norm_title, target_id in title_lookup.items():
            compiled_titles.append((norm_title, target_id, re.compile(r"\b" + re.escape(norm_title) + r"\b")))

        for src_id, text in texts.items():
            norm_text = text.lower()
            for norm_title, target_id, pattern in compiled_titles:
                if target_id == src_id:
                    continue
                # Fast substring check before regex word-boundary evaluation
                if norm_title in norm_text and pattern.search(norm_text):
                    # Title mention is strong semantic link
                    candidate_edges[src_id][target_id] = ("title_mention", default_permeability)

    # -------------------------------------------------------------
    # 2. IDF-Weighted Named Entity & Technical Term Co-occurrence
    # -------------------------------------------------------------
    if method != "learned" and "ner" in method:
        doc_entities: Dict[str, Set[str]] = {}
        entity_df: Dict[str, int] = defaultdict(int)

        for did, text in texts.items():
            full_text = f"{titles.get(did, '')} {text}"
            ents = _extract_candidate_entities(full_text)
            doc_entities[did] = ents
            for e in sorted(ents):
                entity_df[e] += 1

        # Calculate IDF: suppress ubiquitous terms (DF > 25% corpus) or singletons (DF = 1)
        entity_idf: Dict[str, float] = {}
        max_df = max(3, int(0.25 * n_docs))
        for e, df in sorted(entity_df.items()):
            if 2 <= df <= max_df:
                # Standard smoothed IDF
                entity_idf[e] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        # Inverted index: entity -> list of doc_ids
        inverted_index: Dict[str, List[str]] = defaultdict(list)
        for did, ents in doc_entities.items():
            for e in sorted(ents):
                if e in entity_idf:
                    inverted_index[e].append(did)

        # Accumulate pairwise shared entity scores
        pair_scores: Dict[Tuple[str, str], float] = defaultdict(float)
        for e in sorted(inverted_index):
            posting = inverted_index[e]
            idf = entity_idf[e]
            for i in range(len(posting)):
                u = posting[i]
                for j in range(i + 1, len(posting)):
                    v = posting[j]
                    pair_scores[(u, v)] += idf
                    pair_scores[(v, u)] += idf

        # Scale into permeabilities
        for (u, v) in sorted(pair_scores):
            score = pair_scores[(u, v)]
            if v in candidate_edges[u]:
                # If title mention already exists, boost permeability
                rel, p = candidate_edges[u][v]
                boosted_p = min(0.98, p + 0.05)
                candidate_edges[u][v] = (f"{rel}+ner", boosted_p)
            else:
                # Scaled permeability based on entity IDF strength
                perm = min(0.92, max(min_permeability, 0.50 + 0.10 * score))
                candidate_edges[u][v] = ("shared_entity", round(perm, 4))


    # -------------------------------------------------------------
    # 3. Out-Degree Capping (Top-K per source node)
    # -------------------------------------------------------------
    final_edges: List[Tuple[str, str, str, float]] = []
    for src_id in sorted(doc_map.keys()):
        targets = candidate_edges.get(src_id, {})
        if not targets:
            continue
        # Sort targets by permeability descending, tie-breaking deterministically by target_id
        sorted_targets = sorted(targets.items(), key=lambda item: (-item[1][1], item[0]))
        # Cap out-degree
        for target_id, (relation, perm) in sorted_targets[:max_out]:
            final_edges.append((src_id, target_id, relation, perm))

    return final_edges


def compute_edge_statistics(
    edges: List[Tuple[str, str, str, float]],
    total_docs: int,
    gold_pairs: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Computes rigorous graph diagnostics for the mined wormhole topology.
    Gold-pair coverage is computed strictly post-hoc as an observational metric,
    and is never used during edge construction.
    """
    out_degrees: Dict[str, int] = defaultdict(int)
    for src, dst, _, _ in edges:
        out_degrees[src] += 1

    total_edges = len(edges)
    docs_with_edges = len(out_degrees)
    mean_out_deg = total_edges / max(1, total_docs)
    max_out_deg = max(out_degrees.values()) if out_degrees else 0

    stats = {
        "total_edges": total_edges,
        "total_docs": total_docs,
        "docs_with_edges": docs_with_edges,
        "share_docs_with_edges": round(docs_with_edges / max(1, total_docs), 4),
        "mean_out_degree": round(mean_out_deg, 3),
        "max_out_degree": max_out_deg,
    }

    if gold_pairs:
        # Check how many gold pairs were discovered by unlabelled mining
        edge_set = {(src, dst) for src, dst, _, _ in edges}
        # In multi-hop, direction can be either way
        discovered = 0
        for u, v in gold_pairs:
            if (u, v) in edge_set or (v, u) in edge_set:
                discovered += 1
        gold_coverage = discovered / max(1, len(gold_pairs))
        stats["gold_pairs_total"] = len(gold_pairs)
        stats["gold_pairs_discovered"] = discovered
        stats["gold_pair_coverage"] = round(gold_coverage, 4)

    return stats


def generate_random_edges(
    docs: List[Dict[str, Any]],
    template_edges: List[Tuple[str, str, str, float]],
    seed: int = 42,
) -> List[Tuple[str, str, str, float]]:
    """
    Arm 2: Random Edges Control Baseline.
    Generates a random graph with the EXACT same node out-degree distribution
    and permeability distribution as template_edges, but with uniformly random destinations.
    """
    rng = random.Random(seed)
    all_doc_ids = [str(d["id"]) for d in docs]
    if len(all_doc_ids) < 2:
        return []

    # Map out-degrees and permeabilities per source
    out_specs: Dict[str, List[float]] = defaultdict(list)
    for src, _, _, perm in template_edges:
        out_specs[src].append(perm)

    random_edges: List[Tuple[str, str, str, float]] = []
    for src in sorted(out_specs.keys()):
        perms = out_specs[src]
        candidates = sorted(d for d in all_doc_ids if d != src)
        if not candidates:
            continue
        # Sample destinations without replacement up to len(perms)
        k = min(len(perms), len(candidates))
        chosen_dsts = rng.sample(candidates, k)
        for dst, p in zip(chosen_dsts, perms[:k]):
            random_edges.append((src, dst, "random_control", p))

    return random_edges


def generate_oracle_edges(
    gold_pairs: List[Tuple[str, str]],
    permeability: float = 1.0,
) -> List[Tuple[str, str, str, float]]:
    """
    Arm 4: Oracle Edges Upper Bound.
    Builds bidirectional edges exclusively between ground-truth gold pairs.
    STRICTLY MARKED AS AN ORACLE UPPER BOUND.
    """
    oracle_edges = []
    for u, v in gold_pairs:
        oracle_edges.append((str(u), str(v), "ORACLE_UPPER_BOUND", permeability))
        oracle_edges.append((str(v), str(u), "ORACLE_UPPER_BOUND", permeability))
    return oracle_edges


def save_edges_to_file(
    edges: List[Tuple[str, str, str, float]],
    filepath: str,
    stats: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves generated edge topology and diagnostic statistics to JSON."""
    data = {
        "stats": stats or {},
        "edges": [
            {"src": src, "dst": dst, "relation": rel, "permeability": perm}
            for src, dst, rel, perm in edges
        ],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
