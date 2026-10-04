"""
Automatic Corpus-Driven Wormhole Graph Builder
==============================================
Constructs topological shortcut edges across documents based on shared
salient entities, proper noun phrases, and technical terminology extracted
directly from document texts (without gold label leakage).
"""

import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict

from lthcm.core.wormhole import WormholeGraphIndex


# Stopwords to exclude from entity/keyword matching
COMMON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "this", "that", "these", "those", "their", "its", "which",
    "what", "where", "who", "whom", "whose", "why", "how",
}


def extract_salient_entities(text: str) -> Set[str]:
    """
    Extract candidate entities and domain-specific terms from a document text.
    Captures:
      - Capitalized multi-word spans (e.g., 'Dr. Nadia Rostova', 'Carl Zeiss SMT')
      - Technical hyphenated/alphanumeric terms (e.g., 'Kyber-Lattice', 'ROR-gamma', 'NCT0992384', 'KRAS G12C')
      - RFC / Standard references (e.g., 'RFC 9832')
    """
    entities = set()

    # 1. Capitalized multi-word phrases (2-4 words, allowing internal hyphens and dots)
    cap_phrases = re.findall(
        r"\b[A-Z][a-zA-Z0-9]*(?:[-.][A-Za-z0-9]+)*(?:\s+[A-Z][a-zA-Z0-9]*(?:[-.][A-Za-z0-9]+)*){1,3}\b",
        text,
    )
    for p in cap_phrases:
        cleaned = p.strip()
        words = cleaned.lower().split()
        if not all(w in COMMON_STOPWORDS for w in words) and len(cleaned) > 3:
            entities.add(cleaned.lower())

    # 2. Technical identifiers / codes (e.g., NCT0992384, RFC 9832, G12C, TSMC-Fab21, EXE:5200)
    tech_ids = re.findall(
        r"\b(?:NCT\d+|RFC\s*\d+|[A-Z0-9]+-[A-Za-z0-9]+|[A-Z]{2,}[0-9]+|[A-Z][0-9]{2,}[A-Z0-9]*)\b",
        text,
    )
    for tid in tech_ids:
        cleaned = re.sub(r"\s+", " ", tid).strip().lower()
        if len(cleaned) >= 3:
            entities.add(cleaned)

    # 3. Known domain compound terms (e.g. 'kras', 'ror-gamma', 'trem2', 'protac')
    compounds = re.findall(r"\b[a-zA-Z]+-[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text)
    for c in compounds:
        cl = c.strip().lower()
        if len(cl) > 3 and cl not in COMMON_STOPWORDS:
            entities.add(cl)

    return entities


def build_entity_wormholes(
    wormhole_index: WormholeGraphIndex,
    docs: List[Dict[str, Any]],
    permeability: float = 0.85,
    min_entity_len: int = 4,
    max_edges_per_doc: int = 6,
) -> List[Tuple[str, str, str]]:
    """
    Constructs wormhole edges between documents sharing salient entities.
    Processes all provided documents (including distractors) uniformly with zero gold label leakage.

    Args:
        wormhole_index: The WormholeGraphIndex instance to populate with edges.
        docs: List of doc dicts with at least 'id' and 'text' keys.
        permeability: Base transmission permeability for created shortcuts.
        min_entity_len: Minimum character length for an entity to trigger an edge.
        max_edges_per_doc: Maximum outgoing wormhole edges per document to keep graph sparse.

    Returns:
        List of added (source_id, target_id, entity) edge tuples.
    """
    entity_to_docs = defaultdict(set)

    for doc in docs:
        doc_id = doc["id"]
        text = doc.get("text") or doc.get("content") or ""
        ents = {e for e in extract_salient_entities(text) if len(e) >= min_entity_len}
        for e in ents:
            entity_to_docs[e].add(doc_id)

    added_edges = []
    doc_edge_counts = defaultdict(int)

    # Wire edges between documents that share non-trivial entities
    for entity, matched_docs in entity_to_docs.items():
        doc_list = list(matched_docs)
        if len(doc_list) < 2 or len(doc_list) > 10:
            # Skip if only 1 doc (no edge) or too ubiquitous (>10 docs is likely a generic word)
            continue

        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                d1, d2 = doc_list[i], doc_list[j]
                if doc_edge_counts[d1] < max_edges_per_doc and doc_edge_counts[d2] < max_edges_per_doc:
                    wormhole_index.add_wormhole(
                        source_id=d1,
                        target_id=d2,
                        relation_type=f"entity_{entity[:20]}",
                        permeability=permeability,
                    )
                    wormhole_index.add_wormhole(
                        source_id=d2,
                        target_id=d1,
                        relation_type=f"entity_{entity[:20]}_rev",
                        permeability=permeability,
                    )
                    doc_edge_counts[d1] += 1
                    doc_edge_counts[d2] += 1
                    added_edges.append((d1, d2, entity))

    return added_edges
