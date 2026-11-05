"""
Entity Alias Reconciler for Knowledge Graph Ingestion (Item 116).

Inspired by 'FIRME: Registry-Anchored Memory Graphs for Evolving, Heterogeneous Investor Knowledge'
(arXiv:2610.40482). Prevents entity fragmentation across LTHCM topological memory graphs
by canonicalizing aliases, acronyms, and surface mentions to persistent canonical identifiers
at ingest time, before graph edge synthesis and hyperbolic embedding indexing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class CanonicalEntity:
    """A canonical entity registered with its surface aliases and metadata."""
    canonical_id: str
    primary_name: str
    aliases: Set[str] = field(default_factory=set)
    entity_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EntityAliasReconciler:
    """
    Maintains a deterministic alias registry and maps surface mentions, extracted graph
    nodes, and relational edges to canonical persistent identities.
    """

    def __init__(self) -> None:
        self._canonical_entities: Dict[str, CanonicalEntity] = {}
        # Mapping from normalized alias string -> canonical_id
        self._alias_lookup: Dict[str, str] = {}

    @classmethod
    def normalize_mention(cls, text: str) -> str:
        """Normalize an entity mention by stripping punctuation, extra spaces, and lowercasing."""
        clean = re.sub(r"[^\w\s-]", "", text.strip())
        return " ".join(clean.lower().split())

    def register_entity(
        self,
        canonical_id: str,
        primary_name: str,
        aliases: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CanonicalEntity:
        """Register a canonical entity and its alias variations."""
        c_id = canonical_id.strip()
        p_name = primary_name.strip()
        norm_primary = self.normalize_mention(p_name)
        norm_id = self.normalize_mention(c_id)

        all_aliases: Set[str] = set()
        if aliases:
            all_aliases.update(a.strip() for a in aliases if a.strip())

        entity = CanonicalEntity(
            canonical_id=c_id,
            primary_name=p_name,
            aliases=all_aliases,
            entity_type=entity_type,
            metadata=metadata or {},
        )

        self._canonical_entities[c_id] = entity
        self._alias_lookup[norm_primary] = c_id
        self._alias_lookup[norm_id] = c_id

        for alias in all_aliases:
            norm_a = self.normalize_mention(alias)
            if norm_a:
                self._alias_lookup[norm_a] = c_id

        return entity

    def resolve(self, mention_or_id: str) -> str:
        """
        Resolve an entity mention or candidate identifier to its canonical ID.
        If unknown, returns the original identifier unchanged.
        """
        raw = mention_or_id.strip()
        norm = self.normalize_mention(raw)
        return self._alias_lookup.get(norm, raw)

    def reconcile_graph_block(
        self,
        nodes: List[Dict[str, Any]],
        edges: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Reconcile a block of extracted nodes and edges:
        - Resolves node IDs and names to canonical entities.
        - Merges duplicate nodes created by different surface aliases.
        - Rewires edge src and dst identifiers to canonical IDs.
        - Drops self-referential cycles created by alias collapse.
        """
        reconciled_nodes: Dict[str, Dict[str, Any]] = {}

        for n in nodes:
            raw_id = str(n.get("id", ""))
            raw_name = str(n.get("name", raw_id))
            # Check ID resolution, then name resolution
            canon_id = self.resolve(raw_id)
            if canon_id == raw_id and raw_name != raw_id:
                canon_id = self.resolve(raw_name)

            if canon_id not in reconciled_nodes:
                new_node = dict(n)
                new_node["id"] = canon_id
                if canon_id in self._canonical_entities:
                    new_node["name"] = self._canonical_entities[canon_id].primary_name
                reconciled_nodes[canon_id] = new_node
            else:
                # Merge metadata into existing node
                existing = reconciled_nodes[canon_id]
                for k, v in n.items():
                    if k not in existing:
                        existing[k] = v

        reconciled_edges: List[Dict[str, Any]] = []
        seen_edges: Set[Tuple[str, str, str]] = set()

        if edges:
            for e in edges:
                src_raw = str(e.get("src", ""))
                dst_raw = str(e.get("dst", ""))
                rel = str(e.get("rel", "related"))

                canon_src = self.resolve(src_raw)
                canon_dst = self.resolve(dst_raw)

                # Skip self-loops resulting from alias collapse (e.g. "GDM" is "Google DeepMind")
                if canon_src == canon_dst:
                    continue

                edge_key = (canon_src, rel, canon_dst)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    new_edge = dict(e)
                    new_edge["src"] = canon_src
                    new_edge["dst"] = canon_dst
                    reconciled_edges.append(new_edge)

        return list(reconciled_nodes.values()), reconciled_edges
