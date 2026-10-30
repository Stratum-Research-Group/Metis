"""
Deterministic Span and Grounding Verifier for Stoma / LTHCM Graph Ingestion.

Per Tier 1 of ZERO_HALLUCINATION_GRAPH_FILTERING_SPEC.md:
Provides zero-token programmatic verification of extracted graph nodes, timestamps,
and relations against raw source text passages before prompt construction or indexing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


class DeterministicSpanFilter:
    """
    Validates that extracted knowledge graph nodes, timestamps, and relational edges
    have exact or bounded textual grounding in the raw source document/passage.
    """

    @staticmethod
    def verify_event_node(node: Dict[str, Any], source_text: str) -> Tuple[bool, str]:
        """
        Validates that a node's entity name and timestamp attributes are anchored in the source text.
        """
        # 1. Entity name / ID presence
        name = node.get("name", "")
        node_id = node.get("id", "")
        
        # Check either name or node_id explicitly in text
        name_found = False
        if name and name.lower() in source_text.lower():
            name_found = True
        elif node_id and re.search(rf"\b{re.escape(node_id)}\b", source_text, re.IGNORECASE):
            name_found = True
            
        if not name_found and (name or node_id):
            return False, f"Entity '{name or node_id}' has no span grounding in source text."

        # 2. Timestamp grounding check (if timestamp specified)
        ts = node.get("timestamp")
        if ts is not None:
            # Pattern matches: "hour 12", "starts at 12", "12 hours", "at 12", "start_hour=12"
            ts_str = str(ts)
            ts_pattern = (
                rf"\bhour\s+{re.escape(ts_str)}\b|"
                rf"\bstarts?\s+at\s+(?:hour\s+)?{re.escape(ts_str)}\b|"
                rf"\bat\s+hour\s+{re.escape(ts_str)}\b|"
                rf"\bstart_hour={re.escape(ts_str)}\b"
            )
            if not re.search(ts_pattern, source_text, re.IGNORECASE):
                # Also check direct proximity if entity name found
                return False, f"Timestamp {ts} for node '{node_id or name}' lacks textual span support."

        # 3. Duration grounding check (if duration specified)
        dur = node.get("duration")
        if dur is not None:
            dur_str = str(dur)
            dur_pattern = (
                rf"\bduration\s+(?:of\s+)?{re.escape(dur_str)}\s*(?:hours?|h)?\b|"
                rf"\b{re.escape(dur_str)}\s*hours?\s+duration\b|"
                rf"\bduration_hours={re.escape(dur_str)}\b"
            )
            if not re.search(dur_pattern, source_text, re.IGNORECASE):
                return False, f"Duration {dur} for node '{node_id or name}' lacks textual span support."

        return True, "Grounded"

    @staticmethod
    def verify_edge(
        src_id: str,
        dst_id: str,
        source_text: str,
        max_token_distance: int = 250,
    ) -> Tuple[bool, str]:
        """
        Verifies that source and destination entity tokens co-occur within a bounded context window.
        """
        src_matches = [m.start() for m in re.finditer(rf"\b{re.escape(src_id)}\b", source_text, re.IGNORECASE)]
        dst_matches = [m.start() for m in re.finditer(rf"\b{re.escape(dst_id)}\b", source_text, re.IGNORECASE)]

        if not src_matches:
            return False, f"Source node '{src_id}' not found in source text."
        if not dst_matches:
            return False, f"Target node '{dst_id}' not found in source text."

        # Check minimal character distance between any match (approx 4 chars/token)
        max_char_dist = max_token_distance * 4
        min_dist = min(abs(s - d) for s in src_matches for d in dst_matches)

        if min_dist > max_char_dist:
            return False, f"Nodes '{src_id}' and '{dst_id}' co-occur beyond maximum token distance ({min_dist} chars)."

        return True, "Grounded"

    def filter_graph_block(
        self,
        nodes: List[Dict[str, Any]],
        edges: Optional[List[Dict[str, Any]]],
        source_text: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Filters a collection of extracted graph nodes and edges against the raw source text.
        Returns:
            (verified_nodes, verified_edges, consistency_ratio)
        """
        if not nodes:
            return [], [], 1.0

        total_extracted = len(nodes) + (len(edges) if edges else 0)
        verified_nodes: List[Dict[str, Any]] = []
        verified_node_ids: Set[str] = set()

        for node in nodes:
            is_valid, _ = self.verify_event_node(node, source_text)
            if is_valid:
                verified_nodes.append(node)
                if "id" in node:
                    verified_node_ids.add(node["id"])

        verified_edges: List[Dict[str, Any]] = []
        if edges:
            for edge in edges:
                src = edge.get("src") or edge.get("source") or edge.get("from")
                dst = edge.get("dst") or edge.get("target") or edge.get("to")
                if src in verified_node_ids and dst in verified_node_ids:
                    is_edge_valid, _ = self.verify_edge(src, dst, source_text)
                    if is_edge_valid:
                        verified_edges.append(edge)

        total_verified = len(verified_nodes) + len(verified_edges)
        consistency_ratio = total_verified / total_extracted if total_extracted > 0 else 1.0

        return verified_nodes, verified_edges, consistency_ratio

