"""
GraphRAG Community Detection & Evidence Summarization (R4)
===========================================================
Implements Leiden community detection over topological/semantic document graphs
and centroid-salience multi-document community summarization for multi-evidence consistency.

Components:
  - LeidenCommunityDetector: Modularity-maximizing Leiden partition over corpus/subgraph.
  - CommunitySummarizer: Centroid-salience extractive summarizer with MMR diversity.
  - GraphRAGEvidenceBuilder: High-level pipeline bridging graph communities and claim verification.
"""

import os
import re
import math
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import torch
import torch.nn.functional as F

try:
    import igraph as ig
    import leidenalg
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

import networkx as nx


class LeidenCommunityDetector:
    """
    Partitions document graphs into coherent communities using Leiden community detection.
    Guarantees well-connected communities and maximizes Newman-Girvan modularity:
        Q = 1/(2m) * sum_{i,j} [ A_{ij} - (k_i * k_j)/(2m) ] * delta(c_i, c_j)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.25,
        resolution: float = 1.0,
        max_community_size: int = 50,
        seed: int = 42,
    ):
        self.similarity_threshold = similarity_threshold
        self.resolution = resolution
        self.max_community_size = max_community_size
        self.seed = seed

    def detect_communities(
        self,
        doc_ids: List[str],
        docs: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[np.ndarray] = None,
        edges: Optional[List[Tuple[str, str, float]]] = None,
    ) -> Dict[int, List[str]]:
        """
        Partition nodes into communities.

        Args:
            doc_ids: List of unique document identifiers.
            docs: Optional list of doc dicts with 'id' and 'text'.
            embeddings: Optional [N, D] document embeddings.
            edges: Optional list of (src, dst, weight) tuples.

        Returns:
            Dict mapping community_id (int) -> list of doc_ids.
        """
        N = len(doc_ids)
        if N <= 1:
            return {0: list(doc_ids)}

        id_to_idx = {did: idx for idx, did in enumerate(doc_ids)}
        graph_edges: Dict[Tuple[int, int], float] = {}

        # 1. Add explicitly provided edges
        if edges:
            for src, dst, weight in edges:
                if src in id_to_idx and dst in id_to_idx and src != dst:
                    u = id_to_idx[src]
                    v = id_to_idx[dst]
                    edge_key = (min(u, v), max(u, v))
                    graph_edges[edge_key] = max(graph_edges.get(edge_key, 0.0), float(weight))

        # 2. Add embedding cosine similarity edges if available
        if embeddings is not None and embeddings.shape[0] == N:
            # Normalize embeddings
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
            norm_embs = embeddings / norms
            sim_matrix = np.dot(norm_embs, norm_embs.T)

            for i in range(N):
                for j in range(i + 1, N):
                    sim = float(sim_matrix[i, j])
                    if sim >= self.similarity_threshold:
                        edge_key = (i, j)
                        graph_edges[edge_key] = max(graph_edges.get(edge_key, 0.0), sim)

        # 3. Add text overlap edges if no edges exist yet
        if not graph_edges and docs:
            doc_text_map = {d["id"]: d.get("text", "") for d in docs}
            token_sets = {
                did: set(re.findall(r"\b\w{3,}\b", doc_text_map.get(did, "").lower()))
                for did in doc_ids
            }
            for i in range(N):
                for j in range(i + 1, N):
                    s1 = token_sets.get(doc_ids[i], set())
                    s2 = token_sets.get(doc_ids[j], set())
                    if s1 and s2:
                        jaccard = len(s1 & s2) / max(1, len(s1 | s2))
                        if jaccard > 0.08:
                            graph_edges[(i, j)] = jaccard

        # If completely disconnected, each doc forms its own community
        if not graph_edges:
            return {i: [did] for i, did in enumerate(doc_ids)}

        # 4. Partition with Leiden (if available) or NetworkX Louvain fallback
        if HAS_LEIDEN:
            return self._partition_leiden(doc_ids, graph_edges, N)
        else:
            return self._partition_networkx(doc_ids, graph_edges, N)

    def _partition_leiden(
        self,
        doc_ids: List[str],
        graph_edges: Dict[Tuple[int, int], float],
        num_vertices: int,
    ) -> Dict[int, List[str]]:
        """Leiden partition via igraph + leidenalg."""
        g = ig.Graph()
        g.add_vertices(num_vertices)
        
        edge_list = list(graph_edges.keys())
        weights = [graph_edges[e] for e in edge_list]
        g.add_edges(edge_list)
        g.es["weight"] = weights

        try:
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=self.seed,
            )
            membership = partition.membership
        except Exception:
            # Fallback to connected components if partition fails
            membership = g.connected_components().membership

        communities: Dict[int, List[str]] = {}
        for idx, comm_id in enumerate(membership):
            communities.setdefault(comm_id, []).append(doc_ids[idx])

        # Renumber communities 0..K-1 sorted by size descending
        sorted_comms = sorted(communities.values(), key=len, reverse=True)
        return {cid: comm for cid, comm in enumerate(sorted_comms)}

    def _partition_networkx(
        self,
        doc_ids: List[str],
        graph_edges: Dict[Tuple[int, int], float],
        num_vertices: int,
    ) -> Dict[int, List[str]]:
        """Fallback partition via NetworkX."""
        G = nx.Graph()
        G.add_nodes_from(range(num_vertices))
        for (u, v), w in graph_edges.items():
            G.add_edge(u, v, weight=w)

        try:
            from networkx.algorithms.community import louvain_communities
            raw_comms = louvain_communities(G, weight="weight", seed=self.seed)
        except Exception:
            raw_comms = list(nx.connected_components(G))

        communities: Dict[int, List[str]] = {}
        sorted_raw = sorted(raw_comms, key=len, reverse=True)
        for cid, node_set in enumerate(sorted_raw):
            communities[cid] = [doc_ids[idx] for idx in node_set]

        return communities


class CommunitySummarizer:
    """
    Generates structured, coherent summaries for graph communities.
    Uses embedding centroid-salience ranking with Maximal Marginal Relevance (MMR)
    to select the most informative, non-redundant evidence claims.
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        max_sentences: int = 3,
        lambda_diversity: float = 0.65,
    ):
        self.encoder = encoder
        self.max_sentences = max_sentences
        self.lambda_diversity = lambda_diversity
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split passage text into clean constituent sentences."""
        clean = text.replace("\n", " ").strip()
        raw_sents = re.split(r"(?<=[.!?])\s+", clean)
        sentences = [s.strip() for s in raw_sents if len(s.strip()) > 15]
        return sentences if sentences else ([clean] if clean else [])

    def summarize_community(
        self,
        community_id: int,
        docs: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Generate a summary of all documents in the community.

        Args:
            community_id: Numeric community identifier.
            docs: List of doc dicts with 'id' and 'text'.
            embeddings: Optional doc-level or sentence-level embeddings.

        Returns:
            Dict containing:
              - community_id: int
              - summary: str
              - constituent_doc_ids: List[str]
              - key_facts: List[str]
        """
        if not docs:
            return {
                "community_id": community_id,
                "summary": "",
                "constituent_doc_ids": [],
                "key_facts": [],
            }

        doc_ids = [d.get("id", f"doc_{i}") for i, d in enumerate(docs)]
        combined_key = hashlib.md5(" ".join(doc_ids).encode("utf-8")).hexdigest()
        if combined_key in self._cache:
            return self._cache[combined_key]

        # Single doc case: return up to max_sentences
        all_sentences: List[Tuple[str, str]] = []  # (doc_id, sentence)
        for d in docs:
            did = d.get("id", "")
            text = d.get("text", d.get("content", ""))
            for sent in self._split_into_sentences(text):
                all_sentences.append((did, sent))

        if not all_sentences:
            summary_dict = {
                "community_id": community_id,
                "summary": docs[0].get("text", "")[:300],
                "constituent_doc_ids": doc_ids,
                "key_facts": [],
            }
            self._cache[combined_key] = summary_dict
            return summary_dict

        if len(all_sentences) <= self.max_sentences:
            summary_text = " ".join([s for _, s in all_sentences])
            summary_dict = {
                "community_id": community_id,
                "summary": summary_text,
                "constituent_doc_ids": doc_ids,
                "key_facts": [s for _, s in all_sentences],
            }
            self._cache[combined_key] = summary_dict
            return summary_dict

        # Multi-sentence case: Centroid-Salience with MMR
        sentences_text = [s for _, s in all_sentences]
        if self.encoder is not None:
            sent_embs = self.encoder.encode(sentences_text, batch_size=32, normalize_embeddings=True)
        else:
            # Lexical / TF-IDF fallback if no encoder passed
            sent_embs = self._compute_lexical_embeddings(sentences_text)

        # Centroid vector of community
        centroid = np.mean(sent_embs, axis=0, keepdims=True)
        centroid = centroid / (np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-9)

        # Relevance to centroid
        rel_scores = np.dot(sent_embs, centroid.T).flatten()

        # MMR Selection
        selected_indices: List[int] = []
        candidate_indices = list(range(len(sentences_text)))

        # Pick first: highest relevance
        first_idx = int(np.argmax(rel_scores))
        selected_indices.append(first_idx)
        candidate_indices.remove(first_idx)

        while len(selected_indices) < min(self.max_sentences, len(sentences_text)) and candidate_indices:
            best_idx = -1
            best_mmr = -1e9
            for c_idx in candidate_indices:
                rel = rel_scores[c_idx]
                sim_to_selected = max(
                    float(np.dot(sent_embs[c_idx], sent_embs[s_idx]))
                    for s_idx in selected_indices
                )
                mmr = self.lambda_diversity * rel - (1 - self.lambda_diversity) * sim_to_selected
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = c_idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)

        # Order selected sentences by original appearance
        selected_indices.sort()
        selected_sentences = [sentences_text[i] for i in selected_indices]
        summary_text = " ".join(selected_sentences)

        summary_dict = {
            "community_id": community_id,
            "summary": summary_text,
            "constituent_doc_ids": doc_ids,
            "key_facts": selected_sentences,
        }
        self._cache[combined_key] = summary_dict
        return summary_dict

    def _compute_lexical_embeddings(self, texts: List[str]) -> np.ndarray:
        """Lightweight bag-of-words / character n-gram hashing embedding fallback."""
        dim = 128
        embs = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = re.findall(r"\b\w+\b", t.lower())
            for tok in tokens:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
                embs[i, h] += 1.0
            norm = np.linalg.norm(embs[i])
            if norm > 0:
                embs[i] /= norm
        return embs


class GraphRAGEvidenceBuilder:
    """
    Evidence synthesis layer that transforms retrieved multi-passage clusters
    into community-structured evidence summaries for claim verification.
    """

    def __init__(
        self,
        encoder: Optional[Any] = None,
        similarity_threshold: float = 0.25,
        max_sentences_per_community: int = 3,
        seed: int = 42,
    ):
        self.detector = LeidenCommunityDetector(
            similarity_threshold=similarity_threshold,
            seed=seed,
        )
        self.summarizer = CommunitySummarizer(
            encoder=encoder,
            max_sentences=max_sentences_per_community,
        )

    def build_evidence(
        self,
        passages: List[Dict[str, Any]],
        passage_embeddings: Optional[np.ndarray] = None,
        edges: Optional[List[Tuple[str, str, float]]] = None,
    ) -> Dict[str, Any]:
        """
        Group passages into communities and generate synthesized community summaries.

        Args:
            passages: List of dicts with 'id' and 'text'.
            passage_embeddings: Optional [N, D] passage embeddings.
            edges: Optional graph edges between passages.

        Returns:
            Dict containing:
              - raw_concatenated: Direct string concatenation of raw passages.
              - community_summaries: List of community summary dicts.
              - concatenated_summary: String concatenation of community summaries.
              - num_communities: int
              - communities: Dict[int, List[str]]
        """
        if not passages:
            return {
                "raw_concatenated": "",
                "community_summaries": [],
                "concatenated_summary": "",
                "num_communities": 0,
                "communities": {},
            }

        doc_ids = [p.get("id", f"p_{i}") for i, p in enumerate(passages)]
        doc_map = {p.get("id", f"p_{i}"): p for i, p in enumerate(passages)}

        # Raw concatenated baseline
        raw_concatenated = " \n ".join([p.get("text", p.get("content", "")) for p in passages])

        # Detect communities
        communities = self.detector.detect_communities(
            doc_ids=doc_ids,
            docs=passages,
            embeddings=passage_embeddings,
            edges=edges,
        )

        # Summarize each community
        summaries: List[Dict[str, Any]] = []
        for cid, member_ids in communities.items():
            comm_docs = [doc_map[did] for did in member_ids if did in doc_map]
            s_dict = self.summarizer.summarize_community(community_id=cid, docs=comm_docs)
            summaries.append(s_dict)

        concatenated_summary = " \n ".join([s["summary"] for s in summaries if s["summary"]])

        return {
            "raw_concatenated": raw_concatenated,
            "community_summaries": summaries,
            "concatenated_summary": concatenated_summary,
            "num_communities": len(communities),
            "communities": communities,
        }
