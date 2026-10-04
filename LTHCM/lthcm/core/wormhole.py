"""
Topological Wormhole Graph Index
================================
Maintains non-metric express bridges connecting geometrically distant semantic clusters.
Enables multi-hop associative tunneling to knowledge that flat cosine search overlooks.
"""

from typing import List, Dict, Any, Tuple, Optional
import networkx as nx
import torch
import torch.nn.functional as F


class WormholeGraphIndex:
    """
    Hybrid Index combining continuous metric embeddings with discrete topological wormholes.

    In a pure Euclidean vector space, two concepts with low cosine similarity
    (e.g. 'moldy bread' and 'synthetic penicillin manufacturing') are unreachable via k-NN.
    A wormhole edge acts as a non-metric bridge between them.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.graph = nx.DiGraph()
        self.node_embeddings: Dict[str, torch.Tensor] = {}
        self.node_metadata: Dict[str, Dict[str, Any]] = {}
        self._node_id_list: List[str] = []
        self._emb_matrix: Optional[torch.Tensor] = None
        self._matrix_dirty: bool = True

    def add_node(
        self,
        node_id: str,
        embedding: torch.Tensor,
        content: str,
        cluster: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Register a node with its vector coordinate and cluster identity."""
        emb = F.normalize(embedding.detach().cpu().view(-1), p=2, dim=0)
        self.node_embeddings[node_id] = emb
        meta = metadata or {}
        meta["content"] = content
        meta["cluster"] = cluster
        self.node_metadata[node_id] = meta
        self.graph.add_node(node_id, cluster=cluster, content=content)
        self._matrix_dirty = True

    def add_nodes(
        self,
        node_ids: List[str],
        embeddings: torch.Tensor,
        contents: List[str],
        clusters: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ):
        """Batch-register multiple nodes."""
        b = len(node_ids)
        if b == 0:
            return
        norm_embs = F.normalize(embeddings.detach().cpu(), p=2, dim=-1)
        for i in range(b):
            nid = node_ids[i]
            emb = norm_embs[i].view(-1)
            cluster = clusters[i] if clusters and i < len(clusters) else "default"
            content = contents[i]
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            meta["content"] = content
            meta["cluster"] = cluster
            self.node_embeddings[nid] = emb
            self.node_metadata[nid] = meta
            self.graph.add_node(nid, cluster=cluster, content=content)
        self._matrix_dirty = True

    def add_wormhole(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        permeability: float = 1.0,
    ):

        """
        Create a directed topological shortcut (wormhole) between two nodes.
        """

        self.graph.add_edge(
            source_id,
            target_id,
            relation=relation_type,
            permeability=permeability,
            is_wormhole=True,
        )


    def get_bridges(self, node_id: str, min_permeability: float = 0.0) -> List[Dict[str, Any]]:
        """
        Return outgoing wormhole bridges from node_id with permeability >= min_permeability.
        """
        bridges = []
        if node_id in self.graph:
            for neighbor in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, neighbor)
                perm = edge_data.get("permeability", 1.0)
                if perm >= min_permeability:
                    bridges.append({
                        "target": neighbor,
                        "relation": edge_data.get("relation", ""),
                        "permeability": perm,
                    })
        return bridges



    def query(
        self,
        query_vector: torch.Tensor,
        top_k: int = 5,
        max_hops: int = 2,
        hop_decay: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid metric-topological retrieval:
          1. Find metric entry points (direct cosine similarity).
          2. Tunnel through wormholes up to max_hops away.
          3. Aggregate and rank results.
        """
        if len(self.node_embeddings) == 0:
            return []

        q = F.normalize(query_vector.detach().cpu().view(-1), p=2, dim=0)

        # 1. Compute direct metric similarities (vectorized over all nodes)
        if self._matrix_dirty or self._emb_matrix is None or len(self._node_id_list) != len(self.node_embeddings):
            self._node_id_list = list(self.node_embeddings.keys())
            self._emb_matrix = torch.stack([self.node_embeddings[nid] for nid in self._node_id_list])
            self._matrix_dirty = False

        sims = torch.mv(self._emb_matrix, q).clamp_min(0.0)
        sims_np = sims.numpy()
        direct_scores = {nid: float(sims_np[i]) for i, nid in enumerate(self._node_id_list)}

        # Select top entry nodes to seed wormhole exploration
        k_seed = min(len(self._node_id_list), max(3, top_k // 2))
        top_indices = torch.topk(sims, k=k_seed).indices.tolist()
        top_entries = [self._node_id_list[idx] for idx in top_indices]

        # 2. Traverse wormholes
        tunneled_scores: Dict[str, float] = {}
        traversal_paths: Dict[str, List[str]] = {}

        for entry_id in top_entries:
            entry_score = direct_scores[entry_id]
            # BFS exploration through wormholes
            queue = [(entry_id, entry_score, 0, [entry_id])]
            visited = {entry_id: entry_score}

            while queue:
                curr_id, curr_score, hops, path = queue.pop(0)

                if hops >= max_hops:
                    continue

                for neighbor in self.graph.successors(curr_id):
                    edge_data = self.graph.get_edge_data(curr_id, neighbor)
                    perm = edge_data.get("permeability", 1.0)
                    transferred_score = curr_score * perm * hop_decay

                    if neighbor not in visited or transferred_score > visited[neighbor]:
                        visited[neighbor] = transferred_score
                        new_path = path + [neighbor]
                        traversal_paths[neighbor] = new_path
                        if neighbor not in tunneled_scores or transferred_score > tunneled_scores[neighbor]:
                            tunneled_scores[neighbor] = transferred_score
                        queue.append((neighbor, transferred_score, hops + 1, new_path))

        # 3. Combine direct and tunneled scores
        all_candidates = set(direct_scores.keys()).union(tunneled_scores.keys())
        results = []

        for cid in all_candidates:
            d_score = direct_scores.get(cid, 0.0)
            t_score = tunneled_scores.get(cid, 0.0)
            is_tunneled = t_score > d_score
            final_score = max(d_score, t_score)

            meta = self.node_metadata[cid]
            results.append({
                "node_id": cid,
                "score": final_score,
                "direct_score": d_score,
                "tunneled_score": t_score,
                "is_tunneled": is_tunneled,
                "path": traversal_paths.get(cid, [cid]),
                "cluster": meta["cluster"],
                "content": meta["content"],
                "embedding": self.node_embeddings[cid],
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
