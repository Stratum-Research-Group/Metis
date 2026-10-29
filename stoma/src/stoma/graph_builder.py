"""
Offline Knowledge Graph Edge Builder for LTHCM / Stoma Indices.
Computes metric cosine-similarity bridges (tau >= threshold) across passage embeddings
and adds wormhole edges to turn flat index into a multi-hop navigable knowledge graph.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from lthcm.engine.pipeline import LTHCMEngine

logger = logging.getLogger(__name__)


def build_knn_knowledge_graph(
    index_path: Path,
    similarity_threshold: float = 0.75,
    max_k_neighbors: int = 5,
    allow_trusted_pickle: bool = False,
) -> Dict[str, Any]:
    """
    Load index, compute cosine-similarity edges for node pairs with similarity >= threshold,
    and persist updated index atomically.
    """
    path = Path(index_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Index file not found: {path}")

    is_trusted = allow_trusted_pickle or bool(os.environ.get("STOMA_ALLOW_TRUSTED_PICKLE")) or bool(os.environ.get("DIANOIA_ALLOW_TRUSTED_PICKLE"))
    if not is_trusted:
        raise PermissionError(
            f"Untrusted pickle index loading refused for '{path}'. "
            "Pass --allow-trusted-pickle or set STOMA_ALLOW_TRUSTED_PICKLE=1 to authorize."
        )

    with open(path, "rb") as f:
        engine = pickle.load(f)

    if not isinstance(engine, LTHCMEngine):
        raise TypeError(f"Loaded index object is not LTHCMEngine (got {type(engine).__name__})")

    if not hasattr(engine, "wormhole") or not hasattr(engine.wormhole, "node_embeddings"):
        raise ValueError("Engine does not contain a wormhole graph index with node embeddings.")

    node_embeddings = engine.wormhole.node_embeddings
    node_ids = list(node_embeddings.keys())
    n = len(node_ids)

    if n < 2:
        return {
            "nodes_count": n,
            "edges_added": 0,
            "total_edges": (
                engine.wormhole.graph.number_of_edges()
                if hasattr(engine.wormhole.graph, "number_of_edges")
                else 0
            ),
            "threshold": similarity_threshold,
            "message": "Fewer than 2 nodes in index; no edges possible.",
        }

    # Stack normalized embeddings into matrix [n, d]
    emb_list = [F.normalize(node_embeddings[nid].float().view(-1), p=2, dim=0) for nid in node_ids]
    emb_matrix = torch.stack(emb_list, dim=0)

    # Compute similarity matrix [n, n]
    sim_matrix = torch.matmul(emb_matrix, emb_matrix.T)

    edges_added = 0
    for i, src_id in enumerate(node_ids):
        # Mask self-similarity
        sim_row = sim_matrix[i].clone()
        sim_row[i] = -1.0

        # Find entries >= threshold
        mask = sim_row >= similarity_threshold
        valid_indices = torch.nonzero(mask, as_tuple=True)[0]

        if len(valid_indices) == 0:
            continue

        # Sort top-k
        valid_sims = sim_row[valid_indices]
        sorted_order = torch.argsort(valid_sims, descending=True)[:max_k_neighbors]
        chosen_indices = valid_indices[sorted_order]

        for idx in chosen_indices:
            dst_id = node_ids[idx.item()]
            sim_val = float(sim_row[idx].item())
            engine.wormhole.add_wormhole(
                source_id=src_id,
                target_id=dst_id,
                relation_type="semantic_knn",
                permeability=sim_val,
            )
            edges_added += 1

    total_edges = (
        engine.wormhole.graph.number_of_edges()
        if hasattr(engine.wormhole.graph, "number_of_edges")
        else len(engine.wormhole.graph.edges)
    )

    # Atomically persist updated index
    tmp_path = path.with_suffix(".tmp_graph")
    with open(tmp_path, "wb") as f:
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)

    return {
        "nodes_count": n,
        "edges_added": edges_added,
        "total_edges": total_edges,
        "threshold": similarity_threshold,
        "max_k_neighbors": max_k_neighbors,
        "message": f"Successfully linked {edges_added} graph edges across {n} nodes (threshold={similarity_threshold}).",
    }
