"""
Spectral Chebyshev Graph Convolution for Adaptive Wormhole Refinement
======================================================================
Replaces static graph traversal in wormhole graphs with learnable Spectral Graph
Convolutions using truncated Chebyshev polynomial expansions of the graph Laplacian:
    Y = sum_{k=0}^{K-1} T_k(\\tilde{L}) X W_k

Where:
  - L = I - D^{-1/2} A D^{-1/2} is the symmetric normalized graph Laplacian.
  - \\tilde{L} = (2 / \\lambda_{max}) L - I is the scaled Laplacian (with \\lambda_{max} ~ 2.0).
  - T_0(X) = X
  - T_1(X) = \\tilde{L} X
  - T_k(X) = 2 \\tilde{L} T_{k-1}(X) - T_{k-2}(X)
  - W_k in R^{D_{in} x D_{out}} are learnable spectral filter coefficients.

Allows K-hop adaptive message passing over topological wormholes without
expensive eigendecompositions, dynamically amplifying useful bridges and suppressing
noisy shortcuts conditioned on the query.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from lthcm.core.wormhole import WormholeGraphIndex


class ChebyshevSpectralGraphConv(nn.Module):
    """
    Spectral Graph Convolutional Layer using K-order Chebyshev Polynomials.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        bias: bool = True,
    ):
        """
        Args:
            in_channels: Input node feature dimension D_in.
            out_channels: Output node feature dimension D_out.
            K: Chebyshev filter order (K-1 hops of message passing).
            bias: Whether to add a learnable bias vector.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = max(1, K)

        # Learnable filter weights for each polynomial order k = 0, ..., K-1
        self.weights = nn.Parameter(torch.empty(self.K, in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Glorot uniform initialization across polynomial slices
        nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def compute_scaled_laplacian(adj: torch.Tensor, lambda_max: float = 2.0) -> torch.Tensor:
        """
        Compute scaled normalized graph Laplacian:
            L_norm = I - D^{-1/2} A D^{-1/2}
            L_scaled = (2 / lambda_max) * L_norm - I
        Args:
            adj: Adjacency matrix [N, N] (with self-loops or non-negative weights).
            lambda_max: Upper bound on the largest eigenvalue (default 2.0).
        Returns:
            L_scaled: [N, N] scaled Laplacian tensor.
        """
        N = adj.shape[-1]
        device = adj.device
        dtype = adj.dtype

        # Add self-loops to ensure positive semi-definiteness
        A = adj + torch.eye(N, device=device, dtype=dtype)
        deg = torch.sum(A, dim=-1)  # [N]
        deg_inv_sqrt = torch.pow(deg.clamp_min(1e-12), -0.5)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)

        # L_norm = I - D^{-1/2} A D^{-1/2}
        L_norm = torch.eye(N, device=device, dtype=dtype) - D_inv_sqrt @ A @ D_inv_sqrt

        # L_scaled = (2 / lambda_max) * L_norm - I
        L_scaled = (2.0 / lambda_max) * L_norm - torch.eye(N, device=device, dtype=dtype)
        return L_scaled

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Chebyshev polynomial spectral graph filtering.
        Args:
            x: Node feature matrix [N, D_in].
            adj: Adjacency matrix [N, N].
        Returns:
            out: Filtered node features [N, D_out].
        """
        N, D = x.shape
        L_tilde = self.compute_scaled_laplacian(adj)

        # Chebyshev recurrence
        # T_0(x) = x
        t_0 = x
        out = torch.matmul(t_0, self.weights[0])  # [N, D_out]

        if self.K > 1:
            # T_1(x) = L_tilde @ x
            t_1 = torch.matmul(L_tilde, x)
            out = out + torch.matmul(t_1, self.weights[1])

        t_prev2 = t_0
        t_prev1 = t_1 if self.K > 1 else None

        for k in range(2, self.K):
            # T_k(x) = 2 * L_tilde @ T_{k-1} - T_{k-2}
            t_k = 2.0 * torch.matmul(L_tilde, t_prev1) - t_prev2
            out = out + torch.matmul(t_k, self.weights[k])
            t_prev2 = t_prev1
            t_prev1 = t_k

        if self.bias is not None:
            out = out + self.bias

        return out


class SpectralWormholeRefiner(nn.Module):
    """
    Adaptive Wormhole Refiner using Spectral Chebyshev Graph Networks.
    Wraps existing WormholeGraphIndex topologies, learning to dynamically
    amplify or suppress multi-hop wormhole shortcuts conditioned on the query.
    """

    def __init__(
        self,
        d_model: int = 64,
        K_order: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.K_order = K_order

        # 2-layer Spectral Chebyshev GCN with residual connection
        self.conv1 = ChebyshevSpectralGraphConv(d_model, d_model, K=K_order)
        self.conv2 = ChebyshevSpectralGraphConv(d_model, d_model, K=K_order)
        self.norm = nn.LayerNorm(d_model)

        # Query-bridge attention gating
        self.bridge_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def extract_subgraph_tensors(
        self,
        wormhole_index: WormholeGraphIndex,
        candidate_ids: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Extract node embedding matrix X and wormhole adjacency matrix A for candidate set.
        """
        valid_ids = [nid for nid in candidate_ids if nid in wormhole_index.node_embeddings]
        N = len(valid_ids)
        if N == 0:
            return torch.empty(0, self.d_model), torch.empty(0, 0), []

        # Stack node embeddings
        X = torch.stack([wormhole_index.node_embeddings[nid] for nid in valid_ids], dim=0)

        # Construct adjacency matrix incorporating topological wormholes
        A = torch.zeros(N, N, dtype=torch.float32)
        id_to_idx = {nid: i for i, nid in enumerate(valid_ids)}

        for nid in valid_ids:
            u = id_to_idx[nid]
            if nid in wormhole_index.graph:
                for target, edge_data in wormhole_index.graph[nid].items():
                    if target in id_to_idx:
                        v = id_to_idx[target]
                        perm = edge_data.get("permeability", 1.0)
                        A[u, v] = perm
                        A[v, u] = perm  # Undirected propagation

        return X, A, valid_ids

    def refine_candidates(
        self,
        query: torch.Tensor,
        wormhole_index: WormholeGraphIndex,
        candidate_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Execute spectral Chebyshev message passing across the candidate wormhole subgraph.
        Returns refined node representations and query-conditioned relevance scores.
        """
        X, A, valid_ids = self.extract_subgraph_tensors(wormhole_index, candidate_ids)
        if len(valid_ids) == 0:
            return {"results": []}

        device = query.device
        X = X.to(device)
        A = A.to(device)
        q = F.normalize(query, p=2, dim=-1)

        # Spectral Chebyshev convolution pass
        H1 = F.gelu(self.conv1(X, A))
        H2 = self.conv2(H1, A)
        # Residual connection + LayerNorm
        X_refined = self.norm(X + H2)
        X_refined_norm = F.normalize(X_refined, p=2, dim=-1)

        # Dynamic query-conditioned bridge scoring
        q_exp = q.unsqueeze(0).expand(len(valid_ids), -1)
        gate_weights = self.bridge_gate(torch.cat([q_exp, X_refined_norm], dim=-1)).squeeze(-1)  # [N]
        spectral_sim = torch.sum(q_exp * X_refined_norm, dim=-1)  # [N]
        refined_scores = gate_weights * spectral_sim

        results = []
        for i, nid in enumerate(valid_ids):
            content = wormhole_index.node_metadata.get(nid, {}).get("content", "")
            results.append({
                "node_id": nid,
                "refined_score": float(refined_scores[i].item()),
                "spectral_similarity": float(spectral_sim[i].item()),
                "gate_weight": float(gate_weights[i].item()),
                "refined_vector": X_refined_norm[i].cpu(),
                "content": content,
            })

        results.sort(key=lambda r: r["refined_score"], reverse=True)
        return {"results": results}


class BidirectionalWormholeEngine(nn.Module):
    """
    Bidirectional Self-Healing Wormhole Traversal & Backtracking Engine.
    
    Navigates non-metric topological shortcuts while executing downstream verification
    (e.g., Sheaf Laplacian Dirichlet energy, contradiction filtering, or custom verifiers).
    
    If an traversed item or step fails downstream verification:
      1. Performs back-hops along the traversal trajectory back to the previous nexus node.
      2. Dynamically penalizes the failed edge permeability (decay factor).
      3. Re-wormholes via alternative candidate bridges scored by the Spectral Chebyshev Refiner.
      4. Continuously self-heals multi-hop reasoning paths.
    """

    def __init__(
        self,
        wormhole_index: WormholeGraphIndex,
        refiner: Optional[SpectralWormholeRefiner] = None,
        penalty_factor: float = 0.1,
        max_backtracks: int = 5,
        hop_decay: float = 0.90,
    ):
        super().__init__()
        self.wormhole_index = wormhole_index
        self.refiner = refiner
        self.penalty_factor = penalty_factor
        self.max_backtracks = max_backtracks
        self.hop_decay = hop_decay

    def traverse_and_verify(
        self,
        query: torch.Tensor,
        start_node: str,
        verifier_fn: Callable[[List[str]], Tuple[bool, float, str]],
        target_depth: int = 3,
        top_k_alternatives: int = 3,
    ) -> Dict[str, Any]:
        """
        Traverse the wormhole index starting at `start_node` up to `target_depth` with self-healing backtracking.
        
        Args:
            query: Query tensor [D] or [1, D]
            start_node: Node ID to begin traversal from
            verifier_fn: Callable taking the current path `List[str]` and returning:
                         (is_valid: bool, metric_score: float, reason: str)
            target_depth: Target number of nodes in the path
            top_k_alternatives: Number of candidate branches to evaluate at each step
            
        Returns:
            Dictionary containing:
                - status: "SUCCESS" | "MAX_BACKTRACKS_EXCEEDED" | "SEARCH_EXHAUSTED"
                - path: Final verified path of node IDs
                - path_nodes: List of metadata dicts for nodes in path
                - backtracks: Total number of back-hops executed
                - history: Chronological log of forward hops, back-hops, and re-wormholing events
                - edge_penalties: Edge penalty weights dynamically applied
        """
        if start_node not in self.wormhole_index.node_embeddings:
            return {
                "status": "START_NODE_NOT_FOUND",
                "path": [],
                "path_nodes": [],
                "backtracks": 0,
                "history": [],
                "edge_penalties": {},
            }

        q = F.normalize(query.detach().cpu().view(-1), p=2, dim=0)
        path: List[str] = [start_node]
        backtrack_count = 0
        history: List[Dict[str, Any]] = [
            {"action": "start", "node": start_node}
        ]
        edge_penalties: Dict[Tuple[str, str], float] = {}
        status = "IN_PROGRESS"

        while len(path) < target_depth and backtrack_count < self.max_backtracks:
            curr_node = path[-1]
            neighbors = [n for n in self.wormhole_index.graph.successors(curr_node) if n not in path]

            if not neighbors:
                # Dead end reached with no unvisited successors -> Backtrack!
                if len(path) > 1:
                    failed_node = path.pop()
                    backtrack_count += 1
                    history.append({
                        "action": "backtrack_hop",
                        "from": failed_node,
                        "to": path[-1],
                        "reason": "dead_end_no_viable_successors",
                        "backtrack_index": backtrack_count,
                    })
                    continue
                else:
                    status = "SEARCH_EXHAUSTED"
                    break

            # Score candidate successors
            candidate_scores: List[Tuple[str, float]] = []

            if self.refiner is not None and len(neighbors) > 0:
                try:
                    ref_res = self.refiner.refine_candidates(
                        query=q, wormhole_index=self.wormhole_index, candidate_ids=neighbors
                    )
                    scored_map = {r["node_id"]: r["refined_score"] for r in ref_res.get("results", [])}
                except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                    import logging
                    logging.getLogger(__name__).warning("Spectral refiner failed: %s", e)
                    scored_map = {}
            else:
                scored_map = {}

            for nxt in neighbors:
                penalty = edge_penalties.get((curr_node, nxt), 1.0)
                if penalty < 1e-4:
                    continue  # Severed edge

                edge_data = self.wormhole_index.graph.get_edge_data(curr_node, nxt) or {}
                perm = edge_data.get("permeability", 1.0)

                if nxt in scored_map:
                    base_score = scored_map[nxt]
                else:
                    emb = self.wormhole_index.node_embeddings.get(nxt)
                    direct_sim = max(0.0, float(torch.dot(q, emb).item())) if emb is not None else 0.5
                    base_score = direct_sim * perm

                effective_score = base_score * penalty
                candidate_scores.append((nxt, effective_score))

            if not candidate_scores:
                # All candidate edges have been pruned/penalized -> Backtrack!
                if len(path) > 1:
                    failed_node = path.pop()
                    backtrack_count += 1
                    history.append({
                        "action": "backtrack_hop",
                        "from": failed_node,
                        "to": path[-1],
                        "reason": "all_candidate_edges_exhausted",
                        "backtrack_index": backtrack_count,
                    })
                    continue
                else:
                    status = "SEARCH_EXHAUSTED"
                    break

            # Select top candidate to explore
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            best_candidate, best_score = candidate_scores[0]

            # Tentatively perform forward hop
            path.append(best_candidate)

            # Downstream verification
            is_valid, metric_val, reason = verifier_fn(path)

            if not is_valid:
                # Downstream verification failed -> Back-hop and apply dynamic edge penalty
                failed_node = path.pop()
                prev_penalty = edge_penalties.get((curr_node, failed_node), 1.0)
                new_penalty = prev_penalty * self.penalty_factor
                edge_penalties[(curr_node, failed_node)] = new_penalty
                backtrack_count += 1

                history.append({
                    "action": "backtrack_and_penalize",
                    "from": failed_node,
                    "to": curr_node,
                    "reason": reason,
                    "metric": metric_val,
                    "new_penalty": new_penalty,
                    "backtrack_index": backtrack_count,
                })

                if backtrack_count >= self.max_backtracks:
                    status = "MAX_BACKTRACKS_EXCEEDED"
                    break
                # Loop continues with path[-1] == curr_node to re-wormhole via alternative bridge!
            else:
                # Verification passed -> Forward hop confirmed
                history.append({
                    "action": "forward_hop",
                    "from": curr_node,
                    "to": best_candidate,
                    "metric": metric_val,
                    "effective_score": best_score,
                })

        if len(path) >= target_depth:
            status = "SUCCESS"
        elif status == "IN_PROGRESS":
            status = "PARTIAL_PATH"

        path_nodes = []
        for nid in path:
            meta = dict(self.wormhole_index.node_metadata.get(nid, {}))
            meta["node_id"] = nid
            path_nodes.append(meta)

        return {
            "status": status,
            "path": path,
            "path_nodes": path_nodes,
            "backtracks": backtrack_count,
            "history": history,
            "edge_penalties": {f"{u}->{v}": p for (u, v), p in edge_penalties.items()},
        }

    def traverse_with_sheaf(
        self,
        query: torch.Tensor,
        start_node: str,
        sheaf_filter: Any,
        energy_threshold: float = 0.5,
        target_depth: int = 3,
        top_k_alternatives: int = 3,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper integrating Cellular Sheaf Dirichlet energy verification.
        Computes 0-coboundary obstruction delta^0 over the step (u -> v).
        If Dirichlet energy > energy_threshold, flags contradiction and triggers back-hop.
        """
        def sheaf_step_verifier(path: List[str]) -> Tuple[bool, float, str]:
            if len(path) < 2:
                return True, 0.0, "root_valid"
            u_id = path[-2]
            v_id = path[-1]
            emb_u = self.wormhole_index.node_embeddings.get(u_id)
            emb_v = self.wormhole_index.node_embeddings.get(v_id)
            if emb_u is None or emb_v is None:
                return True, 0.0, "no_embedding"

            # Form 2-node graph with 1 directed edge (0 -> 1)
            pair_embs = torch.stack([emb_u, emb_v], dim=0)
            edge_idx = torch.tensor([[0], [1]], dtype=torch.long, device=pair_embs.device)
            sheaf = getattr(sheaf_filter, "sheaf", sheaf_filter)
            with torch.no_grad():
                total_energy, _ = sheaf.compute_dirichlet_energy(pair_embs, edge_idx)
                energy = float(total_energy.item())

            if energy > energy_threshold:
                return False, energy, f"sheaf_dirichlet_obstruction ({energy:.4f} > {energy_threshold})"
            return True, energy, "coherent"

        return self.traverse_and_verify(
            query=query,
            start_node=start_node,
            verifier_fn=sheaf_step_verifier,
            target_depth=target_depth,
            top_k_alternatives=top_k_alternatives,
        )

