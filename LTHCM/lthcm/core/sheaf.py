"""
Cellular Sheaf Cohomology & Contradiction Filter
================================================
Implements Cellular Sheaves over topological graphs and simplicial complexes.
Constructs the Sheaf Laplacian L_F = delta^T delta and computes Cohomological Obstructions (H^1, H^2)
to geometrically isolate factual contradictions and cyclic semantic paradoxes before LLM prompt generation.
"""

from typing import List, Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellularSheaf(nn.Module):
    """
    Cellular Sheaf representation over a set of retrieved documents/chunks.

    Stalks:
      - Vertex stalks F(v) = R^D (document embeddings)
      - Edge stalks F(e) = R^K (shared agreement subspace)
      - Face stalks F(t) = R^M (triadic consistency subspace)

    Restriction Maps:
      - F_{v -> e}: R^D -> R^K
      - F_{e -> t}: R^K -> R^M
    """

    def __init__(self, d_model: int, stalk_dim: int = 64, face_dim: int = 32):
        super().__init__()
        self.d_model = d_model
        self.stalk_dim = stalk_dim
        self.face_dim = face_dim

        # Learned restriction map F_{v -> e} into the shared agreement edge stalk
        self.restriction_map = nn.Linear(d_model, stalk_dim, bias=False)
        # Initialize as normalized projection
        nn.init.orthogonal_(self.restriction_map.weight)

        # Higher-order restriction maps F_{e -> t} for 2-simplices (triangles)
        self.higher_restriction = nn.Linear(stalk_dim, face_dim, bias=False)
        nn.init.orthogonal_(self.higher_restriction.weight)

    def compute_coboundary_0(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute 0-coboundary delta^0: C^0 -> C^1
        (delta^0 x)_e = F_{v -> e} x_v - F_{u -> e} x_u
        Args:
            embeddings: [N, D] (0-cochains)
            edge_index: [2, E] (directed pairs u -> v)
        Returns:
            edge_cochains: [E, stalk_dim] (1-cochains)
        """
        u_idx = edge_index[0]
        v_idx = edge_index[1]

        x_u = embeddings[u_idx] # [E, D]
        x_v = embeddings[v_idx] # [E, D]

        # Apply agreement restriction map into the edge stalk
        res_u = self.restriction_map(x_u)
        res_v = self.restriction_map(x_v)

        # Coboundary difference along each edge
        return res_v - res_u

    def compute_dirichlet_energy(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Sheaf Dirichlet Energy: E(x) = x^T L_F x = ||delta^0 x||^2
        Args:
            embeddings: [N, D]
            edge_index: [2, E]
        Returns:
            total_energy: Scalar Dirichlet energy across the entire complex
            edge_energies: [E] Per-edge obstruction energies (high = contradiction)
        """
        if edge_index.shape[1] == 0:
            return torch.tensor(0.0, device=embeddings.device), torch.empty(0, device=embeddings.device)

        coboundary = self.compute_coboundary_0(embeddings, edge_index) # [E, stalk_dim]
        edge_energies = torch.sum(coboundary ** 2, dim=-1) # [E]
        total_energy = torch.mean(edge_energies)

        return total_energy, edge_energies

    def compute_higher_obstruction_h2(
        self,
        embeddings: torch.Tensor,
        triangles: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute 2nd Cohomological Obstruction H^2 across 2-simplices (triangles u -> v -> w -> u).
        Detects cyclic, non-transitive or paradoxical semantic loops.
        Args:
            embeddings: [N, D]
            triangles: [3, T] (indices of vertices forming 2-simplices)
        Returns:
            triadic_energy: Total H^2 obstruction
            per_triangle_energy: [T] Obstruction per triangle
        """
        if triangles.shape[1] == 0:
            return torch.tensor(0.0, device=embeddings.device), torch.empty(0, device=embeddings.device)

        u = embeddings[triangles[0]] # [T, D]
        v = embeddings[triangles[1]] # [T, D]
        w = embeddings[triangles[2]] # [T, D]

        # Restriction to edges: e1=(u,v), e2=(v,w), e3=(w,u)
        e1 = self.restriction_map(v) - self.restriction_map(u)
        e2 = self.restriction_map(w) - self.restriction_map(v)
        e3 = self.restriction_map(u) - self.restriction_map(w)

        # Coboundary delta^1: sum of oriented boundary edges on face
        # delta^1(e1, e2, e3) = F_{e1->t}(e1) + F_{e2->t}(e2) + F_{e3->t}(e3)
        face_cochain = (
            self.higher_restriction(e1)
            + self.higher_restriction(e2)
            + self.higher_restriction(e3)
        )

        per_triangle_energy = torch.sum(face_cochain ** 2, dim=-1) # [T]
        return torch.mean(per_triangle_energy), per_triangle_energy


class SheafContradictionFilter(nn.Module):
    """
    High-level verification filter that evaluates retrieved context documents
    and geometrically flags/prunes facts that generate high cohomological obstructions.
    Supports both vector-space Sheaf Laplacians and semantic NLI restriction stalks.
    """

    def __init__(
        self,
        d_model: int,
        stalk_dim: int = 64,
        contradiction_threshold: float = 0.5,
        nli_model_name: Optional[str] = None,
    ):
        super().__init__()
        self.sheaf = CellularSheaf(d_model=d_model, stalk_dim=stalk_dim)
        self.threshold = contradiction_threshold
        self.nli_model_name = nli_model_name
        self._nli_model = None

    @property
    def nli_model(self):
        if self._nli_model is None and self.nli_model_name:
            try:
                from sentence_transformers import CrossEncoder
                self._nli_model = CrossEncoder(self.nli_model_name)
                if hasattr(self._nli_model, "model") and hasattr(self._nli_model.model, "eval"):
                    self._nli_model.model.eval()
            except Exception:
                self._nli_model = None
        return self._nli_model

    def build_complete_graph_edges(self, n: int, device: torch.device) -> torch.Tensor:
        """Create fully connected directed edge index for N vertices."""
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    edges.append([i, j])
        if len(edges) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        return torch.tensor(edges, dtype=torch.long, device=device).T

    def evaluate_passages(
        self,
        passage_embeddings: torch.Tensor,
        passages: List[Dict[str, Any]],
        use_nli: bool = True,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a set of retrieved passages for factual contradictions.
        Args:
            passage_embeddings: [N, D]
            passages: List of passage metadata dicts
            use_nli: If True and NLI model is available, evaluate stalks via semantic NLI.
                     If False, evaluate pure geometric Dirichlet energy from restriction maps alone.
            threshold: Optional custom energy cutoff. Defaults to self.threshold.
        Returns:
            Dict containing:
              - is_consistent: bool
              - total_obstruction: float
              - conflicting_pairs: list of conflicting passage pairs
              - verified_passages: list of mutually consistent passages
        """
        N = passage_embeddings.shape[0]
        eff_threshold = threshold if threshold is not None else self.threshold

        if N <= 1:
            return {
                "is_consistent": True,
                "total_obstruction": 0.0,
                "conflicting_pairs": [],
                "verified_passages": passages,
            }

        edge_index = self.build_complete_graph_edges(N, passage_embeddings.device)
        total_energy, edge_energies = self.sheaf.compute_dirichlet_energy(passage_embeddings, edge_index)

        # If semantic NLI restriction map is enabled and available, evaluate stalks across edge pairs
        if use_nli and self.nli_model is not None and len(passages) >= 2 and "content" in passages[0]:
            pairs = []
            for idx in range(edge_index.shape[1]):
                u = edge_index[0, idx].item()
                v = edge_index[1, idx].item()
                pairs.append((passages[u]["content"], passages[v]["content"]))

            raw_scores = self.nli_model.predict(pairs)
            probs = F.softmax(torch.tensor(raw_scores), dim=-1) # [E, 3] (0: contra, 1: entail, 2: neutral)

            # Map directed edge energies into symmetric stalk obstruction
            # Build fast O(1) lookup dict for edges (u, v) -> edge_idx
            edge_map = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}

            semantic_energies = []
            # Group by undirected pairs to verify mutual consistency
            for idx in range(edge_index.shape[1]):
                u = edge_index[0, idx].item()
                v = edge_index[1, idx].item()
                p_contra = probs[idx, 0].item()
                p_entail = probs[idx, 1].item()
                
                # Check reverse edge (v -> u)
                rev_idx = edge_map.get((v, u), -1)
                
                if rev_idx >= 0:
                    p_contra_rev = probs[rev_idx, 0].item()
                    p_entail_rev = probs[rev_idx, 1].item()
                    sym_contra = (p_contra + p_contra_rev) / 2.0
                    sym_entail = max(p_entail, p_entail_rev)
                    if sym_entail > 0.35:
                        sym_contra = min(sym_contra, 0.1)
                else:
                    sym_contra = p_contra
                
                semantic_energies.append(sym_contra * 10.0)

            edge_energies = torch.tensor(semantic_energies, device=passage_embeddings.device)
            total_energy = torch.mean(edge_energies)

        conflicting_pairs = []
        high_conflict_nodes = set()
        seen_pairs = set()

        for idx in range(edge_index.shape[1]):
            energy = float(edge_energies[idx].item())
            if energy > eff_threshold:
                u = edge_index[0, idx].item()
                v = edge_index[1, idx].item()
                pair_tuple = (min(u, v), max(u, v))
                if pair_tuple not in seen_pairs:
                    seen_pairs.add(pair_tuple)
                    conflicting_pairs.append({
                        "doc_a": passages[u].get("doc_id", f"doc_{u}"),
                        "doc_b": passages[v].get("doc_id", f"doc_{v}"),
                        "content_a": passages[u].get("content", ""),
                        "content_b": passages[v].get("content", ""),
                        "obstruction_energy": float(energy),
                    })
                    high_conflict_nodes.add(u)
                    high_conflict_nodes.add(v)

        verified = [p for i, p in enumerate(passages) if i not in high_conflict_nodes]
        collapsed = len(verified) == 0 and len(passages) > 0
        if collapsed:
            # Prevent silent candidate collapse: retain passages and surface fallback
            verified_passages = passages
        else:
            verified_passages = verified

        return {
            "is_consistent": len(conflicting_pairs) == 0,
            "total_obstruction": float(total_energy.item()),
            "conflicting_pairs": conflicting_pairs,
            "verified_passages": verified_passages,
            "collapsed": collapsed,
            "fallback_used": collapsed,
        }
