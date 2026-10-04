"""
Matryoshka (Nested) Dimensional Compression & Adaptive-Resolution Retrieval
===========================================================================
Implements Matryoshka Representation Learning (MRL) concepts for LTHCM.
Represents embeddings in a "Russian doll" nested prefix structure where the first
d' dimensions of a d-dimensional vector (d_1 < d_2 < ... < d_M = d) form a valid,
self-contained metric space.

Key Capabilities:
  1. Prefix Truncation & Normalization: Extracts unit-norm representations at nested scales.
  2. Nested Multi-Granularity Similarity: Evaluates hierarchical similarity across prefix scales.
  3. Coarse-to-Fine Staged Search: Stage 1 prunes large memory stores using low-dimensional
     prefixes (e.g. 4x-8x memory and compute savings), followed by full-dimensional LTHCM
     topological verification on the candidate shortlist.
  4. Adaptive Resolution Gating: Early-exits search if coarse-scale confidence margin
     exceeds threshold, saving computation for unambiguous queries.
  5. SVD/Variance-Ordered Alignment: Orthonormal alignment head to ensure standard embeddings
     maximize information density in leading dimensions if not pre-trained with MRL.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from lthcm.core.superposition import SuperpositionMemoryStore


class MatryoshkaCompressor(nn.Module):
    """
    Matryoshka Dimensional Compression and Staged Retrieval Controller.

    Manages nested dimension granularities and staged coarse-to-fine filtering.
    """

    def __init__(
        self,
        d_model: int = 64,
        nested_dims: Optional[List[int]] = None,
        learnable_alignment: bool = False,
    ):
        """
        Args:
            d_model: Full embedding dimension D.
            nested_dims: List of ascending prefix dimensions [d_1, d_2, ..., d_M].
                         Defaults to [d_model // 4, d_model // 2, d_model].
            learnable_alignment: Whether to use an orthonormal projection head to
                                 concentrate information into leading dimensions.
        """
        super().__init__()
        self.d_model = d_model

        if nested_dims is None:
            # Default to 3 nested scales: quarter, half, full
            q = max(4, d_model // 4)
            h = max(8, d_model // 2)
            dims = sorted(list(set([q, h, d_model])))
            self.nested_dims = dims
        else:
            self.nested_dims = sorted(nested_dims)
            if self.nested_dims[-1] != d_model:
                self.nested_dims.append(d_model)

        self.coarse_dim = self.nested_dims[0]
        self.full_dim = self.nested_dims[-1]

        # Optional linear alignment matrix to maximize leading dimension variance
        if learnable_alignment:
            self.alignment_proj = nn.Linear(d_model, d_model, bias=False)
            nn.init.orthogonal_(self.alignment_proj.weight)
        else:
            self.alignment_proj = None

    def slice_prefix(self, x: torch.Tensor, dim: int, normalize: bool = True) -> torch.Tensor:
        """
        Extract the first `dim` coordinates of tensor x and optionally L2-normalize.
        Args:
            x: Input tensor [..., D].
            dim: Number of prefix dimensions (dim <= D).
            normalize: Whether to project onto the unit sphere S^{dim - 1}.
        Returns:
            Sliced tensor [..., dim].
        """
        dim = min(dim, x.shape[-1])
        if self.alignment_proj is not None:
            x = self.alignment_proj(x)
        sliced = x[..., :dim]
        if normalize:
            sliced = F.normalize(sliced, p=2, dim=-1)
        return sliced

    def multi_scale_embed(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Generate representations for all nested Matryoshka levels.
        Args:
            x: Input tensor [..., D].
        Returns:
            Dictionary mapping dimension d' -> normalized prefix tensor [..., d'].
        """
        return {d: self.slice_prefix(x, d, normalize=True) for d in self.nested_dims}

    def multi_scale_similarity(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        weights: Optional[Dict[int, float]] = None,
    ) -> torch.Tensor:
        """
        Compute weighted multi-scale cosine similarity across all nested dimensions.
        Args:
            query: Query tensor [D].
            candidates: Candidate tensor [N, D].
            weights: Optional dict of weights per scale (uniform if None).
        Returns:
            Combined similarity tensor [N].
        """
        if weights is None:
            # Equal weighting across scales
            w_val = 1.0 / len(self.nested_dims)
            weights = {d: w_val for d in self.nested_dims}

        total_sim = torch.zeros(candidates.shape[0], device=candidates.device, dtype=candidates.dtype)
        for d in self.nested_dims:
            w = weights.get(d, 1.0 / len(self.nested_dims))
            q_d = self.slice_prefix(query, d, normalize=True)
            c_d = self.slice_prefix(candidates, d, normalize=True)
            sim_d = torch.matmul(c_d, q_d)
            total_sim += w * sim_d

        return total_sim

    def fit_svd_alignment(self, data: torch.Tensor):
        """
        Calibrate an orthogonal rotation matrix using PCA/SVD on representative corpus embeddings.
        Aligns the principal axes of variation with the leading coordinate axes, ensuring
        that prefix truncation preserves maximum spectral variance.
        Args:
            data: Sample corpus embeddings [N, D].
        """
        with torch.no_grad():
            centered = data - data.mean(dim=0, keepdim=True)
            # SVD: centered = U @ S @ V.T -> V is [D, D] orthogonal basis
            _, _, V = torch.linalg.svd(centered, full_matrices=True)
            if self.alignment_proj is None:
                self.alignment_proj = nn.Linear(self.d_model, self.d_model, bias=False).to(data.device)
            # V.T maps largest variance components to coordinate indices 0, 1, 2...
            self.alignment_proj.weight.copy_(V.T)

    def coarse_to_fine_search(
        self,
        query: torch.Tensor,
        memory_store: SuperpositionMemoryStore,
        coarse_dim: Optional[int] = None,
        coarse_k: int = 10,
        fine_top_k: int = 5,
        early_exit_margin: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Execute staged Coarse-to-Fine Matryoshka retrieval against SuperpositionMemoryStore.

        Stage 1: Coarse Pruning (d_coarse << D)
          Computes fast POVM scores using only the first `coarse_dim` dimensions of all
          documents. Prunes the entire corpus down to `coarse_k` candidates.
          If the top match leads by more than `early_exit_margin`, early-exits to save compute.

        Stage 2: Full-Dimensional Re-ranking (D_full)
          Evaluates exact full-dimensional POVM expectation scores only on the shortlisted
          `coarse_k` candidates.

        Args:
            query: Query tensor [D].
            memory_store: Instance of SuperpositionMemoryStore.
            coarse_dim: Prefix dimension for Stage 1 (defaults to self.coarse_dim).
            coarse_k: Number of candidates to retain from coarse stage.
            fine_top_k: Final number of ranked results.
            early_exit_margin: Confidence gap between rank 1 and rank 2 to early-exit.
        Returns:
            Dictionary containing final results, stage diagnostics, and compression ratio.
        """
        N = len(memory_store.documents)
        if N == 0:
            return {"status": "empty", "results": [], "compression_ratio": 1.0}

        c_dim = coarse_dim or self.coarse_dim
        device = memory_store.facet_tensor.device
        q = query.to(device)

        # Compression ratio achieved in Stage 1
        compression_ratio = self.full_dim / c_dim

        # --- Stage 1: Coarse Prefix POVM Retrieval ---
        q_coarse = self.slice_prefix(q, c_dim, normalize=True)  # [c_dim]
        # Sliced facet tensor: [N, K, c_dim]
        facets_coarse = self.slice_prefix(memory_store.facet_tensor, c_dim, normalize=True)
        # Inner products in coarse subspace
        inner_coarse = torch.einsum('d, nkd -> nk', q_coarse, facets_coarse)
        probs_coarse = inner_coarse ** 2
        scores_coarse = torch.sum(memory_store.weight_tensor * probs_coarse, dim=-1)  # [N]

        k_coarse = min(coarse_k, N)
        top_coarse_scores, top_coarse_idx = torch.topk(scores_coarse, k=k_coarse)

        # Check for Early Exit
        early_exit = False
        if k_coarse >= 2:
            gap = float((top_coarse_scores[0] - top_coarse_scores[1]).item())
            if gap >= early_exit_margin:
                early_exit = True

        if early_exit:
            # Return coarse results directly, bypassing fine evaluation
            k_out = min(fine_top_k, k_coarse)
            results = []
            for rank, (sc, idx) in enumerate(zip(top_coarse_scores[:k_out].tolist(), top_coarse_idx[:k_out].tolist())):
                doc = memory_store.documents[idx]
                bf_idx = int(torch.argmax(probs_coarse[idx]).item())
                results.append({
                    "rank": rank + 1,
                    "score": float(sc),
                    "doc_id": doc["id"],
                    "content": doc["content"],
                    "collapsed_facet_idx": bf_idx,
                    "collapsed_facet_energy": float(torch.max(probs_coarse[idx]).item()),
                    "collapsed_facet_vector": memory_store.facet_tensor[idx, bf_idx].cpu(),
                    "stage": "coarse_early_exit",
                    "coarse_dim": c_dim,
                    "metadata": doc["metadata"],
                })
            return {
                "results": results,
                "early_exit": True,
                "stage1_candidates": k_coarse,
                "coarse_dim": c_dim,
                "full_dim": self.full_dim,
                "compression_ratio": float(compression_ratio),
                "flops_saved_pct": float((1.0 - (c_dim / self.full_dim)) * 100.0),
            }

        # --- Stage 2: Fine Full-Dimensional POVM Readout on Shortlist ---
        q_full = F.normalize(q, p=2, dim=-1)
        shortlist_idx = top_coarse_idx.tolist()
        shortlist_facets = memory_store.facet_tensor[shortlist_idx]  # [k_coarse, K, D]
        shortlist_weights = memory_store.weight_tensor[shortlist_idx]  # [k_coarse, K]

        inner_full = torch.einsum('d, nkd -> nk', q_full, shortlist_facets)
        probs_full = inner_full ** 2
        scores_full = torch.sum(shortlist_weights * probs_full, dim=-1)  # [k_coarse]

        best_facet_idx = torch.argmax(probs_full, dim=-1)
        best_facet_energy = torch.max(probs_full, dim=-1).values

        k_fine = min(fine_top_k, len(shortlist_idx))
        top_fine_scores, top_fine_ranks = torch.topk(scores_full, k=k_fine)

        results = []
        for rank, (sc, r_idx) in enumerate(zip(top_fine_scores.tolist(), top_fine_ranks.tolist())):
            doc_idx = shortlist_idx[r_idx]
            doc = memory_store.documents[doc_idx]
            bf_idx = int(best_facet_idx[r_idx].item())

            results.append({
                "rank": rank + 1,
                "score": float(sc),
                "coarse_score": float(scores_coarse[doc_idx].item()),
                "doc_id": doc["id"],
                "content": doc["content"],
                "collapsed_facet_idx": bf_idx,
                "collapsed_facet_energy": float(best_facet_energy[r_idx].item()),
                "collapsed_facet_vector": memory_store.facet_tensor[doc_idx, bf_idx].cpu(),
                "stage": "fine_full_reranked",
                "metadata": doc["metadata"],
            })

        # Calculate actual computational savings:
        # Full evaluation cost: N * D
        # Staged evaluation cost: N * c_dim + k_coarse * D
        cost_full = N * self.full_dim
        cost_staged = N * c_dim + k_coarse * self.full_dim
        compute_reduction = (1.0 - cost_staged / max(1, cost_full)) * 100.0

        return {
            "results": results,
            "early_exit": False,
            "stage1_candidates": k_coarse,
            "coarse_dim": c_dim,
            "full_dim": self.full_dim,
            "compression_ratio": float(compression_ratio),
            "flops_saved_pct": max(0.0, float(compute_reduction)),
        }
