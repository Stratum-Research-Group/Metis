"""
Quantum Superposition & POVM Density Matrix Memory
==================================================
Represents multi-faceted documents and knowledge as mixed quantum states (density operators rho).
Eliminates polysemy and single-vector compression loss by resolving queries via
Positive Operator-Valued Measure (POVM) expectation values and wave interference.
"""

from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperpositionMemoryStore(nn.Module):
    """
    Quantum-inspired Density Matrix Memory Store.

    Each document i is stored as a low-rank density operator:
        rho_i = sum_{k=1}^K w_{ik} * |v_{ik}><v_{ik}|
    where sum_k w_{ik} = 1, w_{ik} >= 0, and <v_{ik}|v_{ik}> = 1.

    When query |q> arrives, relevance is the quantum expectation value:
        P(q | rho_i) = Tr(|q><q| * rho_i) = <q| rho_i |q> = sum_{k=1}^K w_{ik} * (<q|v_{ik}>)^2

    Features:
      - Polysemy preservation without single-vector centroid blurring.
      - Fast O(K * D) trace computation without materializing D x D matrices.
      - Dynamic facet collapse extraction: inspect exactly which sense of the document matched.
    """

    def __init__(self, d_model: int, num_facets: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_facets = num_facets

        # Dynamic storage buffers
        self.documents: List[Dict[str, Any]] = []
        self.doc_index: Dict[str, int] = {}
        # Tensors storing all facet vectors and mixture weights
        self.register_buffer("facet_tensor", torch.empty(0, num_facets, d_model))
        self.register_buffer("weight_tensor", torch.empty(0, num_facets))

    def add_document(
        self,
        doc_id: str,
        content: str,
        facet_vectors: torch.Tensor,
        mixture_weights: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
        is_logits: bool = False,
    ):
        """
        Register a multi-faceted document into superposition memory.
        Args:
            doc_id: Unique string identifier.
            content: Raw text content.
            facet_vectors: Tensor of shape [K, D] or [1, K, D].
            mixture_weights: Tensor of shape [K] or None (uniform prior).
            metadata: Additional domain/time metadata.
            is_logits: Whether mixture_weights are raw unnormalized logits.
        """
        if facet_vectors.dim() == 2:
            facet_vectors = facet_vectors.unsqueeze(0) # [1, K, D]

        K = facet_vectors.shape[1]
        if K != self.num_facets:
            # Pad or truncate to self.num_facets
            if K < self.num_facets:
                pad = torch.zeros(1, self.num_facets - K, self.d_model, device=facet_vectors.device)
                facet_vectors = torch.cat([facet_vectors, pad], dim=1)
            else:
                facet_vectors = facet_vectors[:, :self.num_facets, :]

        # Normalize facet vectors to unit norm (pure states <v|v> = 1)
        facet_vectors = F.normalize(facet_vectors, p=2, dim=-1)

        if mixture_weights is None:
            mixture_weights = torch.ones(1, self.num_facets, device=facet_vectors.device) / self.num_facets
        else:
            if mixture_weights.dim() == 1:
                mixture_weights = mixture_weights.unsqueeze(0)
            mixture_weights = mixture_weights.to(facet_vectors.device)
            k_in = mixture_weights.shape[1]
            # If caller passed negative values, they cannot be probabilities; treat as logits
            if not is_logits and torch.any(mixture_weights < 0.0):
                is_logits = True

            if is_logits:
                if k_in < self.num_facets:
                    pad = torch.full((1, self.num_facets - k_in), -1e9, device=mixture_weights.device)
                    mixture_weights = torch.cat([mixture_weights, pad], dim=1)
                else:
                    mixture_weights = mixture_weights[:, :self.num_facets]
                mixture_weights = F.softmax(mixture_weights, dim=-1)
            else:
                # Treat non-negative inputs as direct probabilities
                if k_in > self.num_facets:
                    mixture_weights = mixture_weights[:, :self.num_facets]
                mixture_weights = mixture_weights.clamp_min(0.0)
                sum_w = torch.sum(mixture_weights, dim=-1, keepdim=True)
                if torch.all(sum_w <= 1e-12):
                    mixture_weights = torch.ones_like(mixture_weights) / mixture_weights.shape[1]
                else:
                    mixture_weights = mixture_weights / sum_w.clamp_min(1e-12)
                if k_in < self.num_facets:
                    pad = torch.zeros(1, self.num_facets - k_in, device=mixture_weights.device)
                    mixture_weights = torch.cat([mixture_weights, pad], dim=1)

        # Handle upsert if doc_id already registered
        if doc_id in self.doc_index:
            old_idx = self.doc_index[doc_id]
            self.documents[old_idx] = {
                "idx": old_idx,
                "id": doc_id,
                "content": content,
                "metadata": metadata or {},
            }
            self.facet_tensor[old_idx:old_idx + 1] = facet_vectors.to(self.facet_tensor.device)
            self.weight_tensor[old_idx:old_idx + 1] = mixture_weights.to(self.weight_tensor.device)
            return

        doc_idx = len(self.documents)
        self.doc_index[doc_id] = doc_idx
        self.documents.append({
            "idx": doc_idx,
            "id": doc_id,
            "content": content,
            "metadata": metadata or {},
        })

        self.facet_tensor = torch.cat([self.facet_tensor, facet_vectors.to(self.facet_tensor.device)], dim=0)
        self.weight_tensor = torch.cat([self.weight_tensor, mixture_weights.to(self.weight_tensor.device)], dim=0)

    def add_documents(
        self,
        doc_ids: List[str],
        contents: List[str],
        facet_vectors: torch.Tensor,
        mixture_weights: Optional[torch.Tensor] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        is_logits: bool = False,
    ):
        """
        Batch-register multi-faceted documents into superposition memory.
        Avoids per-document reallocation overhead.
        """
        b = len(doc_ids)
        if b == 0:
            return
        if len(contents) != b:
            raise ValueError(f"Length mismatch: {len(contents)} contents for {b} doc_ids")

        if facet_vectors.dim() == 2:
            # [B, D] -> [B, 1, D]
            facet_vectors = facet_vectors.unsqueeze(1)

        k_in = facet_vectors.shape[1]
        if k_in != self.num_facets:
            if k_in < self.num_facets:
                pad = torch.zeros(b, self.num_facets - k_in, self.d_model, device=facet_vectors.device)
                facet_vectors = torch.cat([facet_vectors, pad], dim=1)
            else:
                facet_vectors = facet_vectors[:, :self.num_facets, :]

        facet_vectors = F.normalize(facet_vectors, p=2, dim=-1)

        if mixture_weights is None:
            mixture_weights = torch.ones(b, self.num_facets, device=facet_vectors.device) / self.num_facets
        else:
            if mixture_weights.dim() == 1:
                mixture_weights = mixture_weights.unsqueeze(0).expand(b, -1)
            mixture_weights = mixture_weights.to(facet_vectors.device)
            mw_k = mixture_weights.shape[1]
            if not is_logits and torch.any(mixture_weights < 0.0):
                is_logits = True

            if is_logits:
                if mw_k < self.num_facets:
                    pad = torch.full((b, self.num_facets - mw_k), -1e9, device=mixture_weights.device)
                    mixture_weights = torch.cat([mixture_weights, pad], dim=1)
                else:
                    mixture_weights = mixture_weights[:, :self.num_facets]
                mixture_weights = F.softmax(mixture_weights, dim=-1)
            else:
                if mw_k > self.num_facets:
                    mixture_weights = mixture_weights[:, :self.num_facets]
                mixture_weights = mixture_weights.clamp_min(0.0)
                sum_w = torch.sum(mixture_weights, dim=-1, keepdim=True)
                zero_mask = (sum_w <= 1e-12).squeeze(-1)
                if torch.any(zero_mask):
                    uniform = torch.ones_like(mixture_weights) / self.num_facets
                    mixture_weights = torch.where(zero_mask.unsqueeze(-1), uniform, mixture_weights / sum_w.clamp_min(1e-12))
                else:
                    mixture_weights = mixture_weights / sum_w.clamp_min(1e-12)
                if mw_k < self.num_facets:
                    pad = torch.zeros(b, self.num_facets - mw_k, device=mixture_weights.device)
                    mixture_weights = torch.cat([mixture_weights, pad], dim=1)

        # Separate new vs upsert documents
        new_doc_indices = []
        for i, did in enumerate(doc_ids):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            if did in self.doc_index:
                old_idx = self.doc_index[did]
                self.documents[old_idx] = {
                    "idx": old_idx,
                    "id": did,
                    "content": contents[i],
                    "metadata": meta,
                }
                self.facet_tensor[old_idx] = facet_vectors[i].to(self.facet_tensor.device)
                self.weight_tensor[old_idx] = mixture_weights[i].to(self.weight_tensor.device)
            else:
                new_doc_indices.append(i)

        if new_doc_indices:
            start_idx = len(self.documents)
            new_facets = facet_vectors[new_doc_indices].to(self.facet_tensor.device)
            new_weights = mixture_weights[new_doc_indices].to(self.weight_tensor.device)
            for offset, idx in enumerate(new_doc_indices):
                d_idx = start_idx + offset
                did = doc_ids[idx]
                meta = metadatas[idx] if metadatas and idx < len(metadatas) else {}
                self.doc_index[did] = d_idx
                self.documents.append({
                    "idx": d_idx,
                    "id": did,
                    "content": contents[idx],
                    "metadata": meta,
                })
            self.facet_tensor = torch.cat([self.facet_tensor, new_facets], dim=0)
            self.weight_tensor = torch.cat([self.weight_tensor, new_weights], dim=0)

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Compute POVM quantum measurement expectation value against all density matrices.
        Args:
            query: Query vector [D] or [1, D].
        Returns:
            List of top matches with expectation scores and active collapsed facets.
        """
        if len(self.documents) == 0:
            return []

        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Normalize query state |q>
        q = F.normalize(query, p=2, dim=-1).to(self.facet_tensor.device) # [1, D]

        # Inner product with each facet: <q | v_{ik}> -> [N_docs, K]
        # facet_tensor is [N, K, D]
        inner_products = torch.einsum('bd, nkd -> nk', q, self.facet_tensor)

        # Quantum transition probability: P_k = (<q|v_{ik}>)^2
        transition_probs = inner_products ** 2

        # Trace expectation value: Tr(M_q * rho_i) = sum_k w_{ik} * P_k
        expectation_scores = torch.sum(self.weight_tensor * transition_probs, dim=-1) # [N_docs]

        # Find best matching facet per document (eigenstate collapse)
        best_facet_idx = torch.argmax(transition_probs, dim=-1) # [N_docs]
        best_facet_energy = torch.max(transition_probs, dim=-1).values

        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(expectation_scores, k=k)

        results = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
            doc = self.documents[idx]
            bf_idx = int(best_facet_idx[idx].item())
            results.append({
                "rank": rank + 1,
                "score": float(score),
                "doc_id": doc["id"],
                "content": doc["content"],
                "collapsed_facet_idx": bf_idx,
                "collapsed_facet_energy": float(best_facet_energy[idx].item()),
                "collapsed_facet_vector": self.facet_tensor[idx, bf_idx].cpu(),
                "metadata": doc["metadata"],
            })

        return results

    # Alias for retrieve
    query = retrieve


    def density_matrix(self, doc_idx: int) -> torch.Tensor:
        """Explicitly construct the full D x D density matrix rho for analysis."""
        V = self.facet_tensor[doc_idx] # [K, D]
        w = self.weight_tensor[doc_idx] # [K]
        rho = torch.zeros(self.d_model, self.d_model, device=V.device)
        for k in range(self.num_facets):
            v_k = V[k].unsqueeze(1) # [D, 1]
            rho += w[k] * (v_k @ v_k.T)
        return rho
