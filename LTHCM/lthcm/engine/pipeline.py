"""
Unified LTHCM Engine & Orchestration Pipeline
==============================================
Orchestrates the entire topological and non-Euclidean memory lifecycle:
  1. Quantum POVM Superposition Retrieval
  2. Wormhole Topological Tunneling & Spectral Refinement
  3. Lie Group Gauge Fibre Parallel Transport
  4. Lorentzian Spacetime Causal Light-Cone Filtering
  5. Cellular Sheaf Cohomological Contradiction Verification
  6. Fisher-Rao & Wasserstein Distribution-Weighted Re-Ranking
  7. Titans Test-Time Neural Memorization
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from lthcm.core.superposition import SuperpositionMemoryStore
from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.core.gauge_bundle import FibreBundle
from lthcm.core.lorentzian import LorentzianCausalAttention, LorentzianMetric
from lthcm.core.sheaf import SheafContradictionFilter
from lthcm.core.titans_memory import TitansNeuralMemory
from lthcm.core.diffeomorphic import HamiltonianMetricFlow
from lthcm.core.fisher_rao import FisherRaoMetric
from lthcm.core.stochastic_resonance import StochasticResonanceQueryLayer
from lthcm.core.matryoshka import MatryoshkaCompressor
from lthcm.core.optimal_transport import WassersteinDocumentAligner
from lthcm.core.hdc_vsa import VectorSymbolicArchitecture
from lthcm.core.hyperbolic import HyperbolicHierarchicalStore, PoincareBall
from lthcm.core.spectral_wormhole import SpectralWormholeRefiner
from lthcm.core.persistent_homology import PersistentHomologyGuard
from lthcm.core.riemannian_langevin import RiemannianLangevinMemory, RiemannianMetricTensor
from lthcm.graph.community import GraphRAGEvidenceBuilder


class LTHCMEngine(nn.Module):
    """
    Long-Term High-Capacity Topological Memory Engine.
    Integrates cutting-edge non-Euclidean representation, causality, and test-time learning.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_facets: int = 4,
        c_speed: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_facets = num_facets
        self.c_speed = c_speed
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


        # Core subsystems
        self.superposition = SuperpositionMemoryStore(d_model=d_model, num_facets=num_facets).to(self.device)
        self.wormhole = WormholeGraphIndex(d_model=d_model)
        self.bundle = FibreBundle(d_model=d_model).to(self.device)
        self.causal_attn = LorentzianCausalAttention(d_model=d_model).to(self.device)
        # Note: Model family must remain cross-encoder/nli-deberta-v3-* only (matches label ordering 0=contradiction, 1=entailment, 2=neutral)
        self.sheaf_filter = SheafContradictionFilter(
            d_model=d_model,
            nli_model_name="cross-encoder/nli-deberta-v3-xsmall",
            contradiction_threshold=2.5,
        ).to(self.device)
        self.titans = TitansNeuralMemory(d_key=d_model, d_val=d_model).to(self.device)
        self.metric_flow = HamiltonianMetricFlow(d_model=d_model).to(self.device)
        self.fisher_rao = FisherRaoMetric(d_model=d_model).to(self.device)
        self.stochastic_resonance = StochasticResonanceQueryLayer(d_model=d_model).to(self.device)
        self.matryoshka = MatryoshkaCompressor(d_model=d_model).to(self.device)
        self.wasserstein = WassersteinDocumentAligner(d_model=d_model).to(self.device)
        self.hdc = VectorSymbolicArchitecture(d_hdc=512, d_model=d_model).to(self.device)
        self.hyperbolic = HyperbolicHierarchicalStore(d_model=d_model, curvature=1.0).to(self.device)
        self.spectral_wormhole = SpectralWormholeRefiner(d_model=d_model, K_order=3).to(self.device)
        self.homology_guard = PersistentHomologyGuard(distance_metric="euclidean", significance_ratio=0.20)
        self.riemannian_memory = RiemannianLangevinMemory(
            d_key=d_model,
            d_val=d_model,
            num_heads=4,
            temperature=0.01,
            base_lr=0.05,
            damping=0.1,
            prior_decay=0.001,
        ).to(self.device)
        self.community_evidence_builder = GraphRAGEvidenceBuilder()

        # Domain coordinate codebook (for fibre bundle base manifold B)
        self.domain_codebook: Dict[str, torch.Tensor] = {}

    def get_or_create_domain(self, domain_name: str) -> torch.Tensor:
        """Fetch or initialize a base manifold coordinate for a specific contextual domain."""
        if domain_name not in self.domain_codebook:
            seed = int.from_bytes(hashlib.sha256(domain_name.encode("utf-8")).digest()[:8], "little")
            g = torch.Generator(device="cpu").manual_seed(seed)
            coord = torch.randn(self.d_model, generator=g).to(self.device)
            coord = F.normalize(coord, p=2, dim=0)
            self.domain_codebook[domain_name] = coord
        return self.domain_codebook[domain_name]

    def index_document(
        self,
        doc_id: str,
        content: str,
        facet_vectors: torch.Tensor,
        domain: str = "general",
        timestamp: float = 0.0,
        wormhole_links: Optional[List[Dict[str, Any]]] = None,
        mixture_weights: Optional[torch.Tensor] = None,
    ):
        """
        Ingest a document into the unified topological memory store.
        """
        facet_vectors = facet_vectors.to(self.device)
        if mixture_weights is not None:
            mixture_weights = mixture_weights.to(self.device)

        meta = {
            "domain": domain,
            "timestamp": timestamp,
        }

        # 1. Store in Quantum Superposition Memory
        self.superposition.add_document(
            doc_id=doc_id,
            content=content,
            facet_vectors=facet_vectors,
            mixture_weights=mixture_weights,
            metadata=meta,
        )

        # 2. Add centroid to Wormhole Graph Index
        primary_emb = facet_vectors[0] if facet_vectors.dim() == 2 else facet_vectors[0, 0]
        self.wormhole.add_node(
            node_id=doc_id,
            embedding=primary_emb.cpu(),
            content=content,
            cluster=domain,
            metadata=meta,
        )

        # 3. Add any explicit wormhole shortcuts
        if wormhole_links:
            for link in wormhole_links:
                self.wormhole.add_wormhole(
                    source_id=doc_id,
                    target_id=link["target_id"],
                    relation_type=link.get("relation", "relational_bridge"),
                    permeability=link.get("permeability", 1.0),
                )

        # 4. Index concept into Hyperbolic Hierarchical Store
        self.hyperbolic.add_concept(
            doc_id=doc_id,
            content=content,
            embedding=primary_emb.to(self.device),
            is_hyperbolic=False,
            metadata=meta,
        )

    def index_documents(
        self,
        doc_ids: List[str],
        contents: List[str],
        facet_vectors: torch.Tensor,
        domains: Optional[List[str]] = None,
        timestamps: Optional[List[float]] = None,
        wormhole_links: Optional[List[Dict[str, Any]]] = None,
        mixture_weights: Optional[torch.Tensor] = None,
    ):
        """
        Batch-ingest documents into the unified topological memory store.
        """
        b = len(doc_ids)
        facet_vectors = facet_vectors.to(self.device)
        if mixture_weights is not None:
            mixture_weights = mixture_weights.to(self.device)

        domains = domains or ["general"] * b
        timestamps = timestamps or [0.0] * b
        metas = [{"domain": domains[i], "timestamp": timestamps[i]} for i in range(b)]

        self.superposition.add_documents(
            doc_ids=doc_ids,
            contents=contents,
            facet_vectors=facet_vectors,
            mixture_weights=mixture_weights,
            metadatas=metas,
        )

        primary_embs = facet_vectors[:, 0] if facet_vectors.dim() == 3 else (
            facet_vectors if facet_vectors.dim() == 2 else facet_vectors[:, 0, 0]
        )
        self.wormhole.add_nodes(
            node_ids=doc_ids,
            embeddings=primary_embs.cpu(),
            contents=contents,
            clusters=domains,
            metadatas=metas,
        )

        for i, did in enumerate(doc_ids):
            self.hyperbolic.add_concept(
                doc_id=did,
                content=contents[i],
                embedding=primary_embs[i].to(self.device),
                is_hyperbolic=False,
                metadata=metas[i],
            )

        if wormhole_links:
            for link in wormhole_links:
                self.wormhole.add_wormhole(
                    source_id=link.get("source_id", doc_ids[0]),
                    target_id=link["target_id"],
                    relation_type=link.get("relation", "relational_bridge"),
                    permeability=link.get("permeability", 1.0),
                )

    def get_retriever(self, config: Optional[Any] = None) -> Any:
        """
        Get a unified deterministic retriever instance backed by this engine's graph and subsystems.
        """
        import numpy as np
        from lthcm.engine.retriever import LTHCMRetriever
        retriever = LTHCMRetriever(d_model=self.d_model, device=self.device, engine=self)
        retriever.wormhole = self.wormhole
        if len(self.wormhole.node_embeddings) > 0:
            doc_list = []
            emb_list = []
            for nid, emb in self.wormhole.node_embeddings.items():
                meta = self.wormhole.node_metadata.get(nid, {})
                doc_list.append({
                    "id": nid,
                    "title": meta.get("cluster", nid),
                    "text": meta.get("content", ""),
                })
                emb_list.append(emb.detach().cpu().numpy())

            existing_edges = []
            for u, v, data in self.wormhole.graph.edges(data=True):
                existing_edges.append((u, v, data.get("relation", "bridge"), float(data.get("permeability", 1.0))))

            retriever.index(doc_list, embeddings=np.stack(emb_list), edges=existing_edges if existing_edges else None, config=config)
        return retriever

    def search(
        self,
        query: Union[str, torch.Tensor, Any],
        top_k: int = 5,
        config: Optional[Any] = None,
    ) -> Any:
        """
        Execute deterministic search through the unified Retriever interface.
        """
        retriever = self.get_retriever(config=config)
        return retriever.search(query=query, top_k=top_k, config=config)

    def retrieve_and_verify(
        self,
        query_vector: torch.Tensor,
        query_domain: str = "general",
        query_timestamp: float = 100.0,
        top_k: int = 5,
        enforce_causality: bool = True,
        enforce_sheaf_verification: bool = True,
        update_titans_memory: bool = False,
        use_stochastic_resonance: bool = True,
        use_matryoshka_staged: bool = False,
        use_spectral_wormhole: bool = True,
        compute_wasserstein: bool = False,
        audit_completeness: bool = False,
        use_riemannian_langevin: bool = False,
        query_facets: Optional[torch.Tensor] = None,
        coarse_dim: Optional[int] = None,
        coarse_k: Optional[int] = None,
        use_nli_sheaf: bool = True,
        build_community_summaries: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end topological retrieval and verification.
        """
        q_vec = query_vector.to(self.device)
        q_domain_coord = self.get_or_create_domain(query_domain)

        # --- Phase 1: Quantum POVM Superposition Readout (Staged Matryoshka or Stochastic Resonance) ---
        matryoshka_diag = None
        if use_matryoshka_staged:
            m_res = self.matryoshka.coarse_to_fine_search(
                query=q_vec,
                memory_store=self.superposition,
                coarse_dim=coarse_dim,
                coarse_k=coarse_k or (top_k * 4),
                fine_top_k=top_k * 2,
            )
            superposition_matches = m_res.get("results", [])
            matryoshka_diag = {
                "early_exit": m_res.get("early_exit", False),
                "coarse_dim": m_res.get("coarse_dim"),
                "full_dim": m_res.get("full_dim"),
                "compression_ratio": m_res.get("compression_ratio"),
                "flops_saved_pct": m_res.get("flops_saved_pct"),
            }
        elif use_stochastic_resonance:
            superposition_matches = self.stochastic_resonance.resonate_retrieval(
                query=q_vec,
                memory_store=self.superposition,
                top_k=top_k * 2,
            )
        else:
            superposition_matches = self.superposition.retrieve(q_vec, top_k=top_k * 2)

        # --- Phase 2: Wormhole Graph Tunneling & Spectral Chebyshev Refinement ---
        wormhole_matches = self.wormhole.query(q_vec, top_k=top_k * 2, max_hops=2)

        spectral_diag = None
        if use_spectral_wormhole and len(wormhole_matches) > 1:
            candidate_nids = [w["node_id"] for w in wormhole_matches]
            spectral_res = self.spectral_wormhole.refine_candidates(
                query=q_vec,
                wormhole_index=self.wormhole,
                candidate_ids=candidate_nids,
            )
            refined_dict = {r["node_id"]: r for r in spectral_res.get("results", [])}
            spectral_diag = {
                "refined_count": len(refined_dict),
                "top_spectral_node": spectral_res["results"][0]["node_id"] if spectral_res.get("results") else None,
            }
            for w in wormhole_matches:
                nid = w["node_id"]
                if nid in refined_dict:
                    ref_info = refined_dict[nid]
                    w["spectral_refined_score"] = ref_info["refined_score"]
                    w["spectral_gate"] = ref_info["gate_weight"]
                    # Blend geometric wormhole score with learned spectral polynomial filter
                    w["score"] = 0.5 * w["score"] + 0.5 * ref_info["refined_score"]

        # Merge candidate documents
        candidate_dict: Dict[str, Dict[str, Any]] = {}
        for m in superposition_matches:
            candidate_dict[m["doc_id"]] = {
                "doc_id": m["doc_id"],
                "content": m["content"],
                "score": m["score"],
                "clean_score": m.get("clean_score", m["score"]),
                "resonance_boost": m.get("resonance_boost", 0.0),
                "sr_diagnostics": m.get("sr_diagnostics"),
                "matryoshka_stage": m.get("stage"),
                "vector": m["collapsed_facet_vector"].to(self.device),
                "domain": m["metadata"].get("domain", "general"),
                "timestamp": m["metadata"].get("timestamp", 0.0),
                "source": "superposition_povm",
                "collapsed_facet": m["collapsed_facet_idx"],
            }

        for w in wormhole_matches:
            nid = w["node_id"]
            if nid not in candidate_dict:
                candidate_dict[nid] = {
                    "doc_id": nid,
                    "content": w["content"],
                    "score": w["score"] * 0.9,
                    "spectral_refined_score": w.get("spectral_refined_score"),
                    "spectral_gate": w.get("spectral_gate"),
                    "vector": w["embedding"].to(self.device),
                    "domain": w["cluster"],
                    "timestamp": self.wormhole.node_metadata[nid].get("timestamp", 0.0),
                    "source": "wormhole_tunnel",
                    "path": w["path"],
                }
            elif "spectral_refined_score" in w:
                candidate_dict[nid]["spectral_refined_score"] = w["spectral_refined_score"]
                candidate_dict[nid]["spectral_gate"] = w.get("spectral_gate")

        raw_candidates = list(candidate_dict.values())
        if len(raw_candidates) == 0:
            return {"status": "empty", "verified_context": []}

        # --- Phase 3: Lie Group Gauge Fibre Parallel Transport ---
        for cand in raw_candidates:
            cand_domain_coord = self.get_or_create_domain(cand["domain"])
            aligned_v, alignment_sim = self.bundle.align_to_query_frame(
                candidate_vector=cand["vector"],
                candidate_domain=cand_domain_coord,
                query_vector=q_vec,
                query_domain=q_domain_coord,
            )
            cand["aligned_vector"] = aligned_v
            cand["gauge_sim"] = alignment_sim

        # --- Phase 4: Lorentzian Spacetime Causal Light-Cone Filtering ---
        causally_valid_candidates = []
        causal_diagnostics = []

        if enforce_causality:
            c_speed = getattr(self, "c_speed", 1.0)
            for cand in raw_candidates:
                cand_t = float(cand.get("timestamp", 0.0))
                dt = query_timestamp - cand_t
                is_past = dt > 0.0

                # Compute normalized spatial separation in gauge-aligned frame (dx in [0, 2])
                q_unit = F.normalize(q_vec.view(-1), p=2, dim=0)
                cand_unit = F.normalize(cand["aligned_vector"].view(-1), p=2, dim=0)
                dx_norm = float(torch.norm(q_unit - cand_unit, p=2).item())
                # Minkowski interval: ds^2 = - (c * dt)^2 + dx^2
                ds2 = - (c_speed * dt) ** 2 + (dx_norm ** 2)
                is_timelike = ds2 <= 0.0
                in_light_cone = is_past and is_timelike


                causal_diagnostics.append({
                    "doc_id": cand["doc_id"],
                    "ds2": float(ds2),
                    "dt": float(dt),
                    "dx": float(dx_norm),
                    "in_light_cone": in_light_cone,
                })

                if in_light_cone:
                    causally_valid_candidates.append(cand)
        else:
            causally_valid_candidates = raw_candidates

        causal_fallback = False
        if enforce_causality and len(causally_valid_candidates) == 0 and len(raw_candidates) > 0:
            causal_fallback = True
            candidates_for_sheaf = raw_candidates[:top_k]
        else:
            candidates_for_sheaf = causally_valid_candidates

        # --- Phase 5: Cellular Sheaf Cohomological Verification ---
        if enforce_sheaf_verification and len(candidates_for_sheaf) > 1:
            stacked_embs = torch.stack([c["aligned_vector"] for c in candidates_for_sheaf])
            sheaf_eval = self.sheaf_filter.evaluate_passages(
                stacked_embs, candidates_for_sheaf, use_nli=use_nli_sheaf
            )
            verified_candidates = sheaf_eval["verified_passages"]
            sheaf_report = {
                "is_consistent": sheaf_eval["is_consistent"],
                "obstruction_energy": sheaf_eval["total_obstruction"],
                "conflicting_pairs": sheaf_eval["conflicting_pairs"],
                "collapsed": sheaf_eval.get("collapsed", False),
                "fallback_used": sheaf_eval.get("fallback_used", False),
                "used_nli": use_nli_sheaf and self.sheaf_filter.nli_model is not None,
            }
        else:
            verified_candidates = candidates_for_sheaf
            sheaf_report = {
                "is_consistent": True,
                "obstruction_energy": 0.0,
                "conflicting_pairs": [],
                "collapsed": False,
                "fallback_used": False,
                "used_nli": use_nli_sheaf and self.sheaf_filter.nli_model is not None,
            }

        # --- Phase 6: Fisher-Rao & Wasserstein Distribution-Weighted Re-Ranking ---
        # Fisher-Rao replaces naive cosine with uncertainty-aware scoring
        if len(verified_candidates) > 0:
            cand_vectors = torch.stack([c["aligned_vector"] for c in verified_candidates])
            fisher_scores = self.fisher_rao.similarity(q_vec, cand_vectors)
            for i, cand in enumerate(verified_candidates):
                cand["fisher_rao_score"] = float(fisher_scores[i].item())

            # Optional: Wasserstein Document Alignment (Entropic Optimal Transport)
            if compute_wasserstein and self.superposition.facet_tensor.shape[0] > 0:
                q_f = query_facets.to(self.device) if query_facets is not None else q_vec.unsqueeze(0)
                for cand in verified_candidates:
                    doc_idx = self.superposition.doc_index.get(cand["doc_id"])
                    if doc_idx is not None and doc_idx < self.superposition.facet_tensor.shape[0]:
                        cand_facets = self.superposition.facet_tensor[doc_idx].to(self.device)
                        cand_weights = self.superposition.weight_tensor[doc_idx].to(self.device) if self.superposition.weight_tensor is not None else None
                        w_dist, _ = self.wasserstein.pairwise_distance(q_f, cand_facets, weights_b=cand_weights)
                        w_sim = float(torch.exp(-w_dist / (self.wasserstein.tau + 1e-6)).item())
                        cand["wasserstein_distance"] = float(w_dist.item())
                        cand["wasserstein_similarity"] = w_sim

            verified_candidates.sort(key=lambda x: x.get("fisher_rao_score", x.get("score", 0.0)), reverse=True)
        final_top = verified_candidates[:top_k]

        # Optional: GraphRAG Community Evidence Synthesis
        community_evidence = None
        if build_community_summaries and len(final_top) > 1:
            p_dicts = [{"id": c["doc_id"], "text": c["content"]} for c in final_top]
            p_embs = torch.stack([c["aligned_vector"] for c in final_top]).cpu().numpy()
            community_evidence = self.community_evidence_builder.build_evidence(
                passages=p_dicts, passage_embeddings=p_embs
            )

        # --- Phase 7: Persistent Homology & Betti Filtration Epistemic Guard ---
        epistemic_audit = None
        if audit_completeness and len(verified_candidates) > 0:
            cand_embs = torch.stack([c["aligned_vector"] for c in verified_candidates])
            cand_ids = [c["doc_id"] for c in verified_candidates]
            epistemic_audit = self.homology_guard.audit_retrieved_context(cand_embs, doc_ids=cand_ids)

        # --- Phase 8: Titans / Riemannian Langevin Test-Time Neural Memorization ---
        titans_surprise = 0.0
        riemannian_langevin_diag = None
        if (update_titans_memory or use_riemannian_langevin) and len(final_top) > 0:
            top_cand = final_top[0]
            if use_riemannian_langevin:
                _, surprise_tensor, riemannian_langevin_diag = self.riemannian_memory.step(
                    k=q_vec.unsqueeze(0),
                    v=top_cand["aligned_vector"].unsqueeze(0),
                )
                titans_surprise = float(surprise_tensor.item())
            else:
                _, surprise_tensor = self.titans.step(
                    k=q_vec.unsqueeze(0),
                    v=top_cand["aligned_vector"].unsqueeze(0),
                )
                titans_surprise = float(surprise_tensor.item())

        return {
            "query": {
                "domain": query_domain,
                "timestamp": query_timestamp,
            },
            "stochastic_resonance_active": use_stochastic_resonance and not use_matryoshka_staged,
            "matryoshka_staged_active": use_matryoshka_staged,
            "matryoshka_diagnostics": matryoshka_diag,
            "spectral_wormhole_active": use_spectral_wormhole,
            "spectral_diagnostics": spectral_diag,
            "wasserstein_alignment_active": compute_wasserstein,
            "persistent_homology_active": audit_completeness,
            "epistemic_audit": epistemic_audit,
            "riemannian_langevin_active": use_riemannian_langevin,
            "riemannian_langevin_diagnostics": riemannian_langevin_diag,
            "candidates_count": len(raw_candidates),
            "causally_valid_count": len(causally_valid_candidates),
            "causal_fallback": causal_fallback,
            "causal_diagnostics": causal_diagnostics,
            "sheaf_fallback": sheaf_report.get("fallback_used", False),
            "sheaf_verification": sheaf_report,
            "community_evidence": community_evidence,
            "titans_surprise": titans_surprise,
            "results": final_top,
        }

    def audit_knowledge_completeness(
        self,
        embeddings: torch.Tensor,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Audit arbitrary candidate passage embeddings for persistent topological voids
        and epistemic fragmentation (Betti numbers beta_0, beta_1).
        """
        return self.homology_guard.audit_retrieved_context(embeddings, doc_ids=doc_ids)

    def retrieve_hyperbolic(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        is_hyperbolic: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Query the Poincaré Ball Hyperbolic Hierarchical Store.
        Retrieves concepts minimizing hyperbolic geodesic distance d_B(q, x).
        """
        return self.hyperbolic.retrieve(query.to(self.device), top_k=top_k, is_hyperbolic=is_hyperbolic)

    def align_documents_wasserstein(
        self,
        doc_id_a: str,
        doc_id_b: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal transport distance and alignment plan between two indexed multi-facet documents.
        """
        idx_a = self.superposition.doc_index[doc_id_a]
        idx_b = self.superposition.doc_index[doc_id_b]
        facets_a = self.superposition.facet_tensor[idx_a].to(self.device)
        facets_b = self.superposition.facet_tensor[idx_b].to(self.device)
        weights_a = self.superposition.weight_tensor[idx_a].to(self.device) if self.superposition.weight_tensor is not None else None
        weights_b = self.superposition.weight_tensor[idx_b].to(self.device) if self.superposition.weight_tensor is not None else None
        return self.wasserstein.pairwise_distance(facets_a, facets_b, weights_a=weights_a, weights_b=weights_b)

    def create_symbolic_proposition(
        self,
        predicate: str,
        subject: Union[str, torch.Tensor],
        object_: Union[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Construct an algebraically exact symbolic proposition using Vector Symbolic Architecture (HDC):
            P = PREDICATE (*) (Pi^0(SUBJECT) + Pi^1(OBJECT))
        """
        return self.hdc.create_proposition(predicate, subject, object_)

    def extract_symbolic_role(
        self,
        proposition: torch.Tensor,
        predicate: str,
        role_index: int = 0,
    ) -> torch.Tensor:
        """
        Unbind and recover an argument from an HDC symbolic proposition.
        """
        return self.hdc.extract_role_filler(proposition.to(self.device), predicate, role_index=role_index)

    def bind_symbolic(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """HDC circular convolution binding a (*) b."""
        return self.hdc.bind(a.to(self.device), b.to(self.device))

    def unbind_symbolic(self, bound: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """HDC circular correlation unbinding bound (*)~ key."""
        return self.hdc.unbind(bound.to(self.device), key.to(self.device))

    def bundle_symbolic(self, vectors: List[torch.Tensor], weights: Optional[List[float]] = None) -> torch.Tensor:
        """HDC bundling / superposition of hypervectors."""
        vecs = [v.to(self.device) for v in vectors]
        return self.hdc.bundle(vecs, weights=weights)

    def compute_causal_attention_mask(
        self,
        query_spacetime: torch.Tensor,
        key_spacetime: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Lorentzian causal metric intervals and attention mask ds^2 <= 0.
        """
        return self.causal_attn.compute_causal_mask(query_spacetime.to(self.device), key_spacetime.to(self.device))

    def step_hamiltonian_flow(
        self,
        candidates: torch.Tensor,
        query: torch.Tensor,
        steps: int = 4,
        total_time: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simulate continuous diffeomorphic Hamiltonian phase-space metric flow.
        Deforms candidate coordinates along geodesic flow lines toward the query manifold.
        """
        return self.metric_flow.warp(candidates.to(self.device), query.to(self.device), steps=steps, total_time=total_time)

    def create_functorial_pipeline(
        self,
        include_causality: bool = True,
        include_fisher_rao: bool = True,
        include_persistent_homology: bool = True,
    ) -> Any:
        """
        Construct a formally verified, composable functorial pipeline from engine subsystems.
        Stages are morphisms in a symmetric monoidal category satisfying associativity and functoriality.
        """
        from lthcm.core.categorical import (
            FunctorialPipeline,
            CausalFilterMorphism,
            FisherRaoRankingMorphism,
            PersistentHomologyAuditMorphism,
        )
        pipe = FunctorialPipeline()
        if include_causality:
            pipe.append(CausalFilterMorphism(c_speed=self.c_speed))
        if include_fisher_rao:
            pipe.append(FisherRaoRankingMorphism(self.fisher_rao))
        if include_persistent_homology:
            pipe.append(PersistentHomologyAuditMorphism(self.homology_guard))
        return pipe

    def step_riemannian_langevin_memory(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Execute test-time memory consolidation governed by Riemannian Langevin Dynamics.
        Emergent forgetting naturally erodes stale representations along flat directions,
        while high-curvature memory manifolds are preserved.
        """
        return self.riemannian_memory.step(key.to(self.device), value.to(self.device))

    def retrieve_riemannian_langevin_memory(
        self,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query test-time fast weights from the Riemannian Langevin memory store.
        """
        return self.riemannian_memory.retrieve(query.to(self.device))

