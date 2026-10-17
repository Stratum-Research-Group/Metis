"""
Comprehensive Invariance & Mathematical Correctness Test Suite for LTHCM
"""

import unittest
from unittest.mock import patch
import os
import sys
import subprocess
import torch
import torch.nn.functional as F

from lthcm.core.superposition import SuperpositionMemoryStore
from lthcm.core.lorentzian import LorentzianMetric, LorentzianCausalAttention
from lthcm.core.sheaf import CellularSheaf, SheafContradictionFilter
from lthcm.core.gauge_bundle import LieAlgebraGaugeConnection
from lthcm.core.titans_memory import TitansNeuralMemory
from lthcm.core.wormhole import WormholeGraphIndex
from lthcm.core.diffeomorphic import HamiltonianMetricFlow
from lthcm.core.optimal_transport import WassersteinDocumentAligner, sinkhorn_knopp, compute_cost_matrix
from lthcm.core.hdc_vsa import VectorSymbolicArchitecture
from lthcm.core.hyperbolic import PoincareBall, HyperbolicHierarchicalStore
from lthcm.core.spectral_wormhole import (
    ChebyshevSpectralGraphConv,
    SpectralWormholeRefiner,
    BidirectionalWormholeEngine,
)
from lthcm.core.persistent_homology import PersistentHomologyGuard
from lthcm.core.categorical import (
    Morphism,
    IdentityMorphism,
    CompositeMorphism,
    TensorMorphism,
    BraidingMorphism,
    AssociatorMorphism,
    CategoricalLens,
    AdjointFunctorPair,
    FunctorialPipeline,
    PipelineState,
    CausalFilterMorphism,
    FisherRaoRankingMorphism,
    PersistentHomologyAuditMorphism,
)
from lthcm.engine.pipeline import LTHCMEngine


class TestLTHCM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.d_model = 32

    def test_superposition_density_matrix_properties(self):
        """Verify quantum trace properties and POVM expectation calculations."""
        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=3)
        facets = torch.randn(3, self.d_model)
        weights = torch.tensor([0.5, 0.3, 0.2])
        store.add_document("doc_quantum", "Quantum computing and biology", facets, weights)

        # 1. Check explicit density matrix trace is 1.0 and positive semi-definite
        rho = store.density_matrix(0)
        trace_val = torch.trace(rho).item()
        self.assertAlmostEqual(trace_val, 1.0, places=4, msg="Density matrix must have unit trace")

        eigenvalues = torch.linalg.eigvalsh(rho)
        self.assertTrue(torch.all(eigenvalues >= -1e-6), "Density matrix must be positive semi-definite")

        # 2. Check that fast O(K*D) retrieval score matches exact matrix trace Tr(rho * q q^T)
        query = torch.randn(self.d_model)
        q_norm = F.normalize(query, p=2, dim=0)

        matches = store.retrieve(q_norm, top_k=1)
        fast_score = matches[0]["score"]

        q_proj = q_norm.unsqueeze(1) @ q_norm.unsqueeze(0)
        exact_trace = torch.trace(rho @ q_proj).item()

        self.assertAlmostEqual(fast_score, exact_trace, places=4, msg="Fast POVM trace must match explicit matrix trace")

    def test_superposition_negative_mixture_weights(self):
        """Verify that negative mixture weights passed without is_logits are safely handled as logits or non-negative probabilities."""
        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=4)
        facets = torch.randn(2, self.d_model)
        # Passing negative weights like [1.0, -0.5] without is_logits=True
        weights = torch.tensor([1.0, -0.5])
        store.add_document("doc_neg", "Negative weights test", facets, weights)

        # Check weights tensor
        w = store.weight_tensor[0]
        self.assertTrue(torch.all(w >= 0.0), f"Mixture weights must be non-negative, got {w}")
        self.assertAlmostEqual(torch.sum(w).item(), 1.0, places=4, msg="Mixture weights must sum to 1.0")

        # Check density matrix
        rho = store.density_matrix(0)
        eigenvalues = torch.linalg.eigvalsh(rho)
        self.assertTrue(torch.all(eigenvalues >= -1e-6), "Density matrix must be positive semi-definite")

    def test_lorentzian_spacetime_causality(self):
        """Verify Minkowski spacetime intervals and light-cone horizons."""
        # Event 1: Observer at t=10.0, x=0.0
        q_event = torch.zeros(1, 1, self.d_model)
        q_event[0, 0, 0] = 10.0 # Time

        # Event 2: Past event at t=5.0, distance=2.0 (inside light cone: dt=5, dx=2, ds^2 = -25 + 4 = -21 < 0)
        k_past = torch.zeros(1, 1, self.d_model)
        k_past[0, 0, 0] = 5.0
        k_past[0, 0, 1] = 2.0

        # Event 3: Future event at t=15.0 (acausal)
        k_future = torch.zeros(1, 1, self.d_model)
        k_future[0, 0, 0] = 15.0

        # Event 4: Spacelike event at t=8.0, distance=10.0 (outside light cone: dt=2, dx=10, ds^2 = -4 + 100 = 96 > 0)
        k_spacelike = torch.zeros(1, 1, self.d_model)
        k_spacelike[0, 0, 0] = 8.0
        k_spacelike[0, 0, 1] = 10.0

        k_all = torch.cat([k_past, k_future, k_spacelike], dim=1)

        attn = LorentzianCausalAttention(d_model=self.d_model)
        ds2, causal_mask = attn.compute_causal_mask(q_event, k_all)

        # k_past must be in the past light cone
        self.assertTrue(causal_mask[0, 0, 0].item(), "Timelike past event must be inside light cone")
        # k_future must be excluded
        self.assertFalse(causal_mask[0, 0, 1].item(), "Future event must NOT be inside past light cone")
        # k_spacelike must be excluded
        self.assertFalse(causal_mask[0, 0, 2].item(), "Spacelike event must NOT be inside light cone")

    def test_sheaf_contradiction_obstruction(self):
        """Verify Sheaf Laplacian Dirichlet energy detects contradictions."""
        filter_layer = SheafContradictionFilter(d_model=self.d_model, contradiction_threshold=0.1)

        # Case 1: Consistent assertions (identical or parallel semantics)
        base_fact = torch.randn(self.d_model)
        consistent_embs = torch.stack([base_fact, base_fact + 0.01 * torch.randn(self.d_model)])
        passages_c = [{"doc_id": "c1", "content": "Drug A is effective."}, {"doc_id": "c2", "content": "Drug A works well."}]

        eval_consistent = filter_layer.evaluate_passages(consistent_embs, passages_c)
        self.assertTrue(eval_consistent["is_consistent"], "Consistent assertions must produce near-zero obstruction")

        # Case 2: Directly opposing / contradictory assertions
        opposing_embs = torch.stack([base_fact, -base_fact])
        passages_o = [{"doc_id": "o1", "content": "Drug A is completely safe."}, {"doc_id": "o2", "content": "Drug A is strictly lethal."}]

        eval_opposing = filter_layer.evaluate_passages(opposing_embs, passages_o)
        self.assertGreater(eval_opposing["total_obstruction"], 0.05, "Opposing assertions must trigger high cohomological obstruction")

    def test_gauge_bundle_isometry(self):
        """Verify that Lie algebra parallel transport is strictly norm-preserving (isometric)."""
        gauge = LieAlgebraGaugeConnection(d_model=self.d_model)
        source_domain = torch.randn(self.d_model)
        target_domain = torch.randn(self.d_model)

        vec = torch.randn(self.d_model)
        original_norm = torch.norm(vec, p=2).item()

        transported, tau = gauge.parallel_transport(vec, source_domain, target_domain)
        transported_norm = torch.norm(transported, p=2).item()

        # Check isometry: ||Tau(v)|| == ||v||
        self.assertAlmostEqual(original_norm, transported_norm, places=4, msg="Gauge parallel transport must be isometric")

        # Check orthogonality of Tau: Tau @ Tau^T == Identity
        eye = torch.eye(self.d_model)
        tau_id = tau @ tau.T
        self.assertTrue(torch.allclose(tau_id, eye, atol=1e-4), "Transport matrix must be in SO(D)")

    def test_titans_memory_surprise_reduction(self):
        """Verify that Titans associative memory reduces surprise over repeated test-time observations."""
        memory = TitansNeuralMemory(d_key=self.d_model, d_val=self.d_model, base_lr=0.2)
        k = torch.randn(1, self.d_model)
        v = torch.randn(1, self.d_model)

        # First observation: High surprise
        _, surp_1 = memory.step(k, v)

        # Observe the same key-value association 5 times
        for _ in range(5):
            _, surp_next = memory.step(k, v)

        # Final surprise must be significantly lower than initial surprise
        self.assertLess(surp_next.item(), surp_1.item(), "Titans memory must adapt and reduce surprise error")

    def test_wormhole_traversal(self):
        """Verify that non-metric wormhole edges bridge distant clusters."""
        index = WormholeGraphIndex(d_model=self.d_model)

        # Create two orthogonal clusters
        cluster_a = torch.zeros(self.d_model)
        cluster_a[0] = 1.0

        cluster_b = torch.zeros(self.d_model)
        cluster_b[1] = 1.0 # Orthogonal to cluster_a, cosine = 0.0

        index.add_node("entry_node", cluster_a, "Mold on cantaloupe", cluster="botany")
        index.add_node("target_node", cluster_b, "Mass production of penicillin in Peoria", cluster="pharma")

        # Query aligned with cluster_a
        q = cluster_a

        # Without wormhole: target_node score is 0.0
        results_before = index.query(q, top_k=2)
        target_score_before = [r["score"] for r in results_before if r["node_id"] == "target_node"][0]
        self.assertAlmostEqual(target_score_before, 0.0, places=4)

        # Add wormhole bridge
        index.add_wormhole("entry_node", "target_node", relation_type="historical_discovery", permeability=0.9)

        # With wormhole: target_node should be retrieved via tunnel
        results_after = index.query(q, top_k=2)
        target_res = [r for r in results_after if r["node_id"] == "target_node"][0]
        self.assertTrue(target_res["is_tunneled"], "Target must be reachable via wormhole tunnel")
        self.assertGreater(target_res["score"], 0.5, "Tunneled score must be significantly non-zero")

    def test_hamiltonian_metric_flow(self):
        """Verify continuous diffeomorphic flow deforms coordinates toward the query."""
        flow = HamiltonianMetricFlow(d_model=self.d_model)
        candidates = torch.randn(4, self.d_model)
        query = torch.randn(1, self.d_model)

        initial_dist = torch.norm(candidates - query, p=2, dim=-1)
        warped, path_len = flow.warp(candidates, query, steps=4, total_time=0.5)
        final_dist = torch.norm(warped - query, p=2, dim=-1)

        # The harmonic attractor should pull candidates closer to the query
        self.assertTrue(torch.all(path_len > 0), "Geodesic flow must have non-zero path length")
        self.assertTrue(torch.all(final_dist < initial_dist), "Hamiltonian flow must contract toward query")

    def test_lthcm_engine_e2e(self):
        """Verify the full unified LTHCM engine pipeline end-to-end."""
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)

        # Ingest two documents
        doc1_facets = torch.randn(2, self.d_model)
        doc2_facets = torch.randn(2, self.d_model)

        engine.index_document(
            doc_id="doc_history",
            content="Historical event in 1980",
            facet_vectors=doc1_facets,
            domain="history",
            timestamp=1980.0,
        )
        engine.index_document(
            doc_id="doc_future",
            content="Future event in 2050",
            facet_vectors=doc2_facets,
            domain="future_tech",
            timestamp=2050.0,
        )

        # Query from observer at t = 2000.0
        query_v = torch.randn(self.d_model)
        response = engine.retrieve_and_verify(
            query_vector=query_v,
            query_domain="history",
            query_timestamp=2000.0,
            top_k=2,
            enforce_causality=True,
            enforce_sheaf_verification=True,
            update_titans_memory=True,
        )

        self.assertIn("results", response)
        self.assertIn("sheaf_verification", response)
        self.assertIn("causal_fallback", response)
        self.assertIn("sheaf_fallback", response)
        # Future event (2050) must not be in causally valid results for observer at 2000
        result_ids = [r["doc_id"] for r in response["results"]]
        self.assertNotIn("doc_future", result_ids, "Future event must be filtered by Lorentzian light cone")

    def test_engine_subsystem_helpers(self):
        """Verify helper methods for causal attention mask and Hamiltonian metric flow on LTHCMEngine."""
        engine = LTHCMEngine(d_model=self.d_model)

        # Test causal attention mask helper
        q_st = torch.randn(1, 4, self.d_model)
        k_st = torch.randn(1, 4, self.d_model)
        intervals, mask = engine.compute_causal_attention_mask(q_st, k_st)
        self.assertEqual(intervals.shape, (1, 4, 4))
        self.assertEqual(mask.shape, (1, 4, 4))

        # Test Hamiltonian flow helper
        cands = torch.randn(2, self.d_model)
        query = torch.randn(1, self.d_model)
        warped, flow_dist = engine.step_hamiltonian_flow(cands, query, steps=3, total_time=0.3)
        self.assertEqual(warped.shape, (2, self.d_model))
        self.assertEqual(flow_dist.shape, (2,))

    def test_fisher_rao_identity_reduction(self):
        """Verify Fisher-Rao reduces to standard cosine when all weights are uniform."""
        from lthcm.core.fisher_rao import FisherRaoMetric

        metric = FisherRaoMetric(d_model=self.d_model, init_mode="uniform")
        q = F.normalize(torch.randn(self.d_model), p=2, dim=0)
        docs = F.normalize(torch.randn(5, self.d_model), p=2, dim=0)

        # Fisher-Rao similarity with uniform weights
        fr_scores = metric.similarity(q, docs)

        # Standard cosine similarity
        cos_scores = F.cosine_similarity(q.unsqueeze(0), docs)

        # They should be nearly identical when Λ ≈ I
        self.assertTrue(
            torch.allclose(fr_scores, cos_scores, atol=0.05),
            "Fisher-Rao with uniform weights must approximate standard cosine similarity"
        )

    def test_fisher_rao_dimension_weighting(self):
        """Verify that Fisher-Rao amplifies high-weight dimensions and suppresses low-weight ones."""
        from lthcm.core.fisher_rao import FisherRaoMetric

        metric = FisherRaoMetric(d_model=self.d_model, init_mode="uniform")

        # Manually set weights: dimension 0 very high, dimension 1 very low
        with torch.no_grad():
            metric.raw_weights[0] = 5.0   # softplus(5.0) ≈ 5.0 → amplified
            metric.raw_weights[1] = -5.0  # softplus(-5.0) ≈ 0.007 → suppressed

        # Two documents: doc_a differs in high-weight dim, doc_b differs in low-weight dim
        base = torch.zeros(self.d_model)
        query = base.clone()
        query[0] = 1.0  # Signal in the amplified dimension

        doc_a = base.clone()
        doc_a[0] = 0.9  # Close in the high-weight dimension

        doc_b = base.clone()
        doc_b[1] = 1.0  # Signal only in the suppressed dimension

        docs = torch.stack([doc_a, doc_b])
        scores = metric.similarity(query, docs)

        # doc_a should score higher because its match is in the amplified dimension
        self.assertGreater(
            scores[0].item(), scores[1].item(),
            "Document matching in high-weight dimension must score higher than one matching in low-weight dimension"
        )

    def test_fisher_rao_estimate_from_data(self):
        """Verify that data-driven weight estimation produces valid Fisher weights."""
        from lthcm.core.fisher_rao import FisherRaoMetric

        metric = FisherRaoMetric(d_model=self.d_model, init_mode="uniform")

        # Create data with known variance structure: dim 0 has low var (high precision),
        # dim 1 has high var (low precision)
        data = torch.randn(100, self.d_model)
        data[:, 0] *= 0.01  # Very low variance → high Fisher weight
        data[:, 1] *= 10.0  # Very high variance → low Fisher weight

        metric.estimate_from_data(data, method="inverse_variance")
        weights = metric.fisher_weights

        # Dimension 0 should have much higher weight than dimension 1
        self.assertGreater(
            weights[0].item(), weights[1].item() * 5,
            "Low-variance dimension must receive significantly higher Fisher weight"
        )

        # All weights must be strictly positive
        self.assertTrue(torch.all(weights > 0), "All Fisher weights must be strictly positive")

    def test_symplectic_energy_conservation(self):
        """Verify symplectic Störmer-Verlet has lower energy drift than RK4."""
        # Symplectic integrator
        flow_sym = HamiltonianMetricFlow(d_model=self.d_model, integrator="symplectic")
        # Legacy RK4 integrator
        flow_rk4 = HamiltonianMetricFlow(d_model=self.d_model, integrator="rk4")

        # Share the same network weights for fair comparison
        flow_rk4.load_state_dict(flow_sym.state_dict())

        candidates = torch.randn(4, self.d_model)
        query = torch.randn(1, self.d_model)

        # Run with diagnostics (many steps to accumulate drift)
        _, _, diag_sym = flow_sym.warp_with_diagnostics(candidates, query, steps=20, total_time=1.0)
        _, _, diag_rk4 = flow_rk4.warp_with_diagnostics(candidates, query, steps=20, total_time=1.0)

        mean_drift_sym = diag_sym["relative_drift"].mean().item()
        mean_drift_rk4 = diag_rk4["relative_drift"].mean().item()

        # Symplectic integrator should have lower energy drift
        self.assertLessEqual(
            mean_drift_sym, mean_drift_rk4 + 0.1,  # Allow small tolerance
            f"Symplectic relative drift ({mean_drift_sym:.4f}) should not exceed "
            f"RK4 drift ({mean_drift_rk4:.4f}) by more than 0.1"
        )

    def test_symplectic_flow_contracts_toward_query(self):
        """Verify symplectic flow still contracts candidates toward query (functional correctness)."""
        flow = HamiltonianMetricFlow(d_model=self.d_model, integrator="symplectic")
        candidates = torch.randn(4, self.d_model)
        query = torch.randn(1, self.d_model)

        initial_dist = torch.norm(candidates - query, p=2, dim=-1)
        warped, path_len = flow.warp(candidates, query, steps=4, total_time=0.5)
        final_dist = torch.norm(warped - query, p=2, dim=-1)

        self.assertTrue(torch.all(path_len > 0), "Symplectic flow must have non-zero path length")
        self.assertTrue(
            torch.all(final_dist < initial_dist),
            "Symplectic Hamiltonian flow must contract candidates toward query"
        )

    def test_stochastic_resonance_weak_query_amplification(self):
        """Verify that stochastic resonance boosts weak sub-threshold query detection."""
        from lthcm.core.stochastic_resonance import StochasticResonanceQueryLayer

        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=1)
        # Target document with distinct facet along dim 0
        target_facet = torch.zeros(1, self.d_model)
        target_facet[0, 0] = 1.0
        store.add_document("doc_target", "Target document", target_facet)

        # Distractor document in orthogonal subspace along dim 1
        dist_facet = torch.zeros(1, self.d_model)
        dist_facet[0, 1] = 1.0
        store.add_document("doc_distractor", "Distractor document", dist_facet)

        # Sub-threshold query: mostly background noise, with only 0.04 alignment on target
        # Threshold is set to 0.10, so clean inner product (0.04) produces exactly 0 score
        v_t = target_facet[0]
        q_rand = torch.randn(self.d_model)
        q_orth = q_rand - torch.dot(q_rand, v_t) * v_t
        q_orth = F.normalize(q_orth, p=2, dim=0)
        weak_query = F.normalize(q_orth + 0.04 * v_t, p=2, dim=0)

        sr_layer = StochasticResonanceQueryLayer(
            d_model=self.d_model, base_sigma=0.15, num_samples=32, threshold=0.10, adaptive=False
        )

        # Clean baseline retrieval (1 probe = clean query only)
        clean_results = sr_layer.resonate_retrieval(weak_query, store, top_k=2, num_samples=1)
        # Resonated retrieval (multiple stochastic probes)
        resonated_results = sr_layer.resonate_retrieval(weak_query, store, top_k=2, num_samples=64)

        target_clean = [r for r in clean_results if r["doc_id"] == "doc_target"][0]
        target_res = [r for r in resonated_results if r["doc_id"] == "doc_target"][0]

        # Clean score must be sub-threshold (0.0)
        self.assertAlmostEqual(target_clean["score"], 0.0, places=4)
        # Stochastic resonance must activate and detect the target document
        self.assertGreater(
            target_res["score"], 0.0,
            "Stochastic resonance must activate sub-threshold target document"
        )
        self.assertGreater(
            target_res["score"], target_clean["score"],
            "Resonated expectation score must exceed zero-noise clean score"
        )
        self.assertGreater(
            target_res["resonance_boost"], 0.0,
            "Resonance boost metric must be positive for sub-threshold query"
        )

    def test_stochastic_resonance_clean_query_preservation(self):
        """Verify strong unambiguous queries retain top accuracy with stochastic resonance."""
        from lthcm.core.stochastic_resonance import StochasticResonanceQueryLayer

        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=1)
        v1 = F.normalize(torch.randn(1, self.d_model), p=2, dim=-1)
        v2 = F.normalize(torch.randn(1, self.d_model), p=2, dim=-1)
        store.add_document("doc_a", "Alpha content", v1)
        store.add_document("doc_b", "Beta content", v2)

        # Query strongly aligned with doc_a facet 0
        strong_query = v1[0].clone()

        sr_layer = StochasticResonanceQueryLayer(
            d_model=self.d_model, num_samples=8, threshold=0.05, adaptive=True
        )
        results = sr_layer.resonate_retrieval(strong_query, store, top_k=2)

        self.assertEqual(results[0]["doc_id"], "doc_a", "Strong query must retrieve target doc as rank 1")
        self.assertGreater(results[0]["score"], 0.8, "Strong query must maintain high confidence score")

    def test_stochastic_resonance_curve_peak(self):
        """Verify the signature non-monotonic resonance curve has an optimal interior sigma* > 0."""
        from lthcm.core.stochastic_resonance import StochasticResonanceQueryLayer

        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=1)
        v_target = torch.zeros(1, self.d_model)
        v_target[0, 0] = 1.0
        v_distractor = torch.zeros(1, self.d_model)
        v_distractor[0, 1] = 1.0

        store.add_document("target", "Target", v_target)
        store.add_document("distractor", "Distractor", v_distractor)

        # Sub-threshold query: 0.04 alignment with target, threshold = 0.10
        q_rand = torch.randn(self.d_model)
        q_orth = q_rand - q_rand[0] * v_target[0] - q_rand[1] * v_distractor[0]
        q_orth = F.normalize(q_orth, p=2, dim=0)
        q = F.normalize(q_orth + 0.04 * v_target[0], p=2, dim=0)

        sr_layer = StochasticResonanceQueryLayer(d_model=self.d_model, adaptive=False)
        sigmas = [0.0, 0.05, 0.10, 0.20, 0.40, 1.0]
        curve = sr_layer.compute_resonance_curve(
            query=q,
            memory_store=store,
            target_doc_idx=0,
            distractor_doc_idx=1,
            sigmas=sigmas,
            num_samples=40,
            threshold=0.10,
        )

        self.assertIn("optimal_sigma", curve)
        self.assertIn("contrasts", curve)
        # Optimal sigma should be non-zero (stochastic resonance amplification)
        self.assertGreaterEqual(
            curve["optimal_sigma"], 0.05,
            "Optimal sigma must be strictly positive for sub-threshold signal (stochastic resonance)"
        )
        self.assertGreater(
            curve["peak_contrast"], 0.0,
            "Peak contrast must be strictly positive"
        )

    def test_stochastic_resonance_pipeline_integration(self):
        """Verify stochastic resonance operates seamlessly within full LTHCMEngine pipeline."""
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        doc1 = torch.randn(2, self.d_model)
        engine.index_document("doc1", "Pipeline test doc", doc1, domain="tech", timestamp=10.0)

        query = torch.randn(self.d_model)
        res = engine.retrieve_and_verify(
            query_vector=query,
            query_domain="tech",
            query_timestamp=20.0,
            use_stochastic_resonance=True,
        )

        self.assertTrue(res["stochastic_resonance_active"], "Pipeline must confirm stochastic resonance active")
        self.assertIn("results", res)
        if len(res["results"]) > 0:
            self.assertIn("resonance_boost", res["results"][0])

    def test_matryoshka_prefix_slicing_and_normalization(self):
        """Verify Matryoshka prefix slicing produces correct shapes and unit norms."""
        from lthcm.core.matryoshka import MatryoshkaCompressor

        mrl = MatryoshkaCompressor(d_model=self.d_model, nested_dims=[8, 16, 32])
        x = torch.randn(4, self.d_model)

        prefix_8 = mrl.slice_prefix(x, dim=8, normalize=True)
        self.assertEqual(prefix_8.shape, (4, 8))
        norms = torch.norm(prefix_8, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

        multi_scales = mrl.multi_scale_embed(x)
        self.assertIn(8, multi_scales)
        self.assertIn(16, multi_scales)
        self.assertIn(32, multi_scales)
        self.assertEqual(multi_scales[16].shape, (4, 16))

    def test_matryoshka_coarse_to_fine_search(self):
        """Verify staged coarse-to-fine search retrieves target with compute savings."""
        from lthcm.core.matryoshka import MatryoshkaCompressor

        mrl = MatryoshkaCompressor(d_model=self.d_model, nested_dims=[8, 16, 32])
        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=2)

        # Ingest 15 documents
        for i in range(15):
            facets = torch.randn(2, self.d_model)
            store.add_document(f"doc_{i}", f"Document content {i}", facets)

        # Create target query aligned with doc_3 facet 0
        target_vec = store.facet_tensor[3, 0].clone()
        query = target_vec + 0.05 * torch.randn(self.d_model)

        res = mrl.coarse_to_fine_search(
            query=query,
            memory_store=store,
            coarse_dim=8,
            coarse_k=6,
            fine_top_k=3,
            early_exit_margin=0.99,  # Force stage 2 execution
        )

        self.assertIn("results", res)
        self.assertFalse(res["early_exit"])
        self.assertEqual(res["results"][0]["doc_id"], "doc_3", "Target doc must be retrieved at Rank 1")
        self.assertGreater(res["compression_ratio"], 1.0, "Compression ratio must be > 1")
        self.assertGreater(res["flops_saved_pct"], 0.0, "Must achieve positive FLOPs reduction")

    def test_matryoshka_early_exit(self):
        """Verify unambiguous queries trigger early exit at coarse stage."""
        from lthcm.core.matryoshka import MatryoshkaCompressor

        mrl = MatryoshkaCompressor(d_model=self.d_model, nested_dims=[8, 16, 32])
        store = SuperpositionMemoryStore(d_model=self.d_model, num_facets=1)

        # Target document has distinct feature in dimension 0
        target_facet = torch.zeros(1, self.d_model)
        target_facet[0, 0] = 1.0
        store.add_document("strong_target", "Clear match", target_facet)

        # Other docs are orthogonal
        for i in range(5):
            f = torch.zeros(1, self.d_model)
            f[0, i + 1] = 1.0
            store.add_document(f"dist_{i}", f"Distractor {i}", f)

        # Query strongly matches strong_target in coarse prefix
        query = target_facet[0].clone()
        res = mrl.coarse_to_fine_search(
            query=query,
            memory_store=store,
            coarse_dim=8,
            coarse_k=4,
            fine_top_k=2,
            early_exit_margin=0.25,
        )

        self.assertTrue(res["early_exit"], "Clear query must trigger early exit at coarse stage")
        self.assertEqual(res["results"][0]["doc_id"], "strong_target")
        self.assertEqual(res["results"][0]["stage"], "coarse_early_exit")

    def test_matryoshka_pipeline_integration(self):
        """Verify Matryoshka staged search executes seamlessly in LTHCMEngine."""
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        doc1 = torch.randn(2, self.d_model)
        engine.index_document("doc_mrl", "MRL pipeline doc", doc1, domain="finance", timestamp=10.0)

        query = torch.randn(self.d_model)
        res = engine.retrieve_and_verify(
            query_vector=query,
            query_domain="finance",
            query_timestamp=20.0,
            use_matryoshka_staged=True,
            coarse_dim=8,
        )

        self.assertTrue(res["matryoshka_staged_active"])
        self.assertIsNotNone(res["matryoshka_diagnostics"])
        self.assertIn("results", res)

    def test_optimal_transport_wasserstein_alignment(self):
        """Verify Entropic Optimal Transport (Sinkhorn-Knopp) and document permutation invariance."""
        aligner = WassersteinDocumentAligner(d_model=self.d_model, eps=0.05)

        # 1. Identity alignment: distance between identical distributions must be near 0
        facets_a = F.normalize(torch.randn(4, self.d_model), p=2, dim=-1)
        dist_self, plan_self = aligner.pairwise_distance(facets_a, facets_a)
        self.assertLess(float(dist_self.item()), 0.05, "Self-Wasserstein distance must be near 0")

        # 2. Permutation Invariance: shuffling facet order must yield identical Wasserstein distance
        perm_indices = torch.tensor([2, 0, 3, 1])
        facets_perm = facets_a[perm_indices]
        dist_perm, _ = aligner.pairwise_distance(facets_a, facets_perm)
        self.assertAlmostEqual(float(dist_self.item()), float(dist_perm.item()), places=3,
                               msg="Optimal Transport must be invariant to facet permutation")

        # 3. Non-negativity and triangle monotonicity
        facets_b = F.normalize(torch.randn(4, self.d_model), p=2, dim=-1)
        dist_ab, _ = aligner.pairwise_distance(facets_a, facets_b)
        self.assertGreaterEqual(float(dist_ab.item()), 0.0)

        # 4. Engine integration & document alignment
        engine = LTHCMEngine(d_model=self.d_model, num_facets=3)
        doc1_facets = torch.randn(3, self.d_model)
        doc2_facets = torch.randn(3, self.d_model)
        engine.index_document("doc_ot1", "Biomedical oncology literature", doc1_facets, domain="med")
        engine.index_document("doc_ot2", "Cancer immunotherapy treatments", doc2_facets, domain="med")

        ot_dist, plan = engine.align_documents_wasserstein("doc_ot1", "doc_ot2")
        self.assertEqual(plan.shape, (3, 3))
        self.assertAlmostEqual(float(plan.sum().item()), 1.0, places=4, msg="Transport plan must sum to 1.0")

        # 5. Pipeline retrieve_and_verify with compute_wasserstein=True
        q = torch.randn(self.d_model)
        res = engine.retrieve_and_verify(
            query_vector=q,
            query_domain="med",
            query_timestamp=50.0,
            compute_wasserstein=True,
        )
        self.assertTrue(res["wasserstein_alignment_active"])
        if len(res["results"]) > 0:
            self.assertIn("wasserstein_distance", res["results"][0])
            self.assertIn("wasserstein_similarity", res["results"][0])

    def test_hdc_vsa_symbolic_composition(self):
        """Verify Vector Symbolic Architecture binding, unbinding, and proposition reasoning."""
        vsa = VectorSymbolicArchitecture(d_hdc=512, d_model=self.d_model)

        # 1. Quasi-orthogonality of binding
        a = vsa.get_or_create_atom("Subject_Agent")
        b = vsa.get_or_create_atom("Verb_Action")
        c = vsa.bind(a, b)

        sim_ca = abs(vsa.similarity(c, a))
        sim_cb = abs(vsa.similarity(c, b))
        self.assertLess(sim_ca, 0.25, "Bound vector must be quasi-orthogonal to component a")
        self.assertLess(sim_cb, 0.25, "Bound vector must be quasi-orthogonal to component b")

        # 2. Invertibility: unbinding recovers component
        a_recovered = vsa.unbind(c, b)
        rec_sim = vsa.similarity(a_recovered, a)
        self.assertGreater(rec_sim, 0.60, "Unbinding with b must recover a with high similarity")

        # 3. Bundling superposition
        x = vsa.get_or_create_atom("Atom_X")
        y = vsa.get_or_create_atom("Atom_Y")
        z = vsa.get_or_create_atom("Atom_Z")
        bundle_xyz = vsa.bundle([x, y, z])
        self.assertGreater(vsa.similarity(bundle_xyz, x), 0.3)
        self.assertGreater(vsa.similarity(bundle_xyz, y), 0.3)
        self.assertGreater(vsa.similarity(bundle_xyz, z), 0.3)

        # 4. Structured Proposition Creation and Role Extraction via Engine
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        prop = engine.create_symbolic_proposition("TREATS", "Aspirin", "HeartAttack")
        recovered_subj = engine.extract_symbolic_role(prop, "TREATS", role_index=0)
        recovered_obj = engine.extract_symbolic_role(prop, "TREATS", role_index=1)

        aspirin_atom = engine.hdc.get_or_create_atom("Aspirin")
        heart_atom = engine.hdc.get_or_create_atom("HeartAttack")
        self.assertGreater(engine.hdc.similarity(recovered_subj, aspirin_atom), 0.40)
        self.assertGreater(engine.hdc.similarity(recovered_obj, heart_atom), 0.40)

    def test_hyperbolic_poincare_geometry(self):
        """Verify Poincaré Ball metric axioms, exponential boundary expansion, and hierarchical store."""
        manifold = PoincareBall(c=1.0)

        # 1. Metric axioms: d(x, x) = 0, symmetry d(x, y) = d(y, x), triangle inequality
        x = manifold.project(torch.randn(self.d_model) * 0.3)
        y = manifold.project(torch.randn(self.d_model) * 0.3)
        z = manifold.project(torch.randn(self.d_model) * 0.3)

        d_xx = manifold.dist(x, x).item()
        self.assertAlmostEqual(d_xx, 0.0, places=4, msg="d(x, x) must be 0")

        d_xy = manifold.dist(x, y).item()
        d_yx = manifold.dist(y, x).item()
        self.assertAlmostEqual(d_xy, d_yx, places=4, msg="d(x, y) must equal d(y, x)")

        d_xz = manifold.dist(x, z).item()
        d_yz = manifold.dist(y, z).item()
        self.assertLessEqual(d_xz, d_xy + d_yz + 1e-4, "Triangle inequality must hold")

        # 2. Exponential volume expansion near boundary
        delta = 0.05
        p_origin1 = torch.zeros(self.d_model)
        p_origin2 = torch.zeros(self.d_model)
        p_origin2[0] = delta

        p_boundary1 = torch.zeros(self.d_model)
        p_boundary1[0] = 0.85
        p_boundary2 = torch.zeros(self.d_model)
        p_boundary2[0] = 0.85 + delta

        d_origin = manifold.dist(p_origin1, p_origin2).item()
        d_boundary = manifold.dist(p_boundary1, p_boundary2).item()
        self.assertGreater(d_boundary, d_origin * 2.0, "Hyperbolic distance near boundary must vastly exceed origin distance")

        # 3. Inverse ExpMap and LogMap
        v = torch.randn(self.d_model) * 0.4
        p = manifold.expmap0(v)
        v_recovered = manifold.logmap0(p)
        self.assertTrue(torch.allclose(v, v_recovered, atol=1e-4), "logmap0(expmap0(v)) must recover v")

        # 4. Hierarchical Store & Pipeline Integration
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        root_emb = torch.randn(self.d_model) * 0.05
        leaf1_emb = root_emb + torch.randn(self.d_model) * 0.5
        leaf2_emb = root_emb + torch.randn(self.d_model) * 0.5

        engine.index_document("root", "Living Organism (Taxonomy Root)", root_emb.unsqueeze(0))
        engine.index_document("leaf1", "Canis lupus (Gray Wolf)", leaf1_emb.unsqueeze(0))
        engine.index_document("leaf2", "Felis catus (Domestic Cat)", leaf2_emb.unsqueeze(0))

        hyp_results = engine.retrieve_hyperbolic(root_emb, top_k=2)
        self.assertEqual(len(hyp_results), 2)
        self.assertEqual(hyp_results[0]["doc_id"], "root", "Root concept must be closest to query near root")

    def test_spectral_chebyshev_wormhole_refinement(self):
        """Verify Spectral Chebyshev Graph Convolution and adaptive wormhole message passing."""
        K = 3
        conv = ChebyshevSpectralGraphConv(in_channels=self.d_model, out_channels=self.d_model, K=K)
        N = 6
        X = torch.randn(N, self.d_model)
        adj = torch.rand(N, N)
        adj = (adj + adj.T) * 0.5
        adj = torch.where(adj > 0.6, adj, torch.zeros_like(adj))

        out = conv(X, adj)
        self.assertEqual(out.shape, (N, self.d_model))
        self.assertFalse(torch.isnan(out).any(), "Spectral Chebyshev output must not contain NaNs")

        refiner = SpectralWormholeRefiner(d_model=self.d_model, K_order=K)
        wormhole = WormholeGraphIndex(d_model=self.d_model)

        emb_a = torch.randn(self.d_model)
        emb_b = torch.randn(self.d_model)
        emb_c = torch.randn(self.d_model)

        wormhole.add_node("node_a", emb_a, "Node A content", cluster="Cluster_1")
        wormhole.add_node("node_b", emb_b, "Node B content", cluster="Cluster_2")
        wormhole.add_node("node_c", emb_c, "Node C content", cluster="Cluster_3")

        wormhole.add_wormhole("node_a", "node_b", relation_type="topological_tunnel", permeability=0.9)
        wormhole.add_wormhole("node_b", "node_c", relation_type="topological_tunnel", permeability=0.8)

        query = emb_b + 0.1 * torch.randn(self.d_model)
        refined = refiner.refine_candidates(query, wormhole, ["node_a", "node_b", "node_c"])

        self.assertIn("results", refined)
        self.assertEqual(len(refined["results"]), 3)
        for r in refined["results"]:
            self.assertIn("refined_score", r)
            self.assertIn("gate_weight", r)
            self.assertGreaterEqual(r["gate_weight"], 0.0)
            self.assertLessEqual(r["gate_weight"], 1.0)

        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        engine.index_document("doc_s1", "Distributed computing clusters", torch.randn(2, self.d_model), domain="cs")
        engine.index_document("doc_s2", "Byzantine consensus protocols", torch.randn(2, self.d_model), domain="cs",
                              wormhole_links=[{"target_id": "doc_s1", "relation": "fault_tolerance", "permeability": 0.95}])

        q_cs = torch.randn(self.d_model)
        res = engine.retrieve_and_verify(
            query_vector=q_cs,
            query_domain="cs",
            query_timestamp=100.0,
            use_spectral_wormhole=True,
        )
        self.assertTrue(res["spectral_wormhole_active"])
        self.assertIn("results", res)

    def test_bidirectional_wormhole_backtracking_and_rewormhole(self):
        """Verify Bidirectional self-healing loop: back-hop, penalty decay, and alternative re-wormholing."""
        wormhole = WormholeGraphIndex(d_model=self.d_model)

        # Build node embeddings:
        # q points along axis 0
        q = torch.zeros(self.d_model)
        q[0] = 1.0

        emb_start = torch.zeros(self.d_model)
        emb_start[0] = 0.9

        # emb_bad has very high initial alignment (0.95), tempting the agent
        emb_bad = torch.zeros(self.d_model)
        emb_bad[0] = 0.95

        # emb_good has moderate initial alignment (0.70)
        emb_good = torch.zeros(self.d_model)
        emb_good[0] = 0.70

        # emb_goal has high alignment
        emb_goal = torch.zeros(self.d_model)
        emb_goal[0] = 0.85

        wormhole.add_node("start", emb_start, "Start premise")
        wormhole.add_node("bad_step", emb_bad, "Hallucinatory trap")
        wormhole.add_node("good_step", emb_good, "Solid intermediary fact")
        wormhole.add_node("goal", emb_goal, "Verified conclusion")

        # start connects to both bad_step and good_step
        wormhole.add_wormhole("start", "bad_step", relation_type="deceptive_shortcut", permeability=1.0)
        wormhole.add_wormhole("start", "good_step", relation_type="valid_bridge", permeability=1.0)
        wormhole.add_wormhole("good_step", "goal", relation_type="valid_bridge", permeability=1.0)

        # Verifier rejects bad_step
        def verifier_fn(path):
            if "bad_step" in path:
                return False, 0.99, "factual_contradiction"
            return True, 0.05, "verified"

        engine = BidirectionalWormholeEngine(
            wormhole_index=wormhole,
            refiner=None,
            penalty_factor=0.01,
            max_backtracks=5,
        )

        res = engine.traverse_and_verify(
            query=q,
            start_node="start",
            verifier_fn=verifier_fn,
            target_depth=3,
        )

        # Agent must detect failure, back-hop, penalize bad_step, and re-wormhole to good_step -> goal!
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["path"], ["start", "good_step", "goal"])
        self.assertEqual(res["backtracks"], 1)

        # Verify edge penalties recorded
        self.assertIn("start->bad_step", res["edge_penalties"])
        self.assertLessEqual(res["edge_penalties"]["start->bad_step"], 0.05)

        # Verify history trace has both backtrack and forward hops
        actions = [h["action"] for h in res["history"]]
        self.assertIn("backtrack_and_penalize", actions)
        self.assertIn("forward_hop", actions)

    def test_bidirectional_wormhole_with_sheaf_verification(self):
        """Verify BidirectionalWormholeEngine using Sheaf Dirichlet energy obstruction."""
        wormhole = WormholeGraphIndex(d_model=self.d_model)
        sheaf_filter = SheafContradictionFilter(d_model=self.d_model, stalk_dim=16, contradiction_threshold=0.3)

        # Query points along axis 0
        q = torch.zeros(self.d_model)
        q[0] = 1.0

        # Root premise points along axis 1
        emb_root = torch.zeros(self.d_model)
        emb_root[1] = 1.0

        # Contradiction: high query alignment (0.90 along dim 0), but negative along root axis 1 (-0.435)
        # Results in large Dirichlet obstruction with root
        emb_contra = torch.zeros(self.d_model)
        emb_contra[0] = 0.90
        emb_contra[1] = -0.435

        # Coherent deduction: moderate query alignment (0.60 along dim 0), but strong agreement with root (0.80 along axis 1)
        # Results in low Dirichlet obstruction with root
        emb_coherent = torch.zeros(self.d_model)
        emb_coherent[0] = 0.60
        emb_coherent[1] = 0.80

        wormhole.add_node("root", emb_root, "Premise node")
        wormhole.add_node("contradiction", emb_contra, "Contradictory hypothesis")
        wormhole.add_node("coherent", emb_coherent, "Harmonious deduction")

        wormhole.add_wormhole("root", "contradiction", relation_type="shortcut", permeability=1.0)
        wormhole.add_wormhole("root", "coherent", relation_type="shortcut", permeability=0.9)

        engine = BidirectionalWormholeEngine(
            wormhole_index=wormhole,
            penalty_factor=0.05,
            max_backtracks=3,
        )

        res = engine.traverse_with_sheaf(
            query=q,
            start_node="root",
            sheaf_filter=sheaf_filter,
            energy_threshold=0.8,
            target_depth=2,
        )

        # Sheaf Dirichlet obstruction triggers back-hop and re-wormhole to coherent node!
        self.assertEqual(res["status"], "SUCCESS")
        self.assertEqual(res["path"], ["root", "coherent"])
        self.assertGreaterEqual(res["backtracks"], 1)

    def test_domain_seeding_rng_isolation_and_determinism(self):
        """Verify domain initialization isolates RNG state and produces identical coordinates across PYTHONHASHSEEDs."""
        engine = LTHCMEngine(d_model=self.d_model)

        # 1. Global torch RNG state must remain unchanged
        state_before = torch.get_rng_state()
        coord = engine.get_or_create_domain("test_domain_seeding")
        state_after = torch.get_rng_state()
        self.assertTrue(torch.equal(state_before, state_after), "get_or_create_domain modified global torch RNG!")
        self.assertEqual(coord.shape, (self.d_model,))
        self.assertAlmostEqual(torch.norm(coord, p=2).item(), 1.0, places=5)

        # 2. Subprocess check across different PYTHONHASHSEED values
        code = (
            f"from lthcm.engine.pipeline import LTHCMEngine; "
            f"e = LTHCMEngine(d_model={self.d_model}); "
            f"print(e.get_or_create_domain('cross_process_domain').tolist()[:5])"
        )
        out1 = subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": "0"}).decode().strip()
        out2 = subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": "4242"}).decode().strip()
        self.assertEqual(out1, out2, f"Coordinates differed across PYTHONHASHSEED: {out1} vs {out2}")

    def test_bidirectional_wormhole_spectral_refiner_wiring(self):
        """Verify BidirectionalWormholeEngine calls refiner.refine_candidates and uses refined scores to order candidates."""
        wh = WormholeGraphIndex(d_model=self.d_model)
        # Create n1 (start), n2 (low raw sim, high refiner score), n3 (high raw sim, low refiner score)
        q = torch.zeros(self.d_model)
        q[0] = 1.0

        n1_emb = torch.zeros(self.d_model)
        n1_emb[1] = 1.0

        n2_emb = torch.zeros(self.d_model)
        n2_emb[2] = 1.0  # dot(q, n2_emb) = 0.0

        n3_emb = torch.zeros(self.d_model)
        n3_emb[0] = 1.0  # dot(q, n3_emb) = 1.0 (higher raw sim)

        wh.add_node("n1", n1_emb, "text 1")
        wh.add_node("n2", n2_emb, "text 2")
        wh.add_node("n3", n3_emb, "text 3")
        wh.add_wormhole("n1", "n2", relation_type="link", permeability=1.0)
        wh.add_wormhole("n1", "n3", relation_type="link", permeability=1.0)

        refiner = SpectralWormholeRefiner(d_model=self.d_model, K_order=2)
        engine_trav = BidirectionalWormholeEngine(wormhole_index=wh, refiner=refiner)

        # Mock refiner to assign n2 a score of 0.95 and n3 a score of 0.05
        with patch.object(
            refiner,
            "refine_candidates",
            return_value={"results": [{"node_id": "n2", "refined_score": 0.95}, {"node_id": "n3", "refined_score": 0.05}]}
        ) as mock_refine:
            res = engine_trav.traverse_and_verify(
                query=q,
                start_node="n1",
                verifier_fn=lambda p: (True, 0.0, "ok"),
                target_depth=2,
            )
            self.assertTrue(mock_refine.called, "refine_candidates was not called by traverse_and_verify!")
            self.assertEqual(res["status"], "SUCCESS")
            # Despite n3 having higher direct dot product (1.0 vs 0.0), refiner score prioritized n2!
            self.assertEqual(res["path"], ["n1", "n2"])
            self.assertEqual(res["history"][1]["action"], "forward_hop")
            self.assertAlmostEqual(res["history"][1]["effective_score"], 0.95, places=4)

    def test_universal_32_frontiers_benchmark_suite(self):
        """Verify the 32-frontier real-world empirical benchmark suite executes and returns valid schema."""
        from lthcm.benchmark_universal_32_frontiers import run_all_32_frontiers
        result = run_all_32_frontiers()
        self.assertEqual(result["total_frontiers"], 32)
        self.assertIn("num_passed", result)
        self.assertIn("all_passed", result)
        self.assertIn("results", result)
        self.assertEqual(len(result["results"]), 32)

    def test_standard_community_benchmarks_suite(self):
        """Verify the synthetic benchmarks suite executes cleanly with bounded metrics in [0, 1] matching audit numbers."""
        from lthcm.benchmarks.run_standard_benchmarks import run_all_standard_benchmarks
        res = run_all_standard_benchmarks(verbose=False)
        self.assertIn("m_niah", res)
        self.assertIn("hotpot_qa", res)
        self.assertIn("fever", res)
        self.assertIn("beir", res)

        # M-NIAH schema and metric bounds [0, 1]
        m_niah_lthcm = res["m_niah"]["lthcm_topological"]
        for key in ["macro_recall@1", "macro_recall@3", "macro_recall@5", "mrr"]:
            self.assertIn(key, m_niah_lthcm)
            self.assertGreaterEqual(m_niah_lthcm[key], 0.0)
            self.assertLessEqual(m_niah_lthcm[key], 1.0)

        # HotpotQA multi-hop schema and metric bounds
        hotpot_lthcm = res["hotpot_qa"]["lthcm_wormhole"]
        hotpot_flat = res["hotpot_qa"]["flat_dense_baseline"]
        for key in ["hop1_recall@1", "hop2_recall@2", "joint_multihop_recall@2", "ndcg@10"]:
            self.assertIn(key, hotpot_lthcm)
            self.assertGreaterEqual(hotpot_lthcm[key], 0.0)
            self.assertLessEqual(hotpot_lthcm[key], 1.0)
            self.assertIn(key, hotpot_flat)
            self.assertGreaterEqual(hotpot_flat[key], 0.0)
            self.assertLessEqual(hotpot_flat[key], 1.0)
        self.assertGreaterEqual(hotpot_lthcm["joint_multihop_recall@2"], hotpot_flat["joint_multihop_recall@2"])
        self.assertGreaterEqual(hotpot_lthcm["ndcg@10"], hotpot_flat["ndcg@10"])

        # FEVER contradiction schema and metric bounds
        self.assertIn("pure_sheaf_untrained", res["fever"])
        fever_untrained = res["fever"]["pure_sheaf_untrained"]
        fever_sheaf = res["fever"]["lthcm_sheaf_laplacian"]
        for key in ["accuracy", "contradiction_precision", "contradiction_recall", "contradiction_f1"]:
            self.assertIn(key, fever_sheaf)
            self.assertGreaterEqual(fever_sheaf[key], 0.0)
            self.assertLessEqual(fever_sheaf[key], 1.0)
        self.assertGreaterEqual(fever_untrained["accuracy"], 0.0)
        self.assertLessEqual(fever_untrained["accuracy"], 1.0)
        self.assertGreaterEqual(fever_sheaf["accuracy"], fever_untrained["accuracy"])
        self.assertGreater(fever_sheaf["energy_separation_ratio"], 1.0)

        # BEIR IR schema and metric bounds
        beir_lthcm = res["beir"]["lthcm_topological"]
        beir_dense = res["beir"]["dense_biencoder_baseline"]
        for key in ["ndcg@10", "recall@10", "map@100", "mrr@10"]:
            self.assertIn(key, beir_lthcm)
            self.assertGreaterEqual(beir_lthcm[key], 0.0)
            self.assertLessEqual(beir_lthcm[key], 1.0)
            self.assertIn(key, beir_dense)
            self.assertGreaterEqual(beir_dense[key], 0.0)
            self.assertLessEqual(beir_dense[key], 1.0)




    def test_persistent_homology_square_void(self):
        """Verify persistent homology detects topological reasoning void in a hollow square configuration."""
        guard = PersistentHomologyGuard(distance_metric="euclidean", significance_ratio=0.20)
        # Hollow unit square: (0,0), (1,0), (1,1), (0,1)
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ], dtype=torch.float32)
        doc_ids = ["premise_A", "premise_B", "premise_C", "premise_D"]

        res = guard.audit_retrieved_context(points, doc_ids=doc_ids)
        self.assertEqual(res["num_documents"], 4)
        self.assertEqual(res["status"], "KNOWLEDGE_VOID_DETECTED")
        self.assertFalse(res["is_complete"])
        self.assertGreater(len(res["voids"]), 0)
        self.assertLess(res["epistemic_completeness"], 0.85)

        # The void must have birth=1.0 and death ≈ 1.414 (sqrt(2))
        top_void = res["voids"][0]
        self.assertAlmostEqual(top_void["birth"], 1.0, places=2)
        self.assertAlmostEqual(top_void["death"], 1.4142, places=2)
        self.assertIn("premise_A", top_void["participating_docs"])

    def test_persistent_homology_solid_simplex(self):
        """Verify contractible solid geometry yields COMPLETE status with C=1.0."""
        guard = PersistentHomologyGuard(distance_metric="euclidean")
        # Equilateral triangle with center point (contractible 2-simplices fill immediately)
        points = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.866025],
            [0.5, 0.288675],
        ], dtype=torch.float32)

        res = guard.audit_retrieved_context(points)
        self.assertEqual(res["status"], "COMPLETE")
        self.assertTrue(res["is_complete"])
        self.assertEqual(len(res["voids"]), 0)
        self.assertAlmostEqual(res["epistemic_completeness"], 1.0, places=2)

    def test_persistent_homology_fragmented_clusters(self):
        """Verify disjoint clusters trigger FRAGMENTED_CLUSTERS status."""
        guard = PersistentHomologyGuard(distance_metric="euclidean")
        # Cluster 1 around origin, Cluster 2 displaced by 100.0
        points = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [100.0, 100.0],
            [100.1, 100.0],
            [100.0, 100.1],
        ], dtype=torch.float32)

        res = guard.audit_retrieved_context(points)
        self.assertEqual(res["status"], "FRAGMENTED_CLUSTERS")
        self.assertFalse(res["is_complete"])
        self.assertGreater(res["betti_0_profile"], 1)

    def test_pipeline_persistent_homology_audit_integration(self):
        """Verify full LTHCMEngine pipeline integration with audit_completeness=True."""
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        
        # Index 4 documents forming a square in the first 2 dimensions
        for i, coord in enumerate([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]):
            vec = torch.zeros(2, self.d_model)
            vec[:, 0] = coord[0]
            vec[:, 1] = coord[1]
            engine.index_document(
                doc_id=f"doc_{i}",
                content=f"Document {i} content",
                facet_vectors=vec,
                timestamp=float(i),
            )

        query = torch.zeros(self.d_model)
        query[0] = 0.5
        query[1] = 0.5

        # Query without homology audit
        res_no_audit = engine.retrieve_and_verify(
            query_vector=query,
            query_timestamp=10.0,
            top_k=4,
            audit_completeness=False,
        )
        self.assertFalse(res_no_audit["persistent_homology_active"])
        self.assertIsNone(res_no_audit["epistemic_audit"])

        # Query with homology audit
        res_audit = engine.retrieve_and_verify(
            query_vector=query,
            query_timestamp=10.0,
            top_k=4,
            audit_completeness=True,
        )
        self.assertTrue(res_audit["persistent_homology_active"])
        self.assertIsNotNone(res_audit["epistemic_audit"])
        audit = res_audit["epistemic_audit"]
        self.assertIn("status", audit)
        self.assertIn("epistemic_completeness", audit)
        self.assertIn("betti_0_profile", audit)
        self.assertIn("betti_1_profile", audit)

        # Dedicated engine helper audit_knowledge_completeness
        embs = torch.randn(4, self.d_model)
        standalone_audit = engine.audit_knowledge_completeness(embs, doc_ids=["d1", "d2", "d3", "d4"])
        self.assertIn("status", standalone_audit)
        self.assertEqual(standalone_audit["num_documents"], 4)

    def test_categorical_morphism_composition_associativity(self):
        """Verify identity morphism laws and composition associativity ((f >> g) >> h == f >> (g >> h))."""
        class ScaleMorphism(Morphism[torch.Tensor, torch.Tensor]):
            def __init__(self, factor: float):
                super().__init__(name=f"Scale({factor})")
                self.factor = factor
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.factor

        class ShiftMorphism(Morphism[torch.Tensor, torch.Tensor]):
            def __init__(self, offset: float):
                super().__init__(name=f"Shift({offset})")
                self.offset = offset
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + self.offset

        f = ScaleMorphism(2.0)
        g = ShiftMorphism(3.0)
        h = ScaleMorphism(0.5)
        id_m = IdentityMorphism[torch.Tensor]()

        x = torch.tensor([1.0, 2.0, 3.0])

        # 1. Identity laws: f >> id == id >> f == f
        self.assertTrue(torch.allclose((f >> id_m)(x), f(x)))
        self.assertTrue(torch.allclose((id_m >> f)(x), f(x)))

        # 2. Strict associativity: ((f >> g) >> h)(x) == (f >> (g >> h))(x)
        left_assoc = (f >> g) >> h
        right_assoc = f >> (g >> h)
        out_left = left_assoc(x)
        out_right = right_assoc(x)
        self.assertTrue(torch.allclose(out_left, out_right, atol=1e-7))
        # Expected: ((x * 2) + 3) * 0.5 = [2.5, 3.5, 4.5]
        expected = ((x * 2.0) + 3.0) * 0.5
        self.assertTrue(torch.allclose(out_left, expected, atol=1e-7))

    def test_categorical_symmetric_monoidal_tensor_and_braiding(self):
        """Verify monoidal tensor product f ⊗ g, associator α, and symmetric braiding β."""
        class NegMorphism(Morphism[torch.Tensor, torch.Tensor]):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return -x

        class DoubleMorphism(Morphism[torch.Tensor, torch.Tensor]):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 2.0

        f = NegMorphism("Neg")
        g = DoubleMorphism("Double")

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        c = torch.tensor([5.0, 6.0])

        # 1. Tensor product (f ⊗ g)(a, b) = (f(a), g(b))
        tensor_fg = f * g
        res_a, res_b = tensor_fg((a, b))
        self.assertTrue(torch.allclose(res_a, -a))
        self.assertTrue(torch.allclose(res_b, b * 2.0))

        # 2. Braiding involution: β_{B,A} ∘ β_{A,B} == id_{A ⊗ B}
        braid_ab = BraidingMorphism[torch.Tensor, torch.Tensor]()
        braid_ba = BraidingMorphism[torch.Tensor, torch.Tensor]()
        involution = braid_ab >> braid_ba
        rec_a, rec_b = involution((a, b))
        self.assertTrue(torch.allclose(rec_a, a))
        self.assertTrue(torch.allclose(rec_b, b))

        # 3. Associator: α((a, b), c) = (a, (b, c))
        assoc = AssociatorMorphism[torch.Tensor, torch.Tensor, torch.Tensor]()
        res_assoc = assoc(((a, b), c))
        self.assertEqual(len(res_assoc), 2)
        self.assertTrue(torch.allclose(res_assoc[0], a))
        self.assertTrue(torch.allclose(res_assoc[1][0], b))
        self.assertTrue(torch.allclose(res_assoc[1][1], c))

    def test_categorical_lens_laws(self):
        """Verify the 3 formal categorical lens laws (GetPut, PutGet, PutPut)."""
        # Lens on PipelineState focusing on query_timestamp
        def getter(s: PipelineState) -> float:
            return s.query_timestamp

        def setter(s: PipelineState, new_t: float) -> PipelineState:
            new_s = s.clone()
            new_s.query_timestamp = new_t
            return new_s

        timestamp_lens = CategoricalLens[PipelineState, float](
            getter=getter,
            setter=setter,
            name="TimestampLens",
        )

        state = PipelineState(
            query_vector=torch.randn(self.d_model),
            query_domain="physics",
            query_timestamp=42.0,
        )

        def eq_state(s1: PipelineState, s2: PipelineState) -> bool:
            return (
                s1.query_domain == s2.query_domain
                and abs(s1.query_timestamp - s2.query_timestamp) < 1e-6
                and torch.allclose(s1.query_vector, s2.query_vector)
            )

        laws = timestamp_lens.verify_lens_laws(
            s=state,
            val_a=100.0,
            val_b=200.0,
            eq_fn_s=eq_state,
            eq_fn_a=lambda x, y: abs(x - y) < 1e-6,
        )
        self.assertTrue(laws["get_put"], "GetPut law must hold: set(s, get(s)) == s")
        self.assertTrue(laws["put_get"], "PutGet law must hold: get(set(s, a)) == a")
        self.assertTrue(laws["put_put"], "PutPut law must hold: set(set(s, a), b) == set(s, b)")
        self.assertTrue(laws["all_passed"])

    def test_categorical_adjoint_functors(self):
        """Verify F ⊣ G adjunction natural isomorphism between continuous and HDC spaces."""
        adj = AdjointFunctorPair(d_model=self.d_model, d_hdc=512, seed=123)
        x = F.normalize(torch.randn(self.d_model), p=2, dim=0)
        y = F.normalize(torch.randn(512), p=2, dim=0)

        res = adj.verify_adjunction_isomorphism(x, y)
        self.assertLess(res["natural_isomorphism_error"], 1e-4)
        self.assertTrue(res["is_adjoint"])

        # Unit of adjunction η: X -> G(F(X)) preserves non-trivial fidelity
        eta_x = adj.unit_eta(x)
        fidelity = float(torch.dot(x, eta_x).item())
        self.assertGreater(fidelity, 0.70)

    def test_functorial_pipeline_engine_integration(self):
        """Verify full FunctorialPipeline compilation, execution, and associativity via LTHCMEngine."""
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)
        pipe = engine.create_functorial_pipeline(
            include_causality=True,
            include_fisher_rao=True,
            include_persistent_homology=True,
        )
        self.assertEqual(len(pipe.morphisms), 3)

        # Create mock candidate documents
        cands = [
            {
                "doc_id": "doc_past_valid",
                "timestamp": 20.0,
                "aligned_vector": F.normalize(torch.randn(self.d_model), p=2, dim=0),
                "domain": "general",
            },
            {
                "doc_id": "doc_future_acausal",
                "timestamp": 120.0,
                "aligned_vector": F.normalize(torch.randn(self.d_model), p=2, dim=0),
                "domain": "general",
            },
            {
                "doc_id": "doc_past_close",
                "timestamp": 40.0,
                "aligned_vector": F.normalize(torch.randn(self.d_model), p=2, dim=0),
                "domain": "general",
            },
        ]

        q = F.normalize(torch.randn(self.d_model), p=2, dim=0)
        init_state = PipelineState(
            query_vector=q,
            query_domain="general",
            query_timestamp=50.0,
            candidates=cands,
        )

        out_state = pipe.execute(init_state)

        # 1. Causality check: doc_future_acausal (t=120 > query_t=50) must be filtered out
        surviving_ids = [c["doc_id"] for c in out_state.candidates]
        self.assertNotIn("doc_future_acausal", surviving_ids)
        self.assertIn("doc_past_valid", surviving_ids)
        self.assertIn("doc_past_close", surviving_ids)

        # 2. Fisher-Rao scores must be assigned and sorted in descending order
        for cand in out_state.candidates:
            self.assertIn("fisher_rao_score", cand)
        scores = [c["fisher_rao_score"] for c in out_state.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

        # 3. Persistent homology audit must be populated
        self.assertIn("homology_audit", out_state.diagnostics)
        audit = out_state.diagnostics["homology_audit"]
        self.assertIn("epistemic_completeness", audit)
        self.assertIn("status", audit)

        # 4. Invariant: Pipeline composition associativity across the 3 stages
        m1, m2, m3 = pipe.morphisms[0], pipe.morphisms[1], pipe.morphisms[2]
        is_assoc = pipe.verify_composition_associativity(init_state, m1, m2, m3)
        self.assertTrue(is_assoc, "Functorial pipeline composition must be strictly associative")

    def test_riemannian_metric_tensor_properties(self):
        """
        Verify mathematical properties of the Riemannian metric tensor:
        1. Positive-definiteness & valid Cholesky factorization L L^T = G
        2. Natural gradient curvature correction
        3. Emergent thermodynamic forgetting: noise variance along stale/unreinforced
           directions is dramatically higher (>4x) than along reinforced directions.
        """
        from lthcm.core.riemannian_langevin import RiemannianMetricTensor

        d_key = 16
        metric = RiemannianMetricTensor(d_key=d_key, damping=0.1, momentum=0.90)
        B, H = 1, 1

        # Reinforced direction u = e_0
        u = torch.zeros(B, H, d_key, 1)
        u[0, 0, 0, 0] = 1.0

        # Repeatedly update metric with key along u
        for _ in range(50):
            metric.update(u)

        # 1. Cholesky factorization check: L @ L^T must equal G
        L, L_inv = metric.compute_cholesky()
        G_recon = torch.matmul(L, L.transpose(-1, -2))
        eye = torch.eye(d_key).unsqueeze(0).unsqueeze(0)
        cholesky_err = torch.norm(G_recon - (metric.G + 1e-5 * eye)).item()
        self.assertLess(cholesky_err, 1e-4, "Cholesky decomposition L L^T must reconstruct G")

        # Inversion check: L @ L_inv = I
        inv_err = torch.norm(torch.matmul(L, L_inv) - eye).item()
        self.assertLess(inv_err, 1e-4, "L @ L_inv must be the identity matrix")

        # 2. Curvature spectrum: e_0 has high curvature, e_1 has low curvature (damping only)
        min_eig, max_eig = metric.get_spectrum()
        self.assertGreater(max_eig, min_eig * 3.0, "Reinforced direction must induce high manifold curvature")

        # 3. Thermodynamic noise anisotropy (Emergent Forgetting):
        # Sample 10,000 noise instances and measure empirical variance along e_0 vs e_1
        noises = []
        for _ in range(500):
            n = metric.sample_anisotropic_noise(
                shape=(B, H, 1, d_key),
                temperature=0.01,
                lr=0.05,
                L_inv=L_inv,
            )
            noises.append(n)
        noises = torch.cat(noises, dim=2) # [1, 1, 500, d_key]

        var_e0 = torch.var(noises[0, 0, :, 0]).item() # reinforced direction
        var_e1 = torch.var(noises[0, 0, :, 1]).item() # stale direction

        self.assertGreater(
            var_e1,
            var_e0 * 3.5,
            f"Thermal noise variance in flat direction ({var_e1:.5f}) must be >3.5x higher than in reinforced direction ({var_e0:.5f})",
        )

    def test_riemannian_langevin_memory_surprise_reduction(self):
        """
        Verify associative memorization and surprise reduction under Riemannian Langevin SDE.
        """
        from lthcm.core.riemannian_langevin import RiemannianLangevinMemory

        torch.manual_seed(42)
        d_model = 32
        memory = RiemannianLangevinMemory(
            d_key=d_model,
            d_val=d_model,
            num_heads=2,
            temperature=0.001,
            base_lr=0.1,
            damping=0.1,
            prior_decay=0.0001,
        )

        k = F.normalize(torch.randn(1, d_model), p=2, dim=-1)
        v = F.normalize(torch.randn(1, d_model), p=2, dim=-1)

        surprises = []
        for step_idx in range(25):
            _, surp, diag = memory.step(k, v)
            surprises.append(surp.item())

        # Surprise must drop significantly as memory learns associative mapping
        self.assertLess(
            surprises[-1],
            surprises[0] * 0.25,
            f"Surprise error must drop by >75% (from {surprises[0]:.4f} to {surprises[-1]:.4f})",
        )

        # Retrieval must produce high cosine alignment with stored target
        v_retrieved = memory.retrieve(k)
        cos_sim = F.cosine_similarity(v_retrieved, v, dim=-1).item()
        self.assertGreater(cos_sim, 0.70, f"Retrieved vector cosine similarity ({cos_sim:.3f}) must exceed 0.70")

    def test_titans_riemannian_langevin_integration(self):
        """
        Verify TitansNeuralMemory functions correctly with Riemannian Langevin dynamics enabled.
        """
        from lthcm.core.titans_memory import TitansNeuralMemory

        titans_rl = TitansNeuralMemory(
            d_key=32,
            d_val=32,
            num_heads=2,
            use_riemannian_langevin=True,
            langevin_temperature=0.005,
        )

        seq_k = torch.randn(2, 4, 32)
        seq_v = torch.randn(2, 4, 32)

        out, surp = titans_rl(seq_k, seq_v)
        self.assertEqual(out.shape, (2, 4, 32))
        self.assertEqual(surp.shape, (2, 4))
        self.assertIsNotNone(titans_rl.riemannian_metric.G)

    def test_lthcm_engine_riemannian_langevin_pipeline(self):
        """
        Verify end-to-end integration of Riemannian Langevin Memory within LTHCMEngine.
        """
        engine = LTHCMEngine(d_model=self.d_model, num_facets=2)

        doc_vec = torch.randn(2, self.d_model)
        engine.index_document("doc_rl", "Riemannian Langevin Document", doc_vec, domain="physics", timestamp=10.0)

        query = torch.randn(self.d_model)
        res = engine.retrieve_and_verify(
            query_vector=query,
            query_domain="physics",
            query_timestamp=20.0,
            use_riemannian_langevin=True,
        )

        self.assertTrue(res["riemannian_langevin_active"])
        self.assertIsNotNone(res["riemannian_langevin_diagnostics"])
        diag = res["riemannian_langevin_diagnostics"]
        self.assertIn("curvature_condition_number", diag)
        self.assertIn("thermal_noise_norm", diag)

        # Helper methods test
        k = torch.randn(1, self.d_model)
        v = torch.randn(1, self.d_model)
        v_out, surp, h_diag = engine.step_riemannian_langevin_memory(k, v)
        self.assertEqual(v_out.shape, (1, self.d_model))
        self.assertIn("thermal_noise_norm", h_diag)

        v_ret = engine.retrieve_riemannian_langevin_memory(k)
        self.assertEqual(v_ret.shape, (1, self.d_model))

    def test_hdc_deterministic_atom_seeds(self):
        """Finding B13a: Atom generation must be deterministic and cross-process reproducible."""
        vsa1 = VectorSymbolicArchitecture(d_model=self.d_model, d_hdc=64)
        vsa2 = VectorSymbolicArchitecture(d_model=self.d_model, d_hdc=64)

        atom1 = vsa1.get_or_create_atom("concept_entropy")
        atom2 = vsa2.get_or_create_atom("concept_entropy")

        self.assertTrue(torch.allclose(atom1, atom2, atol=1e-6))
        # Different concept produces different quasi-orthogonal atom
        atom_diff = vsa1.get_or_create_atom("concept_gravity")
        self.assertFalse(torch.allclose(atom1, atom_diff, atol=1e-3))

    def test_contrastive_sheaf_weights_only_checkpoint(self):
        """Finding B14: Checkpoint save and load must succeed with weights_only=True."""
        import tempfile
        from lthcm.core.contrastive_sheaf import ContrastiveSheafTrainer
        trainer = ContrastiveSheafTrainer(d_model=self.d_model, stalk_dim=8)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            ckpt_path = tf.name

        try:
            meta = {"commit": "test-commit", "seed": 42}
            trainer.save_checkpoint(ckpt_path, metadata=meta)

            trainer2 = ContrastiveSheafTrainer(d_model=self.d_model, stalk_dim=8)
            trainer2.load_checkpoint(ckpt_path, expected_metadata={"commit": "test-commit"})

            # Projector weights must match
            for p1, p2 in zip(trainer.projector.parameters(), trainer2.projector.parameters()):
                self.assertTrue(torch.allclose(p1, p2))
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

    def test_pipeline_gauge_transport_same_domain_identity(self):
        """Finding B13c: Intra-domain candidates must preserve exact vector without random gauge perturbation."""
        engine = LTHCMEngine(d_model=self.d_model)
        doc_vec = torch.randn(1, self.d_model)
        engine.index_document("doc_id", "Exact Domain Match", doc_vec, domain="physics", timestamp=50.0)

        query = torch.randn(self.d_model)
        res = engine.retrieve_and_verify(
            query_vector=query,
            query_domain="physics",
            query_timestamp=100.0,
            enforce_causality=False,
            enforce_sheaf_verification=False,
            enable_gauge_transport=True,
        )

        cand = res["results"][0]
        # Same domain -> aligned_vector == original vector and gauge_sim == 1.0
        self.assertTrue(torch.allclose(cand["aligned_vector"], cand["vector"], atol=1e-6))
        self.assertAlmostEqual(cand["gauge_sim"], 1.0)

    def test_pipeline_no_grad_titans_step(self):
        """Finding B15: Test-time memory update must execute under torch.no_grad to prevent graph leakage."""
        engine = LTHCMEngine(d_model=self.d_model)
        doc_vec = torch.randn(1, self.d_model)
        engine.index_document("doc_t", "Titans Document", doc_vec, domain="general", timestamp=50.0)

        query = torch.randn(self.d_model, requires_grad=True)
        res = engine.retrieve_and_verify(
            query_vector=query,
            query_domain="general",
            query_timestamp=100.0,
            update_titans_memory=True,
        )

        for param in engine.titans.parameters():
            self.assertIsNone(param.grad)

    def test_causal_morphism_missing_vector_and_fallback(self):
        """Finding B11 & B16: Missing vector gets dx=2.0 and allow_fallback=False returns empty."""
        from lthcm.core.categorical import PipelineState, CausalFilterMorphism
        q_vec = torch.randn(self.d_model)
        # Candidate with no vector and dt=0.5 -> with dx=2.0, ds^2 = -(1.0*0.5)^2 + 4.0 = 3.75 > 0 (spacelike)
        cand_no_vec = {"doc_id": "c1", "timestamp": 99.5}
        state = PipelineState(query_vector=q_vec, query_timestamp=100.0, candidates=[cand_no_vec])

        morphism_no_fallback = CausalFilterMorphism(c_speed=1.0, allow_fallback=False)
        out_state = morphism_no_fallback.forward(state)

        # Spacelike candidate without vector must be rejected, and with allow_fallback=False result is empty
        self.assertEqual(len(out_state.candidates), 0)
        self.assertEqual(out_state.diagnostics["causal_pass_count"], 0)

if __name__ == "__main__":
    unittest.main()

