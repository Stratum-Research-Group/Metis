"""
Stochastic Resonance-Enhanced Query Retrieval
==============================================
Exploits the physical phenomenon of Stochastic Resonance (SR): adding controlled,
calibrated stochastic fluctuations to a nonlinear detection system amplifies
sub-threshold and weak signals rather than degrading them.

In LTHCM's quantum memory retrieval, the Positive Operator-Valued Measure (POVM)
expectation value:
    P(q | rho_i) = <q | rho_i | q> = sum_k w_k * (<q | v_k>)^2
acts as the threshold quadratic nonlinearity.

When a query is sub-threshold (ambiguous, vague, low-norm, or near the noise floor),
standard inner products collapse into uniform low-confidence scores. By injecting
calibrated Gaussian fluctuations xi ~ N(0, I) with learnable scale sigma:
    q^(s) = normalize(q + sigma * xi^(s))
and integrating expectation values over S stochastic probes, weak coherent alignments
are elevated above the activation threshold, breaking degeneracy and recovering retrieval
precision.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lthcm.core.superposition import SuperpositionMemoryStore


class StochasticResonanceQueryLayer(nn.Module):
    """
    Stochastic Resonance Noise-Injection and Ensemble Measurement Layer.

    Features:
      - Learnable per-dimension or scalar noise scale sigma.
      - Adaptive query-norm / signal-strength gating: weak/sub-threshold queries receive
        higher noise assistance, while strong confident queries receive near-zero perturbation.
      - Vectorized multi-probe POVM readout: evaluates S stochastic probes in a single
        einsum batch operation against density matrices.
      - SNR and resonance curve diagnostics to empirically locate optimal sigma*.
    """

    def __init__(
        self,
        d_model: int,
        base_sigma: float = 0.08,
        num_samples: int = 8,
        threshold: float = 0.05,
        adaptive: bool = True,
        subthreshold_cutoff: float = 0.5,
    ):
        """
        Args:
            d_model: Embedding dimension.
            base_sigma: Initial noise standard deviation sigma_0 > 0.
            num_samples: Default number of stochastic probes S to average.
            threshold: Nonlinear POVM activation barrier (values below threshold are sub-threshold).
            adaptive: Whether to dynamically scale sigma based on query signal strength.
            subthreshold_cutoff: Threshold norm or score below which query is marked sub-threshold.
        """
        super().__init__()
        self.d_model = d_model
        self.num_samples = max(1, num_samples)
        self.threshold = threshold
        self.adaptive = adaptive
        self.subthreshold_cutoff = subthreshold_cutoff

        # Parameterize sigma = softplus(raw_sigma) to guarantee sigma > 0
        # Initialize raw_sigma such that softplus(raw_sigma) == base_sigma
        # softplus(x) = log(1 + exp(x)) => x = log(exp(base_sigma) - 1)
        raw_val = math.log(math.expm1(base_sigma)) if base_sigma < 20 else base_sigma
        self.raw_sigma = nn.Parameter(torch.full((d_model,), raw_val, dtype=torch.float32))

        # Query signal strength gating network: estimates query clarity / confidence
        if adaptive:
            self.gate_net = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.gate_net = None

    @property
    def sigma(self) -> torch.Tensor:
        """Effective per-dimension noise scale vector."""
        return F.softplus(self.raw_sigma)

    def estimate_query_strength(self, query: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Estimate query signal strength.
        Args:
            query: Query tensor [D] or [B, D].
        Returns:
            strength: Float or Tensor in [0, 1] (0 = sub-threshold / weak, 1 = strong).
            is_subthreshold: Boolean indicating if strength < subthreshold_cutoff.
        """
        if query.dim() == 1:
            q = query.unsqueeze(0)
        else:
            q = query

        # Norm prior to unit-normalization measures initial embedding energy
        raw_norm = torch.norm(q, p=2, dim=-1, keepdim=True)
        norm_strength = torch.clamp(raw_norm / 1.0, 0.0, 1.0)

        if self.gate_net is not None:
            # Learned confidence / clarity assessment
            learned_strength = self.gate_net(q)
            # Combine geometric norm with learned clarity
            strength = 0.5 * (norm_strength + learned_strength)
        else:
            strength = norm_strength

        mean_str = float(strength.mean().item())
        is_sub = mean_str < self.subthreshold_cutoff
        return strength.squeeze(), is_sub

    def perturb_query(
        self,
        query: torch.Tensor,
        num_samples: Optional[int] = None,
        sigma_override: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Inject calibrated stochastic resonance noise into query vector.
        Args:
            query: Query vector [D] or [B, D].
            num_samples: Number of stochastic probes (default: self.num_samples).
            sigma_override: Optional explicit sigma value (disables learned sigma).
        Returns:
            probes: Tensor of normalized probe queries [S, D] or [B, S, D].
            diagnostics: Dictionary containing effective sigma and signal metrics.
        """
        S = num_samples if num_samples is not None else self.num_samples
        device = query.device
        dtype = query.dtype

        is_1d = (query.dim() == 1)
        if is_1d:
            q = query.unsqueeze(0)  # [1, D]
        else:
            q = query  # [B, D]

        B, D = q.shape

        strength, is_subthreshold = self.estimate_query_strength(q)

        # Compute effective sigma
        if sigma_override is not None:
            effective_sigma = torch.full((D,), float(sigma_override), device=device, dtype=dtype)
        else:
            base_s = self.sigma.to(device)
            if self.adaptive:
                # Stochastic Resonance principle: weak signals (strength -> 0) require
                # higher noise assistance; strong signals (strength -> 1) require minimal noise
                if strength.dim() == 0:
                    scale_factor = 2.0 * (1.0 - strength)
                    effective_sigma = base_s * scale_factor
                else:
                    scale_factor = 2.0 * (1.0 - strength.unsqueeze(-1))
                    effective_sigma = base_s * scale_factor
            else:
                effective_sigma = base_s

        # Generate S stochastic probes
        # Probe 0 is always the clean normalized query (anchoring coherent signal)
        q_norm = F.normalize(q, p=2, dim=-1)  # [B, D]

        if S <= 1:
            probes = q_norm.unsqueeze(1)  # [B, 1, D]
        else:
            noise = torch.randn(B, S - 1, D, device=device, dtype=dtype)  # [B, S-1, D]
            if effective_sigma.dim() == 1:
                scaled_noise = noise * effective_sigma.unsqueeze(0).unsqueeze(0)
            else:
                scaled_noise = noise * effective_sigma.unsqueeze(1)

            # Additive stochastic resonance perturbation
            perturbed = q.unsqueeze(1) + scaled_noise  # [B, S-1, D]
            perturbed_norm = F.normalize(perturbed, p=2, dim=-1)

            # Concatenate clean query at index 0
            probes = torch.cat([q_norm.unsqueeze(1), perturbed_norm], dim=1)  # [B, S, D]

        mean_sigma_val = float(effective_sigma.mean().item())
        diag = {
            "num_probes": S,
            "mean_sigma": mean_sigma_val,
            "signal_strength": float(strength.mean().item()) if strength.dim() > 0 else float(strength.item()),
            "is_subthreshold": is_subthreshold,
        }

        if is_1d:
            return probes.squeeze(0), diag  # [S, D]
        return probes, diag  # [B, S, D]

    def forward(
        self,
        query: torch.Tensor,
        noise_scale: Optional[float] = None,
        num_samples: Optional[int] = None,
        sigma_override: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Forward pass: generates stochastic resonance probes and returns the ensemble-enhanced query.

        Args:
            query: Query tensor [D] or [B, D].
            noise_scale: Alias for sigma_override (Gaussian perturbation scale).
            num_samples: Number of probes S to average (default: self.num_samples).
            sigma_override: Explicit perturbation scale sigma > 0.
        Returns:
            enhanced_query: Unit-normalized ensemble query [D] or [B, D].
        """
        eff_sigma = noise_scale if noise_scale is not None else sigma_override
        probes, _ = self.perturb_query(query, num_samples=num_samples, sigma_override=eff_sigma)
        if probes.dim() == 2:
            enhanced = torch.mean(probes, dim=0)
        else:
            enhanced = torch.mean(probes, dim=1)
        return F.normalize(enhanced, p=2, dim=-1)

    def resonate_retrieval(
        self,
        query: torch.Tensor,
        memory_store: SuperpositionMemoryStore,
        top_k: int = 5,
        num_samples: Optional[int] = None,
        sigma_override: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute Stochastic Resonance-enhanced POVM quantum retrieval.
        Averages expectation values over stochastic probes across all density matrices:
            E_xi[ <q^(s) | rho_i | q^(s)> ]

        Args:
            query: Query tensor [D].
            memory_store: Instance of SuperpositionMemoryStore.
            top_k: Number of top documents to return.
            num_samples: Number of probes S.
            sigma_override: Optional explicit noise standard deviation.
            threshold: Optional override for nonlinear activation threshold.
        Returns:
            List of retrieved documents with stochastic resonance expectation scores,
            variance/confidence metrics, and collapsed facet details.
        """
        if len(memory_store.documents) == 0:
            return []

        S = num_samples if num_samples is not None else self.num_samples
        probes, diag = self.perturb_query(query, num_samples=S, sigma_override=sigma_override)  # [S, D]
        probes = probes.to(memory_store.facet_tensor.device)

        # Batch POVM measurement:
        # probes: [S, D]
        # facet_tensor: [N_docs, K, D]
        # weight_tensor: [N_docs, K]
        # Inner products: <q^(s) | v_{ik}> -> [S, N_docs, K]
        inner_products = torch.einsum('sd, nkd -> snk', probes, memory_store.facet_tensor)

        # Nonlinear thresholding / activation barrier
        eff_threshold = threshold if threshold is not None else self.threshold
        if eff_threshold > 0.0:
            rectified_inner = F.relu(inner_products - eff_threshold)
            transition_probs = rectified_inner ** 2  # [S, N, K]
        else:
            transition_probs = inner_products ** 2  # [S, N, K]

        # Expectation per probe: sum_k w_{ik} * P_{s, i, k} -> [S, N]
        w = memory_store.weight_tensor.unsqueeze(0)  # [1, N, K]
        probe_expectations = torch.sum(w * transition_probs, dim=-1)  # [S, N]

        # Ensemble average expectation score: [N]
        ensemble_scores = torch.mean(probe_expectations, dim=0)
        # Score variance across probes indicates stability/robustness: [N]
        score_stds = torch.std(probe_expectations, dim=0) if S > 1 else torch.zeros_like(ensemble_scores)

        # Baseline clean score (probe 0): [N]
        clean_scores = probe_expectations[0]

        # Contrast enhancement / resonance boost
        resonance_boost = ensemble_scores - clean_scores

        # Collapsed facet identification across ensemble
        mean_transitions = torch.mean(transition_probs, dim=0)  # [N, K]
        best_facet_idx = torch.argmax(mean_transitions, dim=-1)  # [N]
        best_facet_energy = torch.max(mean_transitions, dim=-1).values  # [N]

        k = min(top_k, len(memory_store.documents))
        top_scores, top_indices = torch.topk(ensemble_scores, k=k)

        results = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist())):
            doc = memory_store.documents[idx]
            bf_idx = int(best_facet_idx[idx].item())
            clean_sc = float(clean_scores[idx].item())
            std_sc = float(score_stds[idx].item())
            boost_sc = float(resonance_boost[idx].item())

            results.append({
                "rank": rank + 1,
                "score": float(score),
                "clean_score": clean_sc,
                "score_std": std_sc,
                "resonance_boost": boost_sc,
                "doc_id": doc["id"],
                "content": doc["content"],
                "collapsed_facet_idx": bf_idx,
                "collapsed_facet_energy": float(best_facet_energy[idx].item()),
                "collapsed_facet_vector": memory_store.facet_tensor[idx, bf_idx].cpu(),
                "metadata": doc["metadata"],
                "sr_diagnostics": {
                    "is_subthreshold": diag["is_subthreshold"],
                    "mean_sigma": diag["mean_sigma"],
                    "num_probes": S,
                },
            })

        return results

    def compute_resonance_curve(
        self,
        query: torch.Tensor,
        memory_store: SuperpositionMemoryStore,
        target_doc_idx: int,
        distractor_doc_idx: int,
        sigmas: Optional[List[float]] = None,
        num_samples: int = 16,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Compute the Signal-to-Noise Ratio (SNR) or Contrast curve across various sigma values.
        Demonstrates the signature non-monotonic stochastic resonance peak at sigma* > 0.
        Args:
            query: Query tensor [D].
            memory_store: Instance of SuperpositionMemoryStore.
            target_doc_idx: Index of true relevant document.
            distractor_doc_idx: Index of noise/distractor document.
            sigmas: List of sigma values to probe.
            num_samples: Number of probes per sigma.
            threshold: Optional threshold override.
        Returns:
            Dictionary with sigma values, contrasts, SNRs, and optimal sigma*.
        """
        if sigmas is None:
            sigmas = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2]

        contrasts = []
        snrs = []

        for s in sigmas:
            res = self.resonate_retrieval(
                query=query,
                memory_store=memory_store,
                top_k=len(memory_store.documents),
                num_samples=num_samples,
                sigma_override=s,
                threshold=threshold,
            )
            # Find scores for target and distractor
            t_score = 0.0
            d_score = 0.0
            t_std = 1e-6
            d_std = 1e-6
            for r in res:
                if r["doc_id"] == memory_store.documents[target_doc_idx]["id"]:
                    t_score = r["score"]
                    t_std = max(1e-6, r["score_std"])
                elif r["doc_id"] == memory_store.documents[distractor_doc_idx]["id"]:
                    d_score = r["score"]
                    d_std = max(1e-6, r["score_std"])

            contrast = t_score - d_score
            # SNR defined as contrast divided by variance floor
            snr = contrast / (0.5 * (t_std + d_std))
            contrasts.append(float(contrast))
            snrs.append(float(snr))

        best_idx = int(torch.tensor(contrasts).argmax().item())
        optimal_sigma = sigmas[best_idx]

        return {
            "sigmas": sigmas,
            "contrasts": contrasts,
            "snrs": snrs,
            "optimal_sigma": optimal_sigma,
            "peak_contrast": contrasts[best_idx],
        }
