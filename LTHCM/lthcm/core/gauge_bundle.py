"""
Fibre Bundles & Lie Group Gauge Theory in Latent Space
======================================================
Based on differential geometry and non-Abelian Yang-Mills gauge theories.
Models disparate domains (e.g. medical, legal, vernacular) as fibres F_x over a base
semantic manifold B, with isometric parallel transport governed by an SO(D) Lie algebra connection.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LieAlgebraGaugeConnection(nn.Module):
    """
    Learned Lie-algebra-valued gauge connection A_mu in so(D).
    Produces skew-symmetric generator matrices Omega(x, y) = -Omega(x, y)^T.
    Parallel transport is the matrix exponential:
        Tau_{x -> y} = exp(Omega(x, y)) in SO(D)
    Guarantees strict norm-preservation (isometry): ||Tau(v)|| = ||v||.
    """

    def __init__(self, d_model: int, num_generators: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_generators = num_generators

        # Basis of skew-symmetric generators for so(D)
        # G_a = -G_a^T
        raw_generators = torch.randn(num_generators, d_model, d_model)
        skew_generators = 0.5 * (raw_generators - raw_generators.transpose(-1, -2))
        self.generators = nn.Parameter(skew_generators)

        # Field coupling network: maps base domain coordinates (x, y) to Lie algebra coefficients
        self.coupling_net = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.SiLU(),
            nn.Linear(64, num_generators),
        )

    def compute_connection_matrix(
        self,
        base_x: torch.Tensor,
        base_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute skew-symmetric so(D) connection matrix Omega(x, y) between domains.
        Args:
            base_x: Source base coordinate [..., D]
            base_y: Target base coordinate [..., D]
        Returns:
            Omega: [..., D, D] strictly skew-symmetric
        """
        combined = torch.cat([base_x, base_y], dim=-1)
        # Coefficients c_a for Lie algebra generators: [..., num_generators]
        coeffs = self.coupling_net(combined)

        # Lie algebra element: Omega = sum_a c_a * G_a
        # einsum: [..., G] x [G, D, D] -> [..., D, D]
        omega = torch.einsum("...g, gij -> ...ij", coeffs, self.generators)
        # Enforce strict skew-symmetry numerically
        omega_skew = 0.5 * (omega - omega.transpose(-1, -2))
        return omega_skew

    def parallel_transport(
        self,
        fibre_vector: torch.Tensor,
        base_source: torch.Tensor,
        base_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel transport a vector from source fibre to target fibre.
        Args:
            fibre_vector: Vector in source fibre [..., D]
            base_source: Source domain coordinate [..., D]
            base_target: Target domain coordinate [..., D]
        Returns:
            transported_vector: [..., D] in target fibre's frame of reference
            transport_matrix: SO(D) orthogonal matrix [..., D, D]
        """
        omega = self.compute_connection_matrix(base_source, base_target)

        # Compute matrix exponential Tau = exp(Omega) in SO(D)
        # For so(D), exp(Omega) is an exact rotation matrix (orthogonal)
        tau = torch.linalg.matrix_exp(omega)

        # Transport: v_target = Tau @ v_source
        if fibre_vector.dim() == 1:
            transported = (tau @ fibre_vector.unsqueeze(-1)).squeeze(-1)
        else:
            transported = torch.matmul(tau, fibre_vector.unsqueeze(-1)).squeeze(-1)

        return transported, tau

    def compute_curvature_holonomy(
        self,
        base_x: torch.Tensor,
        base_y: torch.Tensor,
        base_z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Wilson Loop holonomy around a closed triangle x -> y -> z -> x:
            W = Tau_{z -> x} @ Tau_{y -> z} @ Tau_{x -> y}
        If space is flat (zero gauge curvature), W = Identity matrix.
        Departure from Identity ||W - I||_F measures cross-domain curvature mismatch.
        """
        _, tau_xy = self.parallel_transport(base_x, base_x, base_y)
        _, tau_yz = self.parallel_transport(base_y, base_y, base_z)
        _, tau_zx = self.parallel_transport(base_z, base_z, base_x)

        wilson_loop = tau_zx @ tau_yz @ tau_xy
        eye = torch.eye(self.d_model, device=base_x.device)
        curvature_mismatch = torch.norm(wilson_loop - eye, p="fro")
        return curvature_mismatch


class FibreBundle(nn.Module):
    """
    High-level Fibre Bundle manager.
    Coordinates base topic projections, domain fibres, and gauge-equivariant alignment.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.connection = LieAlgebraGaugeConnection(d_model=d_model)

    def align_to_query_frame(
        self,
        candidate_vector: torch.Tensor,
        candidate_domain: torch.Tensor,
        query_vector: torch.Tensor,
        query_domain: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Transport candidate vector into query's contextual reference frame
        and evaluate alignment similarity.
        """
        aligned_v, tau = self.connection.parallel_transport(
            candidate_vector,
            base_source=candidate_domain,
            base_target=query_domain,
        )

        # Cosine similarity in the aligned frame
        sim = F.cosine_similarity(
            aligned_v.unsqueeze(0),
            query_vector.unsqueeze(0),
            dim=-1,
        ).item()

        return aligned_v, sim
