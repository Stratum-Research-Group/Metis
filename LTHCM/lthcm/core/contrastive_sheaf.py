"""
Contrastive Claim-Pair Fine-Tuning & Learned Sheaf Restriction Maps (R5)
========================================================================
Implements semantic stalk fine-tuning and learned cellular sheaf restriction maps:
  1. ContrastiveStalkProjector: InfoNCE fine-tuning over minimal-edit claim pairs
     (claim+, claim-, evidence) from the VitaminC train split.
  2. LearnedRestrictionSheaf: Optimizes sheaf restriction maps F_{v -> e} with
     a margin loss on Dirichlet energy:
         L_sheaf = max(0, m + E(support) - E(refute))
     ensuring harmonic agreement for supporting evidence and high cohomological obstruction
     for contradictory claims.
  3. TrainedSheafVerifier: Evaluates claims against evidence as a calibrated verifier over
     semantically trained stalks.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class ContrastiveStalkProjector(nn.Module):
    """
    Non-linear projection head mapping base MiniLM embeddings into contrastive stalk space R^D
    where factual contradictions are geometrically separated.
    """

    def __init__(self, d_in: int = 384, d_out: int = 384, hidden_dim: int = 512):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, d_out),
        )
        self.residual = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects and unit-normalizes representations into stalk space."""
        h = self.net(x) + self.residual(x)
        return F.normalize(h, p=2, dim=-1)


class LearnedRestrictionSheaf(nn.Module):
    """
    Cellular Sheaf with learned restriction maps F_{v -> e}: R^D -> R^K
    optimized directly for Dirichlet energy separation between supporting and refuting evidence.
    """

    def __init__(self, d_model: int = 384, stalk_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.stalk_dim = stalk_dim
        # Learned restriction map into the edge agreement stalk
        self.restriction_map = nn.Linear(d_model, stalk_dim, bias=False)
        nn.init.orthogonal_(self.restriction_map.weight)

    def compute_dirichlet_energy(self, x_u: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
        """
        Dirichlet energy along an edge: E(u, v) = ||F(x_u) - F(x_v)||^2
        Args:
            x_u: [B, D] or [D]
            x_v: [B, D] or [D]
        Returns:
            Scalar or [B] Dirichlet energy
        """
        res_u = self.restriction_map(x_u)
        res_v = self.restriction_map(x_v)
        diff = res_v - res_u
        return torch.sum(diff ** 2, dim=-1)


class ContrastiveSheafTrainer:
    """
    Trainer implementing the leak-free R5 protocol:
      - Fits ContrastiveStalkProjector with InfoNCE loss over VitaminC train split.
      - Fits LearnedRestrictionSheaf with Dirichlet margin loss over frozen stalks.
    """

    def __init__(
        self,
        d_model: int = 384,
        stalk_dim: int = 64,
        temperature: float = 0.07,
        margin: float = 0.35,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.projector = ContrastiveStalkProjector(d_in=d_model, d_out=d_model).to(self.device)
        self.sheaf = LearnedRestrictionSheaf(d_model=d_model, stalk_dim=stalk_dim).to(self.device)
        self.temperature = temperature
        self.margin = margin

    def infonce_loss(
        self,
        claim_pos: torch.Tensor,
        claim_neg: torch.Tensor,
        evidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        InfoNCE objective with contrastive hard negative claim:
            L = -log [ exp(sim(c+, e)/tau) / ( sum_j exp(sim(c+, e_j)/tau) + exp(sim(c-, e_j)/tau) ) ]
        """
        # Shapes: [B, D]
        # Normalize
        c_pos = F.normalize(claim_pos, p=2, dim=-1)
        c_neg = F.normalize(claim_neg, p=2, dim=-1)
        evid = F.normalize(evidence, p=2, dim=-1)

        sim_pos = torch.sum(c_pos * evid, dim=-1) / self.temperature  # [B]
        sim_neg = torch.sum(c_neg * evid, dim=-1) / self.temperature  # [B]

        # Cross-entropy over positive vs negative + in-batch negatives
        batch_sims_pos = torch.matmul(c_pos, evid.T) / self.temperature  # [B, B]
        batch_sims_neg = torch.matmul(c_neg, evid.T) / self.temperature  # [B, B]

        all_logits = torch.cat([batch_sims_pos, batch_sims_neg], dim=1)  # [B, 2B]
        targets = torch.arange(claim_pos.size(0), device=self.device)

        return F.cross_entropy(all_logits, targets)

    def sheaf_margin_loss(
        self,
        stalk_pos: torch.Tensor,
        stalk_neg: torch.Tensor,
        stalk_evid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dirichlet energy margin loss:
            L_sheaf = max(0, margin + E(c+, e) - E(c-, e))
        """
        e_pos = self.sheaf.compute_dirichlet_energy(stalk_pos, stalk_evid)
        e_neg = self.sheaf.compute_dirichlet_energy(stalk_neg, stalk_evid)
        loss = F.relu(self.margin + e_pos - e_neg)
        return torch.mean(loss)

    def train_on_triplets(
        self,
        triplets: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],  # (c_pos_emb, c_neg_emb, evid_emb)
        epochs: int = 5,
        batch_size: int = 32,
        lr_proj: float = 1e-3,
        lr_sheaf: float = 5e-3,
        seed: Optional[int] = 42,
    ) -> Dict[str, List[float]]:
        """
        Train projector with InfoNCE and sheaf with Dirichlet margin loss.
        Uses a local seeded RNG to guarantee permutation determinism.
        """
        optimizer_proj = torch.optim.AdamW(self.projector.parameters(), lr=lr_proj, weight_decay=1e-4)
        optimizer_sheaf = torch.optim.AdamW(self.sheaf.parameters(), lr=lr_sheaf, weight_decay=1e-4)

        N = len(triplets)
        c_pos_arr = np.array([t[0] for t in triplets], dtype=np.float32)
        c_neg_arr = np.array([t[1] for t in triplets], dtype=np.float32)
        evid_arr = np.array([t[2] for t in triplets], dtype=np.float32)

        rng = np.random.default_rng(seed if seed is not None else 42)
        history = {"proj_loss": [], "sheaf_loss": []}

        for epoch in range(epochs):
            indices = rng.permutation(N)
            epoch_p_loss = 0.0
            epoch_s_loss = 0.0
            batches = 0

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                idx = indices[start:end]

                t_pos = torch.from_numpy(c_pos_arr[idx]).to(self.device)
                t_neg = torch.from_numpy(c_neg_arr[idx]).to(self.device)
                t_evid = torch.from_numpy(evid_arr[idx]).to(self.device)

                # Step 1: InfoNCE training on projector
                self.projector.train()
                optimizer_proj.zero_grad()
                proj_pos = self.projector(t_pos)
                proj_neg = self.projector(t_neg)
                proj_evid = self.projector(t_evid)

                loss_p = self.infonce_loss(proj_pos, proj_neg, proj_evid)
                loss_p.backward()
                optimizer_proj.step()

                # Step 2: Sheaf restriction map margin training (with detached stalks)
                self.sheaf.train()
                optimizer_sheaf.zero_grad()
                with torch.no_grad():
                    frozen_pos = self.projector(t_pos)
                    frozen_neg = self.projector(t_neg)
                    frozen_evid = self.projector(t_evid)

                loss_s = self.sheaf_margin_loss(frozen_pos, frozen_neg, frozen_evid)
                loss_s.backward()
                optimizer_sheaf.step()

                epoch_p_loss += float(loss_p.item())
                epoch_s_loss += float(loss_s.item())
                batches += 1

            avg_p = epoch_p_loss / max(1, batches)
            avg_s = epoch_s_loss / max(1, batches)
            history["proj_loss"].append(avg_p)
            history["sheaf_loss"].append(avg_s)

        self.projector.eval()
        self.sheaf.eval()
        return history

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None):
        """Save projector and sheaf weights along with run verification metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "projector": self.projector.state_dict(),
            "sheaf": self.sheaf.state_dict(),
            "temperature": self.temperature,
            "margin": self.margin,
            "metadata": metadata or {},
        }, path)

    def load_checkpoint(self, path: str, expected_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load projector and sheaf weights, validating metadata constraints if provided."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        data = torch.load(path, map_location=self.device, weights_only=True)
        saved_meta = data.get("metadata", {})
        if expected_metadata:
            if not saved_meta or not isinstance(saved_meta, dict):
                raise ValueError(
                    f"Checkpoint at {path} contains no metadata; refusing to load unverified checkpoint."
                )
            for key, val in expected_metadata.items():
                if key not in saved_meta:
                    raise ValueError(
                        f"Checkpoint metadata missing required field '{key}' (expected {val})"
                    )
                if saved_meta[key] != val:
                    raise ValueError(
                        f"Checkpoint metadata mismatch for '{key}': expected {val}, found {saved_meta[key]}"
                    )
        self.projector.load_state_dict(data["projector"])
        self.sheaf.load_state_dict(data["sheaf"])
        self.projector.eval()
        self.sheaf.eval()
        return saved_meta


class ContrastiveSheafVerifier:
    """
    Evaluator executing claim verification using the trained contrastive stalks
    and learned sheaf restriction maps (no NLI overwrite).
    """

    def __init__(
        self,
        projector: ContrastiveStalkProjector,
        sheaf: LearnedRestrictionSheaf,
        energy_threshold: float = 0.15,
        device: Optional[torch.device] = None,
    ):
        self.projector = projector
        self.sheaf = sheaf
        self.energy_threshold = energy_threshold
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def compute_energy(self, claim_emb: np.ndarray, evidence_emb: np.ndarray) -> float:
        """Compute Dirichlet energy between claim and evidence in stalk space."""
        with torch.no_grad():
            c_t = torch.from_numpy(claim_emb).unsqueeze(0).to(self.device)
            e_t = torch.from_numpy(evidence_emb).unsqueeze(0).to(self.device)
            s_c = self.projector(c_t)
            s_e = self.projector(e_t)
            energy = self.sheaf.compute_dirichlet_energy(s_c, s_e)
            return float(energy.item())

    def predict(self, claim_emb: np.ndarray, evidence_emb: np.ndarray) -> int:
        """
        Returns 0 (SUPPORTS) if energy < threshold else 1 (REFUTES).
        """
        energy = self.compute_energy(claim_emb, evidence_emb)
        return 0 if energy < self.energy_threshold else 1
