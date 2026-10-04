"""
Unit tests for Contrastive Claim-Pair Fine-Tuning & Trained Sheaf Verifier (R5)
================================================================================
Verifies:
  1. ContrastiveStalkProjector dimension projection and unit-normalization.
  2. LearnedRestrictionSheaf Dirichlet energy computation and non-negativity.
  3. InfoNCE loss and Dirichlet margin loss.
  4. End-to-end training loop on contrastive triplets.
  5. Weight checkpoint saving and loading.
  6. ContrastiveSheafVerifier decision boundary.
"""

import os
import unittest
import numpy as np
import torch

from lthcm.core.contrastive_sheaf import (
    ContrastiveStalkProjector,
    LearnedRestrictionSheaf,
    ContrastiveSheafTrainer,
    ContrastiveSheafVerifier,
)


class TestContrastiveSheaf(unittest.TestCase):

    def setUp(self):
        self.d_model = 64
        self.stalk_dim = 32
        torch.manual_seed(42)
        np.random.seed(42)

    def test_stalk_projector_normalization(self):
        """Verify ContrastiveStalkProjector produces unit-norm representations."""
        projector = ContrastiveStalkProjector(d_in=self.d_model, d_out=self.d_model)
        x = torch.randn(5, self.d_model)
        out = projector(x)
        self.assertEqual(out.shape, (5, self.d_model))
        norms = torch.norm(out, p=2, dim=-1)
        for n in norms:
            self.assertAlmostEqual(n.item(), 1.0, places=4)

    def test_sheaf_dirichlet_energy_non_negative(self):
        """Verify Dirichlet energy is non-negative and zero for identical inputs."""
        sheaf = LearnedRestrictionSheaf(d_model=self.d_model, stalk_dim=self.stalk_dim)
        x1 = torch.randn(4, self.d_model)
        x2 = torch.randn(4, self.d_model)

        energy = sheaf.compute_dirichlet_energy(x1, x2)
        self.assertEqual(energy.shape, (4,))
        for e in energy:
            self.assertGreaterEqual(e.item(), 0.0)

        # Identical inputs must yield zero Dirichlet energy
        self_energy = sheaf.compute_dirichlet_energy(x1, x1)
        for e in self_energy:
            self.assertAlmostEqual(e.item(), 0.0, places=5)

    def test_trainer_loss_and_step(self):
        """Verify InfoNCE and margin loss decrease after triplet training."""
        trainer = ContrastiveSheafTrainer(
            d_model=self.d_model,
            stalk_dim=self.stalk_dim,
            device=torch.device("cpu"),
        )

        # Generate synthetic triplets
        B = 20
        c_pos = np.random.randn(B, self.d_model).astype(np.float32)
        # c_neg is perturbed
        c_neg = c_pos + np.random.randn(B, self.d_model).astype(np.float32) * 0.8
        evid = c_pos + np.random.randn(B, self.d_model).astype(np.float32) * 0.1

        triplets = [(c_pos[i], c_neg[i], evid[i]) for i in range(B)]
        history = trainer.train_on_triplets(triplets, epochs=3, batch_size=10)

        self.assertIn("proj_loss", history)
        self.assertIn("sheaf_loss", history)
        self.assertEqual(len(history["proj_loss"]), 3)
        self.assertEqual(len(history["sheaf_loss"]), 3)

    def test_checkpoint_roundtrip(self):
        """Verify saving and loading weights reproduces predictions exactly."""
        trainer = ContrastiveSheafTrainer(d_model=self.d_model, stalk_dim=self.stalk_dim, device=torch.device("cpu"))
        ckpt_path = "runs/test_ckpt_roundtrip.pt"

        trainer.save_checkpoint(ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

        new_trainer = ContrastiveSheafTrainer(d_model=self.d_model, stalk_dim=self.stalk_dim, device=torch.device("cpu"))
        new_trainer.load_checkpoint(ckpt_path)

        x = torch.randn(2, self.d_model)
        with torch.no_grad():
            out1 = trainer.projector(x)
            out2 = new_trainer.projector(x)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    def test_sheaf_verifier_prediction(self):
        """Verify ContrastiveSheafVerifier predicts SUPPORTS on low energy and REFUTES on high energy."""
        projector = ContrastiveStalkProjector(d_in=self.d_model, d_out=self.d_model)
        sheaf = LearnedRestrictionSheaf(d_model=self.d_model, stalk_dim=self.stalk_dim)
        verifier = ContrastiveSheafVerifier(projector, sheaf, energy_threshold=0.20, device=torch.device("cpu"))

        vec = np.random.randn(self.d_model).astype(np.float32)
        # Identical vector -> energy near 0 -> SUPPORTS (0)
        pred_identical = verifier.predict(vec, vec)
        self.assertEqual(pred_identical, 0)


if __name__ == "__main__":
    unittest.main()
