"""Unit tests for second-order matrix preconditioner (Shampoo) and benchmark."""

import numpy as np
import pytest

from paideia.training.optimizer import (
    AdamWOptimizer,
    ShampooConfig,
    ShampooOptimizer,
    benchmark_shampoo_vs_adamw,
    matrix_inverse_fourth_root,
)


def test_matrix_inverse_fourth_root():
    """Verify matrix inverse 4th root on identity and positive definite matrices."""
    # Identity matrix: I^{-1/4} = I
    eye = np.eye(4, dtype=np.float32)
    inv4_eye = matrix_inverse_fourth_root(eye)
    assert np.allclose(inv4_eye, eye, atol=1e-5)

    # Diagonal matrix with known eigenvalues: diag(16, 81) -> diag(1/2, 1/3)
    diag_mat = np.diag([16.0, 81.0]).astype(np.float32)
    inv4_diag = matrix_inverse_fourth_root(diag_mat)
    expected = np.diag([0.5, 1.0 / 3.0]).astype(np.float32)
    assert np.allclose(inv4_diag, expected, atol=1e-4)

    # General symmetric positive definite matrix: M^{-1/4} @ M^{1/2} @ M^{-1/4} = I
    rng = np.random.RandomState(42)
    A = rng.randn(5, 5).astype(np.float32)
    M = A @ A.T + 0.1 * np.eye(5, dtype=np.float32)
    inv4 = matrix_inverse_fourth_root(M)
    M_half = inv4 @ inv4 # (M^{-1/4})^2 = M^{-1/2}
    M_recovered = np.linalg.inv(M_half @ M_half)
    assert np.allclose(M, M_recovered, atol=1e-3)


def test_shampoo_optimizer_step_2d_and_1d():
    """Verify Shampoo updates 1D vectors and 2D parameter matrices."""
    rng = np.random.RandomState(42)
    w_2d = rng.randn(8, 16).astype(np.float32)
    b_1d = rng.randn(8).astype(np.float32)

    cfg = ShampooConfig(lr=0.05, precondition_freq=2, weight_decay=1e-4)
    opt = ShampooOptimizer([w_2d, b_1d], config=cfg)

    w_initial = w_2d.copy()
    b_initial = b_1d.copy()

    # Step 1: Preconditioner update on first step
    grad_w = rng.randn(8, 16).astype(np.float32)
    grad_b = rng.randn(8).astype(np.float32)
    opt.step([grad_w, grad_b])

    assert not np.allclose(w_2d, w_initial)
    assert not np.allclose(b_1d, b_initial)
    assert opt.step_count == 1

    # Step 2: Second step (triggers preconditioner update at freq=2)
    opt.step([grad_w, grad_b])
    assert opt.step_count == 2
    assert opt.inv_left[0] is not None
    assert opt.inv_right[0] is not None


def test_adamw_optimizer_step():
    """Verify AdamW reference optimizer step on 2D matrix."""
    rng = np.random.RandomState(42)
    w = rng.randn(4, 4).astype(np.float32)
    w_init = w.copy()
    opt = AdamWOptimizer([w], lr=0.01, weight_decay=1e-4)

    grad = np.ones_like(w)
    opt.step([grad])
    assert opt.step_count == 1
    assert not np.allclose(w, w_init)


def test_benchmark_shampoo_vs_adamw_convergence():
    """Verify controlled benchmark runs and achieves loss reduction."""
    res = benchmark_shampoo_vs_adamw(dim_out=8, dim_in=16, num_steps=20, seed=123)

    assert res["num_steps"] == 20
    assert res["adamw_final_loss"] < res["adamw_initial_loss"]
    assert res["shampoo_final_loss"] < res["shampoo_initial_loss"]
    assert isinstance(res["shampoo_relative_gain_percent"], float)
    assert len(res["losses_adamw"]) == 20
    assert len(res["losses_shampoo"]) == 20
