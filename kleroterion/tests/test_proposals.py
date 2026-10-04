"""
Tests for Gaussian proposals, dimension-aware budgets, and radius clipping (Section 3.1, 3.2, 10.4).
"""

import numpy as np
import pytest

from kleroterion.errors import InvalidConfigError, InvalidProposalError
from kleroterion.proposals import (
    clip_increment_to_radius,
    generate_budgeted_gaussian_increment,
    verify_proposed_features,
)


def test_zero_budget_produces_zero_increment():
    rng = np.random.default_rng(42)
    inc, L = generate_budgeted_gaussian_increment(
        d=64,
        dt=1.0,
        noise_rms_budget=0.0,
        rng=rng,
    )
    assert np.all(inc == 0.0)
    assert np.all(L == 0.0)


@pytest.mark.parametrize("d", [16, 64, 256])
def test_dimension_independent_rms_budget(d):
    """
    Verify that across different dimensions d, the isotropic parameterization
    maintains the same expected RMS total displacement b.
    E[||inc||_2^2] = b^2 => RMS = b.
    """
    b = 0.5
    dt = 1.0
    n_samples = 5000
    rng = np.random.default_rng(123)

    sq_norms = []
    for _ in range(n_samples):
        inc, _ = generate_budgeted_gaussian_increment(
            d=d,
            dt=dt,
            noise_rms_budget=b,
            rng=rng,
        )
        sq_norms.append(np.sum(inc ** 2))

    empirical_rms = np.sqrt(np.mean(sq_norms))
    # Empirical RMS should match b = 0.5 within standard error
    assert np.isclose(empirical_rms, b, atol=0.02)


def test_raw_increments_match_covariance():
    """
    Verify raw increments match mean zero and covariance dt * L * L^T.
    """
    d = 4
    dt = 2.0
    b = 1.0
    rng = np.random.default_rng(777)
    n_samples = 15000

    increments = []
    for _ in range(n_samples):
        inc, L = generate_budgeted_gaussian_increment(
            d=d,
            dt=dt,
            noise_rms_budget=b,
            rng=rng,
        )
        increments.append(inc)

    samples = np.array(increments)
    empirical_mean = np.mean(samples, axis=0)
    empirical_cov = np.cov(samples, rowvar=False)
    expected_cov = dt * (L @ L.T)

    np.testing.assert_allclose(empirical_mean, 0.0, atol=0.04)
    np.testing.assert_allclose(empirical_cov, expected_cov, atol=0.04)


def test_fixed_L_variance_scales_with_dt():
    """
    Verify that holding L fixed, variance scales proportionally with dt.
    """
    d = 2
    b = 2.0
    rng = np.random.default_rng(999)
    # Fixed L
    L = 0.3 * np.eye(d)

    dt1 = 1.0
    dt2 = 4.0

    samples1 = [
        generate_budgeted_gaussian_increment(d, dt1, b, rng, diffusion_matrix=L)[0]
        for _ in range(10000)
    ]
    samples2 = [
        generate_budgeted_gaussian_increment(d, dt2, b, rng, diffusion_matrix=L)[0]
        for _ in range(10000)
    ]

    var1 = np.var(samples1, axis=0)
    var2 = np.var(samples2, axis=0)

    # var2 / var1 should be approx dt2 / dt1 = 4.0
    np.testing.assert_allclose(var2 / var1, dt2 / dt1, rtol=0.1)


def test_anisotropic_matrix_trace_budget_enforcement():
    d = 3
    dt = 1.0
    b = 1.0  # b^2 = 1.0
    rng = np.random.default_rng(42)

    # Valid L: dt * tr(LL^T) = 0.5^2 * 3 = 0.75 <= 1.0
    valid_L = 0.5 * np.eye(d)
    inc, _ = generate_budgeted_gaussian_increment(d, dt, b, rng, diffusion_matrix=valid_L)
    assert inc.shape == (d,)

    # Invalid L: dt * tr(LL^T) = 1.0^2 * 3 = 3.0 > 1.0
    invalid_L = np.eye(d)
    with pytest.raises(InvalidConfigError, match="violates RMS budget"):
        generate_budgeted_gaussian_increment(d, dt, b, rng, diffusion_matrix=invalid_L)


def test_clipping_to_hard_radius():
    increment = np.array([3.0, 4.0])  # Norm = 5.0
    hard_radius = 2.5

    clipped, was_clipped = clip_increment_to_radius(increment, hard_radius)
    assert was_clipped
    assert np.isclose(np.linalg.norm(clipped), hard_radius)
    # Direction preserved
    np.testing.assert_allclose(clipped / np.linalg.norm(clipped), increment / 5.0)

    # Not clipped if within radius
    within_inc = np.array([1.0, 1.0])  # Norm = sqrt(2) ~= 1.414 < 2.5
    not_clipped, was_clipped = clip_increment_to_radius(within_inc, hard_radius)
    assert not was_clipped
    np.testing.assert_allclose(not_clipped, within_inc)


def test_verify_proposed_features():
    orig = np.array([1.0, 2.0, 3.0])
    valid_prop = np.array([1.1, 2.1, 3.0])
    verify_proposed_features(orig, valid_prop, hard_radius=0.5)

    # Violation: displacement exceeds hard radius
    invalid_disp = np.array([2.0, 2.0, 3.0])  # disp = 1.0 > 0.5
    with pytest.raises(InvalidProposalError, match="exceeds hard radius"):
        verify_proposed_features(orig, invalid_disp, hard_radius=0.5)

    # Violation: non-finite values
    nan_prop = np.array([1.0, np.nan, 3.0])
    with pytest.raises(InvalidProposalError, match="non-finite"):
        verify_proposed_features(orig, nan_prop, hard_radius=0.5)
