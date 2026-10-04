"""Pytest fixtures and configuration."""

import pytest


@pytest.fixture
def sample_scores_a():
    """Known fixture scores for run A (mean = 0.6)."""
    return [1.0, 1.0, 0.0, 1.0, 0.0]


@pytest.fixture
def sample_scores_b():
    """Known fixture scores for run B (mean = 0.8)."""
    return [1.0, 1.0, 1.0, 1.0, 0.0]
