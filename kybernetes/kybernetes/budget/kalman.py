"""Kalman filter for dynamic resource burn-rate estimation."""

import math
from typing import Dict, Tuple

from kybernetes.contracts.enums import BudgetUnit


class KalmanBurnRateFilter:
    """
    1-D Kalman Filter tracking the instantaneous resource burn rate
    (resource units consumed per unit of completed progress).
    Maintains explicit units, bounded startup behavior, and variance estimation.
    """

    def __init__(
        self,
        unit: BudgetUnit,
        initial_rate: float,
        initial_variance: float = 1000.0,
        process_noise_q: float = 25.0,
        measurement_noise_r: float = 50.0,
    ):
        self.unit = unit
        self.rate = max(1.0, float(initial_rate))
        self.variance = max(1.0, float(initial_variance))
        self.q = process_noise_q
        self.r = measurement_noise_r
        self.observation_count = 0

    def predict(self) -> None:
        """Time update step: increase uncertainty by process noise."""
        self.variance += self.q

    def update(self, delta_usage: float, delta_progress: float) -> None:
        """
        Measurement update step.
        delta_usage: resources consumed during interval.
        delta_progress: progress fraction achieved [0, 1].
        """
        if delta_progress <= 1e-5 or delta_usage <= 0:
            return

        measured_rate = delta_usage / delta_progress
        self.predict()

        # Kalman gain
        kalman_gain = self.variance / (self.variance + self.r)
        # State update
        self.rate = self.rate + kalman_gain * (measured_rate - self.rate)
        # Covariance update
        self.variance = (1.0 - kalman_gain) * self.variance

        # Ensure physically meaningful non-negative rates
        self.rate = max(0.1, self.rate)
        self.observation_count += 1

    def estimate_remaining(self, remaining_fraction: float) -> Tuple[float, float]:
        """
        Estimate mean remaining usage and standard deviation.
        Returns: (remaining_mean, remaining_std)
        """
        rem_frac = max(0.0, min(1.0, remaining_fraction))
        expected_remaining = self.rate * rem_frac
        # Variance scales with remaining fraction squared plus process uncertainty
        std_remaining = math.sqrt(max(0.01, self.variance * (rem_frac ** 2)))
        return expected_remaining, std_remaining
