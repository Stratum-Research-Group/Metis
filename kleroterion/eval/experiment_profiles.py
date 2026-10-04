"""
Required experimental arms and configurations (Section 10.2).
"""

from typing import Dict
from kleroterion.interfaces import KleroterionConfig, SelectionMode

# Pre-defined configurations for required evaluation arms
ARM_CONFIGS: Dict[str, KleroterionConfig] = {
    # Arm 3: Dianoia scorer-only
    "dianoia_scorer_only": KleroterionConfig(
        proposal_mode="independent",
        dt=1.0,
        noise_rms_budget=0.0,
        proposal_count=1,
        selection_mode=SelectionMode.ARGMAX,
        temperature_initial=1.0,
        adapt_temperature=False,
    ),
    # Arm 4: Softmax-only (sample with fixed temperature, no noise)
    "softmax_only": KleroterionConfig(
        proposal_mode="independent",
        dt=1.0,
        noise_rms_budget=0.0,
        proposal_count=1,
        selection_mode=SelectionMode.SAMPLE,
        temperature_initial=1.0,
        adapt_temperature=False,
    ),
    # Arm 5: Gaussian-only (noise on, argmax of proposal scorer, M=1)
    "gaussian_only": KleroterionConfig(
        proposal_mode="independent",
        dt=1.0,
        noise_rms_budget=0.1,
        noise_hard_radius=0.2,
        proposal_count=1,
        selection_mode=SelectionMode.ARGMAX,
        temperature_initial=1.0,
        adapt_temperature=False,
    ),
    # Arm 6: Gaussian + softmax (noise on, sample marginal policy, M >= 2, feedback off)
    "gaussian_softmax": KleroterionConfig(
        proposal_mode="independent",
        dt=1.0,
        noise_rms_budget=0.1,
        noise_hard_radius=0.2,
        proposal_count=4,
        selection_mode=SelectionMode.SAMPLE,
        temperature_initial=1.0,
        adapt_temperature=False,
    ),
    # Arm 7: Full Kleroterion controller (noise on, sample marginal policy, marginal entropy feedback)
    "full_kleroterion_controller": KleroterionConfig(
        proposal_mode="independent",
        dt=1.0,
        noise_rms_budget=0.1,
        noise_hard_radius=0.2,
        proposal_count=4,
        selection_mode=SelectionMode.SAMPLE,
        temperature_initial=1.0,
        temperature_bounds=(0.2, 5.0),
        adapt_temperature=True,
        entropy_target=0.7,
        entropy_ema_alpha=0.2,
        controller_gain=0.1,
        entropy_deadband=0.05,
    ),
}
