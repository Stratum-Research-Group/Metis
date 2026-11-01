"""Paideia training and fine-tuning runner package."""

from paideia.training.config import TrainingConfig
from paideia.training.hardware import (
    HardwareProfile,
    inspect_hardware,
    acquire_gpu_lock,
    release_gpu_lock,
)
from paideia.training.runner import (
    QLoRATrainingRunner,
    TrainingPreflightError,
    compute_dir_sha256,
)
from paideia.training.optimizer import (
    AdamWOptimizer,
    ShampooConfig,
    ShampooOptimizer,
    benchmark_shampoo_vs_adamw,
    matrix_inverse_fourth_root,
)

from paideia.training.grpo import (
    CandidateCompletion,
    DianoiaStateReconstructor,
    GRPOConfig,
    GRPORewardEngine,
    GroupEvaluation,
)

from paideia.training.loop import (
    ContinuousPolicyRetrainingCoordinator,
    RetrainingCoordinatorConfig,
    RetrainingCycleResult,
)

__all__ = [
    "TrainingConfig",
    "HardwareProfile",
    "inspect_hardware",
    "acquire_gpu_lock",
    "release_gpu_lock",
    "QLoRATrainingRunner",
    "TrainingPreflightError",
    "compute_dir_sha256",
    "ShampooOptimizer",
    "ShampooConfig",
    "AdamWOptimizer",
    "matrix_inverse_fourth_root",
    "benchmark_shampoo_vs_adamw",
    "GRPOConfig",
    "GRPORewardEngine",
    "DianoiaStateReconstructor",
    "CandidateCompletion",
    "GroupEvaluation",
    "ContinuousPolicyRetrainingCoordinator",
    "RetrainingCoordinatorConfig",
    "RetrainingCycleResult",
]
