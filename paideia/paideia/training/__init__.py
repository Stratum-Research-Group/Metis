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

from paideia.training.grpo import (
    CandidateCompletion,
    DianoiaStateReconstructor,
    GRPOConfig,
    GRPORewardEngine,
    GroupEvaluation,
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
    "GRPOConfig",
    "GRPORewardEngine",
    "DianoiaStateReconstructor",
    "CandidateCompletion",
    "GroupEvaluation",
]
