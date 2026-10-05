"""Tests verifying Paideia honest failure for unimplemented real training."""

import pytest
from pathlib import Path
from paideia.data.splitter import DatasetSplits
from paideia.training.config import TrainingConfig
from paideia.training.runner import QLoRATrainingRunner, TrainingUnavailableError


def test_real_training_unimplemented_raises(tmp_path: Path):
    """Verify that requesting real training (allow_simulated=False) fails with TrainingUnavailableError."""
    cfg = TrainingConfig(
        output_dir=str(tmp_path),
        allow_simulated=False,
    )
    splits = DatasetSplits(train=[], dev=[], test=[])
    runner = QLoRATrainingRunner(cfg, splits)

    manifest = runner.run()
    assert manifest.status == "FAILED"
    assert "TrainingUnavailableError" in manifest.failure_reason
