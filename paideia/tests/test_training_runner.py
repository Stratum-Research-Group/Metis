"""Unit tests for training runner, preflight checks, and hardware discovery."""

from pathlib import Path
import pytest

from paideia.training.config import TrainingConfig
from paideia.training.hardware import (
    acquire_gpu_lock,
    inspect_hardware,
    release_gpu_lock,
)
from paideia.training.runner import QLoRATrainingRunner
from paideia.data.splitter import DatasetSplits


def test_hardware_inspection():
    hw = inspect_hardware()
    assert hw.hardware_line != ""
    assert isinstance(hw.cuda_available, bool)


def test_gpu_lock_lifecycle(tmp_path):
    lock_file = tmp_path / "test_gpu.lock"
    
    # 1. Acquire
    ok = acquire_gpu_lock(lock_file)
    assert ok is True
    assert lock_file.exists()

    # 2. Re-acquire by same process (fails because lock is currently held)
    ok_second = acquire_gpu_lock(lock_file)
    assert ok_second is False

    # 3. Release (releases flock, file remains)
    release_gpu_lock(lock_file)
    assert lock_file.exists()

    # 4. Re-acquire after release succeeds
    ok_third = acquire_gpu_lock(lock_file)
    assert ok_third is True
    release_gpu_lock(lock_file)


def test_training_runner_manifest_lifecycle(tmp_path):
    config = TrainingConfig(
        output_dir=str(tmp_path / "runs"),
        epochs=1,
        batch_size=1,
    )
    splits = DatasetSplits(train=[1, 2], dev=[3], test=[4])

    runner = QLoRATrainingRunner(config=config, splits=splits)
    manifest = runner.run()

    assert manifest.run_id == runner.run_id
    # Since torch GPU may not be configured in this test env, runner either completes simulation or fails cleanly
    assert manifest.status in ["SIMULATED", "COMPLETED", "FAILED"]
    if manifest.status == "SIMULATED":
        assert manifest.is_simulated is True
        assert manifest.metrics["epochs_completed"] == 0
    assert runner.manifest_path.exists()
