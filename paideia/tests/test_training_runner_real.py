"""Unit tests for real training execution path in QLoRATrainingRunner."""

import json
from pathlib import Path
import pytest

from paideia.data.splitter import DatasetSplits
from paideia.training.config import TrainingConfig
from paideia.training.runner import QLoRATrainingRunner


def test_real_training_shampoo_optimizer(tmp_path: Path):
    """Verify real training runs with Shampoo optimizer, updates weights, and outputs COMPLETED manifest."""
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "runs_shampoo"),
        task="router",
        optimizer="shampoo",
        epochs=5,
        batch_size=2,
        lr=0.001,
        rank=8,
        shampoo_precondition_freq=2,
        allow_simulated=False,
        seed=42,
    )

    # 4 synthetic training samples
    splits = DatasetSplits(train=[1, 2, 3, 4], dev=[], test=[])
    runner = QLoRATrainingRunner(cfg, splits)

    manifest = runner.run()

    assert manifest.status == "COMPLETED"
    assert manifest.is_simulated is False
    assert manifest.metrics["epochs_completed"] == 5
    assert manifest.metrics["optimizer"] == "shampoo"
    assert manifest.metrics["final_train_loss"] < manifest.metrics["initial_train_loss"]
    assert manifest.adapter_digest is not None

    checkpoint_dir = runner.output_dir / "checkpoints"
    assert (checkpoint_dir / "adapter_config.json").exists()
    assert (checkpoint_dir / "adapter_model.bin").exists()

    with open(checkpoint_dir / "adapter_config.json") as f:
        meta = json.load(f)
    assert meta["optimizer"] == "shampoo"
    assert meta["rank"] == 8


def test_real_training_adamw_optimizer(tmp_path: Path):
    """Verify real training runs with AdamW optimizer and updates weights."""
    cfg = TrainingConfig(
        output_dir=str(tmp_path / "runs_adamw"),
        task="walk_policy",
        optimizer="adamw",
        epochs=2,
        batch_size=2,
        lr=0.01,
        rank=4,
        allow_simulated=False,
        seed=101,
    )

    splits = DatasetSplits(train=["step_1", "step_2", "step_3"], dev=[], test=[])
    runner = QLoRATrainingRunner(cfg, splits)

    manifest = runner.run()

    assert manifest.status == "COMPLETED"
    assert manifest.is_simulated is False
    assert manifest.metrics["epochs_completed"] == 2
    assert manifest.metrics["optimizer"] == "adamw"
    assert manifest.metrics["final_train_loss"] < manifest.metrics["initial_train_loss"]
