"""Unit tests for Paideia train and benchmark-shampoo CLI subcommands."""

import json
from pathlib import Path
from click.testing import CliRunner
import pytest

from paideia.cli import cli


def test_cli_train_simulated(tmp_path: Path):
    """Verify paideia train runs cleanly in simulated mode."""
    runner = CliRunner()
    out_dir = tmp_path / "sim_run"
    result = runner.invoke(
        cli,
        [
            "train",
            "--output-dir",
            str(out_dir),
            "--task",
            "router",
            "--optimizer",
            "shampoo",
            "--simulate",
        ],
    )
    assert result.exit_code == 0
    assert "Starting Paideia QLoRA Training Run" in result.output
    assert "Status: SIMULATED" in result.output
    manifest_files = list(out_dir.glob("*/manifest.json"))
    assert len(manifest_files) == 1
    assert manifest_files[0].exists()


def test_cli_train_real_with_fixture(tmp_path: Path):
    """Verify paideia train runs in real mode when provided with training samples."""
    runner = CliRunner()
    out_dir = tmp_path / "real_run"
    fixture_path = Path("paideia/tests/fixtures/lthcm_sample.json")

    result = runner.invoke(
        cli,
        [
            "train",
            "--data",
            str(fixture_path),
            "--output-dir",
            str(out_dir),
            "--task",
            "router",
            "--optimizer",
            "shampoo",
            "--epochs",
            "2",
            "--no-simulate",
        ],
    )
    assert result.exit_code == 0
    assert "Status: COMPLETED" in result.output
    assert "optimizer: shampoo" in result.output
    manifest_files = list(out_dir.glob("*/manifest.json"))
    assert len(manifest_files) == 1
    assert manifest_files[0].exists()
    checkpoint_dirs = list(out_dir.glob("*/checkpoints/adapter_model.bin"))
    assert len(checkpoint_dirs) == 1


def test_cli_benchmark_shampoo():
    """Verify paideia benchmark-shampoo CLI outputs comparative results."""
    runner = CliRunner()
    result = runner.invoke(cli, ["benchmark-shampoo", "--steps", "15"])
    assert result.exit_code == 0
    assert "Benchmark Results" in result.output
    assert "Shampoo Relative Gain" in result.output


def test_cli_sample_boundary_with_dokime_report(tmp_path: Path):
    """Verify paideia dataset sample-boundary handles Dokime run JSON report."""
    runner = CliRunner()
    report_file = tmp_path / "dokime_report.json"
    report_file.write_text(
        json.dumps({
            "benchmark": "dokime_eval",
            "attempt_records": [
                {"case_id": "c1", "attempt_index": 0, "seed": 42, "score": 1.0},
                {"case_id": "c1", "attempt_index": 1, "seed": 43, "score": 0.0},
                {"case_id": "c2", "attempt_index": 0, "seed": 42, "score": 1.0},
                {"case_id": "c2", "attempt_index": 1, "seed": 43, "score": 1.0},
            ],
        }),
        encoding="utf-8",
    )
    out_json = tmp_path / "selected.json"

    result = runner.invoke(
        cli,
        [
            "dataset",
            "sample-boundary",
            str(report_file),
            "--output",
            str(out_json),
            "--budget",
            "1",
            "--boundary-ratio",
            "1.0",
            "--retention-ratio",
            "0.0",
            "--hard-ratio",
            "0.0",
        ],
    )
    assert result.exit_code == 0
    assert "Paideia Boundary Practice Selection Audit" in result.output
    assert out_json.exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["selected_cases"] == ["c1"]
