"""Tests for manifest creation, SHA-256 checksums, and hardware detection."""

from pathlib import Path
from dokime.manifest import (
    capture_hardware,
    create_manifest,
    sha256_file,
    sha256_text,
)


def test_sha256_checksums(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("hello dokime", encoding="utf-8")

    file_hash = sha256_file(test_file)
    text_hash = sha256_text("hello dokime")

    assert len(file_hash) == 64
    assert file_hash == text_hash


def test_hardware_capture():
    hw = capture_hardware()
    assert "platform" in hw
    assert "python_version" in hw
    assert "cpu_count" in hw
    assert hw["cpu_count"] is not None and hw["cpu_count"] > 0


def test_create_manifest(tmp_path: Path):
    results_file = tmp_path / "results.jsonl"
    results_file.write_text('{"query_id": "1", "score": 1.0}\n', encoding="utf-8")

    dataset_file = tmp_path / "dataset.json"
    dataset_file.write_text('[{"id": "1"}]', encoding="utf-8")

    manifest_file = tmp_path / "manifest.json"

    manifest = create_manifest(
        run_id="run_test123",
        model_ids=["test-model"],
        task_ids=["test-task"],
        seed=42,
        temperature=0.0,
        max_tokens=256,
        results_jsonl_path=results_file,
        dataset_paths={"test-task": dataset_file},
        prompt_template="Solve: {question}",
        output_path=manifest_file,
    )

    assert manifest["run_id"] == "run_test123"
    assert manifest["seed"] == 42
    assert manifest["temperature"] == 0.0
    assert manifest["max_tokens"] == 256
    assert "results_jsonl" in manifest["checksums"]
    assert "dataset_test-task" in manifest["checksums"]
    assert manifest["checksums"]["results_jsonl"] == sha256_file(results_file)
    assert manifest["checksums"]["dataset_test-task"] == sha256_file(dataset_file)


def test_create_manifest_with_labeling_criterion(tmp_path: Path):
    """Verify labeling_criterion is captured in manifest (arXiv:2610.08026)."""
    results_file = tmp_path / "results.jsonl"
    results_file.write_text('{"query_id": "1", "score": 1.0}\n', encoding="utf-8")
    manifest_file = tmp_path / "manifest.json"

    manifest = create_manifest(
        run_id="run_criterion_test",
        model_ids=["test-model"],
        task_ids=["test-task"],
        seed=42,
        temperature=0.0,
        max_tokens=256,
        results_jsonl_path=results_file,
        dataset_paths={"test-task": None},
        labeling_criterion="reference_faithfulness",
        output_path=manifest_file,
    )

    assert manifest["labeling_criterion"] == "reference_faithfulness"
    assert manifest_file.exists()
