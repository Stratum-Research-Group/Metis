"""Unit tests for adapter export, Hestia registry entries, and privacy sanitization."""

from pathlib import Path
import pytest

from paideia.export.registry import export_hestia_adapter_registry
from paideia.export.privacy import (
    RetentionPolicy,
    sanitize_text,
    extract_sanitized_features,
)


def test_export_hestia_adapter_registry(tmp_path):
    adapter_dir = tmp_path / "mock_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    # Create mock manifest and weight files
    (adapter_dir / "manifest.json").write_text('{"status": "SUCCESS", "is_simulated": false}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"dummy_weights_12345")
    (adapter_dir / "adapter_config.json").write_text('{"rank": 16}')

    entry = export_hestia_adapter_registry(
        adapter_id="router-qlora-v1",
        adapter_dir=adapter_dir,
        base_model_name="meta-llama/Llama-3.1-8B-Instruct",
        task="question_router",
        training_data_fingerprint="data_sha256_mock",
        evaluation_run_id="run-eval-42",
        resource_footprint={"peak_vram_mb": 13500, "p95_latency_ms": 14.2},
    )

    assert entry.adapter_id == "router-qlora-v1"
    assert entry.adapter_digest != ""
    assert entry.base_model_digest.startswith("unresolved_alias:")
    assert (adapter_dir / "hestia_registry_entry.json").exists()


def test_export_rejects_missing_manifest(tmp_path):
    adapter_dir = tmp_path / "no_manifest"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="missing required manifest.json"):
        export_hestia_adapter_registry(
            adapter_id="test-1",
            adapter_dir=adapter_dir,
            base_model_name="llama",
            task="router",
            training_data_fingerprint="fp",
            evaluation_run_id="eval-1",
            resource_footprint={},
        )


def test_export_rejects_corrupt_manifest(tmp_path):
    adapter_dir = tmp_path / "corrupt_manifest"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "manifest.json").write_text("{corrupted json...")
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="corrupt or unreadable manifest"):
        export_hestia_adapter_registry(
            adapter_id="test-2",
            adapter_dir=adapter_dir,
            base_model_name="llama",
            task="router",
            training_data_fingerprint="fp",
            evaluation_run_id="eval-1",
            resource_footprint={},
        )


def test_export_rejects_config_only_without_weights(tmp_path):
    adapter_dir = tmp_path / "config_only"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "manifest.json").write_text('{"status": "SUCCESS", "is_simulated": false}')
    (adapter_dir / "adapter_config.json").write_text('{"rank": 16}')

    with pytest.raises(ValueError, match="contains no trained weight files"):
        export_hestia_adapter_registry(
            adapter_id="test-3",
            adapter_dir=adapter_dir,
            base_model_name="llama",
            task="router",
            training_data_fingerprint="fp",
            evaluation_run_id="eval-1",
            resource_footprint={},
        )


def test_export_rejects_simulated_run_without_explicit_flag(tmp_path):
    adapter_dir = tmp_path / "simulated"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "manifest.json").write_text('{"status": "SIMULATED", "is_simulated": true}')
    (adapter_dir / "adapter_model.safetensors").write_bytes(b"weights")

    with pytest.raises(ValueError, match="produced by a simulated run"):
        export_hestia_adapter_registry(
            adapter_id="test-4",
            adapter_dir=adapter_dir,
            base_model_name="llama",
            task="router",
            training_data_fingerprint="fp",
            evaluation_run_id="eval-1",
            resource_footprint={},
            allow_simulated=False,
        )


def test_export_rejects_empty_directory(tmp_path):
    adapter_dir = tmp_path / "empty"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError):
        export_hestia_adapter_registry(
            adapter_id="test-5",
            adapter_dir=adapter_dir,
            base_model_name="llama",
            task="router",
            training_data_fingerprint="fp",
            evaluation_run_id="eval-1",
            resource_footprint={},
        )


def test_privacy_sanitization():
    raw_text = "Contact user john.doe@example.com from IP 192.168.1.50 about secret."
    sanitized = sanitize_text(raw_text)
    assert "john.doe@example.com" not in sanitized
    assert "[EMAIL_REDACTED]" in sanitized
    assert "192.168.1.50" not in sanitized
    assert "[IP_REDACTED]" in sanitized

    features = extract_sanitized_features("Are both Paris and London capital cities?")
    assert features["has_both"] is True
    assert features["token_count"] > 0
    assert "question_hash" in features
