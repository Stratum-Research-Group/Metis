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
    # Create mock weight files
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
    assert (adapter_dir / "hestia_registry_entry.json").exists()


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
