"""Tests for LTHCMAdapter verifying Stoma-to-memory contract, generational indices, and failure modes."""

import hashlib
import json
import pytest
from pathlib import Path

from kybernetes.adapters.lthcm import LTHCMAdapter


def test_stoma_generational_index_loading(tmp_path: Path):
    """Verify loading real index via Stoma's current.json generational pointer."""
    gen_dir = tmp_path / "generations" / "gen_000001"
    gen_dir.mkdir(parents=True)

    manifest_content = {"manifest_version": "1.0", "generation": 1}
    manifest_bytes = json.dumps(manifest_content).encode("utf-8")
    (gen_dir / "manifest.json").write_bytes(manifest_bytes)
    man_hash = hashlib.sha256(manifest_bytes).hexdigest()

    passages_content = [
        {
            "passage_id": "pass-001",
            "source_id": "src-001",
            "source_title": "Test Doc",
            "text": "Kybernetes connects local memory through Stoma generational exports.",
            "keywords": ["kybernetes", "stoma", "memory"],
            "created_epoch": 1000.0,
        }
    ]
    (gen_dir / "passages_export.json").write_text(json.dumps(passages_content), encoding="utf-8")

    current_data = {
        "generation": 1,
        "updated_at": 1050.0,
        "manifest_hash": man_hash,
        "passages_count": 1,
    }
    (tmp_path / "current.json").write_text(json.dumps(current_data), encoding="utf-8")

    adapter = LTHCMAdapter(memory_root=str(tmp_path))
    assert not adapter.is_simulated
    assert adapter.generation_id == "gen-1"

    # Test retrieval
    citations = adapter.retrieve("What connects local memory?", as_of=2000.0)
    assert len(citations) == 1
    assert citations[0].passage_id == "pass-001"
    assert not citations[0].is_simulated


def test_stoma_manifest_hash_mismatch_fails(tmp_path: Path):
    """Verify tampering with manifest.json fails closed with ValueError."""
    gen_dir = tmp_path / "generations" / "gen_000002"
    gen_dir.mkdir(parents=True)

    (gen_dir / "manifest.json").write_text('{"tampered": true}', encoding="utf-8")
    (gen_dir / "passages_export.json").write_text(json.dumps([{"passage_id": "p1", "text": "Valid text"}]), encoding="utf-8")

    current_data = {
        "generation": 2,
        "manifest_hash": "e" * 64,  # mismatched hash
    }
    (tmp_path / "current.json").write_text(json.dumps(current_data), encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest hash mismatch"):
        LTHCMAdapter(memory_root=str(tmp_path))


def test_stoma_list_format_passages_json(tmp_path: Path):
    """Verify list-format passages.json does not raise AttributeError: 'list' object has no attribute 'get'."""
    passages_list = [
        {
            "passage_id": "p-100",
            "source_id": "s-1",
            "source_title": "List Format Doc",
            "text": "Phylax enforces execution capability bounds at point of action.",
            "keywords": ["phylax", "capability"],
            "created_epoch": 500.0,
        }
    ]
    (tmp_path / "passages.json").write_text(json.dumps(passages_list), encoding="utf-8")

    adapter = LTHCMAdapter(memory_root=str(tmp_path))
    assert not adapter.is_simulated

    results = adapter.retrieve("phylax capability", as_of=1000.0)
    assert len(results) == 1
    assert results[0].passage_id == "p-100"


def test_future_dated_passages_excluded(tmp_path: Path):
    """Verify passages with created_epoch > as_of are strictly excluded."""
    passages = [
        {
            "passage_id": "past_pass",
            "text": "Known historical evidence before cutoff.",
            "keywords": ["evidence"],
            "created_epoch": 1000.0,
        },
        {
            "passage_id": "future_pass",
            "text": "Future evidence created after query time.",
            "keywords": ["evidence"],
            "created_epoch": 2500.0,
        },
    ]
    (tmp_path / "passages.json").write_text(json.dumps(passages), encoding="utf-8")
    adapter = LTHCMAdapter(memory_root=str(tmp_path))

    results = adapter.retrieve("evidence", as_of=1500.0)
    assert len(results) == 1
    assert results[0].passage_id == "past_pass"


def test_corrupt_passage_missing_text_fails(tmp_path: Path):
    """Verify passages without text fail closed."""
    passages = [{"passage_id": "empty_pass", "text": "   "}]
    (tmp_path / "passages.json").write_text(json.dumps(passages), encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted passage records without text"):
        LTHCMAdapter(memory_root=str(tmp_path))
