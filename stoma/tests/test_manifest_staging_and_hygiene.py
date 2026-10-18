"""
Unit & Integration Tests for Stoma Ingestion Hygiene, Redaction & Staging (Phase 2).
Covers:
- Manifest staging boundary, path containment, and undeclared file rejection.
- Privacy redaction heuristics (API tokens, private IPs, emails, phone numbers).
- Gopher-style content assessment (repetition and boilerplate detection).
- Archive safety guard (decompression limits and path traversal defense).
"""

import io
from pathlib import Path
import tempfile
import zipfile
import pytest

from stoma.manifest_staging import (
    SecurityPathError,
    StagedFileRecord,
    StagingGate,
    StagingManifest,
)
from stoma.redaction import ContentAssessment, PrivacyRedactor
from stoma.safe_extract import ArchiveBombError, ArchiveSafetyGuard, SafeExtractor


def test_staging_gate_atomic_stage_and_validate(tmp_path):
    """Verify atomic staging and cryptographic manifest validation."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    manifest_path = staging_dir / "staging_manifest.json"

    # Create dummy source files
    src1 = tmp_path / "doc1.txt"
    src1.write_text("Document 1 content for staging.", encoding="utf-8")
    src2 = tmp_path / "sub" / "doc2.txt"
    src2.parent.mkdir(parents=True)
    src2.write_text("Document 2 content inside subfolder.", encoding="utf-8")

    manifest = StagingManifest(source_origin="test_pipeline")

    # Stage files
    rec1 = StagingGate.stage_file(src1, "doc1.txt", staging_dir, manifest)
    rec2 = StagingGate.stage_file(src2, "sub/doc2.txt", staging_dir, manifest)

    assert rec1.rel_path == "doc1.txt"
    assert rec2.rel_path == "sub/doc2.txt"
    assert len(manifest.files) == 2

    # Save and reload manifest
    manifest.save(manifest_path)
    loaded_manifest = StagingManifest.load(manifest_path)
    assert len(loaded_manifest.files) == 2

    # Validate staging directory passes
    valid, errors = StagingGate.validate_staging_dir(staging_dir, loaded_manifest)
    assert valid is True
    assert len(errors) == 0


def test_staging_gate_rejects_path_traversal(tmp_path):
    """Verify that path traversal attempts fail closed with SecurityPathError."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    manifest = StagingManifest()

    src = tmp_path / "legit.txt"
    src.write_text("Data", encoding="utf-8")

    # 1. Reject '..' traversal
    with pytest.raises(SecurityPathError):
        StagingGate.stage_file(src, "../escaped.txt", staging_dir, manifest)

    # 2. Reject absolute paths
    with pytest.raises(SecurityPathError):
        StagingGate.stage_file(src, "/etc/passwd", staging_dir, manifest)

    # 3. Reject '.' traversal
    with pytest.raises(SecurityPathError):
        StagingGate.stage_file(src, "folder/./file.txt", staging_dir, manifest)


def test_staging_gate_rejects_symlink_injection(tmp_path):
    """Verify that symlinks inside the staging hierarchy are strictly rejected."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    manifest = StagingManifest()

    # Create external target
    external_secret = tmp_path / "secret.env"
    external_secret.write_text("SECRET=123", encoding="utf-8")

    # Create symlink inside staging
    symlink_entry = staging_dir / "symlink_doc.txt"
    symlink_entry.symlink_to(external_secret)

    with pytest.raises(SecurityPathError):
        StagingGate.verify_path_containment("symlink_doc.txt", staging_dir)


def test_staging_gate_rejects_undeclared_disk_files(tmp_path):
    """
    WALDO Invariant: Stoma refuses to ingest any file not declared by the manifest.
    An unmanifested rogue file on disk causes validation failure.
    """
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    manifest = StagingManifest()

    src = tmp_path / "doc.txt"
    src.write_text("Valid data", encoding="utf-8")
    StagingGate.stage_file(src, "doc.txt", staging_dir, manifest)

    # Plant an unmanifested file directly in staging directory
    rogue_file = staging_dir / "rogue_payload.sh"
    rogue_file.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    valid, errors = StagingGate.validate_staging_dir(staging_dir, manifest)
    assert valid is False
    assert any("undeclared" in e.lower() for e in errors)


def test_staging_gate_detects_hash_tampering(tmp_path):
    """Verify that any post-staging content tampering is caught."""
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    manifest = StagingManifest()

    src = tmp_path / "doc.txt"
    src.write_text("Initial text", encoding="utf-8")
    StagingGate.stage_file(src, "doc.txt", staging_dir, manifest)

    # Tamper with file after staging
    staged_file = staging_dir / "doc.txt"
    staged_file.write_text("Modified malicious payload", encoding="utf-8")

    valid, errors = StagingGate.validate_staging_dir(staging_dir, manifest)
    assert valid is False
    assert any("hash mismatch" in e.lower() for e in errors)


def test_privacy_redactor_credentials_and_tokens():
    """Verify regex masking of credentials, tokens, and keys."""
    raw = (
        "Here is my OpenAI key: sk-abcdef1234567890abcdef1234567890\n"
        "GitHub token: ghp_123456789012345678901234567890123456\n"
        "AWS key: AKIAIOSFODNN7EXAMPLE\n"
        "Auth header: Bearer abcdef1234567890abcdef1234567890\n"
        "Config: api_key='supersecretpassword123'\n"
    )

    clean, count = PrivacyRedactor.redact(raw)
    assert count >= 5
    assert "sk-abcdef1234567890" not in clean
    assert "[REDACTED_API_KEY]" in clean
    assert "ghp_1234567890" not in clean
    assert "AKIAIOSFODNN7EXAMPLE" not in clean
    assert "[REDACTED_BEARER_TOKEN]" in clean
    assert "supersecretpassword123" not in clean


def test_privacy_redactor_pii_and_networks():
    """Verify masking of internal IPs, emails, and phone numbers."""
    raw = (
        "Contact me at alice.smith+dev@example.corp.org or call 415-555-2671.\n"
        "The internal server is running at 192.168.1.45 and gateway is 10.0.0.1.\n"
    )

    clean, count = PrivacyRedactor.redact(raw)
    assert count >= 4
    assert "alice.smith+dev@example.corp.org" not in clean
    assert "[REDACTED_EMAIL]" in clean
    assert "415-555-2671" not in clean
    assert "[REDACTED_PHONE]" in clean
    assert "192.168.1.45" not in clean
    assert "10.0.0.1" not in clean
    assert "[REDACTED_IP]" in clean


def test_content_assessment_repetition_and_boilerplate():
    """Verify Gopher-style repetition filters on pathological text."""
    # 1. Excessive character repetition
    pathological_chars = "Heading\n" + ("a" * 100) + "\nNormal sentence."
    is_spam, reason = ContentAssessment.is_boilerplate_or_spam(pathological_chars)
    assert is_spam is True
    assert "character repetition" in reason.lower()

    # 2. Excessive n-gram repetition (boilerplate loop)
    loop_text = " ".join(["the quick brown fox jumped over the lazy dog and ran away fast"] * 25)
    is_spam_loop, reason_loop = ContentAssessment.is_boilerplate_or_spam(loop_text)
    assert is_spam_loop is True
    assert "10-gram repetition" in reason_loop.lower()

    # 3. Legitimate natural language passes
    legit_text = (
        "In artificial intelligence, continual learning refers to the ability of an autonomous "
        "model to acquire new knowledge while retaining previously learned competencies. "
        "Unlike standard static batch training, online streaming data requires careful architectural "
        "mitigations to prevent catastrophic forgetting, such as topological memory structures."
    )
    is_spam_legit, _ = ContentAssessment.is_boilerplate_or_spam(legit_text)
    assert is_spam_legit is False


def test_archive_safety_guard_decompression_caps(tmp_path):
    """Verify that expansion bombs exceeding safe limits are rejected before extraction."""
    # 1. Normal safe zip passes
    safe_zip = tmp_path / "safe.zip"
    with zipfile.ZipFile(safe_zip, "w") as zf:
        zf.writestr("test.txt", "Hello World safe zip content.")
    ArchiveSafetyGuard.inspect_zip_archive(safe_zip)

    # 2. Archive exceeding file count limit
    many_files_zip = tmp_path / "many.zip"
    with zipfile.ZipFile(many_files_zip, "w") as zf:
        for i in range(1005):
            zf.writestr(f"file_{i}.txt", "x")

    with pytest.raises(ArchiveBombError) as exc:
        ArchiveSafetyGuard.inspect_zip_archive(many_files_zip)
    assert "contains 1005 files" in str(exc.value)

    # 3. Archive containing path traversal entry
    traversal_zip = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal_zip, "w") as zf:
        zf.writestr("../../etc/passwd", "evil content")

    with pytest.raises(ArchiveBombError) as exc:
        ArchiveSafetyGuard.inspect_zip_archive(traversal_zip)
    assert "path traversal" in str(exc.value)
