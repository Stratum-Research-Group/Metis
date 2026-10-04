"""Tests for path canonicalization, containment, and symlink evasion."""

from pathlib import Path
import pytest

from phylax.engine.canonical import canonicalize_path, is_contained_in_root, matches_path_pattern


def test_path_canonicalization_removes_traversal(tmp_path: Path):
    safe_dir = tmp_path / "safe"
    safe_dir.mkdir()
    target_file = safe_dir / "doc.txt"
    target_file.write_text("content")

    traversal_path = safe_dir / "subdir" / ".." / "doc.txt"
    canon = canonicalize_path(traversal_path)
    assert canon == target_file.resolve()


def test_string_prefix_bypass_prevention(tmp_path: Path):
    allowed_root = tmp_path / "documents"
    allowed_root.mkdir()
    
    # Evasion directory with matching string prefix
    evil_dir = tmp_path / "documents_secret"
    evil_dir.mkdir()
    evil_file = evil_dir / "passwords.txt"
    evil_file.write_text("secret")

    # Simple startswith() would pass: str(evil_file).startswith(str(allowed_root))
    assert str(evil_file).startswith(str(allowed_root))
    
    # Phylax is_contained_in_root MUST reject it!
    assert not is_contained_in_root(evil_file, allowed_root)


def test_symlink_pointing_outside_root(tmp_path: Path):
    allowed_root = tmp_path / "docs"
    allowed_root.mkdir()

    sensitive_root = tmp_path / "private"
    sensitive_root.mkdir()
    sensitive_file = sensitive_root / "secret.txt"
    sensitive_file.write_text("top_secret")

    # Symlink placed inside allowed_root, but target is outside
    symlink_file = allowed_root / "symlink_to_secret.txt"
    try:
        symlink_file.symlink_to(sensitive_file)
    except OSError:
        pytest.skip("Symlink creation not permitted in this environment")

    # Canonical resolution must resolve target to sensitive_file
    canon = canonicalize_path(symlink_file)
    assert canon == sensitive_file.resolve()

    # Containment check against allowed_root must return False!
    assert not is_contained_in_root(canon, allowed_root)
    assert not matches_path_pattern(canon, allowed_root, "**/*")
