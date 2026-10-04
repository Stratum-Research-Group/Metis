"""Path canonicalization and containment utilities."""

import os
from pathlib import Path
import fnmatch


def canonicalize_path(target: str | Path, base_dir: Path | None = None) -> Path:
    """Canonicalize a filesystem path by resolving symlinks and relative traversals.
    
    If target is relative, resolves against base_dir (or current working directory).
    """
    path_obj = Path(target)
    if not path_obj.is_absolute():
        base = base_dir.resolve() if base_dir else Path.cwd().resolve()
        path_obj = base / path_obj
    
    # resolve() handles symlinks and '..' removal
    # If file doesn't exist yet, resolve parent and append name
    try:
        return path_obj.resolve()
    except (FileNotFoundError, RuntimeError):
        # Resolve existing ancestors and normalize
        return Path(os.path.abspath(str(path_obj)))


def is_contained_in_root(target: Path, root: Path) -> bool:
    """Verify that target path is strictly contained within root path.
    
    Uses canonical resolution and Path.is_relative_to to prevent string-prefix bypasses
    (e.g., /app/docs-secret vs /app/docs).
    """
    canon_target = target.resolve()
    canon_root = root.resolve()

    try:
        canon_target.relative_to(canon_root)
        return True
    except ValueError:
        return False


def matches_path_pattern(
    target: Path,
    root: Path | None,
    glob_pattern: str | None,
    base_dir: Path | None = None,
) -> bool:
    """Test if target matches root containment and optional glob pattern."""
    canon_target = target.resolve()

    if root and str(root) != "*":
        canon_root = root if root.is_absolute() else (base_dir or Path.cwd()).resolve() / root
        canon_root = canon_root.resolve()
        if not is_contained_in_root(canon_target, canon_root):
            return False

    if glob_pattern:
        pattern = glob_pattern.strip()
        # Support full path matching or basename matching
        target_str = str(canon_target)
        target_name = canon_target.name

        if fnmatch.fnmatch(target_str, pattern):
            return True
        if fnmatch.fnmatch(target_name, pattern):
            return True
        if root:
            try:
                rel = str(canon_target.relative_to(root.resolve()))
                if fnmatch.fnmatch(rel, pattern):
                    return True
            except ValueError:
                pass
        return False

    return True
