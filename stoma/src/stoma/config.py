"""
Stoma Configuration and Path Safety.
Controls permitted ingestion roots, storage directories, and forbidden path boundaries.
Provides deterministic root identification and runtime dependency verification.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml


DEFAULT_CONFIG_PATH = Path(".stoma/config.yaml")
DEFAULT_STORAGE_DIR = Path(".stoma")
DEFAULT_INDEX_PATH = DEFAULT_STORAGE_DIR / "index.pkl"
DEFAULT_MANIFEST_PATH = DEFAULT_STORAGE_DIR / "manifest.json"

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_D_MODEL = 384
DEFAULT_NUM_FACETS = 1
DEFAULT_MAX_CHUNK_TOKENS = 220
DEFAULT_CHUNK_OVERLAP_TOKENS = 30
DEFAULT_BATCH_SIZE = 32

# Folders that must never be traversed or indexed
DEFAULT_FORBIDDEN_DIR_NAMES: Set[str] = {
    ".git",
    ".stoma",
    ".venv",
    "venv",
    "ENV",
    "env",
    "__pycache__",
    "node_modules",
    "deps",
    "submodules",
    "build",
    "dist",
    ".pytest_cache",
}

PINNED_SUBMODULES: Dict[str, str] = {
    "deps/LTHCM": "336f4af9f624aca5b4978a5e3ce222ce9638da35",
    "deps/dianoia": "da7ce83cb05f3b44764fc59ab5c325995c6fd07c",
    "deps/hestia": "edd1bcf411180dd3c9b1171057d03f400b3bfb96",
    "deps/dokime": "43c025348098aa9511f56acfb26ed19a62983177",
}


class DependencyOriginError(RuntimeError):
    """Raised when runtime dependencies (LTHCM, dianoia) do not originate from expected pinned submodules."""
    pass


def compute_root_id(root: Path) -> str:
    """Compute deterministic stable 8-character ID for a root path."""
    canonical = str(root.resolve()).replace("\\", "/")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


LOCKED_LTHCM_COMMIT = "336f4af9f624aca5b4978a5e3ce222ce9638da35"
LOCKED_DIANOIA_COMMIT = "da7ce83cb05f3b44764fc59ab5c325995c6fd07c"


def _read_top_level_gitlink(stoma_root: Path, submodule_rel: str) -> Optional[str]:
    """Read the top-level parent repository gitlink SHA for a submodule."""
    import subprocess
    try:
        res = subprocess.run(
            ["git", "rev-parse", f"HEAD:{submodule_rel}"],
            cwd=stoma_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    # Fallback: check ls-tree
    try:
        res2 = subprocess.run(
            ["git", "ls-tree", "HEAD", submodule_rel],
            cwd=stoma_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res2.returncode == 0 and res2.stdout.strip():
            parts = res2.stdout.strip().split()
            if len(parts) >= 3:
                return parts[2]
    except Exception:
        pass

    return None


def _read_submodule_commit(submodule_path: Path) -> Optional[str]:
    """Read the checked-out commit SHA for a submodule."""
    import subprocess
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=submodule_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass

    # Direct file inspection fallback
    git_file = submodule_path / ".git"
    if git_file.is_file():
        try:
            line = git_file.read_text(encoding="utf-8").strip()
            if line.startswith("gitdir:"):
                gitdir_rel = line.split(":", 1)[1].strip()
                actual_gitdir = (submodule_path / gitdir_rel).resolve()
                head_file = actual_gitdir / "HEAD"
                if head_file.is_file():
                    head_val = head_file.read_text(encoding="utf-8").strip()
                    if head_val.startswith("ref:"):
                        ref_rel = head_val.split(":", 1)[1].strip()
                        ref_file = actual_gitdir / ref_rel
                        if ref_file.is_file():
                            return ref_file.read_text(encoding="utf-8").strip()
                    return head_val
        except Exception:
            pass
    elif git_file.is_dir():
        head_file = git_file / "HEAD"
        if head_file.is_file():
            try:
                head_val = head_file.read_text(encoding="utf-8").strip()
                if head_val.startswith("ref:"):
                    ref_rel = head_val.split(":", 1)[1].strip()
                    ref_file = git_file / ref_rel
                    if ref_file.is_file():
                        return ref_file.read_text(encoding="utf-8").strip()
                return head_val
            except Exception:
                pass
    return None


def compute_component_digest(component_dir: Path, target_subdirs: Optional[List[str]] = None) -> str:
    """
    Compute deterministic SHA-256 digest over code and metadata files in a component directory.
    Ignores ephemeral build/cache/virtualenv directories (.venv, __pycache__, .git, etc.).
    """
    h = hashlib.sha256()
    candidate_files = []
    scan_roots = [component_dir / sub for sub in target_subdirs] if target_subdirs else [component_dir]
    for sroot in scan_roots:
        if sroot.is_file():
            rel = sroot.relative_to(component_dir).as_posix()
            candidate_files.append((rel, sroot))
        elif sroot.is_dir():
            for p in sroot.rglob("*"):
                if not p.is_file():
                    continue
                rel = p.relative_to(component_dir).as_posix()
                parts = rel.split("/")
                if any(x in parts for x in [
                    "__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache",
                    ".venv", "venv", "runs", ".eggs"
                ]):
                    continue
                if any(part.endswith(".egg-info") for part in parts):
                    continue
                if rel.endswith(".pyc") or rel.endswith(".pyo") or rel.endswith(".pt"):
                    continue
                candidate_files.append((rel, p))
    candidate_files.sort(key=lambda x: x[0])
    for rel, p in candidate_files:
        h.update(rel.encode("utf-8"))
        try:
            content = p.read_bytes()
        except OSError as e:
            raise DependencyOriginError(f"Cannot read component file '{p}': {e}") from e
        h.update(hashlib.sha256(content).digest())
    return h.hexdigest()


def verify_runtime_dependencies(
    stoma_root: Optional[Path] = None,
    provenance_manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Verify that lthcm and dianoia are imported from the exact top-level submodule directories
    or verified monorepo sibling directories with honest provenance.
    Fails closed if gitlink or submodule checkout commit is missing, unreadable, or divergent,
    or if monorepo sibling content digest differs from release manifest.
    """
    if stoma_root is None:
        # Resolve Stoma project root from this file: src/stoma/config.py -> project root
        stoma_root = Path(__file__).resolve().parent.parent.parent

    is_monorepo = False
    if (stoma_root / "deps" / "LTHCM").is_dir():
        expected_lthcm_root = (stoma_root / "deps" / "LTHCM").resolve()
    elif (stoma_root.parent / "LTHCM").is_dir():
        expected_lthcm_root = (stoma_root.parent / "LTHCM").resolve()
        is_monorepo = True
    else:
        raise DependencyOriginError(f"Runtime import violation: Required LTHCM not found at {stoma_root / 'deps' / 'LTHCM'} or {stoma_root.parent / 'LTHCM'}")

    if (stoma_root / "deps" / "dianoia").is_dir():
        expected_dianoia_root = (stoma_root / "deps" / "dianoia").resolve()
    elif (stoma_root.parent / "dianoia").is_dir():
        expected_dianoia_root = (stoma_root.parent / "dianoia").resolve()
        is_monorepo = True
    else:
        raise DependencyOriginError(f"Runtime import violation: Required dianoia not found at {stoma_root / 'deps' / 'dianoia'} or {stoma_root.parent / 'dianoia'}")

    lthcm_gitlink = None
    dianoia_gitlink = None
    lthcm_checkout = None
    dianoia_checkout = None
    lthcm_digest = None
    dianoia_digest = None
    monorepo_revision = None

    if not is_monorepo:
        # Submodule deployment: check top-level gitlink SHAs and exact submodule checkout commits
        lthcm_gitlink = _read_top_level_gitlink(stoma_root, "deps/LTHCM")
        if not lthcm_gitlink:
            raise DependencyOriginError("Runtime import violation: Cannot verify top-level gitlink SHA for 'deps/LTHCM'. Uninitialized or unreadable gitlink.")
        if lthcm_gitlink != LOCKED_LTHCM_COMMIT:
            raise DependencyOriginError(
                f"Runtime import violation: Submodule commit violation: LTHCM gitlink is '{lthcm_gitlink}', "
                f"expected locked pin '{LOCKED_LTHCM_COMMIT}'."
            )

        dianoia_gitlink = _read_top_level_gitlink(stoma_root, "deps/dianoia")
        if not dianoia_gitlink:
            raise DependencyOriginError("Runtime import violation: Cannot verify top-level gitlink SHA for 'deps/dianoia'. Uninitialized or unreadable gitlink.")
        if dianoia_gitlink != LOCKED_DIANOIA_COMMIT:
            raise DependencyOriginError(
                f"Runtime import violation: Submodule commit violation: dianoia gitlink is '{dianoia_gitlink}', "
                f"expected locked pin '{LOCKED_DIANOIA_COMMIT}'."
            )

        lthcm_checkout = _read_submodule_commit(expected_lthcm_root)
        if not lthcm_checkout:
            raise DependencyOriginError("Runtime import violation: Cannot verify checked-out commit SHA for 'deps/LTHCM'. Submodule HEAD unreadable.")
        if lthcm_checkout != LOCKED_LTHCM_COMMIT:
            raise DependencyOriginError(
                f"Runtime import violation: Submodule commit violation: LTHCM is at '{lthcm_checkout}', "
                f"expected locked pin '{LOCKED_LTHCM_COMMIT}'."
            )

        dianoia_checkout = _read_submodule_commit(expected_dianoia_root)
        if not dianoia_checkout:
            raise DependencyOriginError("Runtime import violation: Cannot verify checked-out commit SHA for 'deps/dianoia'. Submodule HEAD unreadable.")
        if dianoia_checkout != LOCKED_DIANOIA_COMMIT:
            raise DependencyOriginError(
                f"Runtime import violation: Submodule commit violation: dianoia is at '{dianoia_checkout}', "
                f"expected locked pin '{LOCKED_DIANOIA_COMMIT}'."
            )
    else:
        # Monorepo deployment: verify locked commit pin integrity
        if LOCKED_LTHCM_COMMIT != "336f4af9f624aca5b4978a5e3ce222ce9638da35":
            raise DependencyOriginError(
                f"Runtime import violation: Submodule commit violation: LOCKED_LTHCM_COMMIT constant altered from expected pin."
            )
        if LOCKED_DIANOIA_COMMIT != "da7ce83cb05f3b44764fc59ab5c325995c6fd07c":
            raise DependencyOriginError(
                f"Runtime import violation: Submodule commit violation: LOCKED_DIANOIA_COMMIT constant altered from expected pin."
            )

        # Honest monorepo revision and component tree/content digests from release manifest
        monorepo_root = stoma_root.parent
        monorepo_revision = _read_submodule_commit(monorepo_root)
        if not monorepo_revision:
            raise DependencyOriginError("Runtime import violation: Cannot verify monorepo git revision for vendored dependencies.")

        # Locate vendored provenance manifest
        manifest_path = provenance_manifest_path
        if manifest_path is None:
            candidates = [
                stoma_root / "vendored_provenance.json",
                stoma_root / "release_manifest.json",
                stoma_root / "src" / "stoma" / "vendored_provenance.json",
            ]
            for cand in candidates:
                if cand.is_file():
                    manifest_path = cand
                    break

        if manifest_path is None or not manifest_path.is_file():
            raise DependencyOriginError(
                f"Runtime import violation: Missing vendored provenance manifest for monorepo dependencies. "
                f"Expected manifest at {stoma_root / 'vendored_provenance.json'}."
            )

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            raise DependencyOriginError(f"Runtime import violation: Failed to parse vendored provenance manifest: {e}") from e

        deps_info = manifest_data.get("dependencies", {})
        if "LTHCM" not in deps_info or "dianoia" not in deps_info:
            raise DependencyOriginError("Runtime import violation: Vendored provenance manifest missing LTHCM or dianoia entries.")

        # Validate LTHCM
        lthcm_spec = deps_info["LTHCM"]
        for req_rel in lthcm_spec.get("required_paths", []):
            if not (expected_lthcm_root / req_rel).is_file():
                raise DependencyOriginError(f"Runtime import violation: Required file '{req_rel}' missing in vendored LTHCM at '{expected_lthcm_root}'.")
        lthcm_digest = compute_component_digest(expected_lthcm_root, lthcm_spec.get("target_subdirs"))
        expected_lthcm_digest = lthcm_spec.get("content_sha256")
        if not expected_lthcm_digest or lthcm_digest != expected_lthcm_digest:
            raise DependencyOriginError(
                f"Runtime import violation: Content digest mismatch for vendored LTHCM: "
                f"expected '{expected_lthcm_digest}', got '{lthcm_digest}'. Sibling source may be altered or corrupted."
            )

        # Validate dianoia
        dianoia_spec = deps_info["dianoia"]
        for req_rel in dianoia_spec.get("required_paths", []):
            if not (expected_dianoia_root / req_rel).is_file():
                raise DependencyOriginError(f"Runtime import violation: Required file '{req_rel}' missing in vendored dianoia at '{expected_dianoia_root}'.")
        dianoia_digest = compute_component_digest(expected_dianoia_root, dianoia_spec.get("target_subdirs"))
        expected_dianoia_digest = dianoia_spec.get("content_sha256")
        if not expected_dianoia_digest or dianoia_digest != expected_dianoia_digest:
            raise DependencyOriginError(
                f"Runtime import violation: Content digest mismatch for vendored dianoia: "
                f"expected '{expected_dianoia_digest}', got '{dianoia_digest}'. Sibling source may be altered or corrupted."
            )

    try:
        import lthcm
        import dianoia
    except ImportError as e:
        raise DependencyOriginError(f"Failed to import core dependency: {e}") from e

    lthcm_raw = getattr(lthcm, "__file__", None) or (lthcm.__path__[0] if hasattr(lthcm, "__path__") else None)
    dianoia_raw = getattr(dianoia, "__file__", None) or (dianoia.__path__[0] if hasattr(dianoia, "__path__") else None)
    if not lthcm_raw or not dianoia_raw:
        raise DependencyOriginError("Could not resolve filesystem path for imported core dependency lthcm or dianoia.")

    lthcm_file = Path(lthcm_raw).resolve()
    dianoia_file = Path(dianoia_raw).resolve()


    # Check origin
    if expected_lthcm_root not in lthcm_file.parents and lthcm_file != expected_lthcm_root:
        raise DependencyOriginError(
            f"Runtime import violation: 'lthcm' imported from '{lthcm_file}', "
            f"expected path inside '{expected_lthcm_root}'."
        )

    if expected_dianoia_root not in dianoia_file.parents and dianoia_file != expected_dianoia_root:
        raise DependencyOriginError(
            f"Runtime import violation: 'dianoia' imported from '{dianoia_file}', "
            f"expected path inside '{expected_dianoia_root}'."
        )

    return {
        "layout": "monorepo_vendored" if is_monorepo else "submodule",
        "monorepo_revision": monorepo_revision,
        "lthcm_path": str(lthcm_file),
        "lthcm_gitlink_sha": lthcm_gitlink,
        "lthcm_checkout_sha": lthcm_checkout,
        "lthcm_content_digest": lthcm_digest,
        "dianoia_path": str(dianoia_file),
        "dianoia_gitlink_sha": dianoia_gitlink,
        "dianoia_checkout_sha": dianoia_checkout,
        "dianoia_content_digest": dianoia_digest,
        "lthcm_origin_ok": True,
        "dianoia_origin_ok": True,
    }


class StomaConfig:
    """Configuration container for Stoma pipeline."""

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        storage_dir: Path = DEFAULT_STORAGE_DIR,
        index_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        forbidden_dir_names: Optional[Set[str]] = None,
    ):
        if allowed_roots is None:
            # Default to current working directory, NEVER entire home directory
            allowed_roots = [Path.cwd().resolve()]
        else:
            allowed_roots = [r.resolve() for r in allowed_roots]

        self.allowed_roots = allowed_roots
        self.storage_dir = Path(storage_dir).resolve()
        self.index_path = Path(index_path).resolve() if index_path else (self.storage_dir / "index.pkl")
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else (self.storage_dir / "manifest.json")
        self.model_name = model_name
        self.max_chunk_tokens = max_chunk_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.batch_size = batch_size
        self.forbidden_dir_names = forbidden_dir_names or DEFAULT_FORBIDDEN_DIR_NAMES

        # Generational storage paths
        self.generations_dir = self.storage_dir / "generations"
        self.current_pointer_path = self.storage_dir / "current.json"
        self.commit_chain_path = self.storage_dir / "commit_chain.json"

        # Build root identity map
        self.root_map: Dict[str, Path] = {}
        for r in self.allowed_roots:
            rid = compute_root_id(r)
            self.root_map[rid] = r

    def get_root_id(self, root: Path) -> str:
        """Get deterministic root_id for a root path."""
        return compute_root_id(root)

    def locate_path(self, target: Path) -> Optional[Tuple[str, Path, str]]:
        """
        Locate a path within allowed roots.
        Returns (root_id, root_path, rel_path) or None if target is outside allowed roots.
        If multiple roots match (e.g. nested), picks the most specific (longest) root path.
        """
        try:
            resolved = target.resolve()
        except Exception:
            return None

        candidates: List[Tuple[str, Path, str]] = []
        for root in self.allowed_roots:
            try:
                rel = resolved.relative_to(root)
                rel_str = str(rel).replace("\\", "/")
                candidates.append((compute_root_id(root), root, rel_str))
            except ValueError:
                continue

        if not candidates:
            return None

        # Pick candidate with longest root path (most specific)
        candidates.sort(key=lambda c: len(str(c[1])), reverse=True)
        return candidates[0]

    def resolve_source_path(self, root_id: str, rel_path: str) -> Optional[Path]:
        """Resolve a (root_id, rel_path) to an existing Path on disk."""
        root = self.root_map.get(root_id)
        if not root:
            # Fallback check against allowed_roots
            for r in self.allowed_roots:
                if compute_root_id(r) == root_id:
                    root = r
                    self.root_map[root_id] = r
                    break
        if not root:
            return None

        candidate = root / rel_path
        if candidate.is_file():
            return candidate
        return None

    def get_generation_dir(self, gen_id: int) -> Path:
        """Return directory for a specific generation number."""
        return self.generations_dir / f"gen_{gen_id:06d}"

    def get_generation_paths(self, gen_id: int) -> Tuple[Path, Path]:
        """Return (index_path, manifest_path) for a specific generation."""
        gdir = self.get_generation_dir(gen_id)
        return (gdir / "index.pkl", gdir / "manifest.json")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> StomaConfig:
        """Load config from a YAML file if present, or return defaults."""
        path = config_path or DEFAULT_CONFIG_PATH
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                roots = [Path(r) for r in data.get("allowed_roots", [])] or None
                return cls(
                    allowed_roots=roots,
                    storage_dir=Path(data.get("storage_dir", DEFAULT_STORAGE_DIR)),
                    index_path=Path(data["index_path"]) if "index_path" in data else None,
                    manifest_path=Path(data["manifest_path"]) if "manifest_path" in data else None,
                    model_name=data.get("model_name", DEFAULT_MODEL_NAME),
                    max_chunk_tokens=int(data.get("max_chunk_tokens", DEFAULT_MAX_CHUNK_TOKENS)),
                    chunk_overlap_tokens=int(data.get("chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS)),
                    batch_size=int(data.get("batch_size", DEFAULT_BATCH_SIZE)),
                )
            except Exception as e:
                raise ValueError(f"Failed to parse config file '{path}': {e}") from e
        return cls()

    def save(self, config_path: Optional[Path] = None) -> None:
        """Persist config to a YAML file."""
        path = config_path or DEFAULT_CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "allowed_roots": [str(r) for r in self.allowed_roots],
            "storage_dir": str(self.storage_dir),
            "index_path": str(self.index_path),
            "manifest_path": str(self.manifest_path),
            "model_name": self.model_name,
            "max_chunk_tokens": self.max_chunk_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "batch_size": self.batch_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False)

    def is_path_allowed(self, target: Path) -> bool:
        """
        Safety check:
        1. Resolve target path (dereference symlinks).
        2. Verify that resolved target is inside at least one allowed root.
        3. Verify that none of its parent components belong to forbidden directories.
        """
        try:
            # Check un-resolved path parts for forbidden directory names
            raw_parts = set(target.parts)
            if raw_parts.intersection(self.forbidden_dir_names):
                return False

            # Check if resolved path is inside storage dir (don't index own index/manifest)
            resolved = target.resolve()
            if self.storage_dir in resolved.parents or resolved == self.storage_dir:
                return False

            # Check resolved parts for forbidden directory names (e.g. .git, deps)
            resolved_parts = set(resolved.parts)
            if resolved_parts.intersection(self.forbidden_dir_names):
                return False

            # Must reside within one of the allowed roots
            inside_any = any(
                root == resolved or root in resolved.parents
                for root in self.allowed_roots
            )
            return inside_any
        except (OSError, RuntimeError):
            # Broken symlink, loop, or permissions issue
            return False
