"""Manifest writer capturing reproducibility provenance, hardware, and SHA-256 checksums."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dokime import __version__


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest over raw file bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for checksum: {p}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_text(text: str) -> str:
    """Compute SHA-256 hex digest over utf-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_dokime_repo_root() -> Path | None:
    """Find the root directory of the Dokime repository containing this package."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        git_marker = parent / ".git"
        if git_marker.exists():
            # Verify this repository actually corresponds to Dokime
            if (parent / "pyproject.toml").exists() or (parent / "src" / "dokime").exists():
                return parent
    return None


def get_git_sha(repo_root: Path | None = None) -> str | None:
    """Retrieve git HEAD commit SHA of Dokime's actual source checkout.

    Never inspects or uses the caller's working directory.
    If running from a distribution with no verifiable source checkout, returns None.
    """
    root = repo_root or find_dokime_repo_root()
    if root is None:
        return None

    try:
        res = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        sha = res.stdout.strip()
        if sha:
            return sha
    except Exception:
        pass
    return None


def get_git_dirty(repo_root: Path | None = None) -> bool | None:
    """Check if Dokime's source checkout has uncommitted changes.

    Never inspects or uses the caller's working directory.
    Returns True if dirty, False if clean, None if unavailable.
    """
    root = repo_root or find_dokime_repo_root()
    if root is None:
        return None

    try:
        res = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
        return bool(res.stdout.strip())
    except Exception:
        pass
    return None


def capture_hardware() -> dict[str, Any]:
    """Capture hardware environment via platform module and nvidia-smi if present."""
    hw: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "nvidia_smi": None,
    }

    # Query nvidia-smi if available
    try:
        res = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if res.returncode == 0 and res.stdout.strip():
            hw["nvidia_smi"] = res.stdout.strip()
    except Exception:
        pass

    return hw


def create_manifest(
    run_id: str,
    model_ids: list[str],
    task_ids: list[str],
    seed: int,
    temperature: float,
    max_tokens: int | None,
    results_jsonl_path: Path,
    dataset_paths: dict[str, Path | None],
    summary_json_path: Path | None = None,
    prompt_template: str | None = None,
    prompt_template_path: str | None = None,
    prompt_template_artifact_path: Path | None = None,
    output_path: Path | None = None,
    repo_root: Path | None = None,
    model_identities: dict[str, Any] | None = None,
    parent_run_id: str | None = None,
    branch_point: str | None = None,
    parameter_diff: dict[str, Any] | None = None,
    dataset_boms: dict[str, Any] | None = None,
    labeling_criterion: str | None = None,
) -> dict[str, Any]:
    """Build and write the run manifest JSON file."""
    checksums: dict[str, str] = {}

    # Hash results.jsonl
    if results_jsonl_path.exists():
        checksums["results_jsonl"] = sha256_file(results_jsonl_path)

    # Hash summary.json if provided and exists (Item 7)
    if summary_json_path and summary_json_path.exists():
        checksums["summary_json"] = sha256_file(summary_json_path)

    # Hash task datasets
    for tid, dpath in dataset_paths.items():
        if dpath and Path(dpath).is_file():
            checksums[f"dataset_{tid}"] = sha256_file(dpath)

    # Hash prompt template artifact or raw string
    prompt_hash: str | None = None
    if prompt_template_artifact_path and prompt_template_artifact_path.is_file():
        prompt_hash = sha256_file(prompt_template_artifact_path)
        checksums["prompt_template"] = prompt_hash
    elif prompt_template:
        prompt_hash = sha256_text(prompt_template)
        checksums["prompt_template"] = prompt_hash

    # Resolve git sha & dirty status
    git_sha = get_git_sha(repo_root)
    git_sha_record = git_sha if git_sha is not None else "unavailable"
    git_dirty = get_git_dirty(repo_root)

    manifest = {
        "run_id": run_id,
        "dokime_version": __version__,
        "git_sha": git_sha_record,
        "git_dirty": git_dirty,
        "parent_run_id": parent_run_id,
        "branch_point": branch_point,
        "parameter_diff": parameter_diff,
        "dataset_boms": dataset_boms or {},
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_ids": model_ids,
        "model_identities": model_identities or {},
        "task_ids": task_ids,
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "labeling_criterion": labeling_criterion,
        "prompt_template_path": prompt_template_path,
        "prompt_template_hash": prompt_hash,
        "checksums": checksums,
        "hardware": capture_hardware(),
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
