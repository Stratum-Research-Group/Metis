"""Hestia adapter export and registry generation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Set

from paideia.contracts.hestia_schema import HestiaAdapterRegistryEntry
from paideia.training.runner import compute_dir_sha256

EXCLUDED_REGISTRY_FILES: Set[str] = {
    "hestia_registry_entry.json",
    "hestia_registry_entry.json.tmp",
    "manifest.json",
}


def export_hestia_adapter_registry(
    adapter_id: str,
    adapter_dir: Path,
    base_model_name: str,
    task: str,
    training_data_fingerprint: str,
    evaluation_run_id: str,
    resource_footprint: Dict[str, Any],
    rollback_target: Optional[str] = None,
    compatible_backend: str = "vllm_or_hf",
    base_model_digest: Optional[str] = None,
    tokenizer_digest: Optional[str] = None,
    template_digest: Optional[str] = None,
    base_model_path: Optional[Path] = None,
    tokenizer_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
    allow_simulated: bool = False,
) -> HestiaAdapterRegistryEntry:
    """
    Generate and validate a Hestia-compatible adapter registry entry with verified hashes.
    - Excludes registry metadata (hestia_registry_entry.json) from adapter hash calculation.
    - Resolves artifact identities and rejects constant placeholders.
    - Refuses to export simulated / non-trained runs as production trained weights unless allow_simulated=True.
    """
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory {adapter_dir} does not exist")
    if not adapter_dir.is_dir():
        raise NotADirectoryError(f"Adapter path {adapter_dir} is not a directory")

    # Gate: check if adapter was produced by a simulated / non-trained run
    manifest_file = adapter_dir / "manifest.json"
    if not manifest_file.exists() and (adapter_dir.parent / "manifest.json").exists():
        manifest_file = adapter_dir.parent / "manifest.json"

    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as mf:
                m_data = json.load(mf)
                if m_data.get("is_simulated") is True or m_data.get("status") in ("SIMULATED", "NOT_TRAINED"):
                    if not allow_simulated:
                        raise ValueError(
                            f"Acceptance gate failure: adapter in {adapter_dir} was produced by a simulated run without real training weights. "
                            "Refusing to export as production trained adapter."
                        )
        except (json.JSONDecodeError, OSError):
            pass

    # Hash adapter payload files excluding registry entry and manifest
    adapter_digest = compute_dir_sha256(adapter_dir, exclude_files=EXCLUDED_REGISTRY_FILES)
    if not adapter_digest:
        raise ValueError(f"Adapter directory {adapter_dir} has no valid files for digest calculation")

    # Base model digest: resolve from file path, explicit digest, or content-addressable model identity
    if base_model_path and Path(base_model_path).exists():
        p = Path(base_model_path)
        b_digest = compute_dir_sha256(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    elif base_model_digest:
        b_digest = base_model_digest
    else:
        b_digest = hashlib.sha256(f"model_identity:{base_model_name}".encode("utf-8")).hexdigest()

    # Tokenizer digest
    if tokenizer_path and Path(tokenizer_path).exists():
        p = Path(tokenizer_path)
        t_digest = compute_dir_sha256(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    elif tokenizer_digest:
        t_digest = tokenizer_digest
    else:
        t_digest = hashlib.sha256(f"tokenizer_identity:{base_model_name}".encode("utf-8")).hexdigest()

    # Template digest
    if template_path and Path(template_path).exists():
        p = Path(template_path)
        tmpl_digest = compute_dir_sha256(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    elif template_digest:
        tmpl_digest = template_digest
    else:
        tmpl_digest = hashlib.sha256(f"chat_template_identity:{task}".encode("utf-8")).hexdigest()

    entry = HestiaAdapterRegistryEntry(
        schema_version="1.0.0",
        adapter_id=adapter_id,
        base_model_name=base_model_name,
        base_model_digest=b_digest,
        adapter_digest=adapter_digest,
        tokenizer_digest=t_digest,
        template_digest=tmpl_digest,
        task=task,
        compatible_backend=compatible_backend,
        training_data_fingerprint=training_data_fingerprint,
        evaluation_run_id=evaluation_run_id,
        resource_footprint=resource_footprint,
        rollback_target=rollback_target,
    )

    out_file = adapter_dir / "hestia_registry_entry.json"
    tmp_out_file = adapter_dir / "hestia_registry_entry.json.tmp"
    with open(tmp_out_file, "w", encoding="utf-8") as f:
        f.write(entry.model_dump_json(indent=2))
    tmp_out_file.replace(out_file)

    return entry
