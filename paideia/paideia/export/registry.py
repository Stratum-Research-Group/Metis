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


import struct


def _validate_safetensors_file(path: Path) -> None:
    size = path.stat().st_size
    if size < 8:
        raise ValueError(f"Safetensors weight file {path.name} is too small ({size} bytes).")
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len <= 0 or header_len + 8 > size:
            raise ValueError(f"Safetensors weight file {path.name} has invalid header length {header_len}.")
        header_bytes = f.read(header_len)
        try:
            header_json = json.loads(header_bytes.decode("utf-8"))
            if not isinstance(header_json, dict):
                raise ValueError("Safetensors header is not a valid JSON dict.")
        except Exception as exc:
            raise ValueError(f"Safetensors weight file {path.name} has corrupt JSON header: {exc}") from exc


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

    if not manifest_file.exists():
        raise ValueError(
            f"Acceptance gate failure: missing required manifest.json in {adapter_dir}. "
            "Cannot verify adapter provenance or training completion."
        )

    try:
        with open(manifest_file, "r", encoding="utf-8") as mf:
            m_data = json.load(mf)
    except Exception as exc:
        raise ValueError(
            f"Acceptance gate failure: corrupt or unreadable manifest in {manifest_file}: {exc}"
        ) from exc

    status = m_data.get("status")
    is_sim = m_data.get("is_simulated") is True or status in ("SIMULATED", "NOT_TRAINED")

    if not is_sim:
        if status not in ("COMPLETED", "COMPLETED_REAL", "SUCCESS"):
            raise ValueError(
                f"Acceptance gate failure: manifest status is '{status}'. "
                "Only completed runs (status 'COMPLETED' or 'SUCCESS') can be exported as production adapters."
            )
    elif not allow_simulated:
        raise ValueError(
            f"Acceptance gate failure: adapter in {adapter_dir} was produced by a simulated run without real training weights. "
            "Refusing to export as production trained adapter."
        )

    # Check for expected trained weights
    weight_file_names = {
        "adapter_model.safetensors",
        "adapter_model.bin",
        "adapter.gguf",
        "pytorch_model.bin",
        "model.safetensors",
    }
    found_weights = [f for f in adapter_dir.iterdir() if f.is_file() and f.name in weight_file_names]
    if not found_weights and not (is_sim and allow_simulated):
        raise ValueError(
            f"Acceptance gate failure: adapter directory {adapter_dir} contains no trained weight files. "
            f"Expected at least one of {sorted(weight_file_names)}."
        )

    # For non-simulated runs, ensure weight files are non-empty and validate safetensors header
    if not (is_sim and allow_simulated):
        for wf in found_weights:
            if wf.stat().st_size == 0:
                raise ValueError(f"Acceptance gate failure: weight file {wf.name} is zero bytes.")
            if wf.suffix == ".safetensors":
                _validate_safetensors_file(wf)

    # Verify evaluation evidence
    clean_eval_id = (evaluation_run_id or "").strip()
    if not clean_eval_id or clean_eval_id.lower() in ("n/a", "none", "unknown", "null"):
        raise ValueError("Acceptance gate failure: verified non-placeholder evaluation_run_id required for adapter export")

    if is_sim and allow_simulated:
        compatible_backend = f"simulation_non_release:{compatible_backend}"

    # Hash adapter payload files excluding registry entry and manifest
    adapter_digest = compute_dir_sha256(adapter_dir, exclude_files=EXCLUDED_REGISTRY_FILES)
    if not adapter_digest:
        raise ValueError(f"Adapter directory {adapter_dir} has no valid files for digest calculation")

    # Base model digest: resolve from file path, explicit digest, or explicitly labeled unresolved alias
    if base_model_path and Path(base_model_path).exists():
        p = Path(base_model_path)
        b_digest = compute_dir_sha256(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    elif base_model_digest:
        b_digest = base_model_digest
    elif is_sim and allow_simulated:
        b_digest = f"unresolved_alias:{base_model_name}"
    else:
        raise ValueError(f"Acceptance gate failure: unresolvable base model digest for '{base_model_name}'.")

    # Tokenizer digest
    if tokenizer_path and Path(tokenizer_path).exists():
        p = Path(tokenizer_path)
        t_digest = compute_dir_sha256(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    elif tokenizer_digest:
        t_digest = tokenizer_digest
    elif is_sim and allow_simulated:
        t_digest = f"unresolved_alias:{base_model_name}"
    else:
        raise ValueError(f"Acceptance gate failure: unresolvable tokenizer digest for '{base_model_name}'.")

    # Template digest
    if template_path and Path(template_path).exists():
        p = Path(template_path)
        tmpl_digest = compute_dir_sha256(p) if p.is_dir() else hashlib.sha256(p.read_bytes()).hexdigest()
    elif template_digest:
        tmpl_digest = template_digest
    elif is_sim and allow_simulated:
        tmpl_digest = f"unresolved_alias:{task}"
    else:
        raise ValueError(f"Acceptance gate failure: unresolvable template digest for '{task}'.")

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
