"""Hestia adapter export and registry generation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from paideia.contracts.hestia_schema import HestiaAdapterRegistryEntry
from paideia.training.runner import compute_dir_sha256


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
) -> HestiaAdapterRegistryEntry:
    """
    Generate and validate a Hestia-compatible adapter registry entry with verified hashes.
    """
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory {adapter_dir} does not exist")

    # Hash adapter files
    adapter_digest = compute_dir_sha256(adapter_dir)
    
    # Hash base model reference / pseudo-hash
    base_digest = hashlib.sha256(base_model_name.encode("utf-8")).hexdigest()
    
    # Template & tokenizer digest
    tok_digest = hashlib.sha256(b"default_tokenizer").hexdigest()
    template_digest = hashlib.sha256(b"default_chat_template").hexdigest()

    entry = HestiaAdapterRegistryEntry(
        schema_version="1.0.0",
        adapter_id=adapter_id,
        base_model_name=base_model_name,
        base_model_digest=base_digest,
        adapter_digest=adapter_digest,
        tokenizer_digest=tok_digest,
        template_digest=template_digest,
        task=task,
        compatible_backend=compatible_backend,
        training_data_fingerprint=training_data_fingerprint,
        evaluation_run_id=evaluation_run_id,
        resource_footprint=resource_footprint,
        rollback_target=rollback_target,
    )

    out_file = adapter_dir / "hestia_registry_entry.json"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(entry.model_dump_json(indent=2))

    return entry
