"""Configuration validation handlers for all Metis modules.

Executes each module's native validator without side effects or disk modifications.
Strictly conforms to Section 3.3b of the Metis Control Dashboard specification.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field

# Ensure sibling Metis modules are resolvable for native config validation
_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for _rel in (
    "hestia/src",
    "prosopon",
    "phylax",
    "krisis",
    "stoma/src",
    "dokime/src",
    "paideia",
    "dianoia",
    "kleroterion",
    "LTHCM",
):
    _p = str(_WORKSPACE_ROOT / _rel)
    if _p not in sys.path and (_WORKSPACE_ROOT / _rel).exists():
        sys.path.insert(0, _p)

from kybernetes.admin.descriptors import MODULE_REGISTRY
from kybernetes.config import KybernetesConfig


class ConfigValidationResult(BaseModel):
    module_id: str
    valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    restart_required: bool = True


def validate_kybernetes(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        KybernetesConfig(**proposed)
    except Exception as exc:
        errors.append(f"KybernetesConfig validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_hestia(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        from hestia.config import PortRange, ServiceConfig
        props = dict(proposed)
        if "backend_port_range" in props and isinstance(props["backend_port_range"], dict):
            pr_data = props["backend_port_range"]
            props["backend_port_range"] = PortRange(
                start=int(pr_data.get("start", 9000)),
                end=int(pr_data.get("end", 9019)),
            )
        ServiceConfig(**props)
    except Exception as exc:
        errors.append(f"Hestia ServiceConfig validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_prosopon(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        from prosopon.config import ProsoponConfig
        ProsoponConfig(**proposed)
    except Exception as exc:
        errors.append(f"ProsoponConfig validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_phylax(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        import yaml
        from phylax.linter.linter import LintSeverity, PolicyLinter
        yaml_str = yaml.safe_dump(proposed)
        linter = PolicyLinter()
        doc, issues = linter.lint_yaml_string(yaml_str)
        for issue in issues:
            msg = str(issue)
            if issue.severity == LintSeverity.ERROR:
                errors.append(msg)
            elif issue.severity == LintSeverity.WARNING:
                warnings.append(msg)
        if doc is None and not errors:
            errors.append("Failed to construct PolicyDocument from configuration.")
    except Exception as exc:
        errors.append(f"Phylax policy validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_krisis(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    import jsonschema

    schema_candidates = [
        Path("krisis/krisis-policy.schema.json"),
        Path("../krisis/krisis-policy.schema.json"),
        Path("krisis-policy.schema.json"),
    ]
    schema_path = next((p for p in schema_candidates if p.is_file()), None)
    if schema_path:
        try:
            schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance=proposed, schema=schema_data)
        except jsonschema.ValidationError as ve:
            errors.append(f"Krisis schema validation error: {ve.message}")
        except Exception as exc:
            warnings.append(f"JSON schema check skipped: {exc}")

    try:
        from krisis.contracts.policy import PolicyManifest
        PolicyManifest(**proposed)
    except Exception as exc:
        # If schema already reported an error, only append if different
        err_msg = f"Krisis PolicyManifest contract error: {exc}"
        if not errors:
            errors.append(err_msg)

    return len(errors) == 0, errors, warnings


def validate_stoma(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    max_chunk = proposed.get("max_chunk_tokens", 220)
    chunk_overlap = proposed.get("chunk_overlap_tokens", 30)
    if isinstance(max_chunk, (int, float)) and isinstance(chunk_overlap, (int, float)):
        if int(chunk_overlap) >= int(max_chunk):
            errors.append(
                f"chunk_overlap_tokens ({chunk_overlap}) must be strictly less than max_chunk_tokens ({max_chunk})"
            )
    roots = proposed.get("allowed_roots")
    if roots is not None and not isinstance(roots, list):
        errors.append("allowed_roots must be a list of paths")

    if not errors:
        try:
            from stoma.config import StomaConfig
            StomaConfig(
                allowed_roots=[Path(r) for r in roots] if roots else None,
                storage_dir=Path(proposed.get("storage_dir", ".stoma")),
                model_name=str(proposed.get("model_name", "all-MiniLM-L6-v2")),
                max_chunk_tokens=int(max_chunk),
                chunk_overlap_tokens=int(chunk_overlap),
                batch_size=int(proposed.get("batch_size", 32)),
            )
        except ImportError:
            warnings.append("stoma package not installed; validated configuration structurally")
        except Exception as exc:
            errors.append(f"StomaConfig validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_dokime(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        from dokime.config import RunConfig
        RunConfig.from_dict(proposed)
    except Exception as exc:
        errors.append(f"Dokime RunConfig validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_paideia(proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        from paideia.contracts.manifest_schema import RunManifest
        if "run_id" in proposed or "task_name" in proposed:
            RunManifest(**proposed)
        else:
            for key in ("seed", "epochs", "batch_size"):
                if key in proposed and not isinstance(proposed[key], int):
                    errors.append(f"Field '{key}' must be an integer, got {type(proposed[key]).__name__}")
    except Exception as exc:
        errors.append(f"Paideia validation error: {exc}")
    return len(errors) == 0, errors, warnings


def validate_library(module_id: str, proposed: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    if not proposed:
        return (
            True,
            [],
            [f"Module '{module_id}' operates as an in-process library without a mutable on-disk configuration."],
        )
    return False, [f"Module '{module_id}' is an in-process library without an editable configuration schema."], []


VALIDATOR_DISPATCH = {
    "kybernetes": validate_kybernetes,
    "hestia": validate_hestia,
    "prosopon": validate_prosopon,
    "phylax": validate_phylax,
    "krisis": validate_krisis,
    "stoma": validate_stoma,
    "dokime": validate_dokime,
    "paideia": validate_paideia,
}


def validate_module_configuration(module_id: str, proposed: Dict[str, Any]) -> ConfigValidationResult:
    """Validate proposed configuration dictionary using the owning module's native validator."""
    if module_id not in MODULE_REGISTRY:
        raise KeyError(f"Unknown module identifier: '{module_id}'")

    descriptor = MODULE_REGISTRY[module_id]
    restart_required = descriptor.restart_required

    validator = VALIDATOR_DISPATCH.get(module_id)
    if validator is not None:
        valid, errors, warnings = validator(proposed)
    else:
        valid, errors, warnings = validate_library(module_id, proposed)

    return ConfigValidationResult(
        module_id=module_id,
        valid=valid,
        errors=errors,
        warnings=warnings,
        restart_required=restart_required,
    )
