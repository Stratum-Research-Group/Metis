"""Dataset Bill of Materials (BOM) generator for Dokime evaluations (Phase 3).

Captures:
- Exact data hashes (SHA-256)
- Record count and token volume estimates
- Canonical schema signatures (keys, field types, structural constraints)
- Full dataset lineage (source format, origin type, generation timestamps)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Optional


def compute_sha256_file(path: str | Path) -> str:
    """Compute SHA-256 checksum over file bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for BOM checksum: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def estimate_tokens(text: str) -> int:
    """Fast, reproducible token count estimate (~4 chars/token or whitespace fallback)."""
    if not text:
        return 0
    # Average of whitespace words and char-length / 4 heuristic
    words = len(text.split())
    char_estimate = max(1, len(text) // 4)
    return max(words, char_estimate)


def extract_schema_signature(queries: Iterable[Any]) -> list[str]:
    """Extract sorted, unique attribute keys across all query objects in the dataset."""
    keys: set[str] = set()
    for q in queries:
        if isinstance(q, dict):
            keys.update(q.keys())
        elif hasattr(q, "__dataclass_fields__"):
            keys.update(q.__dataclass_fields__.keys())
            if hasattr(q, "metadata") and isinstance(q.metadata, dict):
                for mk in q.metadata.keys():
                    keys.add(f"metadata.{mk}")
        elif hasattr(q, "__dict__"):
            keys.update(q.__dict__.keys())
    return sorted(keys)


@dataclass
class DatasetBOM:
    """Cryptographic Bill of Materials (BOM) for an evaluated dataset."""

    task_id: str
    dataset_path: str
    sha256: str
    record_count: int
    token_count_estimate: int
    schema_signature: list[str]
    lineage: dict[str, Any] = field(default_factory=dict)
    created_at_utc: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "dataset_path": self.dataset_path,
            "sha256": self.sha256,
            "record_count": self.record_count,
            "token_count_estimate": self.token_count_estimate,
            "schema_signature": self.schema_signature,
            "lineage": self.lineage,
            "created_at_utc": self.created_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetBOM:
        return cls(
            task_id=data["task_id"],
            dataset_path=data["dataset_path"],
            sha256=data["sha256"],
            record_count=data["record_count"],
            token_count_estimate=data["token_count_estimate"],
            schema_signature=data.get("schema_signature", []),
            lineage=data.get("lineage", {}),
            created_at_utc=data.get("created_at_utc", time.time()),
        )


def generate_dataset_bom(
    task_id: str,
    queries: list[Any],
    dataset_path: Path | str,
    lineage_extra: dict[str, Any] | None = None,
) -> DatasetBOM:
    """Generate a cryptographic DatasetBOM for a task's loaded queries and dataset file."""
    p = Path(dataset_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Cannot generate BOM for non-existent dataset file: {p}")

    sha256 = compute_sha256_file(p)
    record_count = len(queries)

    # Calculate token count estimate across all query prompts, choices, and expected outputs
    total_tokens = 0
    for q in queries:
        if isinstance(q, dict):
            prompt = str(q.get("prompt", ""))
            expected = str(q.get("expected", ""))
            choices = " ".join(str(v) for v in q.get("choices", {}).values()) if isinstance(q.get("choices"), dict) else ""
        else:
            prompt = str(getattr(q, "prompt", ""))
            expected = str(getattr(q, "expected", ""))
            c_dict = getattr(q, "choices", {})
            choices = " ".join(str(v) for v in c_dict.values()) if isinstance(c_dict, dict) else ""

        total_tokens += estimate_tokens(prompt)
        total_tokens += estimate_tokens(expected)
        total_tokens += estimate_tokens(choices)

    schema_sig = extract_schema_signature(queries)

    stat = p.stat()
    lineage = {
        "file_size_bytes": stat.st_size,
        "file_mtime": stat.st_mtime,
        "format": p.suffix.lstrip(".").lower() or "raw",
        "is_materialized": "dataset_" in p.name,
    }
    if lineage_extra:
        lineage.update(lineage_extra)

    return DatasetBOM(
        task_id=task_id,
        dataset_path=str(p),
        sha256=sha256,
        record_count=record_count,
        token_count_estimate=total_tokens,
        schema_signature=schema_sig,
        lineage=lineage,
    )
