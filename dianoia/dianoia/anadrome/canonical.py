"""RFC 8785 Canonical JSON Serialization and Observation Hashing for Anadrome.

Ensures deterministic, bit-exact observation hashing across different environments
and platforms by enforcing lexicographical key sorting, float stabilization
(4 decimal places), and standard JSON formatting.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalize_for_canonical_json(obj: Any) -> Any:
    """Recursively normalize objects for canonical JSON serialization."""
    if isinstance(obj, float):
        # Round floats to 4 decimal places to prevent representation drift
        rounded = round(obj, 4)
        if rounded == -0.0:
            rounded = 0.0
        return rounded
    elif isinstance(obj, dict):
        return {
            str(k): _normalize_for_canonical_json(v)
            for k, v in sorted(obj.items(), key=lambda item: str(item[0]))
        }
    elif isinstance(obj, (list, tuple)):
        return [_normalize_for_canonical_json(x) for x in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    else:
        # Fallback for dataclasses or custom types
        if hasattr(obj, "to_dict"):
            return _normalize_for_canonical_json(obj.to_dict())
        return str(obj)


def canonical_json_dumps(obj: Any) -> str:
    """
    Serialize obj to canonical JSON string according to RFC 8785 guidelines:
      - UTF-8 representation without unnecessary escaping
      - Lexicographical object key ordering
      - Standard delimiters with zero extraneous whitespace: separators=(',', ':')
      - Deterministic float rounding (4 decimals)
    """
    normalized = _normalize_for_canonical_json(obj)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def compute_observation_hash(observation: Any) -> str:
    """Compute an immutable SHA-256 hex digest of an observation dict or object."""
    canonical_str = canonical_json_dumps(observation)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
