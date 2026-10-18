"""RFC 8785 Canonical JSON Serialization and Observation Hashing for Anadrome.

Ensures deterministic, bit-exact observation hashing across different environments
and platforms by enforcing lexicographical key sorting, float stabilization
(4 decimal places), and standard JSON formatting.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


try:
    import unittest.mock as _mock
    _MOCK_TYPES = (_mock.Base,)
except Exception:
    _MOCK_TYPES = ()


def _is_mock(obj: Any) -> bool:
    return bool(_MOCK_TYPES and isinstance(obj, _MOCK_TYPES)) or hasattr(obj, "_mock_return_value")


def _normalize_for_canonical_json(obj: Any, seen: set | None = None) -> Any:
    """Recursively normalize objects for canonical JSON serialization."""
    if _is_mock(obj):
        return str(obj)

    if isinstance(obj, float):
        # Round floats to 4 decimal places to prevent representation drift
        rounded = round(obj, 4)
        if rounded == -0.0:
            rounded = 0.0
        return rounded
    elif isinstance(obj, dict):
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return str(obj)
        seen.add(obj_id)
        return {
            str(k): _normalize_for_canonical_json(v, seen)
            for k, v in sorted(obj.items(), key=lambda item: str(item[0]))
        }
    elif isinstance(obj, (list, tuple)):
        return [_normalize_for_canonical_json(x, seen) for x in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    else:
        # Fallback for dataclasses or custom types
        if hasattr(obj, "to_dict") and not _is_mock(obj):
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return str(obj)
            seen.add(obj_id)
            to_dict_fn = getattr(obj, "to_dict")
            if callable(to_dict_fn):
                res = to_dict_fn()
                if not _is_mock(res):
                    return _normalize_for_canonical_json(res, seen)
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
