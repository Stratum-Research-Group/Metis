"""Phylax engine package."""

from phylax.engine.canonical import canonicalize_path, is_contained_in_root, matches_path_pattern
from phylax.engine.context import (
    get_current_subject,
    inspect_caller_identity,
    trusted_caller,
    verify_caller_authorization,
)
from phylax.engine.evaluator import PolicyEvaluator, compute_argument_digest
from phylax.engine.manager import PolicyManager

__all__ = [
    "canonicalize_path",
    "is_contained_in_root",
    "matches_path_pattern",
    "get_current_subject",
    "inspect_caller_identity",
    "trusted_caller",
    "verify_caller_authorization",
    "PolicyEvaluator",
    "compute_argument_digest",
    "PolicyManager",
]
