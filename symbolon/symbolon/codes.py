"""Scoped discrete code identity and overlap metrics for Symbolon.

Per Symbolon Integration Spec Section 2 and Roadmap Section 2.2:
A code integer is valid and matchable ONLY when scoped by:
(checkpoint_id, tokenizer_id, level, window_idx, pos, code_int).
Cross-book, cross-level, or cross-window equality is never treated as similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Set


@dataclass(frozen=True, slots=True)
class ScopedCode:
    """
    Immutable representation of an NCM discrete multiscale code token.
    Enforces the expanded 6-tuple identity rule across checkpoints, codebooks,
    levels, passage-relative windows, and intra-window positions.
    """
    checkpoint_id: str
    tokenizer_id: str
    level: int
    window_idx: int
    pos: int
    code_int: int

    def format_key(self) -> str:
        """Format deterministic canonical string key."""
        return (
            f"{self.checkpoint_id}:{self.tokenizer_id}:"
            f"lvl{self.level}:w{self.window_idx}:p{self.pos}:c{self.code_int}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "tokenizer_id": self.tokenizer_id,
            "level": self.level,
            "window_idx": self.window_idx,
            "pos": self.pos,
            "code_int": self.code_int,
            "key": self.format_key(),
        }

    def __str__(self) -> str:
        return self.format_key()


def compute_scoped_code_overlap(
    query_codes: Iterable[ScopedCode],
    doc_codes: Iterable[ScopedCode],
) -> int:
    """
    Compute count of strictly matching discrete codes between query and document.
    Only codes sharing the identical 6-tuple scope match.
    """
    q_set: Set[ScopedCode] = set(query_codes)
    d_set: Set[ScopedCode] = set(doc_codes)
    return len(q_set & d_set)


def compute_scoped_code_jaccard(
    query_codes: Iterable[ScopedCode],
    doc_codes: Iterable[ScopedCode],
) -> float:
    """
    Compute Jaccard similarity over strictly scoped discrete codes:
    |Q ∩ D| / |Q ∪ D|
    """
    q_set: Set[ScopedCode] = set(query_codes)
    d_set: Set[ScopedCode] = set(doc_codes)
    union_len = len(q_set | d_set)
    if union_len == 0:
        return 0.0
    return len(q_set & d_set) / float(union_len)
