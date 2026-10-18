"""
Stoma Token-Bounded Structure-Aware Chunker.
Splits extracted raw passages into structure-aware chunks strictly bounded by the encoder's
effective tokenizer and sequence limits (including special-token overhead),
preventing silent truncation during embedding.
Retains all characters without discarding text on oversized strings.
Supports streaming chunk generation.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from stoma.embedder import StomaEmbedder
from stoma.plugins.base import RawPassage
from stoma.provenance import compute_content_sha256


@dataclass
class PassageChunk:
    """A prepared, token-bounded passage chunk ready for embedding and indexing."""
    chunk_index: int
    content: str
    token_count: int
    char_count: int
    chunk_hash: str
    location: Dict[str, Any]
    section: Optional[str] = None


class Chunker:
    """
    Structure-aware chunker bounded by model tokenizer and effective sequence limits.
    Never discards or truncates text.
    """

    def __init__(
        self,
        embedder: StomaEmbedder,
        target_tokens: int = 200,
        overlap_tokens: int = 30,
        redactor: Optional[Any] = None,
        filter_boilerplate: bool = False,
    ):
        self.embedder = embedder
        # Effective max tokens accounts for special tokens overhead
        self.hard_max_tokens = getattr(self.embedder, "effective_max_tokens", self.embedder.max_seq_length - 2)
        self.target_tokens = min(target_tokens, self.hard_max_tokens - 10)
        self.overlap_tokens = min(overlap_tokens, self.target_tokens // 4)
        self.redactor = redactor
        self.filter_boilerplate = filter_boilerplate

    def chunk_raw_passages_stream(
        self,
        raw_passages: Iterable[RawPassage],
        deduplicate_within_file: bool = True,
    ) -> Iterator[PassageChunk]:
        """
        Stream raw passages into token-bounded chunks.
        Eliminates duplicate and empty chunks while keeping memory strictly bounded (dedup capped at 10k items).
        """
        max_seen = 10000
        seen_hashes: Set[str] = set()
        seen_queue: collections.deque[str] = collections.deque()
        chunk_idx = 0

        def _check_and_record(ch: str) -> bool:
            if not deduplicate_within_file:
                return True
            if ch in seen_hashes:
                return False
            if len(seen_hashes) >= max_seen:
                oldest = seen_queue.popleft()
                seen_hashes.discard(oldest)
            seen_hashes.add(ch)
            seen_queue.append(ch)
            return True

        for raw in raw_passages:
            text = raw.content.strip()
            if not text:
                continue

            if self.filter_boilerplate:
                from stoma.redaction import ContentAssessment
                is_spam, _ = ContentAssessment.is_boilerplate_or_spam(text)
                if is_spam:
                    continue

            if self.redactor:
                text, _ = self.redactor.redact_text(text)
                text = text.strip()
                if not text:
                    continue

            num_tokens = self.embedder.count_tokens(text)

            # If the raw passage already fits within our target token boundary
            if num_tokens <= self.target_tokens:
                c_hash = compute_content_sha256(text)
                if not _check_and_record(c_hash):
                    continue
                yield PassageChunk(
                    chunk_index=chunk_idx,
                    content=text,
                    token_count=num_tokens,
                    char_count=len(text),
                    chunk_hash=c_hash,
                    location=raw.location.copy(),
                    section=raw.section,
                )
                chunk_idx += 1
            else:
                # Subdivide oversized passages along structural boundaries
                sub_chunks = self._subdivide_passage(text, raw.location, raw.section)
                for sc in sub_chunks:
                    c_hash = compute_content_sha256(sc.content)
                    if not _check_and_record(c_hash):
                        continue
                    sc.chunk_index = chunk_idx
                    yield sc
                    chunk_idx += 1

    def chunk_raw_passages(
        self,
        raw_passages: Iterable[RawPassage],
        deduplicate_within_file: bool = True,
    ) -> List[PassageChunk]:
        """Return all chunks as a list."""
        return list(self.chunk_raw_passages_stream(raw_passages, deduplicate_within_file=deduplicate_within_file))

    def _subdivide_passage(
        self,
        text: str,
        base_location: Dict[str, Any],
        section: Optional[str],
    ) -> List[PassageChunk]:
        """
        Subdivide an oversized text into chunks each strictly <= hard_max_tokens.
        Uses sentence/line splitting with small token overlap.
        """
        units = [u.strip() for u in re.split(r"(\n{2,}|\.\s+|\?\s+|\!\s+)", text) if u.strip()]
        if not units:
            units = text.splitlines()

        result_chunks: List[PassageChunk] = []
        current_units: List[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self.embedder.count_tokens(unit)

            # If a single unit itself exceeds hard_max_tokens
            if unit_tokens > self.hard_max_tokens:
                # Flush existing buffer
                if current_units:
                    block = " ".join(current_units).strip()
                    if block:
                        tokens = self.embedder.count_tokens(block)
                        result_chunks.append(
                            PassageChunk(
                                chunk_index=0,
                                content=block,
                                token_count=tokens,
                                char_count=len(block),
                                chunk_hash=compute_content_sha256(block),
                                location=base_location.copy(),
                                section=section,
                            )
                        )
                    current_units = []
                    current_tokens = 0

                # Subdivide oversized unit without discarding any characters
                sub_slices = self._slice_oversized_unit(unit)
                for s in sub_slices:
                    tokens = self.embedder.count_tokens(s)
                    result_chunks.append(
                        PassageChunk(
                            chunk_index=0,
                            content=s,
                            token_count=tokens,
                            char_count=len(s),
                            chunk_hash=compute_content_sha256(s),
                            location=base_location.copy(),
                            section=section,
                        )
                    )
                continue

            if current_tokens + unit_tokens > self.target_tokens:
                # Flush current buffer
                if current_units:
                    block = " ".join(current_units).strip()
                    if block:
                        tokens = self.embedder.count_tokens(block)
                        result_chunks.append(
                            PassageChunk(
                                chunk_index=0,
                                content=block,
                                token_count=tokens,
                                char_count=len(block),
                                chunk_hash=compute_content_sha256(block),
                                location=base_location.copy(),
                                section=section,
                            )
                        )

                # Keep overlap if configured
                overlap_units = []
                overlap_count = 0
                for prev in reversed(current_units):
                    p_tok = self.embedder.count_tokens(prev)
                    if overlap_count + p_tok <= self.overlap_tokens:
                        overlap_units.insert(0, prev)
                        overlap_count += p_tok
                    else:
                        break

                current_units = overlap_units + [unit]
                current_tokens = overlap_count + unit_tokens
            else:
                current_units.append(unit)
                current_tokens += unit_tokens

        # Flush remainder
        if current_units:
            block = " ".join(current_units).strip()
            if block:
                tokens = self.embedder.count_tokens(block)
                result_chunks.append(
                    PassageChunk(
                        chunk_index=0,
                        content=block,
                        token_count=tokens,
                        char_count=len(block),
                        chunk_hash=compute_content_sha256(block),
                        location=base_location.copy(),
                        section=section,
                    )
                )

        # Enforce hard guarantee: no chunk can ever exceed hard_max_tokens
        final_chunks: List[PassageChunk] = []
        for c in result_chunks:
            if c.token_count > self.hard_max_tokens:
                slices = self._slice_oversized_unit(c.content)
                for s in slices:
                    tokens = self.embedder.count_tokens(s)
                    final_chunks.append(
                        PassageChunk(
                            chunk_index=0,
                            content=s,
                            token_count=tokens,
                            char_count=len(s),
                            chunk_hash=compute_content_sha256(s),
                            location=base_location.copy(),
                            section=section,
                        )
                    )
            else:
                final_chunks.append(c)

        return final_chunks

    def _slice_oversized_unit(self, unit: str) -> List[str]:
        """
        Slices an oversized unit of text so each slice is strictly <= hard_max_tokens.
        Uses words where possible, and subdivides unbroken strings without losing characters.
        """
        words = unit.split(" ")
        slices: List[str] = []
        current: List[str] = []

        for w in words:
            if not w:
                continue

            w_tok = self.embedder.count_tokens(w)
            if w_tok > self.hard_max_tokens:
                # Flush previous words
                if current:
                    slices.append(" ".join(current))
                    current = []
                # Subdivide the giant unbroken word completely
                sub_parts = self._subdivide_unbroken_string(w, self.hard_max_tokens)
                slices.extend(sub_parts)
                continue

            candidate = " ".join(current + [w]) if current else w
            if self.embedder.count_tokens(candidate) > self.hard_max_tokens:
                if current:
                    slices.append(" ".join(current))
                    current = [w]
                else:
                    # Single word exceeded candidate check
                    sub_parts = self._subdivide_unbroken_string(w, self.hard_max_tokens)
                    slices.extend(sub_parts)
                    current = []
            else:
                current.append(w)

        if current:
            slices.append(" ".join(current))

        return slices or (self._subdivide_unbroken_string(unit, self.hard_max_tokens) if unit else [])

    def _subdivide_unbroken_string(self, text: str, max_tokens: int) -> List[str]:
        """
        Subdivide an unbroken string into consecutive substrings such that each substring
        has token_count <= max_tokens, without discarding ANY characters.
        """
        if not text:
            return []

        parts: List[str] = []
        idx = 0
        total_len = len(text)

        while idx < total_len:
            # Binary search for the largest end index where count_tokens(text[idx:end]) <= max_tokens
            low = idx + 1
            high = total_len
            best = low

            while low <= high:
                mid = (low + high) // 2
                sub = text[idx:mid]
                tok = self.embedder.count_tokens(sub)
                if tok <= max_tokens:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1

            if best == idx:
                best = idx + 1

            parts.append(text[idx:best])
            idx = best

        return parts
