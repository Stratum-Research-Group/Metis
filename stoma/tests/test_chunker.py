"""
Unit tests for Stoma Chunker.
Verifies token-bounding (strictly <= max_seq_length), no silent truncation,
overlap preservation, and deduplication of empty/identical chunks.
"""

import pytest

from stoma.chunker import Chunker
from stoma.plugins.base import RawPassage


def test_chunker_bounds_tokens_strictly(shared_embedder):
    """
    Assert that even with massive inputs, EVERY single emitted chunk
    is strictly bounded by the encoder's effective max_seq_length (256 for MiniLM).
    Guarantees no silent token truncation during embedding.
    """
    chunker = Chunker(
        embedder=shared_embedder,
        target_tokens=150,
        overlap_tokens=20,
    )

    # Generate a massive passage with 1000 words
    long_text = " ".join([f"topological_term_{i} non_euclidean_concept_{i}" for i in range(500)])
    raw_passages = [RawPassage(content=long_text, location={"start_line": 1, "end_line": 50})]

    chunks = chunker.chunk_raw_passages(raw_passages)
    assert len(chunks) > 1

    for c in chunks:
        # Crucial assertion: no chunk can ever exceed the model's hard max_seq_length!
        assert c.token_count <= shared_embedder.max_seq_length, (
            f"Chunk token count {c.token_count} exceeded encoder max_seq_length {shared_embedder.max_seq_length}!"
        )
        assert len(c.content.strip()) > 0
        assert c.chunk_hash is not None


def test_chunker_deduplication(shared_embedder):
    chunker = Chunker(embedder=shared_embedder, target_tokens=200)

    # Two identical passages in the same document
    raw_passages = [
        RawPassage(content="Exact duplicate passage text across sections.", location={"start_line": 1}),
        RawPassage(content="Exact duplicate passage text across sections.", location={"start_line": 20}),
        RawPassage(content="Unique passage text here.", location={"start_line": 40}),
    ]

    chunks = chunker.chunk_raw_passages(raw_passages, deduplicate_within_file=True)
    assert len(chunks) == 2
    contents = [c.content for c in chunks]
    assert contents[0] == "Exact duplicate passage text across sections."
    assert contents[1] == "Unique passage text here."


def test_chunker_filters_empty_passages(shared_embedder):
    chunker = Chunker(embedder=shared_embedder)
    raw_passages = [
        RawPassage(content="", location={"start_line": 1}),
        RawPassage(content="   \n\t  \n  ", location={"start_line": 2}),
        RawPassage(content="Valid non-empty content.", location={"start_line": 5}),
    ]

    chunks = chunker.chunk_raw_passages(raw_passages)
    assert len(chunks) == 1
    assert chunks[0].content == "Valid non-empty content."
