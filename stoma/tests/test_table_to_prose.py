"""Tests for Table-to-Prose Normalization in Stoma (Item 111 / arXiv:2609.24220)."""

import pytest
from stoma.table_normalizer import (
    ParsedTable,
    TableProseNormalizer,
    normalize_markdown_tables,
)
from stoma.plugins.base import RawPassage
from stoma.chunker import Chunker


SAMPLE_TABLE_MD = """# System Configuration

The table below describes timeout parameters:

| Parameter | Default | Timeout Limit | Description |
|-----------|---------|---------------|-------------|
| connect   | 5s      | 30s           | TCP connect timeout |
| read      | 15s     | 60s           | Read socket deadline |
| retry     | 3       | -             | Maximum retries |

Following the table, standard operations resume.
"""


def test_extract_tables():
    """Verify markdown table extraction with title context detection."""
    tables = TableProseNormalizer.extract_tables(SAMPLE_TABLE_MD)
    assert len(tables) == 1

    t = tables[0]
    assert t.headers == ["Parameter", "Default", "Timeout Limit", "Description"]
    assert len(t.rows) == 3
    assert t.rows[0] == ["connect", "5s", "30s", "TCP connect timeout"]
    assert t.rows[1] == ["read", "15s", "60s", "Read socket deadline"]
    assert t.rows[2] == ["retry", "3", "-", "Maximum retries"]
    # Verify title detection
    assert t.title == "System Configuration" or "table" in (t.title or "").lower()


def test_table_to_prose_assertions():
    """Verify rows are rendered as self-contained factual sentences."""
    tables = TableProseNormalizer.extract_tables(SAMPLE_TABLE_MD)
    t = tables[0]
    assertions = TableProseNormalizer.table_to_prose_assertions(t)

    assert len(assertions) == 3
    # Check row 1
    assert "Row 1:" in assertions[0]
    assert "Parameter is 'connect'" in assertions[0]
    assert "Default is '5s'" in assertions[0]
    assert "Timeout Limit is '30s'" in assertions[0]
    assert "Description is 'TCP connect timeout'" in assertions[0]

    # Check row 3 with '-' handling
    assert "Row 3:" in assertions[2]
    assert "Parameter is 'retry'" in assertions[2]
    assert "Timeout Limit is unspecified" in assertions[2]


def test_normalize_markdown_tables_replace_and_append():
    """Verify replace and append modes."""
    # Replace mode
    replaced = normalize_markdown_tables(SAMPLE_TABLE_MD, mode="replace")
    assert "| Parameter | Default |" not in replaced
    assert "|-----------|" not in replaced
    assert "Row 1: Parameter is 'connect'" in replaced
    assert "Following the table, standard operations resume." in replaced

    # Append mode
    appended = normalize_markdown_tables(SAMPLE_TABLE_MD, mode="append")
    assert "| Parameter | Default |" in appended
    assert "Row 1: Parameter is 'connect'" in appended


def test_chunker_with_table_normalization():
    """Verify Chunker with normalize_tables=True converts tables before splitting."""
    class MockEmbedder:
        effective_max_tokens = 256
        max_seq_length = 256
        def count_tokens(self, text: str) -> int:
            return len(text.split())

    embedder = MockEmbedder()
    chunker = Chunker(
        embedder=embedder,
        target_tokens=100,
        normalize_tables=True,
    )

    raw = RawPassage(
        content=SAMPLE_TABLE_MD,
        location={"line_start": 1, "line_end": 12},
        section="Config",
    )

    chunks = chunker.chunk_raw_passages([raw])
    assert len(chunks) >= 1
    combined_content = " ".join(c.content for c in chunks)
    # The chunks should contain the converted prose, not raw pipes
    assert "Parameter is 'connect'" in combined_content
    assert "|-----------|" not in combined_content
