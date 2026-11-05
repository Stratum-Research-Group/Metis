"""
Table-to-Prose Normalizer for Retrieval-Aware Chunking (Item 111).

Inspired by 'Document Retrieval-Aware Chunking (D-RAC)' (arXiv:2609.24220).
Converts structured Markdown tables into self-contained factual prose statements
prior to chunking and dense vector embedding. Prevents split-boundary context loss
where header semantics are separated from data rows in downstream chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ParsedTable:
    """Structured representation of an extracted markdown table."""
    headers: List[str]
    rows: List[List[str]]
    raw_markdown: str
    title: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


class TableProseNormalizer:
    """Converts markdown table rows into self-contained natural language factual assertions."""

    # Regex matching markdown table rows (pipes with content)
    _ROW_REGEX = re.compile(r"^\s*\|(.+)\|\s*$")
    _DIVIDER_CELL_REGEX = re.compile(r"^:?-+:?$")

    @classmethod
    def _split_row(cls, line: str) -> List[str]:
        """Split a pipe-delimited table line into stripped cell contents."""
        match = cls._ROW_REGEX.match(line.strip())
        if not match:
            # Fallback for tables without outer pipes: col1 | col2 | col3
            cells = line.split("|")
        else:
            cells = match.group(1).split("|")
        return [c.strip() for c in cells]

    @classmethod
    def _is_divider_row(cls, cells: List[str]) -> bool:
        """Check if cells represent a markdown divider line (e.g., |---|:---:|---:|)."""
        if not cells:
            return False
        return all(cls._DIVIDER_CELL_REGEX.match(c) is not None for c in cells if c)

    @classmethod
    def extract_tables(cls, text: str) -> List[ParsedTable]:
        """Extract all markdown tables from the document text with position offsets."""
        lines = text.splitlines(keepends=True)
        tables: List[ParsedTable] = []

        i = 0
        char_offset = 0
        line_offsets: List[int] = []
        for line in lines:
            line_offsets.append(char_offset)
            char_offset += len(line)

        num_lines = len(lines)
        while i < num_lines:
            line = lines[i].strip()
            # A table requires at least header line, divider line, and 1 data row
            if ("|" in line) and i + 1 < num_lines:
                potential_header = cls._split_row(line)
                divider_line = lines[i + 1].strip()
                potential_divider = cls._split_row(divider_line)

                if (
                    len(potential_header) >= 2
                    and len(potential_divider) >= 2
                    and cls._is_divider_row(potential_divider)
                ):
                    # Found table header and divider
                    headers = [
                        h if h else f"Column {idx + 1}"
                        for idx, h in enumerate(potential_header)
                    ]
                    num_cols = len(headers)
                    table_rows: List[List[str]] = []
                    table_start_char = line_offsets[i]

                    # Look backwards skipping empty lines for a preceding title or caption
                    table_title: Optional[str] = None
                    for back_idx in range(i - 1, -1, -1):
                        prev_line = lines[back_idx].strip()
                        if not prev_line:
                            continue
                        if prev_line.startswith("#"):
                            table_title = prev_line.lstrip("#").strip()
                            break
                        elif prev_line.lower().startswith("table:") or prev_line.lower().startswith("table "):
                            table_title = prev_line.split(":", 1)[-1].strip()
                            break
                        elif prev_line.endswith(":") or "table" in prev_line.lower():
                            table_title = prev_line.rstrip(":")
                            break
                        else:
                            break

                    row_idx = i + 2
                    while row_idx < num_lines:
                        r_line = lines[row_idx].strip()
                        if not r_line or "|" not in r_line:
                            break
                        cells = cls._split_row(r_line)
                        if cls._is_divider_row(cells):
                            # Stray divider
                            row_idx += 1
                            continue
                        # Pad or truncate cells to match headers
                        if len(cells) < num_cols:
                            cells = cells + [""] * (num_cols - len(cells))
                        elif len(cells) > num_cols:
                            cells = cells[:num_cols]
                        table_rows.append(cells)
                        row_idx += 1

                    if table_rows:
                        # Compute end char offset
                        last_line_idx = row_idx - 1
                        table_end_char = line_offsets[last_line_idx] + len(lines[last_line_idx])
                        raw_table_md = "".join(lines[i:row_idx])

                        tables.append(
                            ParsedTable(
                                headers=headers,
                                rows=table_rows,
                                raw_markdown=raw_table_md,
                                title=table_title,
                                start_pos=table_start_char,
                                end_pos=table_end_char,
                            )
                        )
                        i = row_idx
                        continue

            i += 1

        return tables

    @classmethod
    def table_to_prose_assertions(
        cls,
        table: ParsedTable,
        include_row_numbers: bool = True,
    ) -> List[str]:
        """Convert table rows into self-contained prose statements with header anchoring."""
        prose_statements: List[str] = []
        table_context = f"In table '{table.title}', " if table.title else "In the table, "

        for row_idx, row in enumerate(table.rows, start=1):
            row_prefix = f"Row {row_idx}: " if include_row_numbers else ""
            field_statements: List[str] = []

            for header, val in zip(table.headers, row):
                clean_val = val.strip()
                if not clean_val or clean_val in {"-", "--", "N/A", "n/a", "None"}:
                    field_statements.append(f"{header} is unspecified")
                else:
                    field_statements.append(f"{header} is '{clean_val}'")

            sentence = f"{table_context}{row_prefix}" + ", ".join(field_statements) + "."
            prose_statements.append(sentence)

        return prose_statements

    @classmethod
    def normalize_tables_in_text(
        cls,
        text: str,
        mode: str = "replace",
        include_row_numbers: bool = True,
    ) -> str:
        """
        Normalize all markdown tables found in the given text.

        Modes:
            'replace': Replaces the raw markdown table with the generated prose assertion block.
            'append': Keeps the raw markdown table and appends the prose block immediately after it.
        """
        tables = cls.extract_tables(text)
        if not tables:
            return text

        # Sort tables by start position descending to perform safe slice replacements
        sorted_tables = sorted(tables, key=lambda t: t.start_pos, reverse=True)
        result = text

        for table in sorted_tables:
            prose_lines = cls.table_to_prose_assertions(table, include_row_numbers=include_row_numbers)
            prose_block = "\n".join(prose_lines) + "\n"

            if mode == "replace":
                result = result[:table.start_pos] + prose_block + result[table.end_pos:]
            elif mode == "append":
                replacement = table.raw_markdown.rstrip() + "\n\n" + prose_block
                result = result[:table.start_pos] + replacement + result[table.end_pos:]
            else:
                raise ValueError(f"Unknown table normalization mode: {mode}")

        return result


def normalize_markdown_tables(text: str, mode: str = "replace") -> str:
    """Functional convenience wrapper for normalizing markdown tables to prose assertions."""
    return TableProseNormalizer.normalize_tables_in_text(text, mode=mode)
