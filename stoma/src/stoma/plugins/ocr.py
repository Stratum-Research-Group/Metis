"""
OCR Hook Source Plugin.
Consumes pre-extracted OCR text outputs from external scanlation/OCR tools.
Does NOT perform OCR itself; parses pre-extracted json/text artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List, Set

from stoma.plugins.base import ExtractionResult, RawPassage, SourcePlugin
from stoma.provenance import compute_file_sha256


class OCRPlugin(SourcePlugin):
    """Plugin to consume pre-extracted OCR artifacts from external pipelines."""

    name: str = "ocr"
    supported_extensions: Set[str] = {".ocr.json", ".ocr.txt"}

    def supports_file(self, path: Path) -> bool:
        name = path.name.lower()
        if name.endswith(".ocr.json") or name.endswith(".ocr.txt"):
            return True
        # Also support any .json or .txt directly inside an 'ocr' or 'scanlation' folder
        parent_names = {p.lower() for p in path.parent.parts}
        if "ocr" in parent_names or "scanlation" in parent_names:
            return path.suffix.lower() in {".json", ".txt"}
        return False

    MAX_SAFE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB safe size cap for JSON/OCR parsing

    def extract(self, path: Path, rel_path: str) -> ExtractionResult:
        try:
            stat = path.stat()
            file_hash = compute_file_sha256(path)
            if stat.st_size > self.MAX_SAFE_SIZE_BYTES:
                return ExtractionResult(
                    path=path,
                    rel_path=rel_path,
                    content_hash=file_hash,
                    size_bytes=stat.st_size,
                    mtime=stat.st_mtime,
                    passages=[],
                    plugin_name=self.name,
                    is_supported=False,
                    error_message=(
                        f"OCR artifact size {stat.st_size} bytes exceeds safe 25MB limit. "
                        f"Memory-unbounded JSON/OCR parsing unsupported; source BLOCKED to protect process memory."
                    ),
                )
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return self._create_error_result(path, rel_path, f"Failed to read OCR artifact: {e}")

        passages: List[RawPassage] = []

        if path.suffix.lower() == ".json" or path.name.lower().endswith(".ocr.json"):
            try:
                data = json.loads(raw_text)
                passages = self._parse_ocr_json(data)
            except Exception as e:
                return self._create_error_result(path, rel_path, f"Malformed OCR JSON: {e}")
        else:
            passages = self._parse_ocr_text(raw_text)

        if not passages:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                warning_message="OCR artifact contains no extracted text.",
            )

        return ExtractionResult(
            path=path,
            rel_path=rel_path,
            content_hash=file_hash,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            passages=passages,
            plugin_name=self.name,
        )

    def extract_stream(self, path: Path, rel_path: str) -> Iterator[RawPassage]:
        """Stream passages from OCR artifact."""
        from stoma.plugins.base import ExtractionError, UnsupportedDocumentError
        res = self.extract(path, rel_path)
        if not res.is_supported:
            raise UnsupportedDocumentError(res.error_message or f"Unsupported OCR artifact '{rel_path}'")
        if res.error_message:
            raise ExtractionError(res.error_message)
        yield from res.passages

    def _parse_ocr_json(self, data: object) -> List[RawPassage]:
        passages: List[RawPassage] = []
        if isinstance(data, list):
            for idx, item in enumerate(data, start=1):
                if isinstance(item, dict):
                    text = item.get("text", "") or item.get("content", "")
                    page = item.get("page", item.get("page_number", idx))
                    box = item.get("box", item.get("bounding_box"))
                    source_img = item.get("image", item.get("source_image"))
                    if text.strip():
                        passages.append(
                            RawPassage(
                                content=text.strip(),
                                location={"page": page, "box": box, "source_image": source_img, "ocr": True},
                                section=f"Page {page}",
                            )
                        )
                elif isinstance(item, str) and item.strip():
                    passages.append(
                        RawPassage(
                            content=item.strip(),
                            location={"page": idx, "ocr": True},
                            section=f"Page {idx}",
                        )
                    )
        elif isinstance(data, dict):
            # Check for pages array
            if "pages" in data and isinstance(data["pages"], list):
                for idx, page_item in enumerate(data["pages"], start=1):
                    if isinstance(page_item, dict):
                        text = page_item.get("text", "")
                        page_num = page_item.get("page", page_item.get("page_number", idx))
                        if text.strip():
                            passages.append(
                                RawPassage(
                                    content=text.strip(),
                                    location={"page": page_num, "ocr": True},
                                    section=f"Page {page_num}",
                                )
                            )
            elif "text" in data and str(data["text"]).strip():
                passages.append(
                    RawPassage(
                        content=str(data["text"]).strip(),
                        location={"page": data.get("page", 1), "ocr": True},
                        section="OCR Text",
                    )
                )

            # Check for structured tables (DePlot / document intelligence format)
            if "tables" in data and isinstance(data["tables"], list):
                for t_idx, table in enumerate(data["tables"], start=1):
                    if isinstance(table, dict):
                        t_title = table.get("title", table.get("caption", f"Table {t_idx}"))
                        page = table.get("page", 1)
                        headers = table.get("headers", table.get("columns", []))
                        rows = table.get("rows", table.get("data", []))
                        md_table = self._format_markdown_table(headers, rows)
                        if md_table:
                            content = f"### {t_title}\n\n{md_table}"
                            passages.append(
                                RawPassage(
                                    content=content,
                                    location={"page": page, "table_index": t_idx, "ocr": True, "type": "table"},
                                    section=f"Page {page} - {t_title}",
                                )
                            )
                    elif isinstance(table, str) and table.strip():
                        passages.append(
                            RawPassage(
                                content=table.strip(),
                                location={"page": 1, "table_index": t_idx, "ocr": True, "type": "table"},
                                section=f"Table {t_idx}",
                            )
                        )

            # Check for structured charts (DePlot / MatCha format)
            if "charts" in data and isinstance(data["charts"], list):
                for c_idx, chart in enumerate(data["charts"], start=1):
                    if isinstance(chart, dict):
                        c_title = chart.get("title", f"Chart {c_idx}")
                        c_type = chart.get("type", "chart")
                        page = chart.get("page", 1)
                        summary = chart.get("summary", chart.get("description", ""))
                        table_data = chart.get("table", chart.get("series", []))
                        
                        parts = [f"### Chart: {c_title} ({c_type})"]
                        if summary:
                            parts.append(summary)
                        if isinstance(table_data, dict) and "headers" in table_data:
                            md_table = self._format_markdown_table(
                                table_data.get("headers", []), table_data.get("rows", [])
                            )
                            if md_table:
                                parts.append(md_table)
                        elif isinstance(table_data, list) and table_data and isinstance(table_data[0], list):
                            md_table = self._format_markdown_table([], table_data)
                            if md_table:
                                parts.append(md_table)
                        
                        content = "\n\n".join(parts)
                        passages.append(
                            RawPassage(
                                content=content,
                                location={"page": page, "chart_index": c_idx, "ocr": True, "type": "chart"},
                                section=f"Page {page} - Chart {c_title}",
                            )
                        )

            # Check for structured key-value fields (VRDU format)
            kv_source = data.get("fields") or data.get("key_values") or data.get("entities")
            if kv_source:
                page = data.get("page", 1)
                kv_lines = []
                if isinstance(kv_source, dict):
                    for k, v in kv_source.items():
                        kv_lines.append(f"- **{k}**: {v}")
                elif isinstance(kv_source, list):
                    for item in kv_source:
                        if isinstance(item, dict):
                            k = item.get("key", item.get("field", item.get("name", "Unknown")))
                            v = item.get("value", item.get("text", ""))
                            kv_lines.append(f"- **{k}**: {v}")
                
                if kv_lines:
                    content = f"### Structured Fields (Page {page})\n" + "\n".join(kv_lines)
                    passages.append(
                        RawPassage(
                            content=content,
                            location={"page": page, "ocr": True, "type": "key_values"},
                            section=f"Page {page} - Structured Fields",
                        )
                    )

        return passages

    @staticmethod
    def _format_markdown_table(headers: List[Any], rows: List[List[Any]]) -> str:
        """Render a clean markdown table from list of headers and rows."""
        if not headers and not rows:
            return ""
        if not headers and rows:
            headers = [f"Col {i+1}" for i in range(len(rows[0]))]
        
        col_count = len(headers)
        clean_headers = [str(h).strip().replace("|", "\\|") for h in headers]
        header_line = "| " + " | ".join(clean_headers) + " |"
        sep_line = "| " + " | ".join([":---"] * col_count) + " |"
        
        row_lines = []
        for r in rows:
            cells = [str(c).strip().replace("|", "\\|") for c in r[:col_count]]
            if len(cells) < col_count:
                cells.extend([""] * (col_count - len(cells)))
            row_lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join([header_line, sep_line] + row_lines)

    def _parse_ocr_text(self, text: str) -> List[RawPassage]:
        passages: List[RawPassage] = []
        lines = text.splitlines()
        page_re = re.compile(r"^[-=\s]*(?:page|p\.)\s*(\d+)[:\s-]*$", re.IGNORECASE)

        current_page = 1
        current_lines: List[str] = []

        for line in lines:
            m = page_re.match(line.strip())
            if m:
                if current_lines:
                    block = "\n".join(current_lines).strip()
                    if block:
                        passages.append(
                            RawPassage(
                                content=block,
                                location={"page": current_page, "ocr": True},
                                section=f"Page {current_page}",
                            )
                        )
                    current_lines = []
                current_page = int(m.group(1))
            else:
                current_lines.append(line)

        if current_lines:
            block = "\n".join(current_lines).strip()
            if block:
                passages.append(
                    RawPassage(
                        content=block,
                        location={"page": current_page, "ocr": True},
                        section=f"Page {current_page}",
                    )
                )

        return passages
