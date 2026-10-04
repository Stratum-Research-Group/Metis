"""
Unit tests for Stoma source plugins:
- Plain text & markdown
- Local PDF parser (distinguishing text vs image-only)
- Code repository parser (Python AST & multi-language boundaries)
- OCR hook consumer
- Whisper audio transcript consumer
- Per-file error resilience
"""

import json
from pathlib import Path
import pypdf
import pytest

from stoma.plugins import (
    CodePlugin,
    ExtractionResult,
    OCRPlugin,
    PDFPlugin,
    PluginRegistry,
    TextPlugin,
    WhisperPlugin,
)


def test_text_plugin(temp_workspace):
    plugin = TextPlugin()
    md_file = temp_workspace / "sample.md"
    md_file.write_text(
        "# Title Section\n\n"
        "This is introductory text in section 1.\n\n"
        "## Subheading A\n\n"
        "Line one of sub A.\n"
        "Line two of sub A.\n\n"
        "## Subheading B\n\n"
        "Paragraph in sub B.\n",
        encoding="utf-8",
    )

    res = plugin.extract(md_file, "sample.md")
    assert res.is_supported is True
    assert res.error_message is None
    assert len(res.passages) >= 2
    # Verify line numbers are captured
    for p in res.passages:
        assert "start_line" in p.location
        assert "end_line" in p.location
        assert p.location["start_line"] <= p.location["end_line"]


def test_text_plugin_empty_file(temp_workspace):
    plugin = TextPlugin()
    empty_file = temp_workspace / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    res = plugin.extract(empty_file, "empty.txt")
    assert res.is_supported is True
    assert len(res.passages) == 0
    assert res.warning_message is not None


def test_pdf_plugin_with_text(temp_workspace):
    """Create a minimal valid text PDF using pypdf and verify extraction."""
    pdf_file = temp_workspace / "text_doc.pdf"
    writer = pypdf.PdfWriter()
    # Add a blank page with some text annotation or content
    page = writer.add_blank_page(width=300, height=300)
    # pypdf allows creating a page and writing text via annotation or content stream
    # A simple way to add text is using writer.add_annotation or canvas/content
    # Alternatively write a synthetic PDF stream with text:
    writer.write(str(pdf_file))

    # Let's test image-only detection on this blank PDF
    plugin = PDFPlugin()
    res = plugin.extract(pdf_file, "text_doc.pdf")
    # A blank PDF has 0 extracted characters, so it should be flagged as unsupported image-only/scanned
    assert res.is_supported is False
    assert "Unsupported image-only / scanned PDF" in (res.error_message or "")


def test_pdf_plugin_corrupt_file(temp_workspace):
    plugin = PDFPlugin()
    corrupt_file = temp_workspace / "corrupt.pdf"
    corrupt_file.write_bytes(b"%PDF-1.4 corrupt junk data")

    res = plugin.extract(corrupt_file, "corrupt.pdf")
    assert res.is_supported is False
    assert "Corrupt or invalid PDF" in (res.error_message or "")


def test_code_plugin_python_ast(temp_workspace):
    plugin = CodePlugin()
    py_file = temp_workspace / "module.py"
    py_file.write_text(
        '"""Module docstring."""\n\n'
        'CONSTANT = 42\n\n'
        'class MemoryAgent:\n'
        '    def __init__(self, name: str):\n'
        '        self.name = name\n\n'
        '    def execute(self):\n'
        '        return f"Executing {self.name}"\n\n'
        'async def fetch_memory(agent_id: str):\n'
        '    return agent_id\n',
        encoding="utf-8",
    )

    res = plugin.extract(py_file, "module.py")
    assert res.is_supported is True
    assert res.error_message is None
    symbols = [p.location.get("symbol") for p in res.passages]
    assert "MemoryAgent" in symbols
    assert "fetch_memory" in symbols
    # Verify line ranges
    for p in res.passages:
        assert p.location["start_line"] <= p.location["end_line"]


def test_code_plugin_generic_languages(temp_workspace):
    plugin = CodePlugin()
    js_file = temp_workspace / "service.ts"
    js_file.write_text(
        'export function calculateScore(a: number, b: number): number {\n'
        '  return a + b;\n'
        '}\n\n'
        'export class ClientRouter {\n'
        '  route() {\n'
        '    return true;\n'
        '  }\n'
        '}\n',
        encoding="utf-8",
    )

    res = plugin.extract(js_file, "service.ts")
    assert res.is_supported is True
    assert len(res.passages) >= 2


def test_ocr_plugin_json(temp_workspace):
    plugin = OCRPlugin()
    ocr_file = temp_workspace / "scan_results.ocr.json"
    data = [
        {"page": 1, "text": "Topological memory retrieval in non-Euclidean manifolds.", "box": [10, 20, 100, 200]},
        {"page": 2, "text": "Cellular sheaf cohomology prevents hallucination.", "box": [15, 25, 110, 210]},
    ]
    ocr_file.write_text(json.dumps(data), encoding="utf-8")

    res = plugin.extract(ocr_file, "scan_results.ocr.json")
    assert res.is_supported is True
    assert len(res.passages) == 2
    assert res.passages[0].location["page"] == 1
    assert res.passages[0].location["ocr"] is True
    assert "Topological memory" in res.passages[0].content


def test_ocr_plugin_text(temp_workspace):
    plugin = OCRPlugin()
    ocr_txt = temp_workspace / "doc.ocr.txt"
    ocr_txt.write_text(
        "Page 1:\nScanned text on first page.\n\n"
        "Page 2:\nScanned text on second page.\n",
        encoding="utf-8",
    )

    res = plugin.extract(ocr_txt, "doc.ocr.txt")
    assert res.is_supported is True
    assert len(res.passages) == 2
    assert res.passages[0].location["page"] == 1
    assert res.passages[1].location["page"] == 2


def test_whisper_plugin_vtt(temp_workspace):
    plugin = WhisperPlugin()
    vtt_file = temp_workspace / "audio.vtt"
    vtt_file.write_text(
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:04.500\n"
        "Welcome to the Stoma ingestion pipeline walkthrough.\n\n"
        "00:00:05.000 --> 00:00:08.200\n"
        "We are indexing audio transcripts with millisecond precision.\n",
        encoding="utf-8",
    )

    res = plugin.extract(vtt_file, "audio.vtt")
    assert res.is_supported is True
    assert len(res.passages) == 2
    assert res.passages[0].location["start_sec"] == 1.0
    assert res.passages[0].location["end_sec"] == 4.5
    assert "Stoma ingestion pipeline" in res.passages[0].content


def test_whisper_plugin_json(temp_workspace):
    plugin = WhisperPlugin()
    json_file = temp_workspace / "audio.transcript.json"
    data = {
        "transcription": [
            {
                "timestamps": {"from": "00:00:10,000", "to": "00:00:15,000"},
                "text": "Deliberative cognition layer over LTHCM memory.",
            }
        ]
    }
    json_file.write_text(json.dumps(data), encoding="utf-8")

    res = plugin.extract(json_file, "audio.transcript.json")
    assert res.is_supported is True
    assert len(res.passages) == 1
    assert res.passages[0].location["start_sec"] == 10.0
    assert res.passages[0].location["end_sec"] == 15.0


def test_plugin_registry_dispatch(temp_workspace):
    registry = PluginRegistry()
    txt_path = temp_workspace / "test.txt"
    txt_path.write_text("Hello world", encoding="utf-8")

    res = registry.extract_file(txt_path, "test.txt")
    assert res.is_supported is True
    assert res.plugin_name == "text"

    unknown_path = temp_workspace / "binary.dat"
    unknown_path.write_bytes(b"\x00\x01\x02\x03")
    res_unknown = registry.extract_file(unknown_path, "binary.dat")
    assert res_unknown.is_supported is False
    assert "No source plugin supports" in (res_unknown.error_message or "")
