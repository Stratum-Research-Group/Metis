"""
Stoma Source Plugins Package.
Provides plugin registry and file dispatcher.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

from stoma.plugins.base import ExtractionResult, RawPassage, SourcePlugin
from stoma.plugins.code import CodePlugin
from stoma.plugins.ocr import OCRPlugin
from stoma.plugins.pdf import PDFPlugin
from stoma.plugins.text import TextPlugin
from stoma.plugins.whisper import WhisperPlugin


class PluginRegistry:
    """Registry of active source plugins."""

    def __init__(self, plugins: Optional[List[SourcePlugin]] = None):
        if plugins is None:
            self.plugins: List[SourcePlugin] = [
                OCRPlugin(),       # Check OCR hook first (specific filename patterns)
                WhisperPlugin(),   # Check Whisper transcripts next
                PDFPlugin(),       # Local PDF parser
                CodePlugin(),      # Code files
                TextPlugin(),      # Plain text & markdown fallback
            ]
        else:
            self.plugins = plugins

    def get_plugin_for_file(self, path: Path) -> Optional[SourcePlugin]:
        """Find the first plugin that supports the given file."""
        for plugin in self.plugins:
            if plugin.supports_file(path):
                return plugin
        return None

    def extract_file(self, path: Path, rel_path: str) -> ExtractionResult:
        """Extract content from a file using the appropriate plugin."""
        plugin = self.get_plugin_for_file(path)
        if plugin is None:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash="",
                size_bytes=path.stat().st_size if path.exists() else 0,
                mtime=path.stat().st_mtime if path.exists() else 0.0,
                passages=[],
                plugin_name="unknown",
                is_supported=False,
                error_message=f"No source plugin supports file format '{path.suffix}'.",
            )
        return plugin.extract(path, rel_path)

    def extract_file_stream(self, path: Path, rel_path: str) -> Iterator[RawPassage]:
        """Stream passages from a file using the appropriate plugin."""
        from stoma.plugins.base import UnsupportedDocumentError
        plugin = self.get_plugin_for_file(path)
        if plugin is None:
            raise UnsupportedDocumentError(f"No source plugin supports file format '{path.suffix}'.")
        yield from plugin.extract_stream(path, rel_path)


from stoma.plugins.base import (
    CorruptedDocumentError,
    EmptyExtractionError,
    ExtractionError,
    ExtractionResult,
    FileModifiedDuringExtractionError,
    ImageOnlyPDFError,
    RawPassage,
    SourcePlugin,
    UnsupportedDocumentError,
    ZeroPagePDFError,
)

__all__ = [
    "SourcePlugin",
    "RawPassage",
    "ExtractionResult",
    "ExtractionError",
    "CorruptedDocumentError",
    "ImageOnlyPDFError",
    "UnsupportedDocumentError",
    "ZeroPagePDFError",
    "EmptyExtractionError",
    "FileModifiedDuringExtractionError",
    "PluginRegistry",
    "TextPlugin",
    "PDFPlugin",
    "CodePlugin",
    "OCRPlugin",
    "WhisperPlugin",
]
