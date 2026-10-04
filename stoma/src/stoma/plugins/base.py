"""
Stoma Source Plugin Base Interface.
Defines the contract for file discovery, text extraction, location tagging, and error handling.
Supports both batch extraction and streaming passage extraction for bounded memory usage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set

from stoma.provenance import compute_file_sha256


class ExtractionError(RuntimeError):
    """Base exception for document extraction failures."""
    pass


class CorruptedDocumentError(ExtractionError):
    """Raised when a document is corrupted, truncated, or unreadable."""
    pass


class ImageOnlyPDFError(ExtractionError):
    """Raised when a PDF contains pages but zero extractable text (scanned document)."""
    pass


class UnsupportedDocumentError(ExtractionError):
    """Raised when a document format or feature is unsupported."""
    pass


class ZeroPagePDFError(UnsupportedDocumentError):
    """Raised when a PDF document contains 0 pages (empty/unsupported document)."""
    pass


class EmptyExtractionError(ExtractionError):
    """Raised when extraction yields 0 passages from a non-empty supported file."""
    pass


class FileModifiedDuringExtractionError(ExtractionError):
    """Raised when a file on disk changes while streaming extraction is in progress."""
    pass


@dataclass
class RawPassage:
    """A raw textual passage extracted from a source file before token chunking."""
    content: str
    location: Dict[str, Any] = field(default_factory=dict)
    section: Optional[str] = None
    extra_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """The result of extracting a document via a source plugin."""
    path: Path
    rel_path: str
    content_hash: str
    size_bytes: int
    mtime: float
    passages: List[RawPassage]
    plugin_name: str
    is_supported: bool = True
    error_message: Optional[str] = None
    warning_message: Optional[str] = None


class SourcePlugin(ABC):
    """Abstract base class for all Stoma source plugins."""

    name: str = "base"
    supported_extensions: Set[str] = set()

    def supports_file(self, path: Path) -> bool:
        """Return True if this plugin can process the given file."""
        return path.suffix.lower() in self.supported_extensions

    @abstractmethod
    def extract(self, path: Path, rel_path: str) -> ExtractionResult:
        """
        Extract passages and location metadata from the given file.
        Must handle errors per-file without crashing the batch.
        """
        pass

    def extract_stream(self, path: Path, rel_path: str) -> Iterator[RawPassage]:
        """
        Stream raw passages lazily from the source file to bound memory consumption.
        Default implementation wraps extract(). Plugins should override with true streaming.
        Fails loud if extraction encounters an error or unsupported format.
        """
        res = self.extract(path, rel_path)
        if not res.is_supported:
            raise UnsupportedDocumentError(res.error_message or f"Unsupported document format in '{rel_path}'")
        if res.error_message:
            raise ExtractionError(res.error_message)
        yield from res.passages

    def _create_error_result(
        self,
        path: Path,
        rel_path: str,
        error_msg: str,
        is_supported: bool = False,
    ) -> ExtractionResult:
        """Helper to create an error extraction result."""
        size = 0
        mtime = 0.0
        try:
            if path.exists():
                stat = path.stat()
                size = stat.st_size
                mtime = stat.st_mtime
        except Exception:
            pass

        return ExtractionResult(
            path=path,
            rel_path=rel_path,
            content_hash="",
            size_bytes=size,
            mtime=mtime,
            passages=[],
            plugin_name=self.name,
            is_supported=is_supported,
            error_message=error_msg,
        )
