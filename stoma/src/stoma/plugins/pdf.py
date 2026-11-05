"""
PDF Source Plugin.
Uses local pypdf parser.
Explicitly distinguishes text-based PDFs from scanned/image-only PDFs and reports
image-only PDFs as unsupported unless pre-extracted OCR exists.
Supports streaming page-by-page extraction to bound memory on large PDFs.
Fails loud on corrupted PDFs or page decode errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Set

try:
    import pypdf
except ImportError:
    pypdf = None  # type: ignore

from stoma.plugins.base import (
    CorruptedDocumentError,
    ExtractionResult,
    ImageOnlyPDFError,
    RawPassage,
    SourcePlugin,
    ZeroPagePDFError,
)
from stoma.provenance import compute_file_sha256

logger = logging.getLogger(__name__)


class PDFPlugin(SourcePlugin):
    """Plugin for local parsing of PDF documents."""

    name: str = "pdf"
    supported_extensions: Set[str] = {".pdf"}

    def extract_stream(self, path: Path, rel_path: str) -> Iterator[RawPassage]:
        """
        Stream passages from PDF page by page.
        Propagates corrupt PDF and read errors loudly; detects image-only PDFs.
        """
        if pypdf is None:
            raise CorruptedDocumentError(f"Cannot parse PDF '{path}': pypdf package is not installed.")
        try:
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
        except Exception as e:
            raise CorruptedDocumentError(f"Corrupt or invalid PDF file '{rel_path}': {e}") from e

        if total_pages == 0:
            raise ZeroPagePDFError(f"PDF file '{rel_path}' contains 0 pages (empty/unsupported document).")

        extracted_any_text = False
        for page_idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
            except Exception as e:
                raise CorruptedDocumentError(
                    f"Failed to extract text from page {page_idx} of {total_pages} in '{rel_path}': {e}"
                ) from e

            cleaned = (page_text or "").strip()
            if cleaned:
                extracted_any_text = True
                yield RawPassage(
                    content=cleaned,
                    location={
                        "page": page_idx,
                        "total_pages": total_pages,
                    },
                    section=f"Page {page_idx}",
                )

        if total_pages > 0 and not extracted_any_text:
            raise ImageOnlyPDFError(
                f"Unsupported image-only / scanned PDF: '{rel_path}' contains {total_pages} page(s) but 0 extractable text. "
                "Document appears to be image-only / scanned. Pre-extracted OCR is required to ingest this document."
            )

    def extract(self, path: Path, rel_path: str) -> ExtractionResult:
        """Extract all passages into an ExtractionResult using streaming under the hood."""
        try:
            stat = path.stat()
            file_hash = compute_file_sha256(path)
        except Exception as e:
            return self._create_error_result(path, rel_path, f"Failed to access PDF file: {e}")

        passages: List[RawPassage] = []
        try:
            for p in self.extract_stream(path, rel_path):
                passages.append(p)
        except ImageOnlyPDFError as e:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                is_supported=False,
                error_message=str(e),
                warning_message="Pre-extracted OCR is required to ingest this document.",
            )
        except ZeroPagePDFError as e:
            return ExtractionResult(
                path=path,
                rel_path=rel_path,
                content_hash=file_hash,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                passages=[],
                plugin_name=self.name,
                is_supported=False,
                error_message=str(e),
                warning_message="0-page PDF document is empty or unsupported.",
            )
        except CorruptedDocumentError as e:
            return self._create_error_result(path, rel_path, str(e))
        except Exception as e:
            return self._create_error_result(path, rel_path, f"PDF extraction failed: {e}")

        return ExtractionResult(
            path=path,
            rel_path=rel_path,
            content_hash=file_hash,
            size_bytes=stat.st_size,
            mtime=stat.st_mtime,
            passages=passages,
            plugin_name=self.name,
            is_supported=True,
        )
