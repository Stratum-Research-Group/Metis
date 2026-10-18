"""
Stoma Safe Extraction & Archive Bomb Defense (Phase 2 / WALDO model).
Features:
- Decompression safety: defends against ZIP/EPUB expansion bombs (ratio cap, file count cap, byte cap).
- Panic-safe parser wrapping: contains lower-level parser crashes and returns clean ExtractionResult with error status.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import zipfile

from stoma.plugins.base import CorruptedDocumentError, ExtractionResult, RawPassage

logger = logging.getLogger(__name__)


class ArchiveBombError(ValueError):
    """Raised when an archive exceeds safe compression ratio, file count, or total expansion limits."""
    pass


ArchiveSafetyError = ArchiveBombError



class ArchiveSafetyGuard:
    """
    Validates ZIP and EPUB archives before extraction to prevent denial-of-service via expansion bombs.
    """

    MAX_UNCOMPRESSED_BYTES: int = 50 * 1024 * 1024  # 50 MB
    MAX_FILE_COUNT: int = 1000
    MAX_COMPRESSION_RATIO: float = 100.0

    @classmethod
    def inspect_zip_archive(cls, zip_path: Path) -> None:
        """
        Inspects ZipInfo entries without extracting data.
        Raises ArchiveBombError if archive exceeds safe thresholds.
        """
        if not zip_path.is_file():
            raise FileNotFoundError(f"Archive '{zip_path}' not found")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                infolist = zf.infolist()
                if len(infolist) > cls.MAX_FILE_COUNT:
                    raise ArchiveBombError(
                        f"Archive '{zip_path.name}' contains {len(infolist)} files (limit: {cls.MAX_FILE_COUNT})"
                    )

                total_uncompressed = 0
                total_compressed = 0

                for info in infolist:
                    # Check for path traversal inside zip
                    if info.filename.startswith(("/", "\\")) or ".." in Path(info.filename).parts:
                        raise ArchiveBombError(f"Archive entry '{info.filename}' contains path traversal sequence")

                    total_uncompressed += info.file_size
                    total_compressed += info.compress_size

                    if total_uncompressed > cls.MAX_UNCOMPRESSED_BYTES:
                        raise ArchiveBombError(
                            f"Archive '{zip_path.name}' exceeds uncompressed size limit ({total_uncompressed} > {cls.MAX_UNCOMPRESSED_BYTES} bytes)"
                        )

                if total_compressed > 0:
                    ratio = total_uncompressed / total_compressed
                    if ratio > cls.MAX_COMPRESSION_RATIO:
                        raise ArchiveBombError(
                            f"Archive '{zip_path.name}' compression ratio ({ratio:.1f}:1) exceeds limit ({cls.MAX_COMPRESSION_RATIO}:1)"
                        )

        except zipfile.BadZipFile as e:
            raise CorruptedDocumentError(f"Invalid or corrupted zip archive '{zip_path.name}': {e}") from e


class SafeExtractor:
    """
    Wraps plugin extraction with archive validation and panic containment.
    """

    @classmethod
    def safe_extract_file(
        cls,
        plugin: Any,
        path: Path,
        rel_path: str,
    ) -> ExtractionResult:
        """
        Executes document extraction with archive bomb pre-checks and error containment.
        """
        # If it's a zip/epub, preflight archive safety
        ext = path.suffix.lower()
        if ext in (".zip", ".epub"):
            try:
                ArchiveSafetyGuard.inspect_zip_archive(path)
            except Exception as e:
                logger.warning(f"Archive safety check failed for '{rel_path}': {e}")
                return ExtractionResult(
                    passages=[],
                    file_sha256="",
                    plugin_name=getattr(plugin, "name", "unknown"),
                    error_message=str(e),
                )

        try:
            return plugin.extract(path, rel_path)
        except Exception as e:
            logger.error(f"Safe extractor caught error on '{rel_path}': {e}", exc_info=True)
            return ExtractionResult(
                passages=[],
                file_sha256="",
                plugin_name=getattr(plugin, "name", "unknown"),
                error_message=str(e),
            )
