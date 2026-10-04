"""
End-to-End Integration and Demo Test for Stoma.
Mandated by Spec v4, Line 54:
- Mixed folder (PDFs, markdown, code repo)
- Ingest, query with full provenance (path + timestamp + content hash)
- Modify file -> watcher/indexer replaces passages without duplication
- Delete file -> passages gone
- Restart indexer and repeat query
- Image-only PDF reported unsupported clearly
"""

import os
import time
from pathlib import Path
import pypdf
import pytest

from stoma.config import StomaConfig
from stoma.indexer import StomaIndexer
from stoma.provenance import compute_file_sha256


def _create_minimal_pdf_with_text(path: Path, text: str) -> None:
    """Create a minimal valid PDF containing extractable text."""
    text_cleaned = text.replace("(", "").replace(")", "")
    stream_data = f"BT /F1 12 Tf 72 712 Td ({text_cleaned}) Tj ET\n".encode("latin-1")
    stream_len = len(stream_data)

    pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
        + f"5 0 obj << /Length {stream_len} >> stream\n".encode("latin-1")
        + stream_data
        + b"endstream endobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000340 00000 n \n"
        b"trailer << /Size 6 /Root 1 0 R >>\n"
        b"startxref\n450\n%%EOF\n"
    )
    with open(path, "wb") as f:
        f.write(pdf_bytes)


def test_end_to_end_demo_flow(temp_workspace, shared_embedder):
    storage_dir = temp_workspace / ".stoma_e2e"
    storage_dir.mkdir(parents=True, exist_ok=True)
    ingest_root = temp_workspace / "demo_corpus"
    ingest_root.mkdir()

    cfg = StomaConfig(
        allowed_roots=[ingest_root],
        storage_dir=storage_dir,
        index_path=storage_dir / "index.pkl",
        manifest_path=storage_dir / "manifest.json",
    )

    # 1. Prepare mixed local corpus
    md_file = ingest_root / "architecture.md"
    md_file.write_text(
        "# Topological Architecture\n\n"
        "LTHCM utilizes Poincare hyperbolic embeddings and Lorentzian light cone causality.",
        encoding="utf-8",
    )

    code_file = ingest_root / "quantum_memory.py"
    code_file.write_text(
        "class QuantumMemoryStore:\n"
        "    def readout_superposition(self, query_state):\n"
        "        return 'Superposition readout successful'\n",
        encoding="utf-8",
    )

    text_pdf_file = ingest_root / "spec_sheet.pdf"
    _create_minimal_pdf_with_text(text_pdf_file, "Topological manifold specification document.")

    image_pdf_file = ingest_root / "scanned_doc.pdf"
    # Pure blank PDF = image-only / no text
    writer_blank = pypdf.PdfWriter()
    writer_blank.add_blank_page(width=300, height=300)
    with open(image_pdf_file, "wb") as f:
        writer_blank.write(f)

    # 2. Ingest corpus
    indexer = StomaIndexer(config=cfg, embedder=shared_embedder)
    summary = indexer.scan_and_ingest_directory(ingest_root)

    assert summary.total_files_scanned == 4
    assert summary.files_indexed >= 2  # md, code, text_pdf
    assert summary.files_skipped_unsupported >= 1  # image_pdf
    assert summary.files_failed == 0

    # Verify image-only PDF was flagged
    unsupported_reports = [r for r in summary.reports if r.status == "skipped_unsupported"]
    assert any("scanned_doc.pdf" in r.rel_path for r in unsupported_reports)
    assert any("Pre-extracted OCR is required" in (r.error_message or "") for r in unsupported_reports)

    # 3. Query through Stoma adapter and assert full provenance
    # Query 10 seconds into the future of ingestion so dt * c > dx_norm (inside past light cone)
    res = indexer.query("hyperbolic embeddings and Lorentzian causality", query_timestamp=time.time() + 10.0)
    assert len(res["results"]) > 0
    arch_items = [r for r in res["results"] if r.get("provenance", {}).get("rel_path") == "architecture.md"]
    assert len(arch_items) > 0, "architecture.md was not retrieved!"
    prov = arch_items[0]["provenance"]
    assert prov is not None
    assert prov["rel_path"] == "architecture.md"
    assert prov["file_content_hash"] == compute_file_sha256(md_file)
    assert prov["ingested_at"] > 0.0
    assert "start_line" in prov["location"]
    assert res["causal_fallback"] is False

    # 4. Modify one file and show replacement without duplication
    md_file.write_text(
        "# Updated Topological Architecture\n\n"
        "Completely refreshed text explaining sheaf contradiction filters.",
        encoding="utf-8",
    )
    rep_mod = indexer.ingest_file(md_file)
    assert rep_mod.status == "indexed"

    # Query for old content -> must NOT be returned!
    res_old = indexer.query("Poincare hyperbolic embeddings")
    assert not any("Poincare hyperbolic embeddings" in item["content"] for item in res_old["results"])

    # Query for new content -> MUST be returned
    res_new = indexer.query("sheaf contradiction filters")
    assert any("sheaf contradiction filters" in item["content"] for item in res_new["results"])

    # 5. Delete one file and show passages gone
    code_file.unlink()
    deleted = indexer.delete_file("quantum_memory.py")
    assert deleted is True

    res_code = indexer.query("QuantumMemoryStore readout")
    assert not any("QuantumMemoryStore" in item["content"] for item in res_code["results"])

    # 6. Restart indexer in fresh instance and repeat query
    indexer_restarted = StomaIndexer(config=cfg, embedder=shared_embedder)
    res_restarted = indexer_restarted.query("sheaf contradiction filters")
    assert len(res_restarted["results"]) > 0
    arch_restarted = [r for r in res_restarted["results"] if "architecture.md" in r["doc_id"]]
    assert len(arch_restarted) > 0, "architecture.md not found in restarted query results!"
    assert arch_restarted[0]["provenance"]["file_content_hash"] == compute_file_sha256(md_file)
