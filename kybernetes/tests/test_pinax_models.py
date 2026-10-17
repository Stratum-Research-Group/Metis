"""Tests for Pinax Tool Catalogue data models."""

import pytest
from kybernetes.pinax.models import (
    RiskTier,
    ToolKind,
    EntryKind,
    InstallState,
    ToolRecord,
    ToolDoc,
    ToolCapability,
    ToolRiskOverride,
    ToolSignature,
)


def test_tool_record_defaults_and_dict():
    record = ToolRecord(
        id="ripgrep",
        name="rg",
        kind=ToolKind.BINARY,
        summary="Recursively searches directories for a regex pattern",
        category="search",
        invocation_path="/usr/bin/rg",
        resolved_path="/usr/bin/rg",
        risk_tier=RiskTier.T0,
    )
    d = record.to_dict()
    assert d["id"] == "ripgrep"
    assert d["name"] == "rg"
    assert d["kind"] == "binary"
    assert d["risk_tier"] == "T0"
    assert d["install_state"] == "unknown"
    assert d["host_id"] == "localhost"


def test_tool_doc_serialization():
    doc = ToolDoc(
        tool_id="ripgrep",
        doc_kind="man",
        locator="/usr/share/man/man1/rg.1.gz",
        size_bytes=14200,
    )
    d = doc.to_dict()
    assert d["tool_id"] == "ripgrep"
    assert d["doc_kind"] == "man"
    assert d["locator"] == "/usr/share/man/man1/rg.1.gz"
    assert d["size_bytes"] == 14200


def test_tool_capability_and_risk_override():
    cap = ToolCapability(tool_id="git", capability="network-write", detail="git push")
    assert cap.to_dict()["capability"] == "network-write"

    override = ToolRiskOverride(
        tool_id="git",
        pattern_kind="subcommand",
        pattern="push",
        tier=RiskTier.T3,
        note="Push modifies remote repositories",
    )
    od = override.to_dict()
    assert od["pattern"] == "push"
    assert od["tier"] == "T3"


def test_tool_signature():
    sig = ToolSignature(
        tool_id="read_file_range",
        call_name="read_file_range",
        arg_schema_json='{"type": "object", "properties": {"path": {"type": "string"}}}',
    )
    sd = sig.to_dict()
    assert sd["call_name"] == "read_file_range"
    assert "properties" in sd["arg_schema_json"]
