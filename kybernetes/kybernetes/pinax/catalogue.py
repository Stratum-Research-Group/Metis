"""Pinax Tool Catalogue coordinator for Kybernetes orchestrator.
Manages database, seed inventory, passive scanning, retrieval levels 0-2, and invocation risk evaluation.
"""

import gzip
from pathlib import Path
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

from kybernetes.pinax.db import PinaxDatabase
from kybernetes.pinax.models import (
    EntryKind,
    InstallState,
    ProbeResult,
    ProbeStatus,
    RiskTier,
    ToolCapability,
    ToolDoc,
    ToolKind,
    ToolRecord,
    ToolRiskOverride,
    ToolSignature,
)
from kybernetes.pinax.probe import PinaxProber
from kybernetes.pinax.scanner import PinaxScanner
from kybernetes.pinax.seed import CURATED_TOOLS


class PinaxCatalogue:
    """
    Central repository and query engine for all available tools in Metis.
    Provides multi-level retrieval without prompt drowning (Section 7).
    """

    def __init__(self, db_path: Optional[Path] = None, host_id: str = "localhost"):
        workspace = Path.cwd()
        resolved_db = db_path or (workspace / ".metis" / "pinax.db")
        self.db = PinaxDatabase(resolved_db)
        self.scanner = PinaxScanner(host_id=host_id)
        self.host_id = host_id
        # Automatically seed curated tools if database is empty
        if len(self.db.list_tools(limit=1)) == 0:
            self.seed_curated_tools()

    def seed_curated_tools(self) -> int:
        """Loads curated seed inventory into the database."""
        count = 0
        for item in CURATED_TOOLS:
            record = ToolRecord(
                id=item["id"],
                host_id=self.host_id,
                name=item["name"],
                kind=item.get("kind", ToolKind.BINARY),
                entry_kind=item.get("entry_kind", EntryKind.CANDIDATE),
                summary=item.get("summary", ""),
                category=item.get("category", "general"),
                risk_tier=item.get("risk_tier", RiskTier.T3),
                install_state=item.get("install_state", InstallState.UNKNOWN),
                discovered_by="curated",
                notes=item.get("notes"),
            )
            self.db.upsert_tool(record)
            count += 1

            for cap in item.get("capabilities", []):
                self.db.add_capability(ToolCapability(tool_id=record.id, capability=cap))

            for ovr in item.get("overrides", []):
                self.db.add_risk_override(ToolRiskOverride(
                    tool_id=record.id,
                    pattern_kind=ovr["pattern_kind"],
                    pattern=ovr["pattern"],
                    tier=ovr.get("tier", RiskTier.T3),
                    note=ovr.get("note", ""),
                ))

            if "signature" in item:
                sig_data = item["signature"]
                self.db.add_signature(ToolSignature(
                    tool_id=record.id,
                    call_name=sig_data["call_name"],
                    arg_schema_json=sig_data["arg_schema_json"],
                ))
        return count

    def scan_environment(self, path_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs passive PATH scan and updates database.
        Never executes any target binaries.
        """
        start_time = time.time()
        curated_tools = self.db.list_tools(limit=1000)
        curated_map = {t.name: t for t in curated_tools}

        discovered = self.scanner.scan_path_directories(
            path_dirs=path_dirs,
            curated_names=set(curated_map.keys()),
        )

        added = 0
        changed = 0
        tools_to_upsert: List[ToolRecord] = []
        docs_to_add: List[ToolDoc] = []

        for disc in discovered:
            if disc.name in curated_map:
                existing = curated_map[disc.name]
                # Promote curated candidate to installed with real binary attributes
                existing.invocation_path = disc.invocation_path
                existing.resolved_path = disc.resolved_path
                existing.sha256 = disc.sha256
                existing.install_state = InstallState.INSTALLED
                existing.entry_kind = EntryKind.INSTALLED
                existing.last_verified = time.time()
                tools_to_upsert.append(existing)
                changed += 1

                # Locate local man documentation passively
                doc = self.scanner.locate_doc_page(disc.name)
                if doc:
                    doc.tool_id = existing.id
                    docs_to_add.append(doc)
            else:
                # Unvetted newcomer on PATH
                tools_to_upsert.append(disc)
                added += 1

        self.db.batch_upsert_tools(tools_to_upsert)
        for d in docs_to_add:
            try:
                self.db.add_doc(d)
            except Exception:
                pass

        run_id = f"scan-{uuid.uuid4().hex[:8]}"
        finish_time = time.time()
        elapsed = round(finish_time - start_time, 4)
        summary = {
            "id": run_id,
            "host_id": self.host_id,
            "started_at": start_time,
            "finished_at": finish_time,
            "elapsed_seconds": elapsed,
            "scanner_version": "0.1.0",
            "added": added,
            "removed": 0,
            "changed": changed,
            "installed_updated": changed,
            "unvetted_candidates": added,
            "total_discovered": len(discovered),
        }
        self.db.record_scan_run(summary)
        return summary

    def lookup(
        self,
        query: str = "",
        category: Optional[str] = None,
        risk_tier: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Level 0 compact digest retrieval:
        Provides clean, size-bounded tool recommendations without prompt bloat.
        """
        if query:
            tools = self.db.search_tools(query, limit=limit)
            if category:
                tools = [t for t in tools if t.category == category]
            if risk_tier:
                tier_val = risk_tier.value if hasattr(risk_tier, "value") else str(risk_tier)
                tools = [t for t in tools if (t.risk_tier.value if hasattr(t.risk_tier, "value") else str(t.risk_tier)) == tier_val]
        else:
            tools = self.db.list_tools(category=category, risk_tier=risk_tier, limit=limit)

        results = []
        for t in tools:
            sigs = self.db.get_signatures(t.id)
            results.append({
                "id": t.id,
                "name": t.name,
                "summary": t.summary,
                "category": t.category,
                "risk_tier": t.risk_tier.value,
                "install_state": t.install_state.value,
                "kind": t.kind.value,
                "has_structured_signature": len(sigs) > 0,
            })
        return results

    def get_tool_detail(self, name_or_id: str) -> Optional[Dict[str, Any]]:
        """
        Level 1 tool signature & capability view:
        Returns structured schemas and capability attributes for a chosen tool.
        """
        tool = self.db.get_tool(name_or_id)
        if not tool:
            return None

        caps = self.db.get_capabilities(tool.id)
        overrides = self.db.get_risk_overrides(tool.id)
        sigs = self.db.get_signatures(tool.id)
        docs = self.db.get_docs(tool.id)

        data = tool.to_dict()
        data["capabilities"] = [c.to_dict() for c in caps]
        data["risk_overrides"] = [o.to_dict() for o in overrides]
        data["signatures"] = [s.to_dict() for s in sigs]
        data["docs"] = [d.to_dict() for d in docs]
        return data

    def get_tool_docs(self, name_or_id: str, max_bytes: int = 4096) -> Dict[str, Any]:
        """
        Level 2 doc slice:
        Reads local man page or help excerpt, size-capped and labeled as UNTRUSTED DATA.
        """
        tool = self.db.get_tool(name_or_id)
        if not tool:
            return {
                "tool": name_or_id,
                "found": False,
                "content": "",
                "untrusted_data": True,
                "max_bytes": max_bytes,
                "error": f"Tool '{name_or_id}' not found in catalogue",
            }

        docs = self.db.get_docs(tool.id)
        if not docs:
            return {
                "tool": tool.name,
                "found": False,
                "content": f"# {tool.name}\n\nSummary: {tool.summary}\nCategory: {tool.category}\nRisk Tier: {tool.risk_tier.value}\n(No local man pages recorded).",
                "untrusted_data": True,
                "truncated": False,
                "max_bytes": max_bytes,
            }

        doc = docs[0]
        content = ""
        truncated = False
        locator_path = Path(doc.locator)

        if locator_path.is_file():
            try:
                raw_bytes = b""
                if doc.locator.endswith(".gz"):
                    with gzip.open(locator_path, "rb") as gz:
                        raw_bytes = gz.read(max_bytes + 1)
                else:
                    with open(locator_path, "rb") as f:
                        raw_bytes = f.read(max_bytes + 1)

                if len(raw_bytes) > max_bytes:
                    raw_bytes = raw_bytes[:max_bytes]
                    truncated = True

                content = raw_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                content = f"Error reading document locator {doc.locator}: {e}"
        else:
            content = f"Document locator '{doc.locator}' is not locally present."

        return {
            "tool": tool.name,
            "found": True,
            "doc_kind": doc.doc_kind,
            "locator": doc.locator,
            "content": content,
            "truncated": truncated,
            "max_bytes": max_bytes,
            "untrusted_data": True,  # Critical safety label (Section 6)
        }

    def evaluate_invocation(self, tool_name: str, args: List[str]) -> Dict[str, Any]:
        """
        Evaluates risk tier for an invocation (Section 5).
        Matches parsed arguments against curated overrides (not substring matching).
        """
        tool = self.db.get_tool(tool_name)
        if not tool:
            return {
                "tool": tool_name,
                "tier": RiskTier.T3.value,
                "vetted": False,
                "reason": "Uncatalogued tool: default-denied (Section 4 A5)",
                "requires_approval": True,
            }

        base_tier = tool.risk_tier
        effective_tier = base_tier
        override_reason = "Base catalogue tier"

        overrides = self.db.get_risk_overrides(tool.id)
        for ovr in overrides:
            matched = False
            if ovr.pattern_kind == "subcommand":
                if args and args[0] == ovr.pattern:
                    matched = True
            elif ovr.pattern_kind == "flag":
                if ovr.pattern in args:
                    matched = True
            elif ovr.pattern_kind == "regex":
                full_cmd = " ".join(args)
                if re.search(ovr.pattern, full_cmd):
                    matched = True

            if matched:
                # Elevate if override is higher tier
                effective_tier = ovr.tier
                override_reason = f"Override '{ovr.pattern}': {ovr.note}"
                break

        requires_approval = effective_tier in (RiskTier.T1, RiskTier.T2, RiskTier.T3)
        return {
            "tool": tool.name,
            "tool_id": tool.id,
            "tier": effective_tier.value,
            "base_tier": base_tier.value,
            "vetted": tool.discovered_by != "unvetted",
            "reason": override_reason,
            "requires_approval": requires_approval,
        }

    def verify_tool_before_dispatch(self, name_or_id: str) -> Tuple[bool, str]:
        """
        Performs use-time cryptographic re-check before dispatch (Section 2).
        """
        tool = self.db.get_tool(name_or_id)
        if not tool:
            return False, f"Tool '{name_or_id}' not found in catalogue"

        if tool.kind == ToolKind.STRUCTURED:
            # Structured tools run in-process within Kybernetes
            return True, "Structured tool verified"

        if not tool.invocation_path:
            return False, f"Tool '{tool.name}' has no invocation path recorded"

        return self.scanner.verify_executable_integrity(
            invocation_path=tool.invocation_path,
            expected_sha256=tool.sha256,
        )

    def probe_tool(
        self,
        name_or_id: str,
        args: Optional[List[str]] = None,
        timeout_seconds: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Executes a controlled, side-effect-free health probe on a tool and updates DB state.
        Detects broken binaries, missing tools, and stale venv shims.
        """
        tool = self.db.get_tool(name_or_id)
        if not tool:
            return {
                "tool": name_or_id,
                "status": ProbeStatus.MISSING.value,
                "detail": f"Tool '{name_or_id}' not found in catalogue",
                "install_state": InstallState.NOT_INSTALLED.value,
            }

        # For structured tools (in-process Python classes), probe is trivial
        if tool.kind == ToolKind.STRUCTURED:
            return {
                "tool": tool.name,
                "tool_id": tool.id,
                "status": ProbeStatus.OK.value,
                "detail": "In-process structured tool verified",
                "install_state": InstallState.INSTALLED.value,
                "duration_ms": 0.0,
            }

        res = PinaxProber.probe_command(
            tool_id=tool.id,
            invocation_path=tool.invocation_path,
            args=args,
            timeout_seconds=timeout_seconds,
        )

        # Derive updated install state
        new_state = tool.install_state
        if res.status == ProbeStatus.OK:
            new_state = InstallState.INSTALLED
        elif res.status in (ProbeStatus.BROKEN,):
            new_state = InstallState.BROKEN
        elif res.status == ProbeStatus.MISSING:
            new_state = InstallState.NOT_INSTALLED

        self.db.update_probe_result(tool.id, res, new_install_state=new_state)

        return {
            "tool": tool.name,
            "tool_id": tool.id,
            "status": res.status.value,
            "exit_code": res.exit_code,
            "detail": res.detail,
            "is_stale_shim": res.is_stale_shim,
            "duration_ms": res.duration_ms,
            "install_state": new_state.value,
        }
