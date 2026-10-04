"""Dataset card generator conforming to Metis reproducibility contracts."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json


class DatasetCard:
    """Standardized dataset documentation card."""
    def __init__(
        self,
        dataset_name: str,
        task: str,
        license_name: str,
        source_revisions: Dict[str, str],
        audit_summary: Dict[str, Any],
        split_summary: Dict[str, Any],
        privacy_label: str = "internal",
        known_biases: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.task = task
        self.license_name = license_name
        self.source_revisions = source_revisions
        self.audit_summary = audit_summary
        self.split_summary = split_summary
        self.privacy_label = privacy_label
        self.known_biases = known_biases or "None explicitly declared."
        self.created_at_utc = datetime.now(timezone.utc).isoformat()

    def to_markdown(self) -> str:
        lines = [
            f"# Dataset Card: {self.dataset_name}",
            "",
            f"- **Task**: {self.task}",
            f"- **Created At (UTC)**: {self.created_at_utc}",
            f"- **License**: {self.license_name}",
            f"- **Privacy Classification**: `{self.privacy_label}`",
            "",
            "## Source Provenance & Revisions",
        ]
        for src, rev in self.source_revisions.items():
            lines.append(f"- **{src}**: `{rev}`")

        lines.extend([
            "",
            "## Ingestion & Audit Summary",
            f"- Total Input Records: {self.audit_summary.get('total_records', self.audit_summary.get('total_lines', 0))}",
            f"- Valid Records Ingested: {self.audit_summary.get('valid_records', 0)}",
            f"- Duplicates Removed: {self.audit_summary.get('duplicates_removed', 0)}",
            f"- Schema Violations: {self.audit_summary.get('schema_violations', 0)}",
        ])

        if "missing_counterfactuals" in self.audit_summary:
            lines.append(f"- Missing Counterfactuals Excluded: {self.audit_summary['missing_counterfactuals']}")
        if "ties_excluded" in self.audit_summary:
            lines.append(f"- Ambiguous Ties Excluded: {self.audit_summary['ties_excluded']}")
        if "unresolved_conflicts_excluded" in self.audit_summary:
            lines.append(f"- Unresolved Conflicts Excluded: {self.audit_summary['unresolved_conflicts_excluded']}")

        lines.extend([
            "",
            "## Split Checksums & Allocations",
            f"- **Seed**: {self.split_summary.get('seed', 42)}",
            f"- **Train Items**: {self.split_summary.get('train_count', 0)} (SHA-256: `{self.split_summary.get('train_checksum', '')[:16]}...`)",
            f"- **Dev Items**: {self.split_summary.get('dev_count', 0)} (SHA-256: `{self.split_summary.get('dev_checksum', '')[:16]}...`)",
            f"- **Final Test Items**: {self.split_summary.get('test_count', 0)} (SHA-256: `{self.split_summary.get('test_checksum', '')[:16]}...`)",
            "",
            "## Bias & Limitations",
            self.known_biases,
            "",
        ])
        return "\n".join(lines)

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / "DATASET_CARD.md"
        json_path = output_dir / "dataset_card.json"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.to_markdown())
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": self.dataset_name,
                "task": self.task,
                "created_at_utc": self.created_at_utc,
                "license": self.license_name,
                "source_revisions": self.source_revisions,
                "audit": self.audit_summary,
                "splits": self.split_summary,
                "privacy_label": self.privacy_label,
            }, f, indent=2)
        return md_path
