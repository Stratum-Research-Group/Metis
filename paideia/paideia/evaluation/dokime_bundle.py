"""Dokime-compatible evaluation run bundle generator."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from paideia.evaluation.promotion import PromotionVerdict


class DokimeRunBundle(BaseModel):
    """
    Standardized, replayable evaluation bundle consumable by Dokime.
    Allows Dokime to independently recompute the verdict from item-level logs.
    """
    schema_version: str = Field(default="1.0.0", description="Dokime contract schema version")
    run_id: str = Field(..., description="Unique evaluation run identifier")
    task_name: str = Field(..., description="Evaluated task name")
    candidate_name: str = Field(..., description="Candidate policy name")
    baseline_name: str = Field(..., description="Baseline reference name")
    item_ids: List[str] = Field(..., description="Immutable list of test item IDs in order")
    gold_labels: List[str] = Field(..., description="Gold labels for each item")
    baseline_predictions: List[str] = Field(..., description="Baseline predictions")
    candidate_predictions: List[str] = Field(..., description="Candidate predictions")
    baseline_scores: List[float] = Field(..., description="Item-level binary or continuous scores for baseline")
    candidate_scores: List[float] = Field(..., description="Item-level scores for candidate")
    candidate_latencies_ms: List[float] = Field(default_factory=list, description="Per-item candidate latencies in ms")
    verdict: Dict[str, Any] = Field(..., description="Summary verdict and test outcomes")
    bundle_hash: Optional[str] = Field(default=None, description="SHA-256 integrity hash of bundle content")

    def compute_hash(self) -> str:
        """Compute SHA-256 over canonical JSON excluding bundle_hash."""
        data = self.model_dump(exclude={"bundle_hash"})
        encoded = json.dumps(data, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def save(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.bundle_hash = self.compute_hash()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return output_path
