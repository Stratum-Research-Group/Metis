"""Dokime evaluation task for Pinax Tool Catalogue lookup correctness.
Implements Section 8: 50 'which tool does X' questions with known answers;
benchmarks top-1 and top-3 retrieval accuracy without synthetic shortcuts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dokime.tasks.protocol import Query


class PinaxLookup50Task:
    """Evaluation task testing tool discovery and semantic retrieval across 50 canonical tools."""

    task_id: str = "pinax_lookup_50"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "pinax_lookup_50.json"
        self._dataset_path = dataset_path

    @property
    def dataset_path(self) -> Path | None:
        return self._dataset_path

    def load(self) -> list[Query]:
        """Load the 50 tool lookup queries from disk."""
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Pinax lookup dataset not found at: {self._dataset_path}")

        with open(self._dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: list[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    expected=str(item.get("expected", "")).strip().lower(),
                    metadata={
                        "category": item.get("category", ""),
                        "expected": item.get("expected", ""),
                    },
                )
            )
        return queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        """Format a prompt asking which Unix/Linux tool accomplishes the task."""
        if template:
            return template.replace("{question}", query.prompt).replace("{prompt}", query.prompt)
        return (
            f"Identify the standard Unix/Linux tool or CLI command for the following description:\n"
            f"\"{query.prompt}\"\n\n"
            f"Answer with ONLY the lowercase command name (e.g. 'rg', 'curl', 'git')."
        )

    def extract_prediction(self, query: Query, response: str) -> str | None:
        """Extract tool name prediction from model response."""
        if not response or not response.strip():
            return None
        text = response.strip().lower()
        # Look for code block or backticks
        backtick_match = re.search(r"`([a-zA-Z0-9_\-\.\+]+)`", text)
        if backtick_match:
            return backtick_match.group(1).lower()

        # Split into tokens and look for word matching known tool syntax
        words = re.findall(r"\b[a-zA-Z0-9_\-\.\+]+\b", text)
        if words:
            # Common filler prefixes: 'the', 'use', 'command', 'tool', 'is'
            filtered = [w for w in words if w not in ("the", "tool", "command", "use", "is", "a", "an", "for")]
            if filtered:
                return filtered[0]
            return words[0]
        return text.split()[0]

    def score(self, query: Query, response: str) -> float:
        """Score model prediction against expected tool name."""
        pred = self.extract_prediction(query, response)
        if not pred:
            return 0.0
        expected = query.expected.lower()
        # Exact match or normalized tool ID match (e.g. tool:curl -> curl)
        pred_norm = pred.replace("tool:", "")
        return 1.0 if pred_norm == expected else 0.0

    def evaluate_catalogue(self, catalogue_lookup_fn: Any) -> Dict[str, Any]:
        """
        Directly evaluate catalogue retrieval engine on the 50 queries.
        Returns top-1 accuracy, top-3 accuracy, and per-item diagnostic details.
        """
        queries = self.load()
        top1_hits = 0
        top3_hits = 0
        details: List[Dict[str, Any]] = []

        for q in queries:
            expected = q.expected.lower()
            # Perform search in catalogue using prompt keywords
            candidates = catalogue_lookup_fn(query=q.prompt, limit=5)
            # Candidate names normalized
            cand_names = [
                c.get("name", "").lower() or c.get("id", "").replace("tool:", "").lower()
                for c in candidates
            ]

            hit_top1 = len(cand_names) > 0 and cand_names[0] == expected
            hit_top3 = expected in cand_names[:3]

            if hit_top1:
                top1_hits += 1
            if hit_top3:
                top3_hits += 1

            details.append({
                "id": q.id,
                "prompt": q.prompt,
                "expected": expected,
                "top_candidates": cand_names[:3],
                "top1_hit": hit_top1,
                "top3_hit": hit_top3,
            })

        total = len(queries)
        return {
            "total_queries": total,
            "top1_accuracy": round(top1_hits / total, 4) if total > 0 else 0.0,
            "top3_accuracy": round(top3_hits / total, 4) if total > 0 else 0.0,
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "details": details,
        }
