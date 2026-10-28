"""Dokime evaluation task for verifiable instruction-following (IFEval).

Implements programmatic instruction checking without an LLM judge, inspired by
Google Research IFEval (arXiv 2311.07911). Evaluates strict and loose adherence
across counts, formatting, keywords, headers, and structural constraints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dokime.tasks.ifeval_checkers import verify_instruction
from dokime.tasks.protocol import Query


class IFEvalTask:
    """Verifiable instruction following task evaluating programmatic constraints."""

    task_id: str = "ifeval_instruction_following"

    def __init__(self, dataset_path: Optional[Path] = None, loose_mode: bool = False) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "ifeval_suite.json"
        self._dataset_path = dataset_path
        self.loose_mode = loose_mode

    @property
    def dataset_path(self) -> Optional[Path]:
        return self._dataset_path

    def load(self) -> List[Query]:
        """Load the instruction-following queries from disk."""
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"IFEval dataset not found at: {self._dataset_path}")

        with open(self._dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: List[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    expected="all_rules_satisfied",
                    metadata={
                        "rules": item.get("rules", []),
                        "category": item.get("category", "general"),
                    },
                )
            )
        return queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        """Format the instruction prompt."""
        if template:
            return template.replace("{question}", query.prompt).replace("{prompt}", query.prompt)
        return f"{query.prompt}\n\nPlease follow all instructions precisely."

    def extract_prediction(self, query: Query, response: str) -> Optional[str]:
        """Extract response text."""
        return response.strip() if response else None

    def evaluate_rules(self, query: Query, response: str, loose: bool | None = None) -> List[Dict[str, Any]]:
        """Evaluate each rule individually, returning per-rule status and diagnostics."""
        use_loose = self.loose_mode if loose is None else loose
        rules = query.metadata.get("rules", [])
        results: List[Dict[str, Any]] = []

        for rule in rules:
            rtype = rule.get("rule_type", "")
            kwargs = rule.get("kwargs", {})
            passed, message = verify_instruction(rtype, response, kwargs, loose=use_loose)
            results.append({
                "rule_type": rtype,
                "passed": passed,
                "message": message,
                "kwargs": kwargs,
            })
        return results

    def score(self, query: Query, response: str) -> float:
        """Score response. Returns 1.0 if all rules pass, 0.0 otherwise."""
        if not response or not response.strip():
            return 0.0

        eval_results = self.evaluate_rules(query, response)
        if not eval_results:
            return 1.0

        all_passed = all(r["passed"] for r in eval_results)
        return 1.0 if all_passed else 0.0

    def score_detailed(self, query: Query, response: str) -> Dict[str, Any]:
        """Returns comprehensive evaluation metrics including strict and loose passes."""
        strict_results = self.evaluate_rules(query, response, loose=False)
        loose_results = self.evaluate_rules(query, response, loose=True)

        strict_pass = all(r["passed"] for r in strict_results) if strict_results else True
        loose_pass = all(r["passed"] for r in loose_results) if loose_results else True

        strict_rule_acc = (
            sum(1 for r in strict_results if r["passed"]) / len(strict_results)
            if strict_results else 1.0
        )
        loose_rule_acc = (
            sum(1 for r in loose_results if r["passed"]) / len(loose_results)
            if loose_results else 1.0
        )

        return {
            "query_id": query.id,
            "strict_prompt_pass": strict_pass,
            "loose_prompt_pass": loose_pass,
            "strict_rule_accuracy": strict_rule_acc,
            "loose_rule_accuracy": loose_rule_acc,
            "strict_details": strict_results,
            "loose_details": loose_results,
        }
