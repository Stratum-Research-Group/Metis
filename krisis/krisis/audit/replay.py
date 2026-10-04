"""Fixture replay engine verifying decisions against normative JSON fixtures."""

from pathlib import Path
from typing import Any
import json

from krisis.contracts.decision import KrisisDecision
from krisis.engine.selector import KrisisSelector


class FixtureReplayRunner:
    """Replays normative fixtures from krisis-fixtures-0.1.json against KrisisSelector."""

    def __init__(
        self,
        policy_path: Path | str = "krisis-policy-0.1.json",
        fixtures_path: Path | str = "krisis-fixtures-0.1.json",
    ):
        self.policy_path = Path(policy_path)
        self.fixtures_path = Path(fixtures_path)
        self.selector = KrisisSelector(self.policy_path)

        data = json.loads(self.fixtures_path.read_text(encoding="utf-8"))
        self.fixtures = data.get("fixtures", [])

    def run_all(self) -> dict[str, Any]:
        """Replay all fixtures and return comprehensive verification report."""
        results: list[dict[str, Any]] = []
        all_passed = True

        for item in self.fixtures:
            f_id = item["id"]
            f_name = item.get("name", "")

            # Check if fixture contains sub-cases (e.g. Fixture I)
            if "cases" in item:
                for idx, c in enumerate(item["cases"], start=1):
                    case_id = f"{f_id}.{idx}"
                    passed, diffs, decision = self._evaluate_case(c["request"], c["expected"])
                    if not passed:
                        all_passed = False
                    results.append(
                        {
                            "id": case_id,
                            "name": f"{f_name} (case {idx})",
                            "passed": passed,
                            "diffs": diffs,
                            "decision": decision.model_dump(),
                        }
                    )
            else:
                passed, diffs, decision = self._evaluate_case(item["request"], item["expected"])
                if not passed:
                    all_passed = False
                results.append(
                    {
                        "id": f_id,
                        "name": f_name,
                        "passed": passed,
                        "diffs": diffs,
                        "decision": decision.model_dump(),
                    }
                )

        return {
            "all_passed": all_passed,
            "total": len(results),
            "passed_count": sum(1 for r in results if r["passed"]),
            "failed_count": sum(1 for r in results if not r["passed"]),
            "results": results,
        }

    def _evaluate_case(
        self,
        request: dict[str, Any],
        expected: dict[str, Any],
    ) -> tuple[bool, list[str], KrisisDecision]:
        diffs: list[str] = []
        decision = self.selector.evaluate(request)

        # Check status
        if "status" in expected and decision.status != expected["status"]:
            diffs.append(f"Status mismatch: expected '{expected['status']}', got '{decision.status}'")

        # Check selected_id
        if "selected_id" in expected and decision.selected_id != expected["selected_id"]:
            diffs.append(f"selected_id mismatch: expected '{expected['selected_id']}', got '{decision.selected_id}'")

        # Check reason
        if "reason" in expected and decision.reason != expected["reason"]:
            diffs.append(f"reason mismatch: expected '{expected['reason']}', got '{decision.reason}'")

        # Check profile
        if "profile" in expected and decision.profile != expected["profile"]:
            diffs.append(f"profile mismatch: expected '{expected['profile']}', got '{decision.profile}'")

        # Check scores
        if "scores" in expected:
            for k, expected_score in expected["scores"].items():
                actual_score = decision.scores.get(k)
                if actual_score != expected_score:
                    diffs.append(f"score mismatch for {k}: expected '{expected_score}', got '{actual_score}'")

        # Check label
        if "label" in expected and decision.label != expected["label"]:
            diffs.append(f"label mismatch: expected '{expected['label']}', got '{decision.label}'")

        # Check warning
        if "warning" in expected and decision.warning != expected["warning"]:
            diffs.append(f"warning mismatch: expected '{expected['warning']}', got '{decision.warning}'")

        # Check trust_flags
        if "trust_flags" in expected:
            for k, expected_flag in expected["trust_flags"].items():
                actual_flag = decision.trust_flags.get(k)
                if actual_flag != expected_flag:
                    diffs.append(f"trust_flag mismatch for {k}: expected '{expected_flag}', got '{actual_flag}'")

        # Check dedup merged & kept_separate
        if "merged" in expected:
            if len(decision.merged) != len(expected["merged"]):
                diffs.append(f"merged count mismatch: expected {len(expected['merged'])}, got {len(decision.merged)}")

        # Check normalized_question_class
        if "normalized_question_class" in expected:
            if decision.normalized_question_class != expected["normalized_question_class"]:
                diffs.append(
                    f"normalized_question_class mismatch: expected '{expected['normalized_question_class']}', "
                    f"got '{decision.normalized_question_class}'"
                )

        return len(diffs) == 0, diffs, decision
