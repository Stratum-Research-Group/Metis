"""AIME and Olympiad competition mathematics evaluation task with exact answer verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dokime.tasks.protocol import Query


class AIMECompetitionMathTask:
    """AIME and high-school Olympiad competition mathematics benchmark task (Set 1)."""

    task_id: str = "aime_math"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "aime_math.json"
        self._dataset_path = dataset_path

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    def load(self) -> list[Query]:
        """Load mathematical competition queries from JSON dataset."""
        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Task dataset not found at: {self._dataset_path}")

        with open(self._dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        queries: list[Query] = []
        for item in data:
            queries.append(
                Query(
                    id=str(item["id"]),
                    prompt=str(item["prompt"]),
                    expected=str(item.get("expected", "")).strip(),
                    metadata=item.get("metadata", {}),
                )
            )
        return queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        """Format prompt instructing the model to provide derivations and boxed final answer."""
        if template:
            formatted = template
            if "{question}" in template:
                formatted = formatted.replace("{question}", query.prompt)
            if "{prompt}" in template:
                formatted = formatted.replace("{prompt}", query.prompt)
            if "{id}" in template:
                formatted = formatted.replace("{id}", query.id)
            if formatted == template and not any(k in template for k in ("{question}", "{prompt}")):
                return (
                    f"{template.strip()}\n\n"
                    f"Problem: {query.prompt}\n"
                    f"Show your work and conclude with your final integer answer in \\boxed{{...}}."
                )
            return formatted

        return (
            f"Problem: {query.prompt}\n\n"
            f"Solve this mathematics problem step-by-step without repeating sentences. "
            f"Verify each calculation, then state your final answer as an exact integer inside \\boxed{{...}}."
        )

    def _strip_thought_traces(self, response: str) -> str:
        """Strip internal reasoning and channel tokens."""
        cleaned = re.sub(r"<\|channel\>thought.*?<channel\|\>", "", response, flags=re.DOTALL)
        cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL)
        return cleaned.strip() or response.strip()

    def extract_prediction(self, query: Query, response: str) -> str | None:
        """Extract exact numerical answer from response (supporting \\boxed{...}, final answer, etc.)."""
        if not response or not response.strip():
            return None

        clean = self._strip_thought_traces(response)

        # 1. Look for LaTeX \boxed{...}
        boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", clean)
        if boxed_matches:
            val = boxed_matches[-1].strip()
            val = re.sub(r"\\text\{([^}]+)\}", r"\1", val)
            val = re.sub(r"[\$,\s]", "", val)
            num_m = re.search(r"-?\d+(?:\.\d+)?", val)
            if num_m:
                return num_m.group(0)
            return val

        # 2. Look for explicit final answer declaration: "Final Answer: 18"
        pat_final = re.compile(
            r"(?:(?:the|my)\s+)?final\s+answer\s*(?:is\s*:?|:)\s*[\$\*]*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        final_matches = list(pat_final.finditer(clean))
        if final_matches:
            return final_matches[-1].group(1).strip()

        # 3. Look for "the answer is X"
        pat_answer = re.compile(
            r"(?:the\s+answer\s+is\s*:?\s*[\$\*]*)(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        ans_matches = list(pat_answer.finditer(clean))
        if ans_matches:
            return ans_matches[-1].group(1).strip()

        # 4. Fallback: inspect the final lines for trailing integer
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if lines:
            for line in reversed(lines[-3:]):
                nums = re.findall(r"\b(-?\d+)\b", line)
                if nums:
                    return nums[-1]

        return None

    def score(self, query: Query, response: str) -> float:
        """Deterministic exact match scoring for numerical mathematics."""
        pred = self.extract_prediction(query, response)
        if pred is None:
            return 0.0

        try:
            if abs(float(pred) - float(query.expected)) < 1e-6:
                return 1.0
        except ValueError:
            pass

        return 1.0 if str(pred).strip() == str(query.expected).strip() else 0.0


class AIMECompetitionMathV2Task(AIMECompetitionMathTask):
    """AIME and high-school Olympiad competition mathematics benchmark task (Set 2 - Fresh Unseen)."""

    task_id: str = "aime_math_v2"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "aime_math_v2.json"
        super().__init__(dataset_path=dataset_path)
