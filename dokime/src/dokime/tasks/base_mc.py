"""Base class for robust Multiple-Choice Evaluation Tasks with thought-token stripping."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dokime.tasks.protocol import Query


class MultipleChoiceTask:
    """Base class for multiple choice benchmark tasks with thought-token awareness."""

    task_id: str = "base_mc"

    def __init__(self, dataset_path: Path) -> None:
        self._dataset_path = dataset_path

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    def load(self) -> list[Query]:
        """Load multiple-choice queries from JSON dataset."""
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
                    choices=dict(item.get("choices", {})),
                    expected=str(item.get("expected", "")).strip().upper(),
                    metadata=item.get("metadata", {}),
                )
            )
        return queries

    def format_prompt(self, query: Query, template: str | None = None) -> str:
        """Format the multiple-choice prompt with clear deliberation instructions."""
        choice_lines = [f"{key}: {val}" for key, val in sorted(query.choices.items())]
        choices_text = "\n".join(choice_lines)

        if template:
            formatted = template
            if "{choices}" in template:
                formatted = formatted.replace("{choices}", choices_text)
            if "{question}" in template:
                formatted = formatted.replace("{question}", query.prompt)
            if "{prompt}" in template:
                formatted = formatted.replace("{prompt}", query.prompt)
            if "{id}" in template:
                formatted = formatted.replace("{id}", query.id)
            if formatted == template and not any(k in template for k in ("{choices}", "{question}", "{prompt}")):
                return (
                    f"{template.strip()}\n\n"
                    f"Question: {query.prompt}\n"
                    f"{choices_text}\n"
                    f"Conclude your reasoning with 'Final Answer: [Letter]'."
                )
            return formatted

        return (
            f"Question: {query.prompt}\n\n"
            f"Choices:\n"
            f"{choices_text}\n\n"
            f"Analyze the choices step-by-step without repeating statements. State your final answer in the format 'Final Answer: [Letter]'."
        )

    def _strip_thought_traces(self, response: str) -> str:
        """Strip internal reasoning and channel tokens from modern models."""
        cleaned = re.sub(r"<\|channel\>thought.*?<channel\|\>", "", response, flags=re.DOTALL)
        cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL)
        return cleaned.strip() or response.strip()

    def _extract_from_clause(self, clause: str, choices: dict[str, str]) -> str | None:
        """Extract choice letter or unambiguous choice text from a clause."""
        cleaned_clause = clause.strip()
        m_let = re.match(
            r"^[\s\(\[\{\*\_\"\'\`]*([A-Za-z])[\s\)\]\}\*\_\.\:\;\"\'\`]*(.*)$",
            cleaned_clause,
        )
        if m_let:
            cand = m_let.group(1).upper()
            rest = m_let.group(2)
            if cand in choices:
                if cand == "A" and re.match(r"^\s+[a-zA-Z]", rest):
                    # Indefinite article 'a ...'
                    named_cands = []
                    for k, text in choices.items():
                        raw = text.strip()
                        if raw and re.search(rf"(?<!\w){re.escape(raw)}(?!\w)", cleaned_clause, re.IGNORECASE):
                            named_cands.append(k.upper())
                    if len(named_cands) == 1:
                        return named_cands[0]
                    return None
                return cand

        # Named choice text search
        named_cands = []
        for k, text in choices.items():
            raw = text.strip()
            if raw and re.search(rf"(?<!\w){re.escape(raw)}(?!\w)", cleaned_clause, re.IGNORECASE):
                named_cands.append(k.upper())
        if len(named_cands) == 1:
            return named_cands[0]

        return None

    def extract_prediction(self, query: Query, response: str) -> str | None:
        """Extract choice letter from model response, handling thought traces."""
        if not response or not response.strip():
            return None

        # Clean out thought traces
        clean = self._strip_thought_traces(response)

        # 1. Bare letter or bracketed letter: "(A)", "A", "[B]", "**C**"
        bare_match = re.fullmatch(
            r"^[\s\(\[\{\*\_\"\'\`]*([A-Za-z])[\s\)\]\}\*\_\.\:\;\"\'\`]*$",
            clean,
        )
        if bare_match:
            letter = bare_match.group(1).upper()
            if letter in query.choices:
                return letter

        # 2. Look for \boxed{Letter} or \boxed{(Letter)}
        boxed_matches = re.findall(r"\\boxed\{[\s\(\[\*]*([A-Za-z])[\s\)\]\*]*\}", clean)
        if boxed_matches:
            cand = boxed_matches[-1].upper()
            if cand in query.choices:
                return cand

        # 3. Explicit final answer phrases (search from end)
        pat_final = re.compile(
            r"(?:(?:the|my)\s+)?final\s+answer\s*(?:is\s*:?|:)\s*",
            re.IGNORECASE,
        )
        final_matches = list(pat_final.finditer(clean))
        if final_matches:
            for m in reversed(final_matches):
                tail = clean[m.end():]
                cand = self._extract_from_clause(tail, query.choices)
                if cand:
                    return cand

        # 4. Answer is ...
        pat_ans = re.compile(
            r"(?:(?:the\s+)?correct\s+(?:choice|option|answer)\s+is\s*:?|answer\s*:)\s*",
            re.IGNORECASE,
        )
        ans_matches = list(pat_ans.finditer(clean))
        if ans_matches:
            for m in reversed(ans_matches):
                tail = clean[m.end():]
                cand = self._extract_from_clause(tail, query.choices)
                if cand:
                    return cand

        # 5. Look in the final line for any isolated choice letter
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        if lines:
            last_line = lines[-1]
            end_match = re.search(r"\b([A-D])\b", last_line)
            if end_match:
                cand = end_match.group(1).upper()
                if cand in query.choices:
                    return cand

        return None

    def score(self, query: Query, response: str) -> float:
        """Deterministic exact match scoring (1.0 for match, 0.0 otherwise)."""
        pred = self.extract_prediction(query, response)
        if pred is None:
            return 0.0
        return 1.0 if pred.upper() == query.expected.upper() else 0.0
