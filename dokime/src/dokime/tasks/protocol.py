"""Task protocol and Query data structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Query:
    """A single evaluation query."""

    id: str
    prompt: str
    choices: dict[str, str] = field(default_factory=dict)
    expected: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Task(Protocol):
    """Protocol for Dokime task plugins.

    Only task_id, load(), and score() are strictly required.
    format_prompt, extract_prediction, and dataset_path are optional.
    """

    task_id: str

    def load(self) -> list[Query]:
        """Load and return queries for this task."""
        ...

    def score(self, query: Query, response: str) -> float:
        """Score a model response against the expected output. Returns a float (e.g. 1.0 or 0.0)."""
        ...


def default_format_prompt(query: Query, template: str | None = None) -> str:
    """Default fallback prompt formatter when a task plugin does not provide format_prompt."""
    choice_lines = [f"{k}: {v}" for k, v in sorted(query.choices.items())] if query.choices else []
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
            if choices_text:
                return f"{template.strip()}\n\nQuestion: {query.prompt}\n{choices_text}\nAnswer:"
            return f"{template.strip()}\n\nQuestion: {query.prompt}\nAnswer:"
        return formatted

    if choices_text:
        return (
            f"Question: {query.prompt}\n"
            f"{choices_text}\n"
            f"Answer with only the correct choice letter (e.g. A, B, C, or D).\nAnswer:"
        )
    return query.prompt


def default_extract_prediction(query: Query, response: str) -> str | None:
    """Default fallback prediction extractor when a task plugin does not provide extract_prediction."""
    if not response or not response.strip():
        return None
    return response.strip()


def format_task_prompt(task: Any, query: Query, template: str | None = None) -> str:
    """Safely format prompt using task.format_prompt if present, else default_format_prompt."""
    if hasattr(task, "format_prompt") and callable(getattr(task, "format_prompt")):
        return task.format_prompt(query, template=template)
    return default_format_prompt(query, template=template)


def extract_task_prediction(task: Any, query: Query, response: str) -> str | None:
    """Safely extract prediction using task.extract_prediction if present, else default_extract_prediction."""
    if hasattr(task, "extract_prediction") and callable(getattr(task, "extract_prediction")):
        return task.extract_prediction(query, response)
    return default_extract_prediction(query, response)


def materialize_dataset(queries: list[Query], output_path: Path) -> Path:
    """Deterministically serialize loaded queries to JSON artifact when task has no source file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        {
            "id": q.id,
            "prompt": q.prompt,
            "choices": {k: q.choices[k] for k in sorted(q.choices.keys())},
            "expected": q.expected,
            "metadata": q.metadata,
        }
        for q in queries
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, sort_keys=True)
    return output_path
