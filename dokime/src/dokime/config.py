"""Run configuration parsing and validation using TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


def validate_unique_ids(items: Iterable[str] | None, field_name: str) -> None:
    """Validate that an iterable of IDs contains no duplicate entries.

    Raises:
        ValueError: If duplicate IDs are detected, naming the offending field and ID.
    """
    if items is None:
        return
    seen: set[str] = set()
    for item in items:
        norm = item.strip() if isinstance(item, str) else str(item)
        if norm in seen:
            raise ValueError(f"Duplicate {field_name} ID detected: '{item}'")
        seen.add(norm)


@dataclass
class RunConfig:
    """Configuration options for a Dokime evaluation run."""

    models: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=lambda: ["arc_easy_25"])
    prompt_template: str | None = None
    prompt_template_path: str | None = None
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int | None = None
    url: str | None = None
    output_dir: str = "runs"
    parent_run_id: str | None = None
    branch_point: str | None = None
    labeling_criterion: str | None = None  # e.g. "reference_faithfulness" vs "factual_correctness" (arXiv:2610.08026)

    def __post_init__(self) -> None:
        validate_unique_ids(self.models, "model")
        validate_unique_ids(self.tasks, "task")

    @classmethod
    def from_toml_file(cls, path: str | Path) -> RunConfig:
        """Load RunConfig from a TOML configuration file, resolving relative paths against the config's directory."""
        p = Path(path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Configuration file not found: {p}")

        with open(p, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data, config_dir=p.parent)

    @classmethod
    def from_dict(cls, data: dict[str, Any], config_dir: Path | None = None) -> RunConfig:
        """Create RunConfig from a raw dictionary."""
        # Handle single 'model' vs 'models'
        models = data.get("models")
        if models is not None and "model" in data:
            m_list = [models] if isinstance(models, str) else list(models)
            combined = [data["model"]] + m_list
            validate_unique_ids(combined, "model")
            models = combined
        elif models is None and "model" in data:
            models = [data["model"]]
        elif models is None:
            models = []
        elif isinstance(models, str):
            models = [models]

        # Handle single 'task' vs 'tasks'
        tasks = data.get("tasks")
        if tasks is not None and "task" in data:
            t_list = [tasks] if isinstance(tasks, str) else list(tasks)
            combined = [data["task"]] + t_list
            validate_unique_ids(combined, "task")
            tasks = combined
        elif tasks is None and "task" in data:
            tasks = [data["task"]]
        elif tasks is None:
            tasks = ["arc_easy_25"]
        elif isinstance(tasks, str):
            tasks = [tasks]

        template_path_raw = data.get("prompt_template_path") or data.get("prompt_variant")
        template_text = data.get("prompt_template")
        template_path_str: str | None = None

        if template_path_raw:
            tp = Path(template_path_raw)
            if not tp.is_absolute() and config_dir is not None:
                tp = (config_dir / tp).resolve()
            else:
                tp = tp.resolve()

            if not tp.is_file():
                raise FileNotFoundError(f"Prompt template file not found: {tp}")

            template_text = tp.read_text(encoding="utf-8")
            template_path_str = str(tp)

        return cls(
            models=list(models),
            tasks=list(tasks),
            prompt_template=template_text,
            prompt_template_path=template_path_str,
            seed=int(data.get("seed", 42)),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data["max_tokens"]) if data.get("max_tokens") is not None else None,
            url=data.get("url") or data.get("hestia_url"),
            output_dir=str(data.get("output_dir", "runs")),
            parent_run_id=data.get("parent_run_id") or data.get("parent_run"),
            branch_point=data.get("branch_point"),
            labeling_criterion=data.get("labeling_criterion"),
        )

    def merge_cli_args(
        self,
        models: list[str] | None = None,
        tasks: list[str] | None = None,
        prompt_template: str | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        url: str | None = None,
        output_dir: str | None = None,
        parent_run_id: str | None = None,
        branch_point: str | None = None,
        labeling_criterion: str | None = None,
    ) -> RunConfig:
        """Override configuration with explicit CLI flags."""
        merged_models = models if models is not None else self.models
        merged_tasks = tasks if tasks is not None else self.tasks
        merged_seed = seed if seed is not None else self.seed
        merged_temp = temperature if temperature is not None else self.temperature
        merged_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        merged_url = url if url is not None else self.url
        merged_output_dir = output_dir if output_dir is not None else self.output_dir
        merged_parent_run_id = parent_run_id if parent_run_id is not None else self.parent_run_id
        merged_branch_point = branch_point if branch_point is not None else self.branch_point
        merged_criterion = labeling_criterion if labeling_criterion is not None else self.labeling_criterion

        merged_template = self.prompt_template
        merged_template_path = self.prompt_template_path

        if prompt_template is not None:
            pt_path = Path(prompt_template).resolve()
            if pt_path.is_file():
                merged_template_path = str(pt_path)
                merged_template = pt_path.read_text(encoding="utf-8")
            elif (
                any(prompt_template.endswith(ext) for ext in (".txt", ".tmpl", ".jinja", ".jinja2", ".md"))
                or "/" in prompt_template
                or "\\" in prompt_template
            ):
                raise FileNotFoundError(f"Prompt template file not found: {prompt_template}")
            else:
                merged_template = prompt_template
                merged_template_path = None

        return RunConfig(
            models=merged_models,
            tasks=merged_tasks,
            prompt_template=merged_template,
            prompt_template_path=merged_template_path,
            seed=merged_seed,
            temperature=merged_temp,
            max_tokens=merged_max_tokens,
            url=merged_url,
            output_dir=merged_output_dir,
            parent_run_id=merged_parent_run_id,
            branch_point=merged_branch_point,
            labeling_criterion=merged_criterion,
        )
