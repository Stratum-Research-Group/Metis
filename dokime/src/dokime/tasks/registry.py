"""Task discovery via Python entry points and built-in task registry."""

from __future__ import annotations

import importlib.metadata
import inspect
from typing import Any

from dokime.tasks.arc import ArcEasy25Task
from dokime.tasks.epistemic import EpistemicAdversarialTask
from dokime.tasks.gpqa import GPQADiamondTask
from dokime.tasks.math_competition import AIMECompetitionMathTask
from dokime.tasks.protocol import Task
from dokime.tasks.zebra import ZebraHardLogicTask

# Built-in tasks fallback
BUILTIN_TASKS: dict[str, type[Task] | Task] = {
    "arc_easy_25": ArcEasy25Task,
    "gpqa_diamond": GPQADiamondTask,
    "aime_math": AIMECompetitionMathTask,
    "zebra_hard_logic": ZebraHardLogicTask,
    "epistemic_adversarial": EpistemicAdversarialTask,
}


def discover_tasks() -> dict[str, Task]:
    """Discover all registered task plugins via entry points and built-ins."""
    tasks: dict[str, Task] = {}

    # 1. Add built-in tasks
    for name, task_factory in BUILTIN_TASKS.items():
        if inspect.isclass(task_factory):
            tasks[name] = task_factory()
        else:
            tasks[name] = task_factory

    # 2. Add entry points under group 'dokime.tasks'
    try:
        eps = importlib.metadata.entry_points(group="dokime.tasks")
        for ep in eps:
            try:
                loaded = ep.load()
                if inspect.isclass(loaded):
                    instance = loaded()
                else:
                    instance = loaded
                tasks[ep.name] = instance
            except Exception as e:
                # Log or keep fallback
                pass
    except Exception:
        pass

    return tasks


def get_task(task_id: str) -> Task:
    """Retrieve an instantiated task by task_id."""
    available = discover_tasks()
    if task_id not in available:
        raise KeyError(f"Task '{task_id}' not found. Available tasks: {list(available.keys())}")
    return available[task_id]


def list_task_info() -> list[dict[str, Any]]:
    """Return task metadata including item count for each discovered task."""
    tasks = discover_tasks()
    info_list: list[dict[str, Any]] = []

    for name, task in sorted(tasks.items()):
        count = 0
        dataset_path_str = ""
        try:
            queries = task.load()
            count = len(queries)
        except Exception:
            count = -1

        if getattr(task, "dataset_path", None):
            dataset_path_str = str(task.dataset_path)

        info_list.append(
            {
                "task_id": name,
                "items": count,
                "dataset_path": dataset_path_str,
            }
        )
    return info_list
