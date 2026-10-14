"""Dokime task plugin package."""

from dokime.tasks.arc import ArcEasy25Task
from dokime.tasks.base_mc import MultipleChoiceTask
from dokime.tasks.epistemic import EpistemicAdversarialTask
from dokime.tasks.gpqa import GPQADiamondTask
from dokime.tasks.math_competition import AIMECompetitionMathTask
from dokime.tasks.protocol import Query, Task
from dokime.tasks.registry import discover_tasks, get_task, list_task_info
from dokime.tasks.zebra import ZebraHardLogicTask

__all__ = [
    "ArcEasy25Task",
    "AIMECompetitionMathTask",
    "EpistemicAdversarialTask",
    "GPQADiamondTask",
    "MultipleChoiceTask",
    "Query",
    "Task",
    "ZebraHardLogicTask",
    "discover_tasks",
    "get_task",
    "list_task_info",
]
