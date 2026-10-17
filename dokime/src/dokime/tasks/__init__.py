"""Dokime task plugin package."""

from dokime.tasks.arc import ArcEasy25Task
from dokime.tasks.base_mc import MultipleChoiceTask
from dokime.tasks.epistemic import EpistemicAdversarialTask, EpistemicAdversarialV2Task
from dokime.tasks.gpqa import GPQADiamondTask, GPQADiamondV2Task
from dokime.tasks.math_competition import AIMECompetitionMathTask, AIMECompetitionMathV2Task
from dokime.tasks.pinax_lookup import PinaxLookup50Task
from dokime.tasks.protocol import Query, Task
from dokime.tasks.registry import discover_tasks, get_task, list_task_info
from dokime.tasks.zebra import ZebraHardLogicTask, ZebraHardLogicV2Task

__all__ = [
    "ArcEasy25Task",
    "AIMECompetitionMathTask",
    "AIMECompetitionMathV2Task",
    "EpistemicAdversarialTask",
    "EpistemicAdversarialV2Task",
    "GPQADiamondTask",
    "GPQADiamondV2Task",
    "MultipleChoiceTask",
    "PinaxLookup50Task",
    "Query",
    "Task",
    "ZebraHardLogicTask",
    "ZebraHardLogicV2Task",
    "discover_tasks",
    "get_task",
    "list_task_info",
]

