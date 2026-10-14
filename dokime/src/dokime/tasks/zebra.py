"""Zebra and constraint-satisfaction deduction evaluation task."""

from __future__ import annotations

from pathlib import Path
from dokime.tasks.base_mc import MultipleChoiceTask


class ZebraHardLogicTask(MultipleChoiceTask):
    """Zebra logic, Einstein-style puzzles, and multi-variable constraint satisfaction task."""

    task_id: str = "zebra_hard_logic"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "zebra_hard_logic.json"
        super().__init__(dataset_path=dataset_path)
