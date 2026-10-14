"""GPQA Diamond evaluation task for PhD-level scientific reasoning."""

from __future__ import annotations

from pathlib import Path
from dokime.tasks.base_mc import MultipleChoiceTask


class GPQADiamondTask(MultipleChoiceTask):
    """Graduate-Level Google-Proof Q&A (GPQA Diamond) benchmark task."""

    task_id: str = "gpqa_diamond"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "gpqa_diamond.json"
        super().__init__(dataset_path=dataset_path)
