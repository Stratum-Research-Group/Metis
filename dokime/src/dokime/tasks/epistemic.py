"""Epistemic adversarial evaluation task for fallacy detection and lure discrimination."""

from __future__ import annotations

from pathlib import Path
from dokime.tasks.base_mc import MultipleChoiceTask


class EpistemicAdversarialTask(MultipleChoiceTask):
    """Metis epistemic verification benchmark task for detecting false presuppositions and fallacies."""

    task_id: str = "epistemic_adversarial"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "epistemic_adversarial.json"
        super().__init__(dataset_path=dataset_path)


class EpistemicAdversarialV2Task(MultipleChoiceTask):
    """Set 2: Fresh unseen epistemic verification lures and deceptive presuppositions."""

    task_id: str = "epistemic_adversarial_v2"

    def __init__(self, dataset_path: Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "data" / "epistemic_adversarial_v2.json"
        super().__init__(dataset_path=dataset_path)

