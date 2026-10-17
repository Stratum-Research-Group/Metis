"""Tests for task plugin protocol, reference ARC task, and registry discovery."""

from dokime.tasks.arc import ArcEasy25Task
from dokime.tasks.protocol import Query
from dokime.tasks.registry import discover_tasks, get_task, list_task_info


def test_arc_easy_25_task_loading():
    task = ArcEasy25Task()
    queries = task.load()

    assert len(queries) == 25
    first = queries[0]
    assert isinstance(first, Query)
    assert first.id == "arc_001"
    assert "mass of a rock" in first.prompt
    assert len(first.choices) == 4
    assert set(first.choices.keys()) == {"A", "B", "C", "D"}
    assert first.expected in {"A", "B", "C", "D"}


def test_arc_task_prompt_formatting():
    task = ArcEasy25Task()
    query = Query(
        id="test_01",
        prompt="What color is the sky on a clear day?",
        choices={"A": "Blue", "B": "Green", "C": "Red", "D": "Yellow"},
        expected="A",
    )

    # Default format
    fmt_default = task.format_prompt(query)
    assert "Question: What color is the sky on a clear day?" in fmt_default
    assert "A: Blue" in fmt_default
    assert "Answer:" in fmt_default

    # Custom template with placeholders
    template = "Deliberate and select one choice:\n{question}\n{choices}\nAnswer:"
    fmt_custom = task.format_prompt(query, template=template)
    assert "Deliberate and select one choice:" in fmt_custom
    assert "What color is the sky on a clear day?" in fmt_custom


def test_arc_task_prediction_and_scoring():
    task = ArcEasy25Task()
    query = Query(
        id="q1",
        prompt="Sample question",
        choices={"A": "Alpha", "B": "Beta", "C": "Gamma", "D": "Delta"},
        expected="B",
    )

    # Clean letter
    assert task.extract_prediction(query, "B") == "B"
    assert task.score(query, "B") == 1.0

    # Complex response containing answer
    assert task.extract_prediction(query, "The correct answer is (B).") == "B"
    assert task.score(query, "The correct answer is (B).") == 1.0

    # Markdown bold answer
    assert task.extract_prediction(query, "**B**") == "B"
    assert task.score(query, "**B**") == 1.0

    # Wrong answer
    assert task.extract_prediction(query, "A") == "A"
    assert task.score(query, "A") == 0.0

    # Garbage / unparseable
    assert task.extract_prediction(query, "I don't know") is None
    assert task.score(query, "I don't know") == 0.0


def test_task_registry_discovery():
    tasks = discover_tasks()
    assert "arc_easy_25" in tasks
    loaded = get_task("arc_easy_25")
    assert isinstance(loaded, ArcEasy25Task)

    info = list_task_info()
    arc_info = next(item for item in info if item["task_id"] == "arc_easy_25")
    assert arc_info["items"] == 25

    pinax_info = next(item for item in info if item["task_id"] == "pinax_lookup_50")
    assert pinax_info["items"] == 50


def test_pinax_lookup_50_loading_and_scoring():
    from dokime.tasks.pinax_lookup import PinaxLookup50Task
    task = PinaxLookup50Task()
    queries = task.load()
    assert len(queries) == 50

    q1 = queries[0]
    assert q1.id == "pinax-01"
    assert q1.expected == "rg"

    # Prediction extraction and scoring
    assert task.extract_prediction(q1, "rg") == "rg"
    assert task.score(q1, "rg") == 1.0
    assert task.score(q1, "`rg`") == 1.0
    assert task.score(q1, "The tool is `rg`.") == 1.0
    assert task.score(q1, "tool:rg") == 1.0
    assert task.score(q1, "curl") == 0.0

    # Direct catalogue evaluator
    def mock_lookup(query, limit=5):
        if "regex" in query:
            return [{"id": "tool:rg", "name": "rg"}]
        return [{"id": "tool:unknown", "name": "unknown"}]

    res = task.evaluate_catalogue(mock_lookup)
    assert res["total_queries"] == 50
    assert res["top1_hits"] >= 1
    assert "top1_accuracy" in res
    assert "top3_accuracy" in res

