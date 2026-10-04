"""Unit tests for safe tools and ToolRegistry sandboxing."""

import pytest
from kybernetes.contracts.tool_spec import ToolCallRequest
from kybernetes.tools.calculator import SafeCalculator
from kybernetes.tools.file_search import SafeFileSearch
from kybernetes.tools.registry import ToolRegistry


def test_calculator_basic_and_functions():
    calc = SafeCalculator()
    assert calc.evaluate("2 + 2") == 4.0
    assert calc.evaluate("10 * 5 - 8 / 2") == 46.0
    assert calc.evaluate("sqrt(144) + abs(-5)") == 17.0
    assert calc.evaluate("round(pi, 2)") == 3.14


def test_calculator_zero_division():
    calc = SafeCalculator()
    with pytest.raises(ZeroDivisionError):
        calc.evaluate("100 / 0")


def test_calculator_disallow_arbitrary_code():
    calc = SafeCalculator()
    with pytest.raises(ValueError):
        calc.evaluate("__import__('os').system('ls')")
    with pytest.raises(ValueError):
        calc.evaluate("open('/etc/passwd')")
    with pytest.raises(ValueError):
        calc.evaluate("bad_identifier")


def test_file_search_allowed_root(tmp_path):
    sub = tmp_path / "workspace"
    sub.mkdir()
    sample_file = sub / "doc.txt"
    sample_file.write_text("Hello Metis local stack orchestrator", encoding="utf-8")

    searcher = SafeFileSearch(allowed_roots=[sub])
    res = searcher.search(pattern="orchestrator", directory=str(sub))
    assert res["total_matches"] == 1
    assert "orchestrator" in res["matches"][0]["snippet"]


def test_file_search_disallow_path_traversal(tmp_path):
    sub = tmp_path / "safe_dir"
    sub.mkdir()
    forbidden = tmp_path / "forbidden_dir"
    forbidden.mkdir()

    searcher = SafeFileSearch(allowed_roots=[sub])
    with pytest.raises(PermissionError):
        searcher.search(pattern="secret", directory=str(forbidden))


def test_tool_registry_validation_and_execution():
    registry = ToolRegistry()
    assert registry.get_tool("calculator") is not None
    assert registry.get_tool("file_search") is not None

    req = ToolCallRequest(tool_id="calculator", arguments={"expression": "15 * 3"})
    res = registry.execute(req)
    assert res.success is True
    assert res.data["result"] == 45.0

    # Missing required argument
    bad_req = ToolCallRequest(tool_id="calculator", arguments={})
    res_bad = registry.execute(bad_req)
    assert res_bad.success is False
    assert "Missing required parameter" in res_bad.error
