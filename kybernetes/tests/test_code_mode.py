"""Tests for SafeCodeMode (OpenCode 2.0-inspired sandboxed scripting tool)."""

from pathlib import Path
import pytest

from kybernetes.contracts.tool_spec import ToolCallRequest
from kybernetes.tools.code_mode import SafeCodeMode
from kybernetes.tools.registry import ToolRegistry


def test_code_mode_basic_execution(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])
    script = """
a = 10
b = 25
print("Computing sum...")
result = a + b
"""
    res = cmode.execute(script)
    assert res["success"] is True
    assert res["result"] == 35
    assert "Computing sum..." in res["stdout"]


def test_code_mode_main_function_return(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])
    script = """
def main():
    items = [1, 2, 3, 4]
    return [x * 2 for x in items]
"""
    res = cmode.execute(script)
    assert res["success"] is True
    assert res["result"] == [2, 4, 6, 8]


def test_code_mode_batch_file_exploration(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])

    # Create dummy files
    f1 = tmp_path / "mod1.py"
    f1.write_text("class Alpha:\n    def run(self):\n        return 1\n", encoding="utf-8")
    f2 = tmp_path / "mod2.py"
    f2.write_text("class Beta:\n    def test(self):\n        return 2\n", encoding="utf-8")
    f3 = tmp_path / "readme.txt"
    f3.write_text("Documentation\n", encoding="utf-8")

    script = """
py_files = [f['name'] for f in list_dir(pattern='*.py')]
classes = []
for p in py_files:
    ast_info = parse_ast(p)
    if ast_info['syntax_valid']:
        classes.extend(ast_info['classes'])
result = {'files': py_files, 'classes': sorted(classes)}
"""
    res = cmode.execute(script)
    assert res["success"] is True
    assert res["result"]["files"] == ["mod1.py", "mod2.py"]
    assert res["result"]["classes"] == ["Alpha", "Beta"]


def test_code_mode_security_rejects_import(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])
    script = """
import os
result = os.listdir('.')
"""
    res = cmode.execute(script)
    assert res["success"] is False
    assert "Direct module imports are prohibited" in res["error"]


def test_code_mode_security_rejects_dunder_exploit(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])
    script = """
subs = ().__class__.__bases__[0].__subclasses__()
result = len(subs)
"""
    res = cmode.execute(script)
    assert res["success"] is False
    assert "Prohibited access to internal dunder attribute" in res["error"]


def test_code_mode_security_rejects_eval_exec(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])
    script = """
eval("1 + 1")
"""
    res = cmode.execute(script)
    assert res["success"] is False
    assert "Use of builtin function 'eval' is prohibited" in res["error"]


def test_code_mode_timeout_protection(tmp_path: Path):
    cmode = SafeCodeMode(allowed_roots=[tmp_path])
    script = """
while True:
    pass
"""
    res = cmode.execute(script, timeout_seconds=0.5)
    assert res["success"] is False
    assert "CodeModeTimeout" in res["error"]


def test_code_mode_registered_in_tool_registry(tmp_path: Path):
    registry = ToolRegistry(allowed_roots=[tmp_path])
    assert "run_code_mode" in registry.list_tools()

    req = ToolCallRequest(
        tool_id="run_code_mode",
        arguments={
            "script": "result = calculate('15 * 3')",
        },
    )
    result = registry.execute(req)
    assert result.success is True
    assert result.data["result"] == 45.0
