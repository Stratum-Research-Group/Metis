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
    # Tool execution with approvals disabled
    registry = ToolRegistry(allowed_roots=[tmp_path], enforce_approvals=False)
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

    # With enforce_approvals=True, SYSTEM_EXEC tool requires approval token (Finding F-02)
    enforced_registry = ToolRegistry(allowed_roots=[tmp_path], enforce_approvals=True)
    enforced_res = enforced_registry.execute(req)
    assert enforced_res.success is False
    assert "requires verified Phylax approval token" in enforced_res.error


def test_code_mode_rejects_format_spec_traversal(tmp_path: Path):
    """Finding A6 regression: format-spec attribute traversal must be blocked by AST safety validation."""
    cmode = SafeCodeMode(allowed_roots=[tmp_path])

    # 1. Direct string format with dunder attribute
    script1 = 'result = "{0.__class__}".format(1)'
    res1 = cmode.execute(script1)
    assert res1["success"] is False
    assert ("Prohibited format-spec attribute traversal" in res1["error"]
            or "string format/format_map methods is prohibited" in res1["error"]
            or "Prohibited access to internal or dangerous attribute" in res1["error"])

    # 2. String format with mro / forbidden attributes
    script2 = 'result = "{0.__class__.__mro__}".format(1)'
    res2 = cmode.execute(script2)
    assert res2["success"] is False
    assert ("Prohibited format-spec attribute traversal" in res2["error"]
            or "string format/format_map methods is prohibited" in res2["error"]
            or "Prohibited access to internal or dangerous attribute" in res2["error"])


def test_code_mode_rejects_generator_and_frame_traversal(tmp_path: Path):
    """Finding F-02 regression: generator yield and frame traversal escapes must be blocked."""
    cmode = SafeCodeMode(allowed_roots=[tmp_path])

    # 1. Reject generator yield
    script_yield = """
def gen():
    yield 1
g = gen()
result = 1
"""
    res_yield = cmode.execute(script_yield)
    assert res_yield["success"] is False
    assert "Generator and coroutine yield expressions are prohibited" in res_yield["error"]

    # 2. Reject frame traversal attributes
    script_frame = """
g = [1, 2]
res = g.gi_frame
"""
    res_frame = cmode.execute(script_frame)
    assert res_frame["success"] is False
    assert "Prohibited access to internal dunder attribute 'gi_frame'" in res_frame["error"]


def test_code_mode_enforceable_timeout_on_c_extensions(tmp_path: Path):
    """Finding A6 regression: C-extension runaway calls (e.g. math.factorial) must abort on timeout."""
    cmode = SafeCodeMode(allowed_roots=[tmp_path])

    # math.factorial(3000000) takes >45s in un-sandboxed C code
    script = "result = math.factorial(3000000)"
    res = cmode.execute(script, timeout_seconds=0.5)

    assert res["success"] is False
    assert "CodeModeTimeout" in res["error"]
    assert res["latency_ms"] < 3000  # Aborts cleanly within seconds, not 45s

