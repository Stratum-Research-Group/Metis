"""Unit tests for Pi-inspired coding tools: read_file_range, edit_file_block, write_file, sandboxed_bash."""

import os
from pathlib import Path
import pytest

from kybernetes.contracts.tool_spec import ToolCallRequest
from kybernetes.tools.edit_file_block import SafeEditBlock
from kybernetes.tools.read_file_range import SafeReadRange
from kybernetes.tools.registry import ToolRegistry
from kybernetes.tools.sandboxed_bash import SafeSandboxedBash
from kybernetes.tools.write_file import SafeWriteFile


def test_read_file_range(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    sample = ws / "sample.py"
    sample.write_text("\n".join(f"line {i}" for i in range(1, 21)), encoding="utf-8")

    reader = SafeReadRange(allowed_roots=[ws])
    res = reader.read_range(path=str(sample), start_line=5, end_line=10)
    assert res["total_lines"] == 20
    assert res["start_line"] == 5
    assert res["end_line"] == 10
    assert len(res["lines"]) == 6
    assert "line 5" in res["lines"][0]
    assert "line 10" in res["lines"][-1]


def test_read_file_range_traversal_forbidden(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")

    reader = SafeReadRange(allowed_roots=[ws])
    with pytest.raises(PermissionError):
        reader.read_range(path=str(secret))


def test_write_file_and_overwrite(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    target = ws / "created.py"

    writer = SafeWriteFile(allowed_roots=[ws])
    res1 = writer.write(path=str(target), content="def foo(): pass", overwrite=False)
    assert res1["created"] is True
    assert res1["overwritten"] is False
    assert target.read_text(encoding="utf-8") == "def foo(): pass"

    # Fail on overwrite without flag
    with pytest.raises(FileExistsError):
        writer.write(path=str(target), content="new content", overwrite=False)

    # Success with overwrite
    res2 = writer.write(path=str(target), content="def bar(): return 1", overwrite=True)
    assert res2["overwritten"] is True
    assert target.read_text(encoding="utf-8") == "def bar(): return 1"


def test_edit_file_block_exact_and_ambiguous(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    target = ws / "code.py"
    target.write_text(
        "def compute(x):\n"
        "    temp = x * 2\n"
        "    return temp + 1\n",
        encoding="utf-8",
    )

    editor = SafeEditBlock(allowed_roots=[ws])

    # Ambiguity check when target text appears multiple times
    dup_file = ws / "dup.py"
    dup_file.write_text("item = 1\nitem = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="matches 2 times"):
        editor.edit_block(str(dup_file), "item = 1", "item = 2")

    # Not found check
    with pytest.raises(ValueError, match="was not found"):
        editor.edit_block(str(target), "non_existent_code", "replacement")

    # Successful unambiguous replacement
    search_str = "    temp = x * 2\n    return temp + 1"
    replace_str = "    return x * 2 + 1"
    res = editor.edit_block(str(target), search_str, replace_str)
    assert res["replaced"] is True
    assert target.read_text(encoding="utf-8") == "def compute(x):\n    return x * 2 + 1\n"


def test_sandboxed_bash_execution(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()

    runner = SafeSandboxedBash(allowed_roots=[ws])
    res = runner.execute("echo 'hello sovereign metis'", cwd=str(ws))
    assert res["exit_code"] == 0
    assert "hello sovereign metis" in res["stdout"]

    # Traversal cwd error
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    with pytest.raises(PermissionError):
        runner.execute("ls", cwd=str(forbidden))


def test_registry_invokes_new_tools(tmp_path: Path):
    registry = ToolRegistry(allowed_roots=[tmp_path])
    assert registry.get_tool("read_file_range") is not None
    assert registry.get_tool("edit_file_block") is not None
    assert registry.get_tool("write_file") is not None
    assert registry.get_tool("sandboxed_bash") is not None

    # Mutating tool sandboxed_bash without approval fails closed (Finding A1)
    req_no_app = ToolCallRequest(tool_id="sandboxed_bash", arguments={"command": "echo 42"})
    res_no_app = registry.execute(req_no_app)
    assert res_no_app.success is False
    assert "requires verified Phylax approval token" in res_no_app.error

    # Mutating tool with fabricated approval token fails closed (Finding A1)
    req_fake_app = ToolCallRequest(tool_id="sandboxed_bash", arguments={"command": "echo 42"}, approval_id="appr-fake123")
    res_fake_app = registry.execute(req_fake_app)
    assert res_fake_app.success is False
    assert "Invalid or unverified Phylax approval token" in res_fake_app.error

    # Mutating tool with legitimate approved token succeeds
    _, app_req = registry.phylax_adapter.is_action_permitted("sandboxed_bash", "sandboxed_bash", {"command": "echo 42"})
    assert app_req is not None
    app_id = app_req["approval_id"]
    assert registry.phylax_adapter.resolve_approval(app_id, "allow_once") is True

    req_valid = ToolCallRequest(tool_id="sandboxed_bash", arguments={"command": "echo 42"}, approval_id=app_id)
    res = registry.execute(req_valid)
    assert res.success is True
    assert res.data["exit_code"] == 0
    assert "42" in res.data["stdout"]

    # Replay with same token rejected
    res_replay = registry.execute(req_valid)
    assert res_replay.success is False
    assert "replay rejected" in res_replay.error


def test_read_file_range_hashline(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    sample = ws / "test.py"
    sample.write_text("def hello():\n    return 'metis'\n", encoding="utf-8")

    reader = SafeReadRange(allowed_roots=[ws])
    res = reader.read_range(path=str(sample), start_line=1, end_line=2)
    assert len(res["hashed_lines"]) == 2
    # Check format: line_no:4hex| content
    assert ":|" not in res["hashed_lines"][0]
    first_hashed = res["hashed_lines"][0]
    assert first_hashed.startswith("1:")
    assert "|" in first_hashed
    assert "def hello():" in first_hashed


def test_edit_file_block_hashline_stale_anchor_rejection(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    code_file = ws / "target.py"
    code_file.write_text("x = 10\ny = 20\n", encoding="utf-8")

    editor = SafeEditBlock(allowed_roots=[ws])

    # 1. Stale anchor failure with wrong expected hash
    with pytest.raises(ValueError, match="Stale-anchor rejection"):
        editor.edit_block(
            str(code_file),
            search_block="1:0000| x = 10",
            replace_block="x = 15",
        )

    # 2. Correct hashline succeeds
    reader = SafeReadRange(allowed_roots=[ws])
    read_res = reader.read_range(str(code_file))
    first_line_hashed = read_res["hashed_lines"][0]  # "1:xxxx| x = 10"

    edit_res = editor.edit_block(
        str(code_file),
        search_block=first_line_hashed,
        replace_block="x = 100",
    )
    assert edit_res["replaced"] is True
    assert "x = 100\ny = 20\n" == code_file.read_text(encoding="utf-8")


def test_edit_file_block_syntax_diagnostics(tmp_path: Path):
    ws = tmp_path / "workspace"
    ws.mkdir()
    py_file = ws / "broken.py"
    py_file.write_text("def valid():\n    return 1\n", encoding="utf-8")

    editor = SafeEditBlock(allowed_roots=[ws])

    # Edit into invalid syntax
    res = editor.edit_block(
        str(py_file),
        search_block="    return 1",
        replace_block="    return (unclosed",
    )
    assert res["replaced"] is True
    assert res["syntax_valid"] is False
    assert len(res["diagnostics"]) > 0
    assert "SyntaxError" in res["diagnostics"][0]

