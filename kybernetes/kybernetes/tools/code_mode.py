"""OpenCode 2.0-inspired 'Code Mode' Sandboxed Scripting Tool for Metis.
Enables agents to execute multi-file exploration, AST inspection, and tool chaining
programmatically in a single round-trip instead of 20 sequential LLM turns.
"""

import ast
import concurrent.futures
import io
from io import StringIO
import json
import math
import multiprocessing
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolDefinition
from kybernetes.tools.calculator import SafeCalculator
from kybernetes.tools.file_search import SafeFileSearch
from kybernetes.tools.read_file_range import SafeReadRange


class CodeModeSecurityError(Exception):
    """Raised when script violates Code Mode sandbox constraints."""
    pass


class SafeCodeMode:
    """
    Executes sandboxed Python scripts with workspace-bound read/inspect capabilities.
    Allows LLM to orchestrate batch searches, AST analysis, and data filtering in 1 pass.
    """

    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "int": int,
        "isinstance": isinstance,
        "issubclass": issubclass,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }

    # Explicitly forbidden attribute patterns to prevent sandbox escapes (Finding F-02)
    FORBIDDEN_ATTRIBUTES = {
        "__subclasses__",
        "__globals__",
        "__code__",
        "__builtins__",
        "__import__",
        "__class__",
        "__mro__",
        "__bases__",
        "__dict__",
        "__closure__",
        # Generator, coroutine, and frame traversal escapes
        "gi_frame",
        "gi_code",
        "gi_running",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_running",
        "cr_await",
        "ag_frame",
        "ag_code",
        "ag_running",
        "ag_await",
        "f_back",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_code",
        "f_trace",
        # Format string traversal
        "format",
        "format_map",
    }

    def __init__(self, allowed_roots: Optional[List[Path]] = None):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]
        self._reader = SafeReadRange(allowed_roots=self.allowed_roots)
        self._searcher = SafeFileSearch(allowed_roots=self.allowed_roots)
        self._calc = SafeCalculator()

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_code_mode",
            description=(
                "Execute a sandboxed Python script for batch workspace inspection, multi-file discovery, "
                "and tool chaining in a single pass (OpenCode 2.0 Code Mode). "
                "Available functions in script: read_file(path, start_line, end_line), search_files(pattern), "
                "list_dir(path, pattern), parse_ast(path), calculate(expr), print(*args). "
                "Assign output to 'result' or return from a main() function."
            ),
            side_effect_class=ToolSideEffectClass.SYSTEM_EXECUTION,
            parameters_schema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Python script to execute in sandboxed Code Mode. Assign final value to variable 'result'.",
                        "minLength": 1,
                        "maxLength": 10000,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "description": "Maximum execution time in seconds (default 5.0, max 15.0).",
                        "minimum": 0.5,
                        "maximum": 15.0,
                    },
                },
                "required": ["script"],
            },
            timeout_ms=16000,
        )

    def _validate_ast_safety(self, code_ast: ast.AST) -> None:
        """Validate that AST contains no prohibited operations (imports, exec, eval, dunder exploits)."""
        for node in ast.walk(code_ast):
            # Prohibit import statements
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise CodeModeSecurityError(
                    "Direct module imports are prohibited in Code Mode. Use the pre-injected tools and libraries."
                )

            # Prohibit generators and coroutines (generators expose frames without dunders)
            if isinstance(node, (ast.Yield, ast.YieldFrom)):
                raise CodeModeSecurityError(
                    "Generator and coroutine yield expressions are prohibited in Code Mode."
                )

            # Prohibit dangerous attribute access and private internals
            if isinstance(node, ast.Attribute):
                if (
                    node.attr in self.FORBIDDEN_ATTRIBUTES
                    or node.attr.startswith("_")
                    or (node.attr.startswith("__") and node.attr.endswith("__"))
                ):
                    raise CodeModeSecurityError(
                        f"Prohibited access to internal dunder attribute '{node.attr}' in Code Mode."
                    )

            # Prohibit calls to eval, exec, open, __import__
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ("eval", "exec", "open", "compile", "__import__"):
                        raise CodeModeSecurityError(f"Use of builtin function '{node.func.id}' is prohibited.")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("format", "format_map"):
                        raise CodeModeSecurityError("Use of string format/format_map methods is prohibited in Code Mode.")

            # Prohibit format-spec dunder traversal in string literals (Finding A6)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.search(r"\{[^{}]*__", node.value) or any(
                    f in node.value and re.search(rf"\{{[^{{}}]*{re.escape(f)}", node.value)
                    for f in self.FORBIDDEN_ATTRIBUTES
                ):
                    raise CodeModeSecurityError(
                        "Prohibited format-spec attribute traversal in Code Mode string literal."
                    )

            # Prohibit format-spec dunder traversal in f-strings (Finding A6)
            if isinstance(node, ast.FormattedValue) and node.format_spec:
                spec_str = ast.unparse(node.format_spec)
                if "__" in spec_str or any(f in spec_str for f in self.FORBIDDEN_ATTRIBUTES):
                    raise CodeModeSecurityError(
                        "Prohibited format-spec attribute traversal in Code Mode f-string."
                    )

    def _resolve_safe_path(self, raw_path: str) -> Path:
        """Verify that path resides strictly inside allowed roots."""
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (self.allowed_roots[0] / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if not any(candidate == root or root in candidate.parents for root in self.allowed_roots):
            raise PermissionError(f"Access to path '{raw_path}' denied: strictly outside allowed workspace boundaries.")
        return candidate

    def _helper_read_file(self, path: str, start_line: int = 1, end_line: int = 200) -> Dict[str, Any]:
        """Exposed to script: reads a file range safely."""
        return self._reader.read_range(path, start_line=start_line, end_line=end_line)

    def _helper_search_files(self, pattern: str, directory: Optional[str] = None, max_results: int = 50) -> Dict[str, Any]:
        """Exposed to script: searches files matching glob or regex pattern."""
        return self._searcher.search(pattern=pattern, directory=directory, max_results=max_results)

    def _helper_list_dir(self, path: Optional[str] = None, pattern: str = "*") -> List[Dict[str, Any]]:
        """Exposed to script: safely lists directory contents."""
        target_dir = self._resolve_safe_path(path or ".")
        if not target_dir.is_dir():
            raise NotADirectoryError(f"Path '{target_dir}' is not a directory.")

        results = []
        for item in sorted(target_dir.glob(pattern)):
            results.append({
                "name": item.name,
                "path": str(item.relative_to(self.allowed_roots[0]) if item.is_relative_to(self.allowed_roots[0]) else item),
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else None,
            })
            if len(results) >= 200:
                break
        return results

    def _helper_parse_ast(self, path: str) -> Dict[str, Any]:
        """Exposed to script: safely inspects Python AST structure."""
        file_path = self._resolve_safe_path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: '{file_path}'")

        content = file_path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(content, filename=str(file_path))
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            imports = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    imports.extend(alias.name for alias in n.names)
                elif isinstance(n, ast.ImportFrom) and n.module:
                    imports.append(n.module)

            return {
                "syntax_valid": True,
                "classes": classes,
                "functions": functions,
                "imports": sorted(set(imports)),
                "total_lines": len(content.splitlines()),
            }
        except SyntaxError as e:
            return {
                "syntax_valid": False,
                "syntax_error": f"Syntax error at line {e.lineno}: {e.msg}",
            }

    def execute(self, script: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
        """Execute sandboxed script and return results, captured prints, and timing."""
        start_time = time.perf_counter()
        timeout = min(max(float(timeout_seconds), 0.5), 15.0)

        # 1. Parse and validate AST
        try:
            parsed_ast = ast.parse(script, filename="<codemode_script>")
            self._validate_ast_safety(parsed_ast)
        except (SyntaxError, CodeModeSecurityError) as e:
            return {
                "success": False,
                "error": f"CodeModeValidationFailed: {e}",
                "stdout": "",
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
            }

        compiled = compile(parsed_ast, filename="<codemode_script>", mode="exec")

        # Execute in dedicated child process with hard OS timeout killing (Finding A6)
        def _worker_entry(queue: Any) -> None:
            stdout_buf = StringIO()

            def safe_print(*args: Any, **kwargs: Any) -> None:
                sep = kwargs.get("sep", " ")
                end = kwargs.get("end", "\n")
                line = sep.join(str(a) for a in args) + end
                if stdout_buf.tell() < 65536:
                    stdout_buf.write(line)

            env_builtins = dict(self.SAFE_BUILTINS)
            env_builtins["print"] = safe_print

            sandbox_globals: Dict[str, Any] = {
                "__builtins__": env_builtins,
                # Safe libraries
                "json": json,
                "re": re,
                "math": math,
                # Workspaces tool helpers
                "read_file": self._helper_read_file,
                "search_files": self._helper_search_files,
                "list_dir": self._helper_list_dir,
                "parse_ast": self._helper_parse_ast,
                "calculate": lambda expr: self._calc.evaluate(expr),
                # Default result receptacle
                "result": None,
            }

            try:
                exec(compiled, sandbox_globals)  # pylint: disable=exec-used

                if "main" in sandbox_globals and callable(sandbox_globals["main"]):
                    result_val = sandbox_globals["main"]()
                else:
                    result_val = sandbox_globals.get("result")

                serialized_result: Any = None
                try:
                    json.dumps(result_val)
                    serialized_result = result_val
                except (TypeError, OverflowError):
                    serialized_result = str(result_val)

                queue.put({
                    "success": True,
                    "result": serialized_result,
                    "stdout": stdout_buf.getvalue()[:65536],
                })
            except Exception as e:
                queue.put({
                    "success": False,
                    "error": f"CodeModeRuntimeError: {type(e).__name__}: {str(e)}",
                    "stdout": stdout_buf.getvalue()[:65536],
                })

        mp_ctx = multiprocessing.get_context("fork")
        result_queue = mp_ctx.Queue()
        proc = mp_ctx.Process(target=_worker_entry, args=(result_queue,))
        proc.start()
        proc.join(timeout=timeout)

        if proc.is_alive():
            proc.kill()
            proc.join()
            latency = (time.perf_counter() - start_time) * 1000
            return {
                "success": False,
                "error": f"CodeModeTimeout: Script exceeded max execution time of {timeout}s.",
                "stdout": "",
                "latency_ms": round(latency, 2),
            }

        latency = (time.perf_counter() - start_time) * 1000
        try:
            res_dict = result_queue.get_nowait()
            res_dict["latency_ms"] = round(latency, 2)
            return res_dict
        except Exception:
            return {
                "success": False,
                "error": f"CodeModeRuntimeError: Process terminated abnormally (exitcode {proc.exitcode}).",
                "stdout": "",
                "latency_ms": round(latency, 2),
            }
