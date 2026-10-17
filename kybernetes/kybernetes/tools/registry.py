"""Tool registry providing declarative schema validation, capability checks, and execution sandboxing."""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional


from kybernetes.adapters.phylax import PhylaxAdapter
from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolCallRequest, ToolDefinition, ToolResult
from kybernetes.tools.calculator import SafeCalculator
from kybernetes.tools.code_mode import SafeCodeMode
from kybernetes.tools.edit_file_block import SafeEditBlock
from kybernetes.tools.file_search import SafeFileSearch
from kybernetes.tools.read_file_range import SafeReadRange
from kybernetes.tools.sandboxed_bash import SafeSandboxedBash
from kybernetes.tools.write_file import SafeWriteFile


class ToolRegistry:
    """Central registry of vetted, declarative tools for Kybernetes orchestrator."""

    def __init__(
        self,
        allowed_roots: Optional[List[Path]] = None,
        phylax_adapter: Optional[PhylaxAdapter] = None,
        enforce_approvals: bool = True,
    ):
        self.allowed_roots = [r.resolve() for r in (allowed_roots or [Path.cwd()])]
        self.phylax_adapter = phylax_adapter or PhylaxAdapter()
        self.enforce_approvals = enforce_approvals
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        calc = SafeCalculator()
        self.register(
            calc.get_definition(),
            lambda args: {"result": calc.evaluate(args.get("expression", ""))},
        )

        fsearch = SafeFileSearch(allowed_roots=self.allowed_roots)
        self.register(
            fsearch.get_definition(),
            lambda args: fsearch.search(
                pattern=args.get("pattern", ""),
                directory=args.get("directory"),
                max_results=int(args.get("max_results", 20)),
            ),
        )

        rrange = SafeReadRange(allowed_roots=self.allowed_roots)
        self.register(
            rrange.get_definition(),
            lambda args: rrange.read_range(
                path=args.get("path", ""),
                start_line=int(args.get("start_line", 1)),
                end_line=int(args.get("end_line", 200)),
            ),
        )

        eblock = SafeEditBlock(allowed_roots=self.allowed_roots)
        self.register(
            eblock.get_definition(),
            lambda args: eblock.edit_block(
                path=args.get("path", ""),
                search_block=args.get("search_block", ""),
                replace_block=args.get("replace_block", ""),
                expected_start_hash=args.get("expected_start_hash"),
                expected_end_hash=args.get("expected_end_hash"),
                allow_syntax_errors=bool(args.get("allow_syntax_errors", False)),
            ),
        )

        wfile = SafeWriteFile(allowed_roots=self.allowed_roots)
        self.register(
            wfile.get_definition(),
            lambda args: wfile.write(
                path=args.get("path", ""),
                content=args.get("content", ""),
                overwrite=bool(args.get("overwrite", False)),
            ),
        )

        sbash = SafeSandboxedBash(allowed_roots=self.allowed_roots)
        self.register(
            sbash.get_definition(),
            lambda args: sbash.execute(
                command=args.get("command", ""),
                cwd=args.get("cwd"),
                timeout_seconds=args.get("timeout_seconds"),
            ),
        )

        cmode = SafeCodeMode(allowed_roots=self.allowed_roots)
        self.register(
            cmode.get_definition(),
            lambda args: cmode.execute(
                script=args.get("script", ""),
                timeout_seconds=float(args.get("timeout_seconds", 5.0)),
            ),
        )



    def register(self, definition: ToolDefinition, handler: Callable[[Dict[str, Any]], Any]) -> None:
        """Register a vetted tool with schema and handler."""
        if not definition.name:
            raise ValueError("Tool definition must have a non-empty name")
        self._tools[definition.name] = definition
        self._handlers[definition.name] = handler

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> Dict[str, ToolDefinition]:
        return dict(self._tools)

    def validate_arguments(self, name: str, arguments: Dict[str, Any]) -> None:
        """Validate argument dictionary against parameter JSON schema."""
        tool = self.get_tool(name)
        if not tool:
            raise KeyError(f"Tool '{name}' is not registered")

        schema = tool.parameters_schema
        required = schema.get("required", [])
        for req in required:
            if req not in arguments:
                raise ValueError(f"Missing required parameter '{req}' for tool '{name}'")

        props = schema.get("properties", {})
        for k, v in arguments.items():
            if k in props:
                prop_spec = props[k]
                expected_type = prop_spec.get("type")
                if expected_type == "string":
                    if not isinstance(v, str):
                        raise TypeError(f"Parameter '{k}' expected string, got {type(v).__name__}")
                    if "maxLength" in prop_spec and len(v) > prop_spec["maxLength"]:
                        raise ValueError(f"Parameter '{k}' exceeds maxLength {prop_spec['maxLength']}")
                    if "minLength" in prop_spec and len(v) < prop_spec["minLength"]:
                        raise ValueError(f"Parameter '{k}' below minLength {prop_spec['minLength']}")
                elif expected_type == "integer":
                    if not isinstance(v, int) or isinstance(v, bool):
                        raise TypeError(f"Parameter '{k}' expected integer, got {type(v).__name__}")
                    if "minimum" in prop_spec and v < prop_spec["minimum"]:
                        raise ValueError(f"Parameter '{k}' below minimum {prop_spec['minimum']}")
                    if "maximum" in prop_spec and v > prop_spec["maximum"]:
                        raise ValueError(f"Parameter '{k}' exceeds maximum {prop_spec['maximum']}")
                elif expected_type == "number":
                    if not isinstance(v, (int, float)) or isinstance(v, bool):
                        raise TypeError(f"Parameter '{k}' expected number, got {type(v).__name__}")
                    if "minimum" in prop_spec and v < prop_spec["minimum"]:
                        raise ValueError(f"Parameter '{k}' below minimum {prop_spec['minimum']}")
                    if "maximum" in prop_spec and v > prop_spec["maximum"]:
                        raise ValueError(f"Parameter '{k}' exceeds maximum {prop_spec['maximum']}")

    def execute(self, request: ToolCallRequest) -> ToolResult:
        """Execute registered tool safely with timing and error isolation."""
        start_time = time.perf_counter()
        tool = self.get_tool(request.tool_id)
        if not tool:
            return ToolResult(
                tool_id=request.tool_id,
                success=False,
                error=f"Unregistered tool: '{request.tool_id}'",
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # 1. Enforce Phylax gating for mutating tools (Finding A1)
        is_mutating = tool.side_effect_class != ToolSideEffectClass.READ_ONLY
        if is_mutating and self.enforce_approvals:
            if not request.approval_id:
                latency = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    tool_id=request.tool_id,
                    success=False,
                    error=f"Tool '{request.tool_id}' requires verified Phylax approval token (side_effect_class={tool.side_effect_class.value})",
                    latency_ms=round(latency, 2),
                )

            consumed, msg = self.phylax_adapter.consume_approval(
                approval_id=request.approval_id,
                tool_id=request.tool_id,
                arguments=request.arguments,
            )
            if not consumed:
                latency = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    tool_id=request.tool_id,
                    success=False,
                    error=f"Invalid or unverified Phylax approval token: '{request.approval_id}' - {msg}",
                    latency_ms=round(latency, 2),
                )
        elif request.approval_id:
            # If an approval_id was explicitly provided for any tool, verify and consume it
            consumed, msg = self.phylax_adapter.consume_approval(
                approval_id=request.approval_id,
                tool_id=request.tool_id,
                arguments=request.arguments,
            )
            if not consumed:
                latency = (time.perf_counter() - start_time) * 1000
                return ToolResult(
                    tool_id=request.tool_id,
                    success=False,
                    error=f"Invalid or unverified Phylax approval token: '{request.approval_id}' - {msg}",
                    latency_ms=round(latency, 2),
                )

        try:
            self.validate_arguments(request.tool_id, request.arguments)
            handler = self._handlers[request.tool_id]
            data = handler(request.arguments)
            latency = (time.perf_counter() - start_time) * 1000

            if tool.timeout_ms and latency > tool.timeout_ms:
                return ToolResult(
                    tool_id=request.tool_id,
                    success=False,
                    error=f"Tool '{request.tool_id}' exceeded declared timeout of {tool.timeout_ms}ms (took {latency:.1f}ms)",
                    latency_ms=round(latency, 2),
                )

            return ToolResult(
                tool_id=request.tool_id,
                success=True,
                data=data,
                latency_ms=round(latency, 2),
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_id=request.tool_id,
                success=False,
                error=str(e),
                latency_ms=round(latency, 2),
            )
