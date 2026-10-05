"""Tool registry providing declarative schema validation, capability checks, and execution sandboxing."""

import time
from typing import Any, Callable, Dict, Optional

from kybernetes.contracts.enums import ToolSideEffectClass
from kybernetes.contracts.tool_spec import ToolCallRequest, ToolDefinition, ToolResult
from kybernetes.tools.calculator import SafeCalculator
from kybernetes.tools.file_search import SafeFileSearch


class ToolRegistry:
    """Central registry of vetted, declarative tools for Kybernetes orchestrator."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        calc = SafeCalculator()
        self.register(
            calc.get_definition(),
            lambda args: {"result": calc.evaluate(args.get("expression", ""))},
        )

        fsearch = SafeFileSearch()
        self.register(
            fsearch.get_definition(),
            lambda args: fsearch.search(
                pattern=args.get("pattern", ""),
                directory=args.get("directory"),
                max_results=int(args.get("max_results", 20)),
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
