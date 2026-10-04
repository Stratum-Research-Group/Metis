"""Guarded tool adapter for Kybernetes and internal tool invocations."""

from typing import Any, Callable

from phylax.contracts.enums import ResourceClass, Verb
from phylax.gateway.gateway import GuardedGateway


class GuardedToolAdapter:
    """Brokers tool calls through Phylax authorization."""

    def __init__(self, gateway: GuardedGateway):
        self.gateway = gateway
        self._registered_tools: dict[str, Callable[..., Any]] = {}

    def register_tool(self, tool_id: str, handler: Callable[..., Any]) -> None:
        """Register a local tool implementation."""
        self._registered_tools[tool_id] = handler

    def invoke(
        self,
        tool_id: str,
        arguments: dict[str, Any] | None = None,
        subject: str = "kybernetes",
        approval_id: str | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """Authorize and invoke a registered tool."""
        if tool_id not in self._registered_tools:
            raise KeyError(f"Tool '{tool_id}' is not registered in GuardedToolAdapter")

        handler = self._registered_tools[tool_id]
        args = arguments or {}

        result, _ = self.gateway.authorize_and_execute(
            subject=subject,
            verb=Verb.EXECUTE,
            resource_class=ResourceClass.TOOL,
            target=tool_id,
            action_fn=handler,
            arguments=args,
            approval_id=approval_id,
            trace_id=trace_id,
        )
        return result
