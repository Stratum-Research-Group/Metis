"""Kybernetes orchestrator integration shim."""

from pathlib import Path
from typing import Any, Callable

from phylax.contracts.enums import ResourceClass, Verb
from phylax.gateway.gateway import GuardedGateway
from phylax.gateway.file_adapter import GuardedFileAdapter
from phylax.gateway.tool_adapter import GuardedToolAdapter
from phylax.gateway.send_adapter import GuardedSendAdapter


class KybernetesShim:
    """Standardized capability broker for Kybernetes orchestrator."""

    def __init__(self, gateway: GuardedGateway):
        self.gateway = gateway
        self.files = GuardedFileAdapter(gateway)
        self.tools = GuardedToolAdapter(gateway)
        self.senders = GuardedSendAdapter(gateway)

    def read_file(
        self,
        path: Path | str,
        trace_id: str | None = None,
        approval_id: str | None = None,
    ) -> str:
        """Read a local file subject to Phylax policy."""
        return self.files.read_text(
            path=path,
            subject="kybernetes",
            approval_id=approval_id,
            trace_id=trace_id,
        )

    def write_file(
        self,
        path: Path | str,
        content: str,
        trace_id: str | None = None,
        approval_id: str | None = None,
    ) -> int:
        """Write a local file subject to Phylax policy."""
        return self.files.write_text(
            path=path,
            content=content,
            subject="kybernetes",
            approval_id=approval_id,
            trace_id=trace_id,
        )

    def invoke_tool(
        self,
        tool_id: str,
        arguments: dict[str, Any] | None = None,
        trace_id: str | None = None,
        approval_id: str | None = None,
    ) -> Any:
        """Execute a registered tool subject to Phylax policy."""
        return self.tools.invoke(
            tool_id=tool_id,
            arguments=arguments,
            subject="kybernetes",
            approval_id=approval_id,
            trace_id=trace_id,
        )

    def send(
        self,
        destination: str,
        payload: dict[str, Any],
        sender_fn: Callable[..., Any],
        trace_id: str | None = None,
        approval_id: str | None = None,
    ) -> Any:
        """Send an outbound network message subject to Phylax policy."""
        return self.senders.send(
            destination=destination,
            payload=payload,
            sender_fn=sender_fn,
            subject="kybernetes",
            approval_id=approval_id,
            trace_id=trace_id,
        )
