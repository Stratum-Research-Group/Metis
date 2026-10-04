"""Guarded send/outbound network adapter."""

from typing import Any, Callable

from phylax.contracts.enums import ResourceClass, Verb
from phylax.gateway.gateway import GuardedGateway


class GuardedSendAdapter:
    """Brokers outbound communications (HTTP, webhooks, messages) through Phylax."""

    def __init__(self, gateway: GuardedGateway):
        self.gateway = gateway

    def send(
        self,
        destination: str,
        payload: dict[str, Any],
        sender_fn: Callable[..., Any],
        subject: str = "kybernetes",
        approval_id: str | None = None,
        trace_id: str | None = None,
    ) -> Any:
        """Authorize and send payload to destination."""
        result, _ = self.gateway.authorize_and_execute(
            subject=subject,
            verb=Verb.SEND,
            resource_class=ResourceClass.NETWORK,
            target=destination,
            action_fn=sender_fn,
            arguments=payload,
            approval_id=approval_id,
            trace_id=trace_id,
        )
        return result
