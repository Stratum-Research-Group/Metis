"""Prosopon approval adapter.

Bridges Phylax approval requests to Prosopon's ApprovalCard schema.
"""

from typing import Any
from phylax.approvals.manager import ApprovalManager


class ProsoponApprovalAdapter:
    """Adapts Phylax pending requests for the Prosopon UI and SSE stream."""

    def __init__(self, approval_manager: ApprovalManager):
        self.approval_manager = approval_manager

    def get_pending_cards(self) -> list[dict[str, Any]]:
        """Return list of pending approval cards formatted for Prosopon frontend."""
        pending = self.approval_manager.list_pending()
        cards: list[dict[str, Any]] = []
        for req in pending:
            cards.append(
                {
                    "approval_id": req.approval_id,
                    "turn_id": req.turn_id,
                    "action": req.action,
                    "target": req.target,
                    "argument_summary": req.argument_summary,
                    "argument_digest": req.argument_digest,
                    "policy_rule": req.policy_rule,
                    "expiry_utc": req.expiry_utc,
                }
            )
        return cards

    def resolve(
        self,
        approval_id: str,
        approved: bool,
        reason: str | None = None,
        resolved_by: str = "owner",
    ) -> dict[str, Any]:
        """Process owner decision from Prosopon UI."""
        res = self.approval_manager.resolve_request(
            approval_id=approval_id,
            approved=approved,
            reason=reason,
            resolved_by=resolved_by,
        )
        return res.model_dump()
