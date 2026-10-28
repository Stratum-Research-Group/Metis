"""Speculative Cascades Engine for Hestia.

Implements drafter/verifier serving cascades with early deferral rules and
cost-quality telemetry (ICLR 2025, 'Faster Cascades via Speculative Decoding').
"""

from hestia.cascade.engine import (
    CascadeConfig,
    CascadeDecision,
    SpeculativeCascadeEngine,
)

__all__ = [
    "CascadeConfig",
    "CascadeDecision",
    "SpeculativeCascadeEngine",
]
