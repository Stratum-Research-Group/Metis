"""Component status model with canonical 5-state contract and probe tracking."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ComponentState(str, Enum):
    LIVE = "live"
    SIMULATED = "simulated"
    MOCK = "mock"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class Freshness(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


class ComponentStatus(BaseModel):
    """
    Truthful component status adhering to the Metis Dashboard Contract.
    Enforces the 5 canonical states: live | simulated | mock | offline | unknown.
    """
    state: ComponentState = Field(..., description="Canonical component state")
    detail: str = Field(default="", description="Short human-readable status detail")
    freshness: Freshness = Field(default=Freshness.UNKNOWN, description="Freshness of the probe result")
    version: Optional[str] = Field(default=None, description="Reported or detected component version")
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of last probe check",
    )
    latency_ms: Optional[int] = Field(default=None, description="Latency of probe in milliseconds")
    evidence: str = Field(default="", description="Evidence supporting the reported state")
    is_simulated: bool = Field(default=False, description="Whether execution runs via simulated adapter")
    
    # Backward compatibility fields for legacy clients
    status: str = Field(default="unknown", description="Legacy status string")
    mode: str = Field(default="unknown", description="Legacy mode string")

    def model_post_init(self, __context: Any) -> None:
        # Sync backward compatibility fields if not explicitly provided
        if self.status == "unknown":
            if self.state == ComponentState.LIVE:
                self.status = "ready"
            elif self.state == ComponentState.SIMULATED:
                self.status = "simulated"
            elif self.state == ComponentState.OFFLINE:
                self.status = "offline"
            else:
                self.status = str(self.state.value)
        if self.mode == "unknown":
            self.mode = str(self.state.value)
