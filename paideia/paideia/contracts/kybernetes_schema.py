"""Kybernetes route definition and permitted context contract."""

from typing import List, Optional
from pydantic import BaseModel, Field


class KybernetesRouteDefinition(BaseModel):
    """Versioned route set and allowed context features exposed by Kybernetes."""
    version: str = Field(default="1.0.0", description="Schema version of route definition")
    route_set_id: str = Field(..., description="Unique identifier for this route configuration")
    routes: List[str] = Field(..., min_length=2, description="List of valid route identifiers")
    permitted_context_features: List[str] = Field(
        default_factory=lambda: ["query_text", "query_length", "entity_count"],
        description="Explicitly permitted context features (prevent leakage of prohibited signals)",
    )
    fallback_route: str = Field(..., description="Default fallback route if routing fails or abstains")

    def is_valid_route(self, route: str) -> bool:
        return route in self.routes
