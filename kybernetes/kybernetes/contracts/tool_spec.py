"""Tool specification and execution contracts."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from kybernetes.contracts.enums import ToolSideEffectClass


class ToolDefinition(BaseModel):
    """Declarative specification of a safe Kybernetes tool."""
    name: str = Field(..., description="Unique tool identifier, e.g., 'calculator'")
    version: str = Field(default="1.0.0", description="Semantic tool version")
    description: str = Field(..., description="Human-readable description of what the tool does")
    parameters_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing expected arguments",
    )
    returns_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema describing return structure",
    )
    declared_capabilities: List[str] = Field(
        default_factory=list,
        description="Phylax capabilities required, e.g. ['read_file', 'compute']",
    )
    side_effect_class: ToolSideEffectClass = Field(
        default=ToolSideEffectClass.READ_ONLY,
        description="Side effect categorization",
    )
    timeout_ms: int = Field(default=5000, description="Execution timeout in ms")
    idempotent: bool = Field(default=True, description="Whether re-execution with same args is side-effect safe")


class ToolCallRequest(BaseModel):
    """Invocation request for a tool."""
    tool_id: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None
    approval_id: Optional[str] = None


class ToolResult(BaseModel):
    """Result of a tool execution."""
    tool_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    @property
    def output(self) -> Any:
        return self.data

