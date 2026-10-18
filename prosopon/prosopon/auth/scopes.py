"""Scope definitions and role-based presets for Prosopon access control."""

from typing import List, Optional

VALID_SCOPES: List[str] = [
    "chat:read",
    "chat:operate",
    "workbench:read",
    "workbench:write",
    "agent:operate",
    "approvals:resolve",
    "approvals:resolve_high",
    "squad:read",
    "squad:operate",
    "pinax:read",
    "settings:read",
    "settings:write",
    "models:operate",
    "diagnostics:read",
    "access:read",
    "access:write",
    "remote:configure",
]

PRESET_SCOPES = {
    "viewer": [
        "chat:read",
        "workbench:read",
        "squad:read",
        "diagnostics:read",
    ],
    "companion": [
        "chat:read",
        "workbench:read",
        "squad:read",
        "diagnostics:read",
        "chat:operate",
    ],
    "operator": [
        "chat:read",
        "workbench:read",
        "squad:read",
        "diagnostics:read",
        "chat:operate",
        "agent:operate",
        "squad:operate",
        "approvals:resolve",
        "models:operate",
    ],
    "owner": list(VALID_SCOPES),
}


def get_scopes_for_preset(preset: str, custom_scopes: Optional[List[str]] = None) -> List[str]:
    """Resolve granted scopes for a given preset or custom list."""
    preset_lower = preset.lower().strip()
    if preset_lower in PRESET_SCOPES:
        return list(PRESET_SCOPES[preset_lower])
    if custom_scopes:
        return [s for s in custom_scopes if s in VALID_SCOPES]
    return list(PRESET_SCOPES["companion"])


def has_scope(granted_scopes: List[str], required_scope: str) -> bool:
    """Check if granted scopes satisfy required scope."""
    if "owner" in granted_scopes or "access:write" in granted_scopes:
        # Owner full access
        return True
    return required_scope in granted_scopes
