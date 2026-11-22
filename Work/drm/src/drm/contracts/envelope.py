"""Versioned JSON envelope and error records (spec Section 4, rule 2).

Every machine-facing document carries a ``contract`` field so consumers can
detect schema generations. JSON mode never embeds prose a consumer must
parse; humans get the same facts rendered as tables by the CLI.
"""

from __future__ import annotations

from typing import Any

CONTRACT_VERSION = 1


class ErrorCodes:
    """Stable error codes for JSON error records."""

    DAEMON_UNREACHABLE = "daemon_unreachable"
    UNEXPECTED_ERROR = "unexpected_error"
    POLICY_BLOCKED = "policy_blocked"
    AWAITING_USER = "awaiting_user"
    RECONCILE_REQUIRED = "reconcile_required"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"


def envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Add the contract version to a payload document."""
    return {"contract": CONTRACT_VERSION, **payload}


def error_record(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Build an enveloped error record with a stable code."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return envelope({"error": error})
