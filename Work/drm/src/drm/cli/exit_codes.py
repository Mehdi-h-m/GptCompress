"""Stable exit codes (spec Section 4, rule 3).

Agents branch on exit codes without reading prose:
0 success, 1 unexpected error, 2 blocked by policy/budget/validation,
3 awaiting a user action, 4 reconciliation required.
"""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    SUCCESS = 0
    UNEXPECTED_ERROR = 1
    POLICY_BLOCKED = 2
    AWAITING_USER = 3
    RECONCILE_REQUIRED = 4
