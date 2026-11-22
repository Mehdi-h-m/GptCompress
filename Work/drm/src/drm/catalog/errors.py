"""Catalog failures carrying stable error codes.

Codes map to CLI exit codes: ``validation`` and ``policy_blocked`` exit 2,
``not_found``, ``verify_failed`` and ``adapter_error`` exit 1 (spec
Section 4, rule 3).
"""

from __future__ import annotations


class CatalogError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
