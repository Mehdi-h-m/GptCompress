"""Short, prefixed identifiers for records, e.g. ``op-1a2b3c4d``."""

from __future__ import annotations

import secrets


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(4)}"
