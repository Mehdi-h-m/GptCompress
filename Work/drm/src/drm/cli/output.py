"""Output helpers: one versioned JSON document, or a human-readable line.

JSON mode always emits a single enveloped document on stdout and never
embeds prose a consumer must parse (spec Section 4, rule 2). Human mode
prints tables or short ``error [code]: message`` lines.
"""

from __future__ import annotations

import json

from drm.contracts import envelope, error_record


def print_payload(json_mode: bool, payload: dict, human: str | None = None) -> None:
    if json_mode:
        print(json.dumps(envelope(payload), indent=2))
    else:
        print(human if human is not None else json.dumps(envelope(payload), indent=2))


def print_error(json_mode: bool, code: str, message: str, **details) -> None:
    if json_mode:
        print(json.dumps(error_record(code, message, **details), indent=2))
    else:
        print(f"error [{code}]: {message}")
