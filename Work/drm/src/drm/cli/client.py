"""Minimal HTTP client for the loopback daemon (stdlib only)."""

from __future__ import annotations

import json
import urllib.request


def get_json(url: str, timeout: float = 3.0) -> dict:
    """GET a JSON document; raises URLError/OSError/ValueError on failure."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)
