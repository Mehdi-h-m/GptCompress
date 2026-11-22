"""Minimal HTTP client for the loopback daemon (stdlib only)."""

from __future__ import annotations

import json
import urllib.request


def get_json(url: str, timeout: float = 3.0) -> dict:
    """GET a JSON document; raises URLError/OSError/ValueError on failure."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post_json(url: str, payload: dict, timeout: float = 10.0) -> dict:
    """POST a JSON document and return the JSON response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)
