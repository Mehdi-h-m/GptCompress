"""Local-machine compute adapter (spec Section 5, 'Local machine' profile).

Sprint 1 registers and probes the local runner; execution arrives with the
Sprint 2 planner, so the job verbs raise instead of pretending to run.
"""

from __future__ import annotations

import os
import platform
import sys

from drm.providers.base import AdapterError, ComputeAdapter
from drm.store.db import utcnow


class LocalCompute(ComputeAdapter):
    adapter_name = "local-runner"

    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def discover(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "local"),
            "advertised": {
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
            },
            "observed": {},
        }

    def probe(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "local"),
            "observed": {
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "python": sys.version.split()[0],
            },
            "observed_at": utcnow(),
        }

    def quote(self, request: dict) -> dict:
        raise AdapterError("local quoting arrives with the Sprint 6 planner")

    def submit(self, task_bundle: dict) -> dict:
        raise AdapterError("local execution arrives in Sprint 2")

    def status(self, attempt_id: str) -> dict:
        raise AdapterError("local execution arrives in Sprint 2")

    def cancel(self, attempt_id: str) -> dict:
        raise AdapterError("local execution arrives in Sprint 2")

    def list_usage(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.config.get("name", "local"),
            "entries": [],
            "unverified_manual_entries": [],
            "settlement_delay": None,
        }
