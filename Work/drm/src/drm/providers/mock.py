"""Mock providers with failure injection.

Spec Section 10: use mock providers for failure tests before relying on a
real account. The mocks here back every later sprint's fault tests —
interruption, duplicate results and corrupt transfers — through a
deterministic injection queue rather than a real provider.
"""

from __future__ import annotations

from drm.providers.base import (
    AdapterError,
    AmbiguousOutcome,
    ComputeAdapter,
    StorageAdapter,
    sha256_hex,
)
from drm.store.db import utcnow


class FailureInjector:
    """Queue deterministic failures onto adapter methods for tests."""

    def __init__(self) -> None:
        self._rules: dict[str, list[tuple[int, Exception]]] = {}

    def fail_on(self, method: str, error: Exception | None = None, times: int = 1) -> None:
        """Make the next ``times`` calls of ``method`` raise ``error``."""
        self._rules.setdefault(method, []).append(
            (times, error or AdapterError(f"injected failure: {method}"))
        )

    def check(self, method: str) -> None:
        rules = self._rules.get(method)
        if not rules:
            return
        times, error = rules[0]
        if times <= 0:
            rules.pop(0)
            return
        rules[0] = (times - 1, error)
        raise error


class MockCompute(ComputeAdapter):
    """In-memory compute provider with canned capabilities and quota."""

    adapter_name = "mock"

    def __init__(self, name: str = "mock-a", capabilities: dict | None = None) -> None:
        self.name = name
        self.capabilities = capabilities or {
            "gpu": "mock-gpu",
            "vram_mib": 4096,
            "precisions": ["fp32", "bf16"],
        }
        self.injector = FailureInjector()
        self._attempts: dict[str, dict] = {}
        self._counter = 0
        self._ambiguous_once = False

    def discover(self) -> dict:
        return {
            "adapter": self.adapter_name,
            "provider": self.name,
            "advertised": dict(self.capabilities),
            "observed": {},
        }

    def probe(self) -> dict:
        self.injector.check("probe")
        return {
            "adapter": self.adapter_name,
            "provider": self.name,
            "observed": dict(self.capabilities),
            "observed_at": utcnow(),
        }

    def quote(self, request: dict) -> dict:
        self.injector.check("quote")
        return {
            "adapter": self.adapter_name,
            "provider": self.name,
            "currency": "USD",
            "expires_at": None,
            "billing_increment_seconds": 1,
            "prices": {
                "compute_per_hour": 0.0,
                "storage_gb_month": 0.0,
                "transfer_gb": 0.0,
            },
            "credits": {"eligible": False, "available": None},
            "upper_estimate": {"amount": 0.0, "assumptions": ["free mock tier"]},
            "unknowns": ["settlement_delay"],
        }

    def submit(self, task_bundle: dict) -> dict:
        self.injector.check("submit")
        self._counter += 1
        attempt_id = f"{self.name}-attempt-{self._counter}"
        self._attempts[attempt_id] = {
            "attempt_id": attempt_id,
            "request_id": attempt_id,
            "state": "queued",
            "task_type": task_bundle.get("task_type"),
        }
        if self._ambiguous_once:
            self._ambiguous_once = False
            raise AmbiguousOutcome(
                f"{self.name} may have accepted the submission; reconcile before retrying"
            )
        return dict(self._attempts[attempt_id])

    def ambiguous_next_submit(self) -> None:
        """Make the next submit record an attempt and then raise AmbiguousOutcome."""
        self._ambiguous_once = True

    def status(self, attempt_id: str) -> dict:
        self.injector.check("status")
        if attempt_id not in self._attempts:
            raise AdapterError(f"unknown attempt {attempt_id}")
        return dict(self._attempts[attempt_id])

    def set_attempt_state(self, attempt_id: str, state: str) -> None:
        """Test helper: move an attempt to an arbitrary state."""
        self._attempts[attempt_id]["state"] = state

    def cancel(self, attempt_id: str) -> dict:
        self.injector.check("cancel")
        if attempt_id not in self._attempts:
            raise AdapterError(f"unknown attempt {attempt_id}")
        self._attempts[attempt_id]["state"] = "cancelled"
        return {"attempt_id": attempt_id, "cancelled": True, "confirmed": True}

    def list_usage(self) -> dict:
        self.injector.check("list_usage")
        return {
            "adapter": self.adapter_name,
            "provider": self.name,
            "entries": [],
            "unverified_manual_entries": [],
            "settlement_delay": None,
        }


class MockStorage(StorageAdapter):
    """In-memory object store with checksum verification on every path."""

    adapter_name = "mock-storage"

    def __init__(self, name: str = "mock-store") -> None:
        self.name = name
        self.injector = FailureInjector()
        self._objects: dict[str, bytes] = {}
        self._digests: dict[str, str] = {}

    def head(self, locator: str) -> dict:
        self.injector.check("head")
        exists = locator in self._objects
        return {
            "locator": locator,
            "exists": exists,
            "byte_count": len(self._objects.get(locator, b"")) if exists else None,
            "checksum": self._digests.get(locator),
        }

    def write(self, locator: str, data: bytes, checksum: str | None = None) -> dict:
        self.injector.check("write")
        digest = sha256_hex(data)
        if checksum is not None and checksum != digest:
            raise AdapterError(
                f"checksum mismatch writing {locator}: expected {checksum}, got {digest}"
            )
        self._objects[locator] = bytes(data)
        self._digests[locator] = digest
        return {"locator": locator, "checksum": digest, "byte_count": len(data)}

    def read(self, locator: str) -> bytes:
        self.injector.check("read")
        if locator not in self._objects:
            raise AdapterError(f"no object at {locator}")
        data = self._objects[locator]
        if sha256_hex(data) != self._digests[locator]:
            raise AdapterError(f"stored bytes at {locator} fail checksum verification")
        return data

    def list(self, prefix: str = "") -> list[str]:
        self.injector.check("list")
        return sorted(key for key in self._objects if key.startswith(prefix))

    def delete(self, locator: str) -> dict:
        self.injector.check("delete")
        if locator not in self._objects:
            raise AdapterError(f"no object at {locator}")
        del self._objects[locator]
        self._digests.pop(locator, None)
        return {"locator": locator, "deleted": True}

    def corrupt(self, locator: str) -> None:
        """Test hook: mutate stored bytes so read-time verification fails."""
        data = bytearray(self._objects[locator])
        data[-1] ^= 0xFF
        self._objects[locator] = bytes(data)
