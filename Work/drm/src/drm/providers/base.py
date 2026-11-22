"""Provider adapter contract (spec Section 5).

Adapters separate the adapter type from configured instances: two accounts
on one service are two instances of the same adapter, independently
selectable by the scheduler. Compute and storage capabilities are
configured independently. Execution is declared as api, assisted or manual;
unsupported features stay unavailable to the planner.

Required behavior (spec Section 5 table):

- ``discover``/``probe``: account/session eligibility, accelerator type and
  count, VRAM per device, supported precisions and operations, CPU/RAM/disk,
  queue and session limits, concurrency, network reachability and quota.
  Record timestamps and distinguish advertised values from observations.
- ``quote``: currency, quote expiration, billing increments, known prices,
  credit eligibility and balances, upper cost estimates with assumptions.
  Represent unknown values explicitly.
- ``submit``/``status``/``cancel``: external attempt ID or manual launch
  instructions, stable request ID where available, cancellation support and
  confirmation. On an ambiguous submission, reconcile before retrying —
  raise :class:`AmbiguousOutcome` rather than guessing.
- ``list_usage``: usage and invoice/balance events where supported,
  settlement delays, unverified manual entries kept separate.
- ``head``/``read``/``write``/``list``/``delete``: inspect, stream or copy
  artifacts with version/size/checksum verification. ``delete`` acts on
  specific copies after catalog checks.
"""

from __future__ import annotations

import abc
import hashlib


class AdapterError(Exception):
    """Provider adapter failure with no ambiguity about the outcome."""


class AmbiguousOutcome(AdapterError):
    """The provider may have acted; reconcile provider state before retrying."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class ComputeAdapter(abc.ABC):
    """Compute face of a provider instance."""

    adapter_name: str = "abstract"

    @abc.abstractmethod
    def discover(self) -> dict:
        """Advertised account limits and capabilities, without side effects."""

    @abc.abstractmethod
    def probe(self) -> dict:
        """Run live checks and return observations with timestamps."""

    @abc.abstractmethod
    def quote(self, request: dict) -> dict:
        """Price a workload shape; unknowns explicit, upper estimate included."""

    @abc.abstractmethod
    def submit(self, task_bundle: dict) -> dict:
        """Submit a bounded task bundle; returns an attempt record."""

    @abc.abstractmethod
    def status(self, attempt_id: str) -> dict:
        """Report the current state of one external attempt."""

    @abc.abstractmethod
    def cancel(self, attempt_id: str) -> dict:
        """Request remote stop and report whether it is confirmed."""

    @abc.abstractmethod
    def list_usage(self) -> dict:
        """Usage and billing events, with unverified entries separated."""


class StorageAdapter(abc.ABC):
    """Storage face of a provider instance."""

    adapter_name: str = "abstract"

    @abc.abstractmethod
    def head(self, locator: str) -> dict:
        """Inspect one object: existence, size, version, checksum."""

    @abc.abstractmethod
    def read(self, locator: str) -> bytes:
        """Stream or read one object, verifying content identity."""

    @abc.abstractmethod
    def write(self, locator: str, data: bytes, checksum: str | None = None) -> dict:
        """Write one object atomically where supported; verify checksums."""

    @abc.abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List object locators under a prefix."""

    @abc.abstractmethod
    def delete(self, locator: str) -> dict:
        """Delete one object; callers check catalog references first."""
