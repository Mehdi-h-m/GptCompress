"""Provider adapter contract and registry (spec Section 5)."""

from drm.providers.base import (
    AdapterError,
    AmbiguousOutcome,
    ComputeAdapter,
    StorageAdapter,
)
from drm.providers.mock import FailureInjector, MockCompute, MockStorage

__all__ = [
    "AdapterError",
    "AmbiguousOutcome",
    "ComputeAdapter",
    "FailureInjector",
    "MockCompute",
    "MockStorage",
    "StorageAdapter",
]
