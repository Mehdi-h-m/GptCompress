"""Record state enums (spec Sections 5, 6 and 9)."""

from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    """Lifecycle of a job (spec Section 9)."""

    QUEUED = "queued"
    STAGING = "staging"
    READY = "ready"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETE = "complete"
    RETRY_PENDING = "retry_pending"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_USER = "awaiting_user"
    RECONCILING = "reconciling"


JOB_TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}
)


class OperationState(StrEnum):
    """Lifecycle of a coordinator operation, the unit ``drm wait`` blocks on."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_USER = "awaiting_user"
    RECONCILING = "reconciling"


OPERATION_TERMINAL_STATES: frozenset[OperationState] = frozenset(
    {OperationState.COMPLETE, OperationState.FAILED, OperationState.CANCELLED}
)


class CopyStatus(StrEnum):
    """Physical copy status (spec Section 6)."""

    DURABLE = "durable"
    STAGED = "staged"
    REBUILDABLE = "rebuildable"


class ExecutionMode(StrEnum):
    """How a provider executes work (spec Section 5)."""

    API = "api"
    ASSISTED = "assisted"
    MANUAL = "manual"


class ProviderInstanceState(StrEnum):
    """Configured provider instance state (spec Section 2, Providers screen)."""

    ACTIVE = "active"
    PAUSED = "paused"
    DRAINING = "draining"
    UNREACHABLE = "unreachable"
