"""Task manifest schema v1 (stub).

Spec Section 7: each task specifies dependencies, partitioning strategy,
artifact IDs, numerical compatibility, resource floor and estimated peak,
checkpoint granularity, maximum attempts, timeout and output format.
Sprint 0 validates structure only; richer checks arrive with the planner.
"""

from __future__ import annotations

REQUIRED_FIELDS = ("id", "task_type", "task_version", "max_attempts", "timeout_seconds")

VALID_PARTITION_STRATEGIES = frozenset(
    {"single", "by_example_ids", "by_sequence", "by_layer_group", "by_matrix"}
)


class TaskManifestError(ValueError):
    """The payload is not a valid task manifest."""


def validate(payload: dict) -> list[str]:
    """Return the list of problems; an empty list means the manifest is valid."""
    if not isinstance(payload, dict):
        return ["task manifest must be a JSON object"]
    problems: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            problems.append(f"missing required field: {field}")
    for field in ("max_attempts", "timeout_seconds"):
        value = payload.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 1
        ):
            problems.append(f"{field} must be a positive integer")
    if "id" in payload and not isinstance(payload["id"], str):
        problems.append("id must be a string")
    if "task_type" in payload and not isinstance(payload["task_type"], str):
        problems.append("task_type must be a string")
    partition = payload.get("partition")
    if partition is not None and partition not in VALID_PARTITION_STRATEGIES:
        problems.append(f"unknown partition strategy: {partition}")
    return problems


def parse(payload: dict) -> dict:
    """Validate and return the manifest; raise TaskManifestError if invalid."""
    problems = validate(payload)
    if problems:
        raise TaskManifestError("; ".join(problems))
    return payload
