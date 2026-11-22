"""drm wait and drm explain — blocking waits and decision transparency (spec Section 4)."""

from __future__ import annotations

import typer

from drm.cli.common import JSON_OPTION, CliState, fail, request, resolve_state, wait_for_operation
from drm.cli.output import print_payload


def wait(
    ctx: typer.Context,
    operation_id: str = typer.Argument(...),
    timeout: int = typer.Option(600, "--timeout", help="Seconds to wait for a terminal state"),
    json_mode: bool = JSON_OPTION,
) -> None:
    """Block until an operation reaches a terminal state, then print its record."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = wait_for_operation(state, operation_id, timeout)
    operation = doc["operation"]
    if operation["state"] == "failed":
        error = operation.get("error") or {}
        fail(
            state,
            error.get("code", "unexpected_error"),
            error.get("message", "operation failed"),
            operation=operation_id,
        )
    result = operation.get("result") or {}
    lines = []
    if result.get("published"):
        for item in result["published"]:
            lines.append(f"published {item['artifact']['id']} ({item['artifact']['kind']})")
    if result.get("job_id"):
        lines.append(f"job {result['job_id']} complete")
    human = "\n".join(lines) or f"operation {operation_id}: {operation['state']}"
    print_payload(state.json_mode, doc, human=human)


def explain(
    ctx: typer.Context,
    decision_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
) -> None:
    """Replay a recorded planner decision: inputs, alternatives, chosen, reason."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "GET", f"/decisions/{decision_id}")
    decision = doc["decision"]
    alternatives = decision.get("alternatives", [])
    selected = [a["provider_id"] for a in alternatives if a.get("selected")]
    human = "\n".join(
        [
            f"decision {decision['id']}  subject {decision['subject']}/{decision['subject_id']}",
            f"  chosen        {selected or decision.get('chosen')}",
            f"  reason        {decision.get('reason') or '-'}",
            f"  alternatives  {len(alternatives)} candidates considered",
        ]
    )
    print_payload(state.json_mode, doc, human=human)
