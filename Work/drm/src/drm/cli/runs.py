"""drm run and drm jobs — submit, watch, cancel, retry (spec Sections 2 and 9)."""

from __future__ import annotations

import typer

from pathlib import Path

from drm.cli.common import (
    DRY_RUN_OPTION,
    IDEMPOTENCY_OPTION,
    JSON_OPTION,
    CliState,
    ExitCode,
    fail,
    request,
    resolve_state,
)
from drm.cli.output import print_payload
from drm.cli.plan import _build_payload, _render_plan_human
from drm.contracts import ErrorCodes

runs_app = typer.Typer(no_args_is_help=True, help="Submit and manage job runs")
jobs_app = typer.Typer(no_args_is_help=True, help="Retry failed jobs")


def _fail_on_failed_operation(state: CliState, operation: dict) -> None:
    if operation.get("state") == "failed":
        error = operation.get("error") or {}
        fail(
            state,
            error.get("code", ErrorCodes.UNEXPECTED_ERROR),
            error.get("message", "operation failed"),
            operation=operation.get("id"),
        )


def _job_line(job: dict) -> str:
    return f"{job['id']}  {job['state']:<12} {job.get('name') or '-'}"


@runs_app.command("submit")
def submit(
    ctx: typer.Context,
    task: str = typer.Option(None, "--task"),
    inputs: list[str] = typer.Option(None, "--input"),
    params: list[str] = typer.Option(None, "--param"),
    store: str = typer.Option(None, "--store"),
    compute: str = typer.Option(None, "--compute", help="Preferred compute provider id"),
    name: str = typer.Option(None, "--name"),
    experiment: str = typer.Option(None, "--experiment"),
    max_attempts: int = typer.Option(None, "--max-attempts"),
    timeout: int = typer.Option(None, "--timeout"),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    idempotency_key: str = IDEMPOTENCY_OPTION,
) -> None:
    """Submit a job: plan it, place it and dispatch the first attempt."""
    state: CliState = resolve_state(ctx, json_mode)
    payload = _build_payload(task, inputs, params, store, name, experiment, max_attempts, timeout)
    if compute is not None:
        payload["compute_provider"] = compute
    payload["dry_run"] = dry_run
    payload["idempotency_key"] = idempotency_key
    doc = request(state, "POST", "/run/submit", payload, timeout=60.0)
    if doc.get("dry_run"):
        print_payload(state.json_mode, doc, human=_render_plan_human(doc["plan"]))
        return
    if doc.get("reused"):
        operation = doc["operation"]
        _fail_on_failed_operation(state, operation)
        print_payload(
            state.json_mode, doc,
            human=f"existing run returned (idempotency key): operation {operation['id']}",
        )
        return
    if doc["job"]["state"] == "awaiting_user":
        # exit 3 (spec Section 4, rule 3): an assisted step awaits the user
        result = doc["attempt"].get("result") or {}
        fail(
            state,
            "awaiting_user",
            "notebook bundle prepared: run it on Kaggle, then drm run import",
            exit_code=ExitCode.AWAITING_USER,
            job_id=doc["job"]["id"],
            attempt_id=doc["attempt"]["id"],
            bundle=result.get("bundle"),
            bundle_dir=result.get("bundle_dir"),
            instructions=result.get("instructions"),
            outputs_expected=result.get("outputs_expected"),
            items_in_bundle=result.get("items_in_bundle"),
        )
        return
    print_payload(
        state.json_mode, doc,
        human=(
            f"submitted {_job_line(doc['job'])}"
            f" attempt {doc['attempt']['id']}"
            f" operation {doc['operation']['id']} (drm wait)"
        ),
    )


@runs_app.command("import")
def import_results(
    ctx: typer.Context,
    attempt_id: str = typer.Argument(...),
    output: Path = typer.Option(..., "--output", help="Downloaded notebook output folder"),
    json_mode: bool = JSON_OPTION,
) -> None:
    """Import declared notebook output files and validate them against the manifest."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(
        state, "POST", f"/attempts/{attempt_id}/import",
        {"output_dir": str(output)}, timeout=60.0,
    )
    result = doc["import"]
    if result.get("already_complete"):
        human = f"attempt {attempt_id} was already accepted; nothing re-imported (reconciled)"
    elif result.get("reconciled"):
        human = f"attempt {attempt_id} reconciled without duplicating outputs"
    else:
        names = ", ".join(i["artifact"]["id"] for i in result.get("published", []))
        human = f"import accepted: published {names}"
        if result.get("undeclared_files_ignored"):
            human += f" (ignored undeclared: {', '.join(result['undeclared_files_ignored'])})"
    print_payload(state.json_mode, doc, human=human)


@runs_app.command("status")
def status(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
) -> None:
    """Show one job with its attempts."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "GET", f"/run/{job_id}")
    job = doc["job"]
    lines = [_job_line(job)]
    for attempt in doc.get("attempts", []):
        error = attempt.get("error") or {}
        suffix = f" error={error.get('code')}" if error else ""
        lines.append(f"  attempt {attempt['id']}  {attempt['state']:<12}{suffix}")
        result = attempt.get("result") or {}
        if attempt["state"] == "awaiting_user" and result.get("bundle"):
            lines.append(f"    bundle       {result['bundle']}")
            lines.append(f"    instructions {result.get('instructions')}")
            if result.get("outputs_expected"):
                lines.append(f"    outputs      {', '.join(result['outputs_expected'])}")
    print_payload(state.json_mode, doc, human="\n".join(lines))


@runs_app.command("cancel")
def cancel(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Request cancellation of a non-terminal job."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "POST", f"/run/{job_id}/cancel", {"dry_run": dry_run})
    if doc.get("dry_run"):
        human = f"dry run: would cancel {doc['plan']['job_id']}"
    elif doc.get("already_cancelled"):
        human = f"job {doc['job']['id']} was already cancelled"
    else:
        human = f"cancel requested for {doc['job_id']} (state {doc['state']})"
    print_payload(state.json_mode, doc, human=human)


@jobs_app.command("retry")
def retry(
    ctx: typer.Context,
    job_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
) -> None:
    """Manually retry a failed or cancelled job with a fresh attempt."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(state, "POST", f"/jobs/{job_id}/retry", {})
    human = (
        f"retry dispatched {_job_line(doc['job'])}"
        f" attempt {doc['attempt']['id']} operation {doc['operation']['id']} (drm wait)"
    )
    print_payload(state.json_mode, doc, human=human)
