"""drm data — ingest, list, pin, evict, verify (spec Sections 2 and 6)."""

from __future__ import annotations

from pathlib import Path

import typer

from drm.cli.common import (
    DRY_RUN_OPTION,
    IDEMPOTENCY_OPTION,
    JSON_OPTION,
    CliState,
    fail,
    request,
    resolve_state,
    wait_for_operation,
)
from drm.cli.output import print_payload
from drm.contracts import ErrorCodes

data_app = typer.Typer(no_args_is_help=True, help="Ingest, inspect and manage artifacts")

TERMINAL_STATES = {"complete", "failed", "cancelled"}


def _artifact_lines(artifacts: list) -> list[str]:
    lines = []
    for artifact in artifacts:
        pinned = " pinned" if artifact.get("pinned") else ""
        lines.append(
            f"{artifact['id']}  {artifact['kind']:<16} {artifact['byte_count'] or 0:>12} B"
            f"  sha256:{artifact['content_hash'][:12]}{pinned}"
        )
        for copy in artifact.get("copies", []):
            lines.append(
                f"    copy {copy['id']}  {copy['provider_id']}  {copy['status']:<8}"
                f"  {copy['locator']}"
            )
    return lines


def _fail_on_failed_operation(state: CliState, operation: dict) -> None:
    if operation.get("state") == "failed":
        error = operation.get("error") or {}
        fail(
            state,
            error.get("code", ErrorCodes.UNEXPECTED_ERROR),
            error.get("message", "operation failed"),
            operation=operation.get("id"),
        )


@data_app.command("ingest")
def ingest(
    ctx: typer.Context,
    source: str = typer.Argument(..., help="Local file path or hf://datasets/<owner>/<repo>[@<rev>]/<file>"),
    store: str = typer.Option(..., "--store", help="Provider id of the destination store"),
    kind: str = typer.Option("dataset_release", "--kind"),
    fmt: str = typer.Option(None, "--format"),
    split_identity: str = typer.Option(None, "--split"),
    access_class: str = typer.Option("private", "--access-class"),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
    artifact_id: str = typer.Option(None, "--id", help="Custom logical artifact id"),
    release: str = typer.Option(None, "--release", help="Alias for --id (e.g. --release calib-v1)"),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    idempotency_key: str = IDEMPOTENCY_OPTION,
    timeout: int = typer.Option(600, "--timeout", help="Seconds to wait for the ingest to finish"),
) -> None:
    """Register a local file or a pinned hf:// shard as a verified artifact."""
    state: CliState = resolve_state(ctx, json_mode)
    payload = {
        "source": source,
        "artifact_id": artifact_id or release,
        "store": store,
        "kind": kind,
        "format": fmt,
        "split_identity": split_identity,
        "access_class": access_class,
        "verify": verify,
        "dry_run": dry_run,
        "idempotency_key": idempotency_key,
    }
    doc = request(state, "POST", "/data/ingest", payload, timeout=15.0)
    if not dry_run:
        operation = doc.get("operation", {})
        if operation.get("state") not in TERMINAL_STATES:
            doc = wait_for_operation(state, operation["id"], timeout)
        _fail_on_failed_operation(state, doc.get("operation", {}))
    print_payload(state.json_mode, doc, human=_ingest_human(doc))


def _ingest_human(doc: dict) -> str:
    if doc.get("dry_run"):
        plan = doc["plan"]
        return (
            f"dry run: would ingest {plan['source']} ({plan['source_size']} B)"
            f" into {plan['store']['display_name']} as {plan['artifact_id']}"
        )
    operation = doc.get("operation", {})
    if operation.get("reused"):
        return f"existing operation returned (idempotency key): {operation['id']}"
    result = operation.get("result") or {}
    copy = result.get("copy") or {}
    artifact = result.get("artifact") or {}
    if artifact:
        return (
            f"ingested {artifact['id']} ({artifact['byte_count']} B,"
            f" sha256:{artifact['content_hash'][:12]})"
            f" copy {copy['id']} on {copy['provider_id']} at {copy['locator']}"
        )
    return f"operation {operation.get('id')}: {operation.get('state')}"


@data_app.command("list")
def list_data(ctx: typer.Context, artifact_id: str = typer.Argument(None), json_mode: bool = JSON_OPTION) -> None:
    """List artifacts and their physical copies, or one artifact in detail."""
    state: CliState = resolve_state(ctx, json_mode)
    path = f"/data/artifacts/{artifact_id}" if artifact_id else "/data/artifacts"
    doc = request(state, "GET", path)
    if artifact_id:
        artifacts = [doc.get("artifact")] if doc.get("artifact") else []
    else:
        artifacts = doc.get("artifacts", [])
    lines = _artifact_lines(artifacts) or ["(no artifacts)"]
    print_payload(state.json_mode, doc, human="\n".join(lines))


@data_app.command("pin")
def pin(
    ctx: typer.Context,
    artifact_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    idempotency_key: str = IDEMPOTENCY_OPTION,
) -> None:
    """Pin an artifact: its durable copies are retained by policy."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(
        state,
        "POST",
        "/data/pin",
        {"artifact_id": artifact_id, "dry_run": dry_run, "idempotency_key": idempotency_key},
    )
    if doc.get("dry_run"):
        human = f"dry run: would pin {doc['plan']['artifact_id']}"
    else:
        human = f"pinned {doc['operation']['result']['id']}"
    print_payload(state.json_mode, doc, human=human)


@data_app.command("evict")
def evict(
    ctx: typer.Context,
    copy_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    idempotency_key: str = IDEMPOTENCY_OPTION,
) -> None:
    """Evict one physical copy (gated on pinning and last-durable-copy rules)."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(
        state,
        "POST",
        "/data/evict",
        {"copy_id": copy_id, "dry_run": dry_run, "idempotency_key": idempotency_key},
    )
    if doc.get("dry_run"):
        plan = doc["plan"]
        suffix = f" (blockers: {'; '.join(plan['blockers'])})" if plan["blockers"] else ""
        human = f"dry run: would evict {plan['copy_id']}{suffix}"
    else:
        result = doc["operation"]["result"]
        suffix = " (bytes were already absent)" if result.get("bytes_absent") else ""
        human = f"evicted {result['copy_id']}{suffix}"
    print_payload(state.json_mode, doc, human=human)


@data_app.command("verify")
def verify(
    ctx: typer.Context,
    copy_id: str = typer.Argument(...),
    json_mode: bool = JSON_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    idempotency_key: str = IDEMPOTENCY_OPTION,
) -> None:
    """Re-read a physical copy and compare its content hash to the catalog."""
    state: CliState = resolve_state(ctx, json_mode)
    doc = request(
        state,
        "POST",
        "/data/verify",
        {"copy_id": copy_id, "dry_run": dry_run, "idempotency_key": idempotency_key},
    )
    if doc.get("dry_run"):
        human = f"dry run: would verify {doc['plan']['copy_id']} at {doc['plan']['locator']}"
    else:
        result = doc["operation"]["result"]
        human = f"verified {result['copy_id']} (sha256:{result['checksum'][:12]}, {result['byte_count']} B)"
    print_payload(state.json_mode, doc, human=human)
