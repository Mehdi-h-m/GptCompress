"""Local attempt runner: stage verified inputs, execute, validate, publish.

Spec Section 3 worker runtime for the local provider, and Section 9's
exactly-one-accepted-output rule: outputs are published under artifact ids
derived from the *job* and output name, so a re-run or an adopted result can
never create a second accepted copy of the same logical output — a rerun
whose bytes differ is a conflict, not a duplicate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from drm.catalog import service
from drm.catalog.errors import CatalogError
from drm.providers import registry
from drm.providers.base import sha256_hex
from drm.store import repos
from drm.store.db import transaction, utcnow
from drm.store.ids import new_id
from drm.workers.tasks import TASK_REGISTRY, TaskCancelled, TaskContext


def _is_cancelled(ctx, job_id: str) -> bool:
    with ctx.db_lock:
        return job_id in ctx.cancelled


def _set_states(ctx, job_id: str, job_state: str, attempt_id: str, attempt_state: str,
                result: dict | None = None, error: dict | None = None) -> None:
    with ctx.db_lock:
        repos.set_job_state(ctx.conn, job_id, job_state)
        repos.set_attempt_state(ctx.conn, attempt_id, attempt_state, result=result, error=error)


def _resolve_input_copy(ctx, artifact_id: str):
    """Prefer a durable copy; any copy with recorded checksum works for staging."""
    with ctx.db_lock:
        artifact = repos.get_artifact(ctx.conn, artifact_id)
        if artifact is None:
            raise CatalogError("not_found", f"input artifact {artifact_id} is not in the catalog")
        copies = repos.list_copies(ctx.conn, artifact_id)
    if not copies:
        raise CatalogError("validation", f"input artifact {artifact_id} has no physical copies")
    durable = [c for c in copies if c["status"] == "durable"]
    chosen = (durable or copies)[0]
    with ctx.db_lock:
        provider = repos.get_provider(ctx.conn, chosen["provider_id"])
    if provider is None:
        raise CatalogError("not_found", f"copy provider {chosen['provider_id']} is gone")
    return chosen, provider


def _stage_inputs(ctx, workdir: Path, spec: dict) -> list[dict]:
    inputs_dir = workdir / "inputs"
    staged = []
    for artifact_id in spec.get("inputs", []):
        copy, provider = _resolve_input_copy(ctx, artifact_id)
        storage = registry.storage_for_row(provider)
        try:
            data = storage.read(copy["locator"])
        except Exception as exc:
            raise CatalogError("adapter_error", f"staging {artifact_id} failed: {exc}")
        if sha256_hex(data) != copy["checksum"]:
            raise CatalogError("verify_failed", f"staged bytes of {artifact_id} do not match the catalog")
        target = inputs_dir / copy["locator"].replace("/", "_")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        staged.append(
            {
                "artifact_id": artifact_id,
                "name": target.name,
                "sha256": copy["checksum"],
                "byte_count": len(data),
            }
        )
    return staged


def _validate_outputs(outputs_dir: Path, declared: list[dict]) -> list[dict]:
    validated = []
    for item in declared:
        name = item["name"]
        raw = (outputs_dir / name).read_bytes()
        validated.append(
            {"name": name, "sha256": sha256_hex(raw), "byte_count": len(raw), "declared_sha256": item.get("sha256")}
        )
    return validated


def _output_artifact_id(job_id: str, output_name: str) -> str:
    material = f"{job_id}:{output_name}".encode()
    return "art-" + hashlib.sha256(material).hexdigest()[:12]


def publish_output(conn, storage, provider_id: str, job_id: str, output_name: str,
                   data: bytes, producer: str | None = None) -> dict:
    """Publish one job output as a verified artifact; idempotent per job+name."""
    artifact_id = _output_artifact_id(job_id, output_name)
    content_hash = sha256_hex(data)
    with transaction(conn):
        existing = repos.get_artifact(conn, artifact_id)
    if existing is not None:
        if existing["content_hash"] != content_hash:
            raise CatalogError(
                "conflict",
                f"output {output_name!r} of job {job_id} was already published with different content",
            )
        return {"artifact": service.artifact_dict(existing), "copy": None, "adopted": True}
    locator = f"{artifact_id}/{output_name}"
    written = storage.write(locator, data, checksum=content_hash)
    with transaction(conn):
        artifact = repos.create_artifact(
            conn,
            artifact_id=artifact_id,
            kind="result",
            content_hash=content_hash,
            byte_count=len(data),
            format="json",
            producer=producer,
        )
        copy = repos.create_copy(
            conn,
            copy_id=new_id("copy"),
            artifact_id=artifact_id,
            provider_id=provider_id,
            locator=locator,
            version=content_hash[:12],
            checksum=written["checksum"],
            byte_count=written["byte_count"],
            status="durable",
        )
    return {"artifact": service.artifact_dict(artifact), "copy": dict(copy), "adopted": False}


def execute_local_attempt(ctx, job_row, attempt_row, spec: dict) -> dict:
    job_id, attempt_id = job_row["id"], attempt_row["id"]
    workdir = ctx.runs_dir / job_id / attempt_id
    outputs_dir = workdir / "outputs"

    task_type = spec.get("task_type")
    task_fn = TASK_REGISTRY.get(task_type)
    if task_fn is None:
        raise CatalogError("validation", f"unknown task type {task_type!r}")

    _set_states(ctx, job_id, "staging", attempt_id, "staging")
    if _is_cancelled(ctx, job_id):
        _set_states(ctx, job_id, "cancelled", attempt_id, "cancelled")
        raise TaskCancelled(f"job {job_id} cancelled before staging")
    staged = _stage_inputs(ctx, workdir, spec)

    _set_states(ctx, job_id, "ready", attempt_id, "queued")
    _set_states(ctx, job_id, "running", attempt_id, "running")
    task_ctx = TaskContext(
        workdir=workdir,
        params=spec.get("params", {}),
        cancelled=lambda: _is_cancelled(ctx, job_id),
    )
    declared = task_fn(task_ctx)["outputs"]
    if _is_cancelled(ctx, job_id):
        _set_states(ctx, job_id, "cancelled", attempt_id, "cancelled")
        raise TaskCancelled(f"job {job_id} cancelled after task execution")

    _set_states(ctx, job_id, "validating", attempt_id, "validating")
    validated = _validate_outputs(outputs_dir, declared)
    result_manifest = {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "task_type": task_type,
        "task_version": spec.get("task_version", 1),
        "numerical_profile": spec.get("numerical_profile"),
        "staged_inputs": staged,
        "outputs": validated,
        "completed_at": utcnow(),
    }
    (workdir / "result_manifest.json").write_text(
        json.dumps(result_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    output_store = spec.get("output_store")
    if not output_store:
        raise CatalogError("validation", "job spec has no output_store placement")
    with ctx.db_lock:
        store_row = repos.get_provider(ctx.conn, output_store)
    if store_row is None:
        raise CatalogError("not_found", f"output store {output_store} is gone")
    storage = registry.storage_for_row(store_row)

    published = []
    for item in validated:
        data = (outputs_dir / item["name"]).read_bytes()
        published.append(
            service_publish(ctx, storage, store_row["id"], job_id, item["name"], data, attempt_id)
        )

    _set_states(ctx, job_id, "complete", attempt_id, "complete", result=result_manifest)
    return {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "result_manifest": result_manifest,
        "published": published,
    }


def service_publish(ctx, storage, provider_id, job_id, output_name, data, attempt_id):
    with ctx.db_lock:
        return publish_output(ctx.conn, storage, provider_id, job_id, output_name, data, producer=attempt_id)


def mark_failed(ctx, job_id: str, attempt_id: str, code: str, message: str) -> None:
    _set_states(ctx, job_id, "failed", attempt_id, "failed", error={"code": code, "message": message})


def mark_cancelled(ctx, job_id: str, attempt_id: str) -> None:
    with ctx.db_lock:
        job = repos.get_job(ctx.conn, job_id)
    if job is not None and job["state"] not in repos.TERMINAL_JOB_STATES:
        _set_states(ctx, job_id, "cancelled", attempt_id, "cancelled")


def adopt_or_fail(ctx, job_row, attempt_row) -> str:
    """Startup reconciliation for one open attempt: adopt a valid result or fail it.

    A result manifest that validates is published under the job-derived output
    ids — adopting after a crash can never duplicate an accepted output.
    """
    job_id, attempt_id = job_row["id"], attempt_row["id"]
    workdir = ctx.runs_dir / job_id / attempt_id
    manifest_path = workdir / "result_manifest.json"
    if not manifest_path.is_file():
        with ctx.db_lock:
            repos.set_attempt_state(
                ctx.conn, attempt_id, "failed",
                error={"code": "interrupted", "message": "coordinator restarted before the attempt finished"},
            )
        return "failed"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        outputs_dir = workdir / "outputs"
        output_store = job_row.get("spec", {}).get("output_store") if isinstance(job_row, dict) else None
        if not output_store:
            raise CatalogError("validation", "job spec has no output_store placement")
        with ctx.db_lock:
            store_row = repos.get_provider(ctx.conn, output_store)
        if store_row is None:
            raise CatalogError("not_found", f"output store {output_store!r} is gone")
        storage = registry.storage_for_row(store_row)
        published = []
        for item in manifest["outputs"]:
            data = (outputs_dir / item["name"]).read_bytes()
            if sha256_hex(data) != item["sha256"]:
                raise CatalogError("verify_failed", f"adopted output {item['name']} fails checksum")
            published.append(
                service_publish(ctx, storage, store_row["id"], job_id, item["name"], data, attempt_id)
            )
        with ctx.db_lock:
            repos.set_attempt_state(ctx.conn, attempt_id, "complete", result=manifest)
            repos.set_job_state(ctx.conn, job_id, "complete")
        return "adopted"
    except (CatalogError, OSError, ValueError, KeyError):
        with ctx.db_lock:
            repos.set_attempt_state(
                ctx.conn, attempt_id, "failed",
                error={"code": "interrupted", "message": "result manifest did not survive the restart"},
            )
        return "failed"
