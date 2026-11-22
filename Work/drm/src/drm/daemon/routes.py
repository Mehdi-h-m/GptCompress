"""HTTP API routes for providers, the data catalog and operations.

Every error is an enveloped error record with a stable code (spec Section 4,
rule 2); the CLI maps codes to exit codes. Mutating endpoints accept
``dry_run`` and ``idempotency_key`` (rules 4 and 5).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from drm.catalog import service
from drm.catalog.errors import CatalogError
from drm.contracts import ErrorCodes, envelope, error_record
from drm.planner import core as planner
from drm.providers import hf, registry
from drm.providers.base import AdapterError
from drm.store import repos
from drm.store.db import transaction

_ERROR_STATUS = {"validation": 400, "policy_blocked": 400, "conflict": 409, "not_found": 404}


def error_response(code: str, message: str, status: int | None = None, **details):
    return JSONResponse(
        status_code=status or _ERROR_STATUS.get(code, 400),
        content=error_record(code, message, **details),
    )


def _catalog_guard(fn, *args, **kwargs):
    """Run a catalog call, mapping errors to enveloped responses."""
    try:
        return fn(*args, **kwargs)
    except CatalogError as exc:
        return error_response(exc.code, exc.message)
    except AdapterError as exc:
        return error_response(ErrorCodes.ADAPTER_ERROR, str(exc))


def _planner_guard(fn, *args, **kwargs):
    """Run a planner call, mapping validation errors to enveloped responses."""
    try:
        return fn(*args, **kwargs)
    except CatalogError as exc:
        return error_response(exc.code, exc.message)


def _run_sync(ctx, kind: str, payload: dict, fn, idempotency_key: str | None = None):
    """Record an operation row, run fn under the db lock, record the outcome."""
    with ctx.db_lock:
        op = dict(repos.create_operation(ctx.conn, kind=kind, idempotency_key=idempotency_key, payload=payload))
    try:
        with ctx.db_lock:
            result = fn()
            repos.set_operation_state(ctx.conn, op["id"], "complete", result=result)
            repos.audit(ctx.conn, "drm", f"operation.{kind}.complete", op["id"], result or {})
    except CatalogError as exc:
        with ctx.db_lock:
            repos.set_operation_state(ctx.conn, op["id"], "failed", error={"code": exc.code, "message": exc.message})
        return error_response(exc.code, exc.message)
    except AdapterError as exc:
        with ctx.db_lock:
            repos.set_operation_state(ctx.conn, op["id"], "failed", error={"code": ErrorCodes.ADAPTER_ERROR, "message": str(exc)})
        return error_response(ErrorCodes.ADAPTER_ERROR, str(exc))
    except Exception as exc:
        with ctx.db_lock:
            repos.set_operation_state(ctx.conn, op["id"], "failed", error={"code": ErrorCodes.UNEXPECTED_ERROR, "message": str(exc)})
        return error_response(ErrorCodes.UNEXPECTED_ERROR, str(exc))
    with ctx.db_lock:
        op = repos.operation_dict(repos.get_operation(ctx.conn, op["id"]))
    return envelope({"operation": op})


def _get_provider_or_404(ctx, provider_id: str):
    with ctx.db_lock:
        row = repos.get_provider(ctx.conn, provider_id)
    if row is None:
        return None, error_response(ErrorCodes.NOT_FOUND, f"unknown provider {provider_id}", status=404)
    return row, None


def register_routes(app: FastAPI, ctx) -> None:
    executor = ctx.executor  # created with the app; reconcile shares it

    # --- plan ------------------------------------------------------------------

    @app.post("/plan/preview")
    def plan_preview(body: dict):
        plan_doc = _planner_guard(lambda: planner.plan(ctx.conn, body))
        if isinstance(plan_doc, JSONResponse):
            return plan_doc
        decision = _planner_guard(
            lambda: planner.record_decision(ctx.conn, "plan.preview", plan_doc["spec"]["id"], plan_doc)
        )
        if isinstance(decision, JSONResponse):
            return decision
        plan_doc["decision_id"] = decision["id"]
        return envelope({"plan": plan_doc})

    # --- runs ------------------------------------------------------------------

    @app.post("/run/submit")
    def submit_run(body: dict):
        plan_doc = _planner_guard(lambda: planner.plan(ctx.conn, body))
        if isinstance(plan_doc, JSONResponse):
            return plan_doc
        if body.get("dry_run"):
            return envelope({"dry_run": True, "plan": plan_doc})
        if plan_doc["blockers"]:
            return error_response(
                "validation", "plan has blockers: " + "; ".join(plan_doc["blockers"])
            )
        spec = dict(plan_doc["spec"])
        spec["output_store"] = plan_doc["chosen"]["output_store"]
        spec["compute_provider"] = plan_doc["chosen"]["compute"]
        key = body.get("idempotency_key") or _derive_run_key(spec)
        with ctx.db_lock:
            existing = repos.find_operation_by_key(ctx.conn, key)
            if existing is not None:
                return envelope(
                    {"operation": repos.operation_dict(existing), "reused": True}
                )
            compute_row = repos.get_provider(ctx.conn, spec["compute_provider"])
            execution = compute_row["execution"] if compute_row is not None else "api"
            job = repos.create_job(
                ctx.conn,
                job_id=repos.new_id("job"),
                name=body.get("name"),
                experiment_id=body.get("experiment_id"),
                spec=spec,
                numerical_profile=spec.get("numerical_profile"),
            )
            attempt = repos.create_attempt(
                ctx.conn,
                attempt_id=repos.new_id("att"),
                job_id=job["id"],
                provider_id=spec["compute_provider"],
            )
            operation = repos.create_operation(
                ctx.conn,
                kind="run.job",
                idempotency_key=key,
                payload={"job_id": job["id"], "attempt_id": attempt["id"]},
            )
            repos.set_attempt_operation(ctx.conn, attempt["id"], operation["id"])
            planner.record_decision(ctx.conn, "job.place", job["id"], plan_doc)
            repos.audit(ctx.conn, "drm", "run.submitted", job["id"],
                        {"operation": operation["id"], "placement": plan_doc["chosen"]})
            # capture everything the response needs before the worker thread runs
            job_doc = repos.job_dict(repos.get_job(ctx.conn, job["id"]))
            attempt_doc = repos.attempt_dict(repos.get_attempt(ctx.conn, attempt["id"]))
            operation_doc = repos.operation_dict(repos.get_operation(ctx.conn, operation["id"]))

        if execution == "assisted":
            # bundle preparation is fast and local: park the run in awaiting_user
            # synchronously so the submit response carries the instructions (exit 3)
            from drm.workers import assisted as assisted_worker

            assisted_result = assisted_worker.prepare_bundle(ctx, job_doc, attempt_doc, spec)
            with ctx.db_lock:
                repos.set_operation_state(ctx.conn, operation["id"], "awaiting_user", result=assisted_result)
                job_doc = repos.job_dict(repos.get_job(ctx.conn, job["id"]))
                attempt_doc = repos.attempt_dict(repos.get_attempt(ctx.conn, attempt["id"]))
                operation_doc = repos.operation_dict(repos.get_operation(ctx.conn, operation["id"]))
            return envelope({"job": job_doc, "attempt": attempt_doc, "operation": operation_doc, "plan": plan_doc})

        executor.submit(operation)
        return envelope({"job": job_doc, "attempt": attempt_doc, "operation": operation_doc, "plan": plan_doc})

    def _derive_run_key(spec: dict) -> str:
        import hashlib

        material = json.dumps(spec, sort_keys=True).encode()
        return "run-" + hashlib.sha256(material).hexdigest()[:16]

    @app.get("/run/{job_id}")
    def get_run(job_id: str):
        with ctx.db_lock:
            job = repos.get_job(ctx.conn, job_id)
            if job is None:
                return error_response(ErrorCodes.NOT_FOUND, f"unknown job {job_id}", status=404)
            attempts = [repos.attempt_dict(a) for a in repos.list_attempts(ctx.conn, job_id)]
        return envelope({"job": repos.job_dict(job), "attempts": attempts})

    @app.get("/jobs")
    def list_jobs():
        with ctx.db_lock:
            jobs = [repos.job_dict(row) for row in repos.list_jobs(ctx.conn)]
        return envelope({"jobs": jobs})

    @app.post("/run/{job_id}/cancel")
    def cancel_run(job_id: str, body: dict | None = None):
        with ctx.db_lock:
            job = repos.get_job(ctx.conn, job_id)
        if job is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown job {job_id}", status=404)
        if body and body.get("dry_run"):
            return envelope({"dry_run": True, "plan": {"job_id": job_id, "would_cancel": True}})
        if job["state"] == "cancelled":
            return envelope({"job": repos.job_dict(job), "already_cancelled": True})
        if job["state"] in repos.TERMINAL_JOB_STATES:
            return error_response(
                "conflict", f"job {job_id} is already {job['state']} and cannot be cancelled", status=409
            )
        with ctx.db_lock:
            ctx.cancelled.add(job_id)
            repos.audit(ctx.conn, "drm", "run.cancel_requested", job_id, {})
        return envelope({"job_id": job_id, "cancel_requested": True, "state": job["state"]})

    @app.post("/jobs/{job_id}/retry")
    def retry_job(job_id: str):
        with ctx.db_lock:
            job = repos.get_job(ctx.conn, job_id)
            if job is None:
                return error_response(ErrorCodes.NOT_FOUND, f"unknown job {job_id}", status=404)
            if job["state"] not in ("failed", "cancelled"):
                return error_response(
                    "conflict",
                    f"job {job_id} is {job['state']}; only failed or cancelled jobs can be retried",
                    status=409,
                )
            spec = json.loads(job["dag_json"] or "{}")
            used = repos.count_attempts(ctx.conn, job_id)
            attempt = repos.create_attempt(
                ctx.conn,
                attempt_id=repos.new_id("att"),
                job_id=job_id,
                provider_id=spec.get("compute_provider"),
            )
            operation = repos.create_operation(
                ctx.conn,
                kind="run.job",
                idempotency_key=f"run-{job_id}-manual-{used + 1}",
                payload={"job_id": job_id, "attempt_id": attempt["id"]},
            )
            repos.set_job_state(ctx.conn, job_id, "queued")
            repos.audit(ctx.conn, "drm", "job.retry_manual", job_id,
                        {"attempt": attempt["id"], "operation": operation["id"]})
            job_doc = repos.job_dict(repos.get_job(ctx.conn, job_id))
            attempt_doc = repos.attempt_dict(attempt)
            operation_doc = repos.operation_dict(repos.get_operation(ctx.conn, operation["id"]))
        executor.submit(operation)
        return envelope({"job": job_doc, "attempt": attempt_doc, "operation": operation_doc})

    # --- attempts (assisted execution) --------------------------------------------

    @app.get("/attempts/{attempt_id}")
    def get_attempt(attempt_id: str):
        with ctx.db_lock:
            row = repos.get_attempt(ctx.conn, attempt_id)
        if row is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown attempt {attempt_id}", status=404)
        data = repos.attempt_dict(row)
        instructions = None
        result = data.get("result") or {}
        if result.get("instructions"):
            path = Path(result["instructions"])
            if path.is_file():
                instructions = path.read_text(encoding="utf-8")
        return envelope({"attempt": data, "instructions": instructions})

    @app.post("/attempts/{attempt_id}/import")
    def import_attempt(attempt_id: str, body: dict):
        output_dir = body.get("output_dir")
        if not output_dir:
            return error_response("validation", "import requires 'output_dir'")
        with ctx.db_lock:
            row = repos.get_attempt(ctx.conn, attempt_id)
        if row is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown attempt {attempt_id}", status=404)
        attempt_data = repos.attempt_dict(row)
        with ctx.db_lock:
            job_row = repos.get_job(ctx.conn, attempt_data["job_id"])
        if job_row is None:
            return error_response(ErrorCodes.NOT_FOUND, f"job {attempt_data['job_id']} is gone", status=404)
        job_data = repos.job_dict(job_row)

        from drm.workers import assisted as assisted_worker

        with ctx.db_lock:
            op = repos.create_operation(
                ctx.conn, kind="attempt.import",
                payload={"attempt_id": attempt_id, "output_dir": output_dir},
            )
        try:
            result = assisted_worker.import_results(ctx, job_data, attempt_data, output_dir)
        except CatalogError as exc:
            with ctx.db_lock:
                repos.set_operation_state(ctx.conn, op["id"], "failed",
                                          error={"code": exc.code, "message": exc.message})
            return error_response(exc.code, exc.message)
        with ctx.db_lock:
            repos.set_operation_state(ctx.conn, op["id"], "complete", result=result)
            op_doc = repos.operation_dict(repos.get_operation(ctx.conn, op["id"]))
        return envelope({"import": result, "operation": op_doc})

    # --- decisions ---------------------------------------------------------------

    @app.get("/decisions/{decision_id}")
    def get_decision(decision_id: str):
        with ctx.db_lock:
            row = repos.get_decision(ctx.conn, decision_id)
        if row is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown decision {decision_id}", status=404)
        return envelope({"decision": repos.decision_dict(row)})

    # --- providers -----------------------------------------------------------

    @app.get("/providers")
    def list_providers() -> dict:
        with ctx.db_lock:
            rows = repos.list_providers(ctx.conn)
        return envelope({"providers": [registry.provider_dict(row) for row in rows]})

    @app.get("/providers/{provider_id}")
    def get_provider(provider_id: str):
        row, err = _get_provider_or_404(ctx, provider_id)
        if err is not None:
            return err
        return envelope({"provider": registry.provider_dict(row)})

    @app.post("/providers")
    def connect_provider(body: dict):
        if body.get("dry_run"):
            config = body.get("config") or {}
            try:
                registry.validate_config(body.get("adapter"), config)
            except CatalogError as exc:
                return error_response(exc.code, exc.message)
            return envelope(
                {
                    "dry_run": True,
                    "plan": {
                        "adapter": body.get("adapter"),
                        "name": body.get("name"),
                        "config": config,
                        "kind": registry.ADAPTERS.get(body.get("adapter"), {}).get("kind"),
                    },
                }
            )
        try:
            with ctx.db_lock, transaction(ctx.conn):
                provider, created = registry.connect(ctx.conn, body)
        except CatalogError as exc:
            return error_response(exc.code, exc.message)
        return envelope({"provider": provider, "already_connected": not created})

    @app.post("/providers/{provider_id}/probe")
    def probe_provider(provider_id: str):
        row, err = _get_provider_or_404(ctx, provider_id)
        if err is not None:
            return err
        result = _catalog_guard(registry.probe, row)
        if isinstance(result, JSONResponse):
            return result
        with ctx.db_lock:
            repos.audit(ctx.conn, "drm", "provider.probe", provider_id, result)
        return envelope({"provider_id": provider_id, "probe": result})

    @app.post("/providers/{provider_id}/pause")
    def pause_provider(provider_id: str, body: dict | None = None):
        return _set_provider_state(ctx, provider_id, "paused", body)

    @app.post("/providers/{provider_id}/resume")
    def resume_provider(provider_id: str, body: dict | None = None):
        return _set_provider_state(ctx, provider_id, "active", body)

    def _set_provider_state(ctx, provider_id: str, state: str, body: dict | None = None):
        row, err = _get_provider_or_404(ctx, provider_id)
        if err is not None:
            return err
        if body and body.get("dry_run"):
            return envelope({"dry_run": True, "plan": {"provider_id": provider_id, "state": state}})
        with ctx.db_lock, transaction(ctx.conn):
            updated = repos.set_provider_state(ctx.conn, provider_id, state)
            repos.audit(ctx.conn, "drm", f"provider.{state}", provider_id, {})
        return envelope({"provider": registry.provider_dict(updated)})

    # --- data catalog ----------------------------------------------------------

    @app.get("/data/artifacts")
    def list_artifacts() -> dict:
        with ctx.db_lock:
            artifacts = service.list_artifacts_with_copies(ctx.conn)
        return envelope({"artifacts": artifacts})

    @app.get("/data/artifacts/{artifact_id}")
    def get_artifact(artifact_id: str):
        with ctx.db_lock:
            row = repos.get_artifact(ctx.conn, artifact_id)
        if row is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown artifact {artifact_id}", status=404)
        with ctx.db_lock:
            data = service.artifact_dict(row)
            data["copies"] = [dict(copy) for copy in repos.list_copies(ctx.conn, artifact_id)]
        return envelope({"artifact": data})

    @app.post("/data/ingest")
    def ingest(body: dict):
        try:
            payload = service.normalize_ingest_payload(body)
        except CatalogError as exc:
            return error_response(exc.code, exc.message)
        if not payload["source"].startswith("hf://") and not Path(payload["source"]).is_file():
            return error_response("validation", f"source is not a file: {payload['source']}")
        row, err = _get_provider_or_404(ctx, payload["store"])
        if err is not None:
            return err
        if body.get("dry_run"):
            return _ingest_plan_or_error(ctx, row, payload)
        key = body.get("idempotency_key") or service.derive_ingest_key(row, payload)
        with ctx.db_lock:
            existing = repos.find_operation_by_key(ctx.conn, key)
            if existing is not None and existing["state"] in ("queued", "running", "complete"):
                return envelope({"operation": repos.operation_dict(existing), "reused": True})
            if existing is not None:
                repos.supersede_operation_key(ctx.conn, existing["id"])
            op = repos.create_operation(ctx.conn, kind="data.ingest", idempotency_key=key, payload=payload)
            op_doc = repos.operation_dict(repos.get_operation(ctx.conn, op["id"]))
        executor.submit(op)
        return envelope({"operation": op_doc})

    def _ingest_plan_or_error(ctx, row, payload):
        try:
            if payload["source"].startswith("hf://"):
                parsed = hf.parse_hf_uri(payload["source"])
                plan = {
                    "source": payload["source"],
                    "source_size": None,
                    "origin": parsed,
                    "store": {"id": row["id"], "display_name": row["display_name"]},
                    "artifact_id": payload.get("artifact_id") or "auto",
                    "locator": "assigned on publish",
                    "verify": payload["verify"],
                    "blockers": [],
                }
            else:
                plan = service.ingest_plan(ctx.conn, row, payload)
        except CatalogError as exc:
            return error_response(exc.code, exc.message)
        except hf.AdapterError as exc:
            return error_response("validation", str(exc))
        return envelope({"dry_run": True, "plan": plan})

    @app.post("/data/pin")
    def pin(body: dict):
        artifact_id = body.get("artifact_id")
        with ctx.db_lock:
            exists = repos.get_artifact(ctx.conn, artifact_id or "") is not None
        if not exists:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown artifact {artifact_id}", status=404)
        if body.get("dry_run"):
            return envelope({"dry_run": True, "plan": {"artifact_id": artifact_id, "would_pin": True}})
        return _run_sync(
            ctx,
            "data.pin",
            body,
            lambda: service.pin_artifact(ctx.conn, artifact_id),
            idempotency_key=body.get("idempotency_key"),
        )

    @app.post("/data/evict")
    def evict(body: dict):
        copy_id = body.get("copy_id")
        with ctx.db_lock:
            copy = repos.get_copy(ctx.conn, copy_id or "")
        if copy is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown copy {copy_id}", status=404)
        if body.get("dry_run"):
            plan = _catalog_guard(service.evict_plan, ctx.conn, copy)
            if isinstance(plan, JSONResponse):
                return plan
            return envelope({"dry_run": True, "plan": plan})
        return _run_sync(
            ctx,
            "data.evict",
            body,
            lambda: service.evict_copy(ctx.conn, registry.storage_for_row(_provider_row(ctx, copy["provider_id"])), copy_id),
            idempotency_key=body.get("idempotency_key"),
        )

    @app.post("/data/verify")
    def verify(body: dict):
        copy_id = body.get("copy_id")
        with ctx.db_lock:
            copy = repos.get_copy(ctx.conn, copy_id or "")
        if copy is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown copy {copy_id}", status=404)
        if body.get("dry_run"):
            return envelope(
                {
                    "dry_run": True,
                    "plan": {
                        "copy_id": copy_id,
                        "locator": copy["locator"],
                        "checksum": copy["checksum"],
                        "would_verify": True,
                    },
                }
            )
        return _run_sync(
            ctx,
            "data.verify",
            body,
            lambda: service.verify_copy(ctx.conn, registry.storage_for_row(_provider_row(ctx, copy["provider_id"])), copy_id),
            idempotency_key=body.get("idempotency_key"),
        )

    def _provider_row(ctx, provider_id: str):
        row = repos.get_provider(ctx.conn, provider_id)
        if row is None:
            raise CatalogError("not_found", f"unknown provider {provider_id}")
        return row

    # --- operations ------------------------------------------------------------

    @app.get("/operations/{op_id}")
    def get_operation(op_id: str):
        with ctx.db_lock:
            row = repos.get_operation(ctx.conn, op_id)
        if row is None:
            return error_response(ErrorCodes.NOT_FOUND, f"unknown operation {op_id}", status=404)
        return envelope({"operation": repos.operation_dict(row)})
