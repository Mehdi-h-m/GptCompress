"""HTTP API routes for providers, the data catalog and operations.

Every error is an enveloped error record with a stable code (spec Section 4,
rule 2); the CLI maps codes to exit codes. Mutating endpoints accept
``dry_run`` and ``idempotency_key`` (rules 4 and 5).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from drm.catalog import service
from drm.catalog.errors import CatalogError
from drm.contracts import ErrorCodes, envelope, error_record
from drm.daemon.executor import Executor
from drm.providers import registry
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
    executor = Executor(ctx)

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
        if not Path(payload["source"]).is_file():
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
        executor.submit(op)
        return envelope({"operation": op})

    def _ingest_plan_or_error(ctx, row, payload):
        try:
            plan = service.ingest_plan(ctx.conn, row, payload)
        except CatalogError as exc:
            return error_response(exc.code, exc.message)
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
