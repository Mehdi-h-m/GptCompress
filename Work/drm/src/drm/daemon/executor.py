"""Operation executor: background threads carrying operations to terminal state.

Operations are the unit ``drm wait`` blocks on and the unit startup
reconciliation closes out. Job runs wrap the attempt runner: failures with
remaining attempts move the job to ``retry_pending`` and dispatch a fresh
attempt; cancellation unwinds cooperatively (spec Section 9).
"""

from __future__ import annotations

import json
import threading

from drm.catalog.errors import CatalogError
from drm.contracts import ErrorCodes
from drm.providers.base import AdapterError
from drm.store import repos
from drm.workers.tasks import TaskCancelled


def run_ingest(ctx, payload: dict) -> dict:
    from drm.catalog import service
    from drm.providers import hf as hf_provider
    from drm.providers import registry

    with ctx.db_lock:
        store_row = repos.get_provider(ctx.conn, payload.get("store", ""))
    if store_row is None:
        raise CatalogError("not_found", f"unknown provider {payload.get('store')!r}")
    storage = registry.storage_for_row(store_row)
    kwargs = {key: value for key, value in payload.items() if key != "store"}
    if payload["source"].startswith("hf://"):
        try:
            local_path, provenance = hf_provider.resolve_public_download(payload["source"])
        except hf_provider.AdapterError as exc:
            raise CatalogError("validation", str(exc))
        kwargs["source"] = str(local_path)
        kwargs["producer"] = provenance["producer"]
    with ctx.db_lock:
        return service.ingest_file(conn=ctx.conn, storage=storage, provider_id=store_row["id"], **kwargs)


def run_job(ctx, payload: dict) -> dict:
    from drm.workers import assisted, runner

    with ctx.db_lock:
        job = repos.get_job(ctx.conn, payload.get("job_id", ""))
        attempt = repos.get_attempt(ctx.conn, payload.get("attempt_id", ""))
    if job is None:
        raise CatalogError("not_found", f"unknown job {payload.get('job_id')!r}")
    if attempt is None:
        raise CatalogError("not_found", f"unknown attempt {payload.get('attempt_id')!r}")
    spec = json.loads(job["dag_json"] or "{}")
    with ctx.db_lock:
        provider_row = repos.get_provider(ctx.conn, attempt["provider_id"] or "")
    execution = provider_row["execution"] if provider_row is not None else "api"
    try:
        if execution == "assisted":
            return assisted.prepare_bundle(ctx, job, attempt, spec)
        return runner.execute_local_attempt(ctx, job, attempt, spec)
    except TaskCancelled:
        runner.mark_cancelled(ctx, job["id"], attempt["id"])
        raise
    except (CatalogError, AdapterError) as exc:
        code = exc.code if isinstance(exc, CatalogError) else ErrorCodes.ADAPTER_ERROR
        message = exc.message if isinstance(exc, CatalogError) else str(exc)
        runner.mark_failed(ctx, job["id"], attempt["id"], code, message)
        maybe_autoretry(ctx, job, spec)
        raise
    except Exception as exc:
        # task code can raise anything; the job must still reach a terminal state
        runner.mark_failed(ctx, job["id"], attempt["id"], ErrorCodes.UNEXPECTED_ERROR, str(exc))
        maybe_autoretry(ctx, job, spec)
        raise


def maybe_autoretry(ctx, job_row, spec: dict) -> None:
    """Dispatch another attempt when the job has retries left (spec Section 9)."""
    with ctx.db_lock:
        fresh = repos.get_job(ctx.conn, job_row["id"])
        if fresh is None or fresh["state"] != "failed":
            return
        used = repos.count_attempts(ctx.conn, job_row["id"])
        max_attempts = int(spec.get("max_attempts", 3))
        if used >= max_attempts:
            return
        compute_provider = spec.get("compute_provider")
        attempt = repos.create_attempt(
            ctx.conn,
            attempt_id=repos.new_id("att"),
            job_id=job_row["id"],
            provider_id=compute_provider,
        )
        operation = repos.create_operation(
            ctx.conn,
            kind="run.job",
            idempotency_key=f"run-{job_row['id']}-{used + 1}",
            payload={"job_id": job_row["id"], "attempt_id": attempt["id"]},
        )
        repos.set_attempt_operation(ctx.conn, attempt["id"], operation["id"])
        repos.set_job_state(ctx.conn, job_row["id"], "retry_pending")
        repos.audit(ctx.conn, "daemon", "job.retry_scheduled", job_row["id"],
                    {"attempt": attempt["id"], "operation": operation["id"], "used": used})
    ctx.executor.submit(dict(operation))


HANDLERS = {"data.ingest": run_ingest, "run.job": run_job}


class Executor:
    def __init__(self, ctx, handlers: dict | None = None) -> None:
        self.ctx = ctx
        self.handlers = handlers if handlers is not None else HANDLERS

    def submit(self, op) -> None:
        op = dict(op)
        payload_json = op["payload_json"] if "payload_json" in op else json.dumps(op.get("payload", {}))
        thread = threading.Thread(
            target=self._run,
            args=(op["id"], op["kind"], payload_json),
            name=f"drm-op-{op['id']}",
            daemon=True,
        )
        thread.start()

    def _run(self, op_id: str, kind: str, payload_json: str) -> None:
        ctx = self.ctx
        with ctx.db_lock:
            repos.set_operation_state(ctx.conn, op_id, "running")
        try:
            handler = self.handlers.get(kind)
            if handler is None:
                raise CatalogError("validation", f"no handler registered for operation kind {kind!r}")
            result = handler(ctx, json.loads(payload_json))
            with ctx.db_lock:
                if isinstance(result, dict) and result.get("awaiting_user"):
                    repos.set_operation_state(ctx.conn, op_id, "awaiting_user", result=result)
                    repos.audit(ctx.conn, "daemon", f"operation.{kind}.awaiting_user", op_id,
                                {"attempt": result.get("attempt_id") or ""})
                else:
                    repos.set_operation_state(ctx.conn, op_id, "complete", result=result)
                    repos.audit(ctx.conn, "daemon", f"operation.{kind}.complete", op_id, result or {})
        except TaskCancelled:
            with ctx.db_lock:
                repos.set_operation_state(ctx.conn, op_id, "cancelled",
                                          error={"code": "cancelled", "message": "job cancelled"})
                repos.audit(ctx.conn, "daemon", "operation.cancelled", op_id, {})
        except CatalogError as exc:
            self._fail(op_id, exc.code, exc.message)
        except AdapterError as exc:
            self._fail(op_id, ErrorCodes.ADAPTER_ERROR, str(exc))
        except Exception as exc:  # unexpected: record it, never crash the daemon
            self._fail(op_id, ErrorCodes.UNEXPECTED_ERROR, str(exc))

    def _fail(self, op_id: str, code: str, message: str) -> None:
        ctx = self.ctx
        with ctx.db_lock:
            repos.set_operation_state(ctx.conn, op_id, "failed", error={"code": code, "message": message})
            repos.audit(ctx.conn, "daemon", "operation.failed", op_id, {"code": code, "message": message})
