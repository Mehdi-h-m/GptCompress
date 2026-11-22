"""Operation executor: background threads carrying operations to terminal state.

Operations are the unit ``drm wait`` blocks on (Sprint 2) and the unit
startup reconciliation closes out. At-least-once execution with exactly one
accepted outcome starts here: an interrupted operation is never half-recorded —
it is either complete with a result manifest or failed with an error record.
"""

from __future__ import annotations

import json
import threading

from drm.catalog.errors import CatalogError
from drm.contracts import ErrorCodes
from drm.providers.base import AdapterError
from drm.store import repos


def run_ingest(ctx, payload: dict) -> dict:
    from drm.catalog import service
    from drm.providers import registry

    with ctx.db_lock:
        store_row = repos.get_provider(ctx.conn, payload.get("store", ""))
    if store_row is None:
        raise CatalogError("not_found", f"unknown provider {payload.get('store')!r}")
    storage = registry.storage_for_row(store_row)
    with ctx.db_lock:
        return service.ingest_file(conn=ctx.conn, storage=storage, provider_id=store_row["id"], **{
            key: value for key, value in payload.items() if key != "store"
        })


HANDLERS = {"data.ingest": run_ingest}


class Executor:
    def __init__(self, ctx, handlers: dict | None = None) -> None:
        self.ctx = ctx
        self.handlers = handlers if handlers is not None else HANDLERS

    def submit(self, op: dict) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(op["id"], op["kind"], op["payload_json"]),
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
                repos.set_operation_state(ctx.conn, op_id, "complete", result=result)
                repos.audit(ctx.conn, "daemon", f"operation.{kind}.complete", op_id, result or {})
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
