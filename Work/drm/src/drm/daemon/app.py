"""FastAPI application for the local control daemon (loopback only).

The desktop app and the CLI are clients of this same daemon (spec Sections 1
and 3). One process owns the SQLite store and the scheduling loop; startup
reconciliation closes out anything a previous process left unfinished.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from pathlib import Path

import drm
from drm.config import db_path as default_db_path
from drm.contracts import envelope
from drm.daemon.scheduler import Scheduler
from drm.store import repos
from drm.store.db import connect, migrate, utcnow

_COUNT_TABLES = ("providers", "artifacts", "artifact_copies", "jobs", "operations")


@dataclass
class DaemonContext:
    """Everything the routes and executor need, one instance per daemon."""

    conn: object
    db_lock: threading.Lock
    scheduler: Scheduler
    started_at: str
    runs_dir: Path
    cancelled: set
    executor: object = None


def create_app(db_path: str | Path | None = None):
    """Build the daemon application bound to its own SQLite connection."""
    resolved_db = Path(db_path) if db_path else default_db_path()
    conn = connect(resolved_db)
    migrate(conn)
    ctx = DaemonContext(
        conn=conn,
        db_lock=threading.Lock(),
        scheduler=Scheduler(),
        started_at=utcnow(),
        runs_dir=resolved_db.parent / "runs",
        cancelled=set(),
    )
    from drm.daemon.executor import Executor

    ctx.executor = Executor(ctx)

    @contextlib.asynccontextmanager
    async def lifespan(_):
        with ctx.db_lock:
            repos.audit(conn, "daemon", "started")
        from drm.daemon.reconcile import reconcile_on_startup

        reconciliation = reconcile_on_startup(ctx)
        if reconciliation["released_operations"] or reconciliation["removed_partials"]:
            with ctx.db_lock:
                repos.audit(conn, "daemon", "reconciled", details=reconciliation)
        ctx.scheduler.start()
        try:
            yield
        finally:
            ctx.scheduler.stop()
            conn.close()

    from fastapi import FastAPI

    app = FastAPI(
        title="Distributed Resource Manager",
        version=drm.__version__,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        return envelope({"status": "ok"})

    @app.get("/status")
    def status() -> dict:
        with ctx.db_lock:
            counts = {
                table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in _COUNT_TABLES
            }
        return envelope(
            {
                "status": "ok",
                "version": drm.__version__,
                "db_path": str(resolved_db),
                "started_at": ctx.started_at,
                "now": utcnow(),
                "scheduler": ctx.scheduler.snapshot(),
                "counts": counts,
            }
        )

    from drm.daemon.routes import register_routes

    register_routes(app, ctx)
    return app
