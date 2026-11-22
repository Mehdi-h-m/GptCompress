"""FastAPI application for the local control daemon (loopback only).

The desktop app and the CLI are clients of this same daemon (spec Sections 1
and 3). Sprint 0 exposes only health and status; catalog and job endpoints
arrive with the planner and provider sprints.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import drm
from drm.config import db_path as default_db_path
from drm.contracts import envelope
from drm.daemon.scheduler import Scheduler
from drm.store.db import connect, migrate, utcnow

_COUNT_TABLES = ("providers", "artifacts", "artifact_copies", "jobs", "operations")


def create_app(db_path: str | Path | None = None):
    """Build the daemon application bound to its own SQLite connection."""
    resolved_db = Path(db_path) if db_path else default_db_path()
    conn = connect(resolved_db)
    migrate(conn)
    db_lock = threading.Lock()  # one connection shared across threads
    scheduler = Scheduler()
    started_at = utcnow()

    @contextlib.asynccontextmanager
    async def lifespan(_):
        with db_lock:
            conn.execute(
                "INSERT INTO audit_log (at, actor, action, subject, details_json)"
                " VALUES (?, 'daemon', 'started', NULL, '{}')",
                (utcnow(),),
            )
        scheduler.start()
        try:
            yield
        finally:
            scheduler.stop()
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
        with db_lock:
            counts = {
                table: conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                for table in _COUNT_TABLES
            }
        return envelope(
            {
                "status": "ok",
                "version": drm.__version__,
                "db_path": str(resolved_db),
                "started_at": started_at,
                "now": utcnow(),
                "scheduler": scheduler.snapshot(),
                "counts": counts,
            }
        )

    return app
