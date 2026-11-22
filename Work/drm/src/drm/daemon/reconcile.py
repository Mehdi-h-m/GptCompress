"""Startup reconciliation (spec Sections 3 and 9).

The coordinator owns every operation's state. When the process dies, its
in-flight operations can neither finish nor record an outcome, so on restart
they are moved to a terminal ``failed`` state with an ``interrupted`` error
record — never left ``running`` forever. Half-written staging files in
approved filesystem stores are swept the same way; cataloged copies were
published atomically and are never touched.

Open jobs get the Section 9 treatment: an attempt whose result manifest
survives is validated and **adopted** (outputs publish under job-derived
ids, so adoption can never duplicate an accepted output); otherwise the
attempt fails as interrupted and the job re-runs while attempts remain.
"""

from __future__ import annotations

import json
from pathlib import Path

from drm.contracts import ErrorCodes
from drm.providers.filesystem import FilesystemStorage
from drm.store import repos
from drm.store.db import transaction, utcnow
from drm.workers import runner


def reconcile_on_startup(ctx) -> dict:
    released: list[str] = []
    with ctx.db_lock, transaction(ctx.conn):
        for row in repos.list_unfinished_operations(ctx.conn):
            ctx.conn.execute(
                "UPDATE operations SET state = 'failed', error_json = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(
                        {
                            "code": ErrorCodes.INTERRUPTED,
                            "message": "coordinator restarted before the operation finished",
                        }
                    ),
                    utcnow(),
                    row["id"],
                ),
            )
            released.append(row["id"])
    removed_partials = 0
    with ctx.db_lock:
        for row in repos.list_providers(ctx.conn):
            if row["adapter"] != "filesystem":
                continue
            root = json.loads(row["config_json"]).get("root")
            if root:
                removed_partials += FilesystemStorage(root).cleanup_partials()
    jobs = reconcile_jobs(ctx)
    return {
        "released_operations": released,
        "removed_partials": removed_partials,
        "adopted_attempts": jobs["adopted"],
        "rerun_jobs": jobs["requeued"],
        "failed_jobs": jobs["failed"],
    }


def reconcile_jobs(ctx) -> dict:
    """Adopt surviving results, otherwise re-run unfinished jobs (spec Section 9)."""
    adopted: list[str] = []
    requeued: list[str] = []
    failed_jobs: list[str] = []
    with ctx.db_lock:
        open_jobs = [
            repos.job_dict(row)
            for row in repos.list_jobs(ctx.conn, states=("queued", "staging", "ready", "running", "validating", "retry_pending"))
        ]
    for job in open_jobs:
        with ctx.db_lock:
            attempts = [repos.attempt_dict(a) for a in repos.list_attempts(ctx.conn, job["id"])]
        terminal = repos.TERMINAL_JOB_STATES
        for attempt in [a for a in attempts if a["state"] not in terminal]:
            if runner.adopt_or_fail(ctx, job, attempt) == "adopted":
                adopted.append(attempt["id"])
        with ctx.db_lock:
            fresh = repos.get_job(ctx.conn, job["id"])
        if fresh is None or fresh["state"] in terminal:
            continue
        spec = job.get("spec", {})
        used = len(attempts)
        if used < int(spec.get("max_attempts", 3)):
            with ctx.db_lock:
                attempt = repos.create_attempt(
                    ctx.conn,
                    attempt_id=repos.new_id("att"),
                    job_id=job["id"],
                    provider_id=_rerun_provider(ctx, attempts, spec),
                )
                operation = repos.create_operation(
                    ctx.conn,
                    kind="run.job",
                    idempotency_key=f"run-{job['id']}-{used + 1}",
                    payload={"job_id": job["id"], "attempt_id": attempt["id"]},
                )
                repos.set_job_state(ctx.conn, job["id"], "queued")
                repos.audit(ctx.conn, "daemon", "job.rerun_scheduled", job["id"],
                            {"attempt": attempt["id"], "operation": operation["id"]})
            requeued.append(job["id"])
            ctx.executor.submit(dict(operation))
        else:
            with ctx.db_lock:
                repos.set_job_state(ctx.conn, job["id"], "failed")
            failed_jobs.append(job["id"])
    return {"adopted": adopted, "requeued": requeued, "failed": failed_jobs}


def _rerun_provider(ctx, attempts: list, spec: dict) -> str | None:
    if attempts and attempts[-1].get("provider_id"):
        return attempts[-1]["provider_id"]
    return spec.get("compute_provider")
