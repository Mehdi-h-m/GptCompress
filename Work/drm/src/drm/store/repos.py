"""Row-level data access for the control-plane tables.

All functions expect the caller to hold the daemon's database lock (routes
and the executor own locking); catalog writes that must be atomic use
:func:`drm.store.db.transaction` internally.
"""

from __future__ import annotations

import json

from drm.store.db import utcnow
from drm.store.ids import new_id


def _json(value) -> str:
    return json.dumps(value, sort_keys=True)


def _load(value: str | None, default):
    if not value:
        return default
    return json.loads(value)


# --- audit log ---------------------------------------------------------------


def audit(conn, actor: str, action: str, subject: str | None = None, details: dict | None = None) -> None:
    conn.execute(
        "INSERT INTO audit_log (at, actor, action, subject, details_json)"
        " VALUES (?, ?, ?, ?, ?)",
        (utcnow(), actor, action, subject, _json(details or {})),
    )


def list_audit(conn, limit: int = 50) -> list:
    return conn.execute(
        "SELECT * FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,)
    ).fetchall()


# --- provider instances ------------------------------------------------------


def create_provider(
    conn,
    *,
    adapter: str,
    display_name: str,
    kind: str,
    execution: str,
    config: dict,
    credential_ref: str | None = None,
    region: str | None = None,
):
    provider_id = new_id("prov")
    now = utcnow()
    conn.execute(
        "INSERT INTO providers (id, adapter, display_name, kind, execution, state,"
        " region, credential_ref, config_json, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
        (provider_id, adapter, display_name, kind, execution, region, credential_ref, _json(config), now, now),
    )
    return get_provider(conn, provider_id)


def get_provider(conn, provider_id: str):
    return conn.execute("SELECT * FROM providers WHERE id = ?", (provider_id,)).fetchone()


def find_provider_by_name(conn, adapter: str, display_name: str):
    return conn.execute(
        "SELECT * FROM providers WHERE adapter = ? AND display_name = ?",
        (adapter, display_name),
    ).fetchone()


def list_providers(conn) -> list:
    return conn.execute("SELECT * FROM providers ORDER BY created_at, id").fetchall()


def set_provider_state(conn, provider_id: str, state: str):
    conn.execute(
        "UPDATE providers SET state = ?, updated_at = ? WHERE id = ?",
        (state, utcnow(), provider_id),
    )
    return get_provider(conn, provider_id)


# --- artifacts and copies ----------------------------------------------------


def create_artifact(
    conn,
    *,
    artifact_id: str,
    kind: str,
    content_hash: str,
    byte_count: int,
    format: str | None = None,
    split_identity: str | None = None,
    access_class: str = "private",
    parents: list | None = None,
    producer: str | None = None,
):
    now = utcnow()
    conn.execute(
        "INSERT INTO artifacts (id, kind, content_hash, hash_algorithm, byte_count,"
        " format, split_identity, access_class, parents_json, producer, pinned,"
        " created_at, updated_at)"
        " VALUES (?, ?, ?, 'sha256', ?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (artifact_id, kind, content_hash, byte_count, format, split_identity, access_class, _json(parents or []), producer, now, now),
    )
    return get_artifact(conn, artifact_id)


def get_artifact(conn, artifact_id: str):
    return conn.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()


def list_artifacts(conn) -> list:
    return conn.execute("SELECT * FROM artifacts ORDER BY created_at, id").fetchall()


def set_artifact_pinned(conn, artifact_id: str, pinned: int) -> None:
    conn.execute(
        "UPDATE artifacts SET pinned = ?, updated_at = ? WHERE id = ?",
        (pinned, utcnow(), artifact_id),
    )


def create_copy(
    conn,
    *,
    copy_id: str,
    artifact_id: str,
    provider_id: str,
    locator: str,
    version: str,
    checksum: str,
    byte_count: int,
    status: str = "durable",
    retention: str | None = None,
):
    now = utcnow()
    conn.execute(
        "INSERT INTO artifact_copies (id, artifact_id, provider_id, locator, version,"
        " checksum, hash_algorithm, byte_count, status, retention, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, 'sha256', ?, ?, ?, ?, ?)",
        (copy_id, artifact_id, provider_id, locator, version, checksum, byte_count, status, retention, now, now),
    )
    return get_copy(conn, copy_id)


def get_copy(conn, copy_id: str):
    return conn.execute("SELECT * FROM artifact_copies WHERE id = ?", (copy_id,)).fetchone()


def list_copies(conn, artifact_id: str) -> list:
    return conn.execute(
        "SELECT * FROM artifact_copies WHERE artifact_id = ? ORDER BY created_at, id",
        (artifact_id,),
    ).fetchall()


def delete_copy(conn, copy_id: str) -> None:
    conn.execute("DELETE FROM artifact_copies WHERE id = ?", (copy_id,))


# --- operations --------------------------------------------------------------


def create_operation(conn, *, kind: str, idempotency_key: str | None = None, payload: dict | None = None, op_id: str | None = None):
    op_id = op_id or new_id("op")
    now = utcnow()
    conn.execute(
        "INSERT INTO operations (id, kind, idempotency_key, state, payload_json, created_at, updated_at)"
        " VALUES (?, ?, ?, 'queued', ?, ?, ?)",
        (op_id, kind, idempotency_key, _json(payload or {}), now, now),
    )
    return get_operation(conn, op_id)


def get_operation(conn, op_id: str):
    return conn.execute("SELECT * FROM operations WHERE id = ?", (op_id,)).fetchone()


def operation_dict(row) -> dict:
    """API boundary view of an operation: JSON columns parsed into dicts."""
    data = dict(row)
    data["payload"] = _load(data.pop("payload_json"), {})
    data["result"] = _load(data.pop("result_json"), None)
    data["error"] = _load(data.pop("error_json"), None)
    return data


def find_operation_by_key(conn, key: str):
    return conn.execute(
        "SELECT * FROM operations WHERE idempotency_key = ?", (key,)
    ).fetchone()


def set_operation_state(conn, op_id: str, state: str, result: dict | None = None, error: dict | None = None) -> None:
    conn.execute(
        "UPDATE operations SET state = ?, result_json = ?, error_json = ?, updated_at = ? WHERE id = ?",
        (state, _json(result) if result is not None else None, _json(error) if error is not None else None, utcnow(), op_id),
    )


def supersede_operation_key(conn, op_id: str) -> None:
    """Free an idempotency key held by a terminal-failed operation so it can run again."""
    row = get_operation(conn, op_id)
    if row is None or not row["idempotency_key"]:
        return
    conn.execute(
        "UPDATE operations SET idempotency_key = idempotency_key || '#' || id, updated_at = ? WHERE id = ?",
        (utcnow(), op_id),
    )


def list_unfinished_operations(conn) -> list:
    return conn.execute(
        "SELECT * FROM operations WHERE state IN ('queued','running') ORDER BY created_at, id"
    ).fetchall()


# --- jobs --------------------------------------------------------------------

TERMINAL_JOB_STATES = ("complete", "failed", "cancelled")


def create_job(conn, *, job_id: str, name: str | None, experiment_id: str | None,
               spec: dict, numerical_profile: str | None = None, state: str = "queued"):
    now = utcnow()
    conn.execute(
        "INSERT INTO jobs (id, experiment_id, name, dag_json, numerical_profile, state, created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, experiment_id, name, _json(spec), numerical_profile, state, now, now),
    )
    return get_job(conn, job_id)


def get_job(conn, job_id: str):
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def set_job_state(conn, job_id: str, state: str) -> None:
    conn.execute(
        "UPDATE jobs SET state = ?, updated_at = ? WHERE id = ?", (state, utcnow(), job_id)
    )


def list_jobs(conn, states: tuple[str, ...] | None = None) -> list:
    if states:
        marks = ",".join("?" for _ in states)
        return conn.execute(
            f"SELECT * FROM jobs WHERE state IN ({marks}) ORDER BY created_at, id", states
        ).fetchall()
    return conn.execute("SELECT * FROM jobs ORDER BY created_at, id").fetchall()


def job_dict(row) -> dict:
    data = dict(row)
    data["spec"] = _load(data.pop("dag_json"), {})
    return data


# --- attempts ----------------------------------------------------------------


def create_attempt(conn, *, attempt_id: str, job_id: str, provider_id: str | None,
                   state: str = "queued", operation_id: str | None = None):
    now = utcnow()
    conn.execute(
        "INSERT INTO attempts (id, job_id, provider_id, state, lease_json, operation_id, created_at)"
        " VALUES (?, ?, ?, ?, '{}', ?, ?)",
        (attempt_id, job_id, provider_id, state, operation_id, now),
    )
    return get_attempt(conn, attempt_id)


def get_attempt(conn, attempt_id: str):
    return conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()


def list_attempts(conn, job_id: str) -> list:
    return conn.execute(
        "SELECT * FROM attempts WHERE job_id = ? ORDER BY created_at, id", (job_id,)
    ).fetchall()


def count_attempts(conn, job_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS n FROM attempts WHERE job_id = ?", (job_id,)
    ).fetchone()["n"]


def set_attempt_state(conn, attempt_id: str, state: str, result: dict | None = None,
                      error: dict | None = None) -> None:
    sets = ["state = ?"]
    params: list = [state]
    if result is not None:
        sets.append("result_json = ?")
        params.append(_json(result))
    if error is not None:
        sets.append("error_json = ?")
        params.append(_json(error))
    params.append(attempt_id)
    conn.execute(f"UPDATE attempts SET {', '.join(sets)} WHERE id = ?", params)


def attempt_dict(row) -> dict:
    data = dict(row)
    data["result"] = _load(data.pop("result_json"), None)
    data["error"] = _load(data.pop("error_json"), None)
    data["lease"] = _load(data.pop("lease_json"), {})
    return data


# --- decisions ---------------------------------------------------------------


def create_decision(conn, *, subject: str, subject_id: str, chosen: str,
                    inputs: dict, alternatives: list, reason: str | None = None):
    decision_id = new_id("dec")
    conn.execute(
        "INSERT INTO decisions (id, subject, subject_id, chosen, inputs_json,"
        " alternatives_json, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (decision_id, subject, subject_id, chosen, _json(inputs), _json(alternatives), reason, utcnow()),
    )
    return get_decision(conn, decision_id)


def get_decision(conn, decision_id: str):
    return conn.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()


def decision_dict(row) -> dict:
    data = dict(row)
    try:
        data["chosen"] = json.loads(data["chosen"])
    except (TypeError, ValueError):
        pass
    data["inputs"] = _load(data.pop("inputs_json"), {})
    data["alternatives"] = _load(data.pop("alternatives_json"), [])
    return data
