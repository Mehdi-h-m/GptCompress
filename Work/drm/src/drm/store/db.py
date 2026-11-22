"""SQLite store: connections, versioned migrations, transactional helper.

SQLite is the transactional store for the single local coordinator
(spec Section 3). Money is recorded as integer micros of the entry currency.
Timestamps are ISO-8601 UTC strings.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import sqlite3
from collections.abc import Iterator
from pathlib import Path

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS providers (
    id TEXT PRIMARY KEY,
    adapter TEXT NOT NULL,
    display_name TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('compute','storage','combined')),
    execution TEXT NOT NULL CHECK (execution IN ('api','assisted','manual')),
    state TEXT NOT NULL CHECK (state IN ('active','paused','draining','unreachable')),
    region TEXT,
    credential_ref TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content_hash TEXT,
    hash_algorithm TEXT NOT NULL DEFAULT 'sha256',
    byte_count INTEGER,
    format TEXT,
    split_identity TEXT,
    access_class TEXT NOT NULL DEFAULT 'private',
    parents_json TEXT NOT NULL DEFAULT '[]',
    producer TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifact_copies (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL REFERENCES artifacts(id),
    provider_id TEXT NOT NULL REFERENCES providers(id),
    locator TEXT NOT NULL,
    version TEXT,
    checksum TEXT,
    hash_algorithm TEXT NOT NULL DEFAULT 'sha256',
    byte_count INTEGER,
    status TEXT NOT NULL CHECK (status IN ('durable','staged','rebuildable')),
    retention TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    name TEXT,
    dag_json TEXT NOT NULL DEFAULT '[]',
    numerical_profile TEXT,
    state TEXT NOT NULL CHECK (state IN (
        'queued','staging','ready','running','validating','complete',
        'retry_pending','failed','cancelled','awaiting_user','reconciling')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    shard_id TEXT,
    provider_id TEXT REFERENCES providers(id),
    state TEXT NOT NULL CHECK (state IN (
        'queued','staging','running','validating','complete','failed',
        'cancelled','awaiting_user','reconciling')),
    lease_json TEXT NOT NULL DEFAULT '{}',
    heartbeat_at TEXT,
    result_json TEXT,
    started_at TEXT,
    ended_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS operations (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    state TEXT NOT NULL CHECK (state IN (
        'queued','running','complete','failed','cancelled',
        'awaiting_user','reconciling')),
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    error_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    chosen TEXT NOT NULL,
    inputs_json TEXT NOT NULL,
    alternatives_json TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_entries (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    job_id TEXT,
    provider_id TEXT,
    category TEXT NOT NULL CHECK (category IN (
        'compute','storage','transfer','requests','credit','free_quota',
        'recurring','other')),
    kind TEXT NOT NULL CHECK (kind IN ('quoted','reserved','provisional','settled')),
    amount_micros INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reservations (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    provider_id TEXT,
    job_id TEXT,
    operation_id TEXT,
    amount_micros INTEGER NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    state TEXT NOT NULL CHECK (state IN ('held','released','settled')),
    created_at TEXT NOT NULL,
    released_at TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    subject TEXT,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_artifact_copies_artifact ON artifact_copies(artifact_id);
CREATE INDEX IF NOT EXISTS idx_attempts_job ON attempts(job_id);
CREATE INDEX IF NOT EXISTS idx_operations_idempotency ON operations(idempotency_key);
"""

_SCHEMA_V2 = """
ALTER TABLE artifacts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
"""

_SCHEMA_V3 = """
ALTER TABLE attempts ADD COLUMN error_json TEXT;
ALTER TABLE attempts ADD COLUMN operation_id TEXT;
"""

# (version, name, script)
MIGRATIONS: list[tuple[int, str, str]] = [
    (1, "core tables for the control plane", _SCHEMA_V1),
    (2, "artifact pinning", _SCHEMA_V2),
    (3, "attempt error and operation linkage", _SCHEMA_V3),
]


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open the store with foreign keys on and WAL journalling.

    The daemon uses one connection across its lifespan and worker threads,
    so ``check_same_thread`` is off; callers serialize access through
    :func:`transaction` or their own lock.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Apply pending migrations; each script is idempotent DDL. Returns the count applied."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
    )
    applied = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
    count = 0
    for version, name, script in MIGRATIONS:
        if version in applied:
            continue
        conn.executescript(script)
        conn.execute(
            "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
            (version, name, utcnow()),
        )
        count += 1
    return count


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block transactionally; roll back on any exception."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
