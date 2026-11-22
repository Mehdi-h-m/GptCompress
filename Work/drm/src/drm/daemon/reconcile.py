"""Startup reconciliation (spec Sections 3 and 9).

The coordinator owns every operation's state. When the process dies, its
in-flight operations can neither finish nor record an outcome, so on restart
they are moved to a terminal ``failed`` state with an ``interrupted`` error
record — never left ``running`` forever. Half-written staging files in
approved filesystem stores are swept the same way; cataloged copies were
published atomically and are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

from drm.contracts import ErrorCodes
from drm.providers.filesystem import FilesystemStorage
from drm.store import repos
from drm.store.db import transaction, utcnow


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
    return {"released_operations": released, "removed_partials": removed_partials}
