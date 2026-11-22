import sqlite3
import tempfile
import unittest
from pathlib import Path

from drm.store.db import connect, migrate, transaction, utcnow
from drm.store.ids import new_id

PROVIDER_SQL = (
    "INSERT INTO providers (id, adapter, display_name, kind, execution, state,"
    " created_at, updated_at) VALUES (?, 'mock', 'Mock', 'combined', 'api', 'active', ?, ?)"
)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)  # release WAL handles for Windows temp cleanup

    def test_migrate_creates_all_tables_once(self):
        applied = migrate(self.conn)
        self.assertGreaterEqual(applied, 1)
        names = {
            row["name"]
            for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in (
            "providers",
            "artifacts",
            "artifact_copies",
            "jobs",
            "attempts",
            "operations",
            "decisions",
            "ledger_entries",
            "reservations",
            "audit_log",
            "schema_migrations",
        ):
            self.assertIn(table, names)
        self.assertEqual(migrate(self.conn), 0)

    def test_transaction_commit_and_rollback(self):
        migrate(self.conn)
        now = utcnow()
        with transaction(self.conn):
            self.conn.execute(PROVIDER_SQL, ("p-ok", now, now))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) AS n FROM providers").fetchone()["n"], 1)
        with self.assertRaises(RuntimeError):
            with transaction(self.conn):
                self.conn.execute(PROVIDER_SQL, ("p-rolled-back", now, now))
                raise RuntimeError("forced failure")
        rows = [row["id"] for row in self.conn.execute("SELECT id FROM providers")]
        self.assertEqual(rows, ["p-ok"])

    def test_foreign_keys_enforced(self):
        migrate(self.conn)
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO artifact_copies (id, artifact_id, provider_id, locator,"
                " status, created_at, updated_at)"
                " VALUES ('c1', 'missing-artifact', 'missing-provider', 'x', 'durable', ?, ?)",
                (utcnow(), utcnow()),
            )

    def test_state_check_constraints(self):
        migrate(self.conn)
        now = utcnow()
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO providers (id, adapter, display_name, kind, execution, state,"
                " created_at, updated_at)"
                " VALUES (?, 'mock', 'Mock', 'combined', 'psychic', 'active', ?, ?)",
                ("p-bad", now, now),
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.conn.execute(
                "INSERT INTO jobs (id, state, created_at, updated_at)"
                " VALUES ('j1', 'teleported', ?, ?)",
                (now, now),
            )

    def test_new_id_shape(self):
        self.assertRegex(new_id("op"), r"^op-[0-9a-f]{8}$")
        self.assertNotEqual(new_id("op"), new_id("op"))


if __name__ == "__main__":
    unittest.main()
