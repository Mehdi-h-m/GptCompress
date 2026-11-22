import tempfile
import unittest
from pathlib import Path

from drm.store import repos
from drm.store.db import connect, migrate


class ReposTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)
        migrate(self.conn)

    def test_provider_roundtrip(self):
        row = repos.create_provider(
            self.conn,
            adapter="filesystem",
            display_name="store-a",
            kind="storage",
            execution="api",
            config={"root": "C:/tmp/a"},
        )
        self.assertEqual(row["state"], "active")
        self.assertEqual(repos.find_provider_by_name(self.conn, "filesystem", "store-a")["id"], row["id"])
        self.assertIsNone(repos.find_provider_by_name(self.conn, "filesystem", "other"))
        updated = repos.set_provider_state(self.conn, row["id"], "paused")
        self.assertEqual(updated["state"], "paused")
        self.assertEqual(len(repos.list_providers(self.conn)), 1)

    def test_artifact_and_copy_roundtrip(self):
        artifact = repos.create_artifact(
            self.conn, artifact_id="art-1", kind="dataset_release",
            content_hash="a" * 64, byte_count=10,
        )
        self.assertFalse(artifact["pinned"])
        provider = repos.create_provider(
            self.conn, adapter="filesystem", display_name="s", kind="storage",
            execution="api", config={},
        )
        copy = repos.create_copy(
            self.conn, copy_id="copy-1", artifact_id="art-1", provider_id=provider["id"],
            locator="art-1/x.bin", version="a" * 12, checksum="a" * 64, byte_count=10,
        )
        self.assertEqual(copy["status"], "durable")
        self.assertEqual(len(repos.list_copies(self.conn, "art-1")), 1)
        repos.delete_copy(self.conn, "copy-1")
        self.assertEqual(repos.list_copies(self.conn, "art-1"), [])

    def test_operation_lifecycle_and_key_supersede(self):
        op = repos.create_operation(self.conn, kind="data.ingest", idempotency_key="k1", payload={"a": 1})
        self.assertEqual(op["state"], "queued")
        self.assertEqual(repos.find_operation_by_key(self.conn, "k1")["id"], op["id"])
        repos.set_operation_state(self.conn, op["id"], "failed", error={"code": "verify_failed", "message": "x"})
        self.assertEqual(repos.get_operation(self.conn, op["id"])["error_json"], '{"code": "verify_failed", "message": "x"}')

        # a terminal-failed operation releases its key for a rerun
        repos.supersede_operation_key(self.conn, op["id"])
        self.assertIsNone(repos.find_operation_by_key(self.conn, "k1"))
        retry = repos.create_operation(self.conn, kind="data.ingest", idempotency_key="k1")
        self.assertEqual(repos.get_operation(self.conn, retry["id"])["state"], "queued")

        # active keys are returned, not superseded
        self.assertIn(repos.get_operation(self.conn, op["id"])["state"], ("failed",))
        self.assertEqual(len(repos.list_unfinished_operations(self.conn)), 1)

    def test_audit_rows_recorded(self):
        repos.audit(self.conn, "test", "did.thing", "subj", {"k": "v"})
        rows = repos.list_audit(self.conn)
        self.assertEqual(rows[0]["action"], "did.thing")


if __name__ == "__main__":
    unittest.main()
