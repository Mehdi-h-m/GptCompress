"""API-level Sprint 1 acceptance: registry, catalog flows and restart recovery."""

import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from drm.daemon.app import create_app
from drm.providers.filesystem import FilesystemStorage
from drm.store import repos
from drm.store.db import connect as db_connect, migrate


def _wait_operation(client, op_id, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/operations/{op_id}").json()
        if body["operation"]["state"] in ("complete", "failed", "cancelled"):
            return body["operation"]
        time.sleep(0.1)
    raise AssertionError(f"operation {op_id} never reached a terminal state")


class Sprint1ApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.app = create_app(Path(self._tmp.name) / "drm.db")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.source = Path(self._tmp.name) / "fixture.bin"
        self.source.write_bytes(b"sprint 1 fixture")
        self.store_a = self._connect_store("store-a")
        self.store_b = self._connect_store("store-b")

    def _connect_store(self, name):
        root = Path(self._tmp.name) / name
        response = self.client.post(
            "/providers",
            json={"adapter": "filesystem", "name": name, "config": {"root": str(root)}},
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["provider"]

    def test_providers_connect_list_pause_resume(self):
        response = self.client.post(
            "/providers", json={"adapter": "local-runner", "name": "laptop-cpu"}
        )
        self.assertEqual(response.status_code, 200)
        provider = response.json()["provider"]
        self.assertFalse(response.json()["already_connected"])

        listed = self.client.get("/providers").json()["providers"]
        self.assertEqual({p["adapter"] for p in listed}, {"filesystem", "local-runner"})
        self.assertEqual(len(listed), 3)

        duplicate = self.client.post(
            "/providers", json={"adapter": "local-runner", "name": "laptop-cpu"}
        )
        self.assertTrue(duplicate.json()["already_connected"])

        paused = self.client.post(f"/providers/{provider['id']}/pause", json={})
        self.assertEqual(paused.json()["provider"]["state"], "paused")
        resumed = self.client.post(f"/providers/{provider['id']}/resume", json={})
        self.assertEqual(resumed.json()["provider"]["state"], "active")

        missing = self.client.post("/providers/prov-nope/pause", json={})
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "not_found")

    def test_probe_reports_observed_state(self):
        response = self.client.post(f"/providers/{self.store_a['id']}/probe", json={})
        self.assertEqual(response.status_code, 200)
        storage = response.json()["probe"]["faces"]["storage"]
        self.assertTrue(storage["writable"])
        self.assertIn("observed_at", storage)

    def test_ingest_verify_and_list_flow(self):
        response = self.client.post(
            "/data/ingest",
            json={"source": str(self.source), "store": self.store_a["id"], "verify": True},
        )
        self.assertEqual(response.status_code, 200, response.text)
        operation = _wait_operation(self.client, response.json()["operation"]["id"])
        self.assertEqual(operation["state"], "complete", operation)
        result = operation["result"]
        artifact_id, copy_id = result["artifact"]["id"], result["copy"]["id"]

        listed = self.client.get("/data/artifacts").json()["artifacts"]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["copies"][0]["id"], copy_id)
        self.assertEqual(listed[0]["copies"][0]["status"], "durable")

        status = self.client.get("/status").json()["counts"]
        self.assertEqual(status["providers"], 2)
        self.assertEqual(status["artifacts"], 1)
        self.assertEqual(status["operations"], 1)

        detail = self.client.get(f"/data/artifacts/{artifact_id}")
        self.assertEqual(detail.status_code, 200)
        unknown = self.client.get("/data/artifacts/art-nope")
        self.assertEqual(unknown.status_code, 404)

    def test_verify_fails_after_corruption(self):
        operation = self._ingest_fixture()
        copy = operation["result"]["copy"]
        locator = copy["locator"]
        root = Path(self.store_a["config"]["root"])
        target = root / locator
        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF
        target.write_bytes(bytes(data))

        response = self.client.post("/data/verify", json={"copy_id": copy["id"]})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "verify_failed")

    def test_ingest_without_verify_still_registers_copy(self):
        response = self.client.post(
            "/data/ingest",
            json={"source": str(self.source), "store": self.store_a["id"], "verify": False},
        )
        operation = _wait_operation(self.client, response.json()["operation"]["id"])
        self.assertEqual(operation["state"], "complete")
        self.assertIsNotNone(operation["result"]["copy"]["checksum"])

    def test_pin_blocks_eviction_but_second_copy_may_go(self):
        operation = self._ingest_fixture()
        artifact_id = operation["result"]["artifact"]["id"]
        copy_id = operation["result"]["copy"]["id"]

        pinned = self.client.post("/data/pin", json={"artifact_id": artifact_id})
        self.assertEqual(pinned.status_code, 200)
        self.assertTrue(pinned.json()["operation"]["result"]["pinned"])

        blocked = self.client.post("/data/evict", json={"copy_id": copy_id})
        self.assertEqual(blocked.status_code, 400)
        self.assertEqual(blocked.json()["error"]["code"], "policy_blocked")

        # a second durable copy lifts the last-copy gate; the original may then go
        db = Path(self._tmp.name) / "drm.db"
        conn = db_connect(db)
        try:
            repos.create_copy(
                conn, copy_id="copy-extra", artifact_id=artifact_id,
                provider_id=self.store_b["id"], locator="extra", version="v",
                checksum="c", byte_count=1, status="durable",
            )
        finally:
            conn.close()

        evicted = self.client.post("/data/evict", json={"copy_id": copy_id})
        self.assertEqual(evicted.status_code, 200)
        self.assertTrue(evicted.json()["operation"]["result"]["evicted"])

    def test_dry_runs_create_nothing(self):
        plan = self.client.post(
            "/data/ingest",
            json={"source": str(self.source), "store": self.store_a["id"], "dry_run": True},
        )
        self.assertEqual(plan.status_code, 200)
        self.assertTrue(plan.json()["dry_run"])
        self.assertGreater(plan.json()["plan"]["source_size"], 0)
        self.assertEqual(self.client.get("/data/artifacts").json()["artifacts"], [])

        connect_plan = self.client.post(
            "/providers",
            json={"adapter": "mock", "name": "would-be", "dry_run": True},
        )
        self.assertTrue(connect_plan.json()["dry_run"])
        self.assertEqual(len(self.client.get("/providers").json()["providers"]), 2)

    def test_idempotency_key_returns_original_operation(self):
        first = self.client.post(
            "/data/ingest",
            json={"source": str(self.source), "store": self.store_a["id"], "idempotency_key": "agent-key-1"},
        )
        _wait_operation(self.client, first.json()["operation"]["id"])
        second = self.client.post(
            "/data/ingest",
            json={"source": str(self.source), "store": self.store_a["id"], "idempotency_key": "agent-key-1"},
        )
        self.assertTrue(second.json()["reused"])
        self.assertEqual(second.json()["operation"]["id"], first.json()["operation"]["id"])
        self.assertEqual(len(self.client.get("/data/artifacts").json()["artifacts"]), 1)

    def test_ingest_unknown_store_is_not_found(self):
        response = self.client.post(
            "/data/ingest", json={"source": str(self.source), "store": "prov-nope"}
        )
        self.assertEqual(response.status_code, 404)

    def test_ingest_missing_source_is_validation(self):
        response = self.client.post(
            "/data/ingest", json={"source": "no/such/file", "store": self.store_a["id"]}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "validation")

    def test_operations_endpoint_unknown(self):
        response = self.client.get("/operations/op-nope")
        self.assertEqual(response.status_code, 404)

    def _ingest_fixture(self):
        response = self.client.post(
            "/data/ingest",
            json={"source": str(self.source), "store": self.store_a["id"], "verify": True},
        )
        return _wait_operation(self.client, response.json()["operation"]["id"])


class RestartRecoveryTests(unittest.TestCase):
    def test_abandoned_operations_fail_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "drm.db"
            conn = db_connect(db)
            try:
                migrate(conn)
                stale = repos.create_operation(conn, kind="data.ingest", payload={})
                repos.set_operation_state(conn, stale["id"], "running")
                stale_id = stale["id"]
            finally:
                conn.close()

            app = create_app(db)
            with TestClient(app) as client:
                body = client.get(f"/operations/{stale_id}").json()
                self.assertEqual(body["operation"]["state"], "failed")
                self.assertEqual(body["operation"]["error"]["code"], "interrupted")

    def test_partial_staging_files_swept_on_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "drm.db"
            store_root = Path(tmp) / "store"
            store_root.mkdir()
            conn = db_connect(db)
            try:
                migrate(conn)
                repos.create_provider(
                    conn, adapter="filesystem", display_name="sweep-me", kind="storage",
                    execution="api", config={"root": str(store_root)},
                )
            finally:
                conn.close()
            leftover = store_root / "blob.bin.abc.drm-partial"
            leftover.write_bytes(b"interrupted staging")
            self.assertEqual(len(list(store_root.rglob("*.drm-partial*"))), 1)

            app = create_app(db)
            with TestClient(app):
                self.assertEqual(list(store_root.rglob("*.drm-partial*")), [])


if __name__ == "__main__":
    unittest.main()
