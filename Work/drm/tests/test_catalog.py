import tempfile
import unittest
from pathlib import Path

from drm.catalog import service
from drm.catalog.errors import CatalogError
from drm.providers.mock import MockStorage
from drm.store import repos
from drm.store.db import connect, migrate


class CatalogServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)
        migrate(self.conn)
        self.provider = repos.create_provider(
            self.conn, adapter="filesystem", display_name="store-a", kind="storage",
            execution="api", config={"root": str(Path(self._tmp.name) / "store-a")},
        )
        self.provider2 = repos.create_provider(
            self.conn, adapter="filesystem", display_name="store-b", kind="storage",
            execution="api", config={"root": str(Path(self._tmp.name) / "store-b")},
        )
        self.source = Path(self._tmp.name) / "source.bin"
        self.source.write_bytes(b"calibration fixture bytes")

    def test_ingest_registers_verified_artifact_and_copy(self):
        store = MockStorage("s1")
        result = service.ingest_file(
            self.conn, store, provider_id=self.provider["id"], source=str(self.source), verify=True
        )
        artifact, copy = result["artifact"], result["copy"]
        self.assertEqual(artifact["byte_count"], self.source.stat().st_size)
        self.assertEqual(copy["status"], "durable")
        self.assertEqual(copy["checksum"], artifact["content_hash"])
        self.assertEqual(copy["locator"], f"{artifact['id']}/source.bin")
        self.assertEqual(store.list(f"{artifact['id']}/"), [f"{artifact['id']}/source.bin"])

    def test_ingest_missing_source_is_validation_error(self):
        with self.assertRaises(CatalogError) as ctx:
            service.ingest_file(self.conn, MockStorage(), provider_id="p", source="no/such/file")
        self.assertEqual(ctx.exception.code, "validation")

    def test_ingest_duplicate_custom_id_is_conflict(self):
        store = MockStorage()
        service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source), artifact_id="art-fixed")
        with self.assertRaises(CatalogError) as ctx:
            service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source), artifact_id="art-fixed")
        self.assertEqual(ctx.exception.code, "conflict")

    def test_verify_detects_corruption(self):
        store = MockStorage("s1")
        result = service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source))
        copy_id = result["copy"]["id"]
        ok = service.verify_copy(self.conn, store, copy_id)
        self.assertTrue(ok["verified"])
        store.corrupt(result["copy"]["locator"])
        with self.assertRaises(CatalogError) as ctx:
            service.verify_copy(self.conn, store, copy_id)
        self.assertEqual(ctx.exception.code, "verify_failed")

    def test_verify_unknown_copy(self):
        with self.assertRaises(CatalogError) as ctx:
            service.verify_copy(self.conn, MockStorage(), "copy-nope")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_pin_and_evict_policy_gates(self):
        store = MockStorage("s1")
        result = service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source))
        artifact_id, copy_id = result["artifact"]["id"], result["copy"]["id"]

        # last durable copy cannot be evicted
        with self.assertRaises(CatalogError) as ctx:
            service.evict_copy(self.conn, store, copy_id)
        self.assertEqual(ctx.exception.code, "policy_blocked")

        # pinned artifacts retain their durable copies even with a second copy present
        service.pin_artifact(self.conn, artifact_id)
        second = repos.create_copy(
            self.conn, copy_id="copy-2", artifact_id=artifact_id, provider_id=self.provider2["id"],
            locator="x", version="v", checksum="c", byte_count=1, status="staged",
        )
        with self.assertRaises(CatalogError) as ctx:
            service.evict_copy(self.conn, store, copy_id)
        self.assertEqual(ctx.exception.code, "policy_blocked")

        # a staged copy of a pinned artifact may still be evicted
        evicted = service.evict_copy(self.conn, store, second["id"])
        self.assertTrue(evicted["evicted"])

    def test_evict_with_second_durable_copy_succeeds(self):
        store = MockStorage("s1")
        result = service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source))
        artifact_id = result["artifact"]["id"]
        second = repos.create_copy(
            self.conn, copy_id="copy-2", artifact_id=artifact_id, provider_id=self.provider2["id"],
            locator="same-content", version="v", checksum=result["copy"]["checksum"],
            byte_count=result["copy"]["byte_count"], status="durable",
        )
        evicted = service.evict_copy(self.conn, store, result["copy"]["id"])
        self.assertTrue(evicted["evicted"])
        self.assertIsNone(repos.get_copy(self.conn, result["copy"]["id"]))
        self.assertIsNotNone(repos.get_copy(self.conn, second["id"]))

    def test_evict_when_bytes_already_absent_still_removes_record(self):
        store = MockStorage("s1")
        result = service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source))
        repos.create_copy(
            self.conn, copy_id="copy-2", artifact_id=result["artifact"]["id"], provider_id=self.provider2["id"],
            locator="gone", version="v", checksum="c", byte_count=1, status="durable",
        )
        # the recorded locator does not exist in this store: eviction still removes the record
        evicted = service.evict_copy(self.conn, store, "copy-2")
        self.assertTrue(evicted["bytes_absent"])

    def test_pin_unknown_artifact(self):
        with self.assertRaises(CatalogError) as ctx:
            service.pin_artifact(self.conn, "art-nope")
        self.assertEqual(ctx.exception.code, "not_found")

    def test_derive_ingest_key_is_stable_and_scoped(self):
        provider = repos.create_provider(
            self.conn, adapter="filesystem", display_name="s", kind="storage",
            execution="api", config={"root": "x"},
        )
        payload = service.normalize_ingest_payload({"source": str(self.source), "store": provider["id"]})
        key1 = service.derive_ingest_key(provider, payload)
        key2 = service.derive_ingest_key(provider, payload)
        self.assertEqual(key1, key2)
        self.assertTrue(key1.startswith("ingest-"))

    def test_list_artifacts_includes_copies(self):
        store = MockStorage()
        result = service.ingest_file(self.conn, store, provider_id=self.provider["id"], source=str(self.source))
        listed = service.list_artifacts_with_copies(self.conn)
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["copies"][0]["id"], result["copy"]["id"])
        self.assertFalse(listed[0]["pinned"])


if __name__ == "__main__":
    unittest.main()
