import tempfile
import unittest
from pathlib import Path

from drm.catalog.errors import CatalogError
from drm.providers.filesystem import FilesystemStorage
from drm.providers.registry import (
    ADAPTERS,
    build_bundle,
    connect,
    probe,
    storage_for_row,
    validate_config,
)
from drm.store import repos
from drm.store.db import connect as db_connect, migrate


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = db_connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)
        migrate(self.conn)

    def test_adapter_catalog(self):
        self.assertIn("filesystem", ADAPTERS)
        self.assertIn("local-runner", ADAPTERS)
        self.assertIn("mock", ADAPTERS)

    def test_validate_config_rejects_unknown_adapter_and_missing_root(self):
        with self.assertRaises(CatalogError):
            validate_config("ftp", {})
        with self.assertRaises(CatalogError):
            validate_config("filesystem", {})

    def test_connect_creates_store_root_and_is_idempotent_by_name(self):
        root = Path(self._tmp.name) / "store-a"
        provider, created = connect(self.conn, {"adapter": "filesystem", "name": "store-a", "config": {"root": str(root)}})
        self.assertTrue(created)
        self.assertTrue(root.is_dir())
        again, created = connect(self.conn, {"adapter": "filesystem", "name": "store-a", "config": {"root": str(root)}})
        self.assertFalse(created)
        self.assertEqual(again["id"], provider["id"])
        self.assertEqual(len(repos.list_providers(self.conn)), 1)

    def test_connect_requires_name(self):
        with self.assertRaises(CatalogError) as ctx:
            connect(self.conn, {"adapter": "filesystem", "config": {"root": "x"}})
        self.assertEqual(ctx.exception.code, "validation")

    def test_build_bundle_per_adapter(self):
        fs = build_bundle("filesystem", {"root": str(Path(self._tmp.name) / "r")})
        self.assertIsInstance(fs.storage, FilesystemStorage)
        self.assertIsNone(fs.compute)
        local = build_bundle("local-runner", {})
        self.assertIsNotNone(local.compute)
        self.assertIsNone(local.storage)
        mock = build_bundle("mock", {"name": "m"})
        self.assertIsNotNone(mock.compute)
        self.assertIsNotNone(mock.storage)

    def test_storage_for_row_rejects_compute_only_provider(self):
        row = repos.create_provider(
            self.conn, adapter="local-runner", display_name="cpu", kind="compute",
            execution="api", config={},
        )
        with self.assertRaises(CatalogError):
            storage_for_row(row)

    def test_probe_reports_faces_with_timestamps(self):
        root = Path(self._tmp.name) / "store-b"
        row = repos.create_provider(
            self.conn, adapter="filesystem", display_name="store-b", kind="storage",
            execution="api", config={"root": str(root)},
        )
        result = probe(row)
        self.assertIn("storage", result["faces"])
        self.assertIn("observed_at", result["faces"]["storage"])


if __name__ == "__main__":
    unittest.main()
