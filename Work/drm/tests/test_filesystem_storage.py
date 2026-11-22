import tempfile
import unittest
from pathlib import Path

from drm.providers.base import AdapterError
from drm.providers.filesystem import FilesystemStorage


class FilesystemStorageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name) / "store"
        self.store = FilesystemStorage(self.root)

    def test_write_read_roundtrip_is_atomic(self):
        blob = b"artifact bytes"
        result = self.store.write("art-1/blob.bin", blob)
        self.assertEqual(result["byte_count"], len(blob))
        self.assertEqual(self.store.read("art-1/blob.bin"), blob)
        self.assertEqual(list(Path(self.root).rglob("*.drm-partial*")), [])
        head = self.store.head("art-1/blob.bin")
        self.assertTrue(head["exists"])
        self.assertEqual(head["byte_count"], len(blob))

    def test_write_checksum_mismatch_leaves_no_object(self):
        with self.assertRaises(AdapterError):
            self.store.write("x", b"data", checksum="deadbeef")
        self.assertFalse((self.root / "x").exists())
        self.assertEqual(list(Path(self.root).rglob("*.drm-partial*")), [])

    def test_read_missing_raises(self):
        with self.assertRaises(AdapterError):
            self.store.read("missing")

    def test_locator_traversal_rejected(self):
        with self.assertRaises(AdapterError):
            self.store.write("../escape", b"x")

    def test_list_prefix(self):
        self.store.write("a/one", b"1")
        self.store.write("b/two", b"2")
        self.assertEqual(self.store.list("a/"), ["a/one"])

    def test_delete_missing_raises(self):
        with self.assertRaises(AdapterError):
            self.store.delete("gone")

    def test_cleanup_partials_removes_staging_files(self):
        staging = self.root / "sub"
        staging.mkdir(parents=True)
        (staging / "blob.bin.abc123.drm-partial").write_bytes(b"half-written")
        self.assertEqual(self.store.cleanup_partials(), 1)
        self.assertEqual(list(self.root.rglob("*drm-partial*")), [])

    def test_probe_reports_root_state(self):
        self.root.mkdir(parents=True)
        probe = self.store.probe()
        self.assertTrue(probe["exists"])
        self.assertTrue(probe["writable"])
        self.assertGreater(probe["free_bytes"], 0)
        self.assertIn("observed_at", probe)


if __name__ == "__main__":
    unittest.main()
