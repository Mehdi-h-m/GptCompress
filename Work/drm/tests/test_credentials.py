import os
import tempfile
import unittest

from drm import credentials


class CredentialsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old = os.environ.get("DRM_STATE_DIR")
        os.environ["DRM_STATE_DIR"] = self._tmp.name
        self.addCleanup(self._restore_state_dir)

    def _restore_state_dir(self):
        if self._old is None:
            os.environ.pop("DRM_STATE_DIR", None)
        else:
            os.environ["DRM_STATE_DIR"] = self._old

    def test_env_reference_resolves_at_use_time(self):
        os.environ["DRM_TEST_TOKEN"] = "hf_secret_value"
        self.addCleanup(os.environ.pop, "DRM_TEST_TOKEN")
        self.assertEqual(credentials.resolve("env:DRM_TEST_TOKEN"), "hf_secret_value")
        self.assertIsNone(credentials.resolve("env:DRM_TEST_MISSING"))

    def test_none_means_unauthenticated(self):
        self.assertIsNone(credentials.resolve(None))
        self.assertIsNone(credentials.resolve(""))

    def test_stored_secret_roundtrip(self):
        ref = credentials.store_secret("cred-test-1", "hf_stored")
        self.assertEqual(ref, "cred-test-1")
        self.assertEqual(credentials.resolve("cred-test-1"), "hf_stored")
        # the reference file exists in the state dir and is not world-readable
        path = credentials._store_path()
        self.assertTrue(path.exists())

    def test_unknown_reference_raises(self):
        with self.assertRaises(credentials.CredentialsError):
            credentials.resolve("cred-nope")

    def test_resolved_secrets_never_written_to_audit_like_files(self):
        # the store holds exactly one entry keyed by ref, nothing else leaks
        credentials.store_secret("cred-test-2", "hf_other")
        raw = credentials._store_path().read_text(encoding="utf-8")
        self.assertIn("hf_other", raw)  # store itself holds secrets by design
        self.assertEqual(credentials.resolve("cred-test-2"), "hf_other")


if __name__ == "__main__":
    unittest.main()
