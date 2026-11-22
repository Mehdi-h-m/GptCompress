import unittest

from drm.providers.base import AdapterError, AmbiguousOutcome
from drm.providers.mock import MockCompute, MockStorage


class MockComputeTests(unittest.TestCase):
    def test_quote_submit_status_flow(self):
        provider = MockCompute("acct-a")
        attempt = provider.submit({"task_type": "noop", "manifest": {"id": "t1"}})
        self.assertEqual(attempt["state"], "queued")
        self.assertEqual(
            provider.status(attempt["attempt_id"])["attempt_id"], attempt["attempt_id"]
        )
        quote = provider.quote({})
        self.assertEqual(quote["currency"], "USD")
        self.assertIn("prices", quote)
        self.assertTrue(quote["unknowns"])
        self.assertIn("assumptions", quote["upper_estimate"])

    def test_injected_failure_then_success(self):
        provider = MockCompute()
        provider.injector.fail_on("submit", times=1)
        with self.assertRaises(AdapterError):
            provider.submit({})
        attempt = provider.submit({})  # second call succeeds
        self.assertIn("attempt_id", attempt)

    def test_ambiguous_submit_records_attempt_for_reconciliation(self):
        provider = MockCompute()
        provider.ambiguous_next_submit()
        with self.assertRaises(AmbiguousOutcome):
            provider.submit({})
        found = provider.status("mock-a-attempt-1")
        self.assertEqual(found["state"], "queued")

    def test_cancel_confirms_and_updates_state(self):
        provider = MockCompute()
        attempt_id = provider.submit({})["attempt_id"]
        self.assertTrue(provider.cancel(attempt_id)["cancelled"])
        self.assertEqual(provider.status(attempt_id)["state"], "cancelled")

    def test_unknown_attempt_raises(self):
        provider = MockCompute()
        with self.assertRaises(AdapterError):
            provider.status("does-not-exist")

    def test_probe_returns_observations(self):
        provider = MockCompute("acct-b")
        observed = provider.probe()
        self.assertEqual(observed["provider"], "acct-b")
        self.assertIn("observed_at", observed)


class MockStorageTests(unittest.TestCase):
    def test_write_read_roundtrip(self):
        store = MockStorage()
        blob = b"hello drm"
        result = store.write("a/blob.bin", blob)
        self.assertEqual(result["byte_count"], len(blob))
        self.assertEqual(store.read("a/blob.bin"), blob)
        self.assertIn("a/blob.bin", store.list("a/"))
        self.assertTrue(store.head("a/blob.bin")["exists"])

    def test_write_checksum_mismatch_rejected(self):
        store = MockStorage()
        with self.assertRaises(AdapterError):
            store.write("x", b"data", checksum="deadbeef")

    def test_write_with_correct_checksum_accepted(self):
        from drm.providers.mock import sha256_hex

        store = MockStorage()
        blob = b"verified bytes"
        result = store.write("x", blob, checksum=sha256_hex(blob))
        self.assertEqual(result["checksum"], sha256_hex(blob))

    def test_corruption_detected_on_read(self):
        store = MockStorage()
        store.write("x", b"stable bytes")
        store.corrupt("x")
        with self.assertRaises(AdapterError):
            store.read("x")

    def test_delete_and_missing_objects(self):
        store = MockStorage()
        with self.assertRaises(AdapterError):
            store.read("missing")
        store.write("gone", b"1")
        self.assertTrue(store.delete("gone")["deleted"])
        with self.assertRaises(AdapterError):
            store.delete("gone")


if __name__ == "__main__":
    unittest.main()
