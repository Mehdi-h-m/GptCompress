"""Hugging Face adapter logic against an in-memory fake hub (no network)."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from drm.providers import hf
from drm.providers.base import AdapterError


class FakeHub:
    """The HfApi surface the adapter uses, in memory."""

    def __init__(self, user="tester", fail_on=None):
        self.user = user
        self.repos = {}  # repo_id -> {path: bytes}
        self.commits = 0
        self.downloads = 0
        self.fail_on = set(fail_on or ())

    def _check(self, method):
        if method in self.fail_on:
            raise RuntimeError(f"injected failure: {method}")

    def whoami(self):
        self._check("whoami")
        return {"name": self.user, "auth": {"accessToken": {"role": "write"}}}

    def repo_info(self, repo_id, repo_type=None):
        self._check("repo_info")
        if repo_id not in self.repos:
            raise RuntimeError(f"Repository not found: {repo_id}")
        return SimpleNamespace(id=repo_id, private=True)

    def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
        self._check("create_repo")
        self.repos.setdefault(repo_id, {})
        return SimpleNamespace(repo_id=repo_id, private=private)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        self._check("upload_file")
        data = bytes(path_or_fileobj)
        self.repos.setdefault(repo_id, {})[path_in_repo] = data
        self.commits += 1
        return SimpleNamespace(commit_hash=f"commit{self.commits:012d}")

    def list_repo_files(self, repo_id, repo_type=None):
        self._check("list_repo_files")
        if repo_id not in self.repos:
            raise RuntimeError(f"Repository not found: {repo_id}")
        return sorted(self.repos[repo_id])

    def get_paths_info(self, repo_id, paths, repo_type=None):
        self._check("get_paths_info")
        repo = self.repos.get(repo_id, {})
        return [SimpleNamespace(path=p, size=len(repo[p])) for p in paths if p in repo]

    def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
        self._check("hf_hub_download")
        self.downloads += 1
        repo = self.repos.get(repo_id, {})
        if filename not in repo:
            raise RuntimeError(f"Entry not found: {filename}")
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        handle.write(repo[filename])
        handle.close()
        return handle.name

    def delete_file(self, path_in_repo, repo_id, repo_type=None):
        self._check("delete_file")
        repo = self.repos.setdefault(repo_id, {})
        if path_in_repo not in repo:
            raise RuntimeError(f"Entry not found: {path_in_repo}")
        del repo[path_in_repo]
        return SimpleNamespace(commit_hash="deleted")


def _store(repo_id="tester/drm-store", api=None, **kwargs):
    return hf.HfStorage(repo_id, api=api if api is not None else FakeHub(**kwargs))


class ParseHfUriTests(unittest.TestCase):
    def test_parse_full_uri_with_encoded_revision(self):
        parsed = hf.parse_hf_uri("hf://datasets/google/boolq@refs%2Fconvert%2Fparquet/default/train.parquet")
        self.assertEqual(parsed["repo_id"], "google/boolq")
        self.assertEqual(parsed["revision"], "refs/convert/parquet")
        self.assertEqual(parsed["path"], "default/train.parquet")

    def test_parse_minimal_uri(self):
        parsed = hf.parse_hf_uri("hf://datasets/owner/repo")
        self.assertIsNone(parsed["revision"])
        self.assertIsNone(parsed["path"])

    def test_invalid_uri_rejected(self):
        for bad in ("https://x", "hf://models/owner/repo/file", "hf://datasets/owner", "hf://"):
            with self.assertRaises(AdapterError):
                hf.parse_hf_uri(bad)


class HfStorageTests(unittest.TestCase):
    def setUp(self):
        self.store = _store()

    def test_write_read_roundtrip_versions_by_commit(self):
        blob = b"private release bytes"
        result = self.store.write("hf://datasets/tester/drm-store/art-1/blob.bin", blob, checksum=None)
        self.assertEqual(result["byte_count"], len(blob))
        self.assertTrue(result["version"].startswith("commit"))
        self.assertEqual(self.store.read("hf://datasets/tester/drm-store/art-1/blob.bin"), blob)

    def test_write_checksum_mismatch_rejected_before_upload(self):
        before = self.store._api.commits
        with self.assertRaises(AdapterError):
            self.store.write("hf://datasets/tester/drm-store/x", b"data", checksum="deadbeef")
        self.assertEqual(self.store._api.commits, before)

    def test_head_and_list(self):
        self.store.write("hf://datasets/tester/drm-store/a/one.bin", b"one")
        self.store.write("hf://datasets/tester/drm-store/a/two.bin", b"two")
        head = self.store.head("hf://datasets/tester/drm-store/a/one.bin")
        self.assertTrue(head["exists"])
        self.assertEqual(head["byte_count"], 3)
        missing = self.store.head("hf://datasets/tester/drm-store/a/nope.bin")
        self.assertFalse(missing["exists"])
        self.assertEqual(
            self.store.list("a/"),
            ["hf://datasets/tester/drm-store/a/one.bin", "hf://datasets/tester/drm-store/a/two.bin"],
        )

    def test_locator_of_another_repo_rejected(self):
        with self.assertRaises(AdapterError):
            self.store.read("hf://datasets/someone/else/file.bin")

    def test_read_missing_raises(self):
        with self.assertRaises(AdapterError):
            self.store.read("hf://datasets/tester/drm-store/missing.bin")

    def test_delete(self):
        self.store.write("hf://datasets/tester/drm-store/gone.bin", b"x")
        self.assertTrue(self.store.delete("hf://datasets/tester/drm-store/gone.bin")["deleted"])
        with self.assertRaises(AdapterError):
            self.store.delete("hf://datasets/tester/drm-store/gone.bin")

    def test_probe_reports_documented_limits_and_observed_usage(self):
        self.store.write("hf://datasets/tester/drm-store/data.bin", b"12345")
        probe = self.store.probe()
        self.assertEqual(probe["advertised"]["private_storage_free_gb"], 100)
        self.assertIn("storage-limits", probe["advertised"]["source"])
        self.assertTrue(probe["observed"]["authenticated"])
        self.assertEqual(probe["observed"]["user"], "tester")
        self.assertEqual(probe["observed"]["repo_files"], 1)
        self.assertEqual(probe["observed"]["repo_bytes"], 5)
        self.assertIn("observed_at", probe)

    def test_probe_without_credentials_still_reports_state(self):
        anonymous = _store(api=FakeHub(user=None, fail_on=("whoami",)))
        probe = anonymous.probe()
        self.assertFalse(probe["observed"]["authenticated"])

    def test_unreachable_probe_degrades_without_raising(self):
        down = _store(api=FakeHub(fail_on=("whoami", "repo_info", "list_repo_files")))
        probe = down.probe()
        self.assertFalse(probe["observed"]["reachable"])

    def test_error_messages_redact_token_shapes(self):
        class LeakyHub(FakeHub):
            def upload_file(self, **kwargs):
                raise RuntimeError("401 Unauthorized for token hf_abcdef1234567890abcdef")

        store = _store(api=LeakyHub())
        with self.assertRaises(AdapterError) as ctx:
            store.write("hf://datasets/tester/drm-store/x", b"d")
        self.assertNotIn("hf_abcdef1234567890abcdef", str(ctx.exception))
        self.assertIn("hf_***", str(ctx.exception))


class PublicResolveTests(unittest.TestCase):
    def test_resolve_public_download_records_provenance(self):
        fake = FakeHub()
        fake.upload_file(b"corpus shard", "data/shard-01.jsonl", "google/boolq", "dataset")

        def downloader(repo_id, filename, repo_type, revision):
            fake.hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
            # simulate the real downloader returning a temp file
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            handle.write(fake.repos[repo_id][filename])
            handle.close()
            return handle.name

        local, provenance = hf.resolve_public_download(
            "hf://datasets/google/boolq@main/data/shard-01.jsonl", downloader=downloader
        )
        self.assertTrue(Path(local).read_bytes() == b"corpus shard")
        self.assertEqual(provenance["repo_id"], "google/boolq")
        self.assertEqual(provenance["revision"], "main")
        self.assertEqual(provenance["producer"], "hf:google/boolq@main")

    def test_resolve_requires_a_path(self):
        with self.assertRaises(AdapterError):
            hf.resolve_public_download("hf://datasets/google/boolq", downloader=lambda **k: None)

    def test_public_fetch_failure_is_redacted(self):
        def downloader(**kwargs):
            raise RuntimeError("401 with token hf_abcdef1234567890abcdef")

        with self.assertRaises(AdapterError) as ctx:
            hf.resolve_public_download("hf://datasets/o/r@main/f.bin", downloader=downloader)
        self.assertIn("hf_***", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
