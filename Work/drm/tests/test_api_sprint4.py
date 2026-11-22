"""API-level Sprint 4 acceptance with an injected fake hub (no network).

The daemon builds HF adapters through ``drm.providers.hf.make_api``; tests
patch that factory so the same route/executor code paths run against the fake.
"""

import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from drm.daemon.app import create_app
from drm.providers import hf as hf_module


def _wait_operation(client, op_id, timeout=25.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/operations/{op_id}").json()
        if body["operation"]["state"] in ("complete", "failed", "cancelled"):
            return body["operation"]
        time.sleep(0.1)
    raise AssertionError(f"operation {op_id} never reached a terminal state")


def _make_hub():
    """An in-memory hub mirroring the adapter's HfApi surface."""
    repos = {}
    state = {"commits": 0}

    class Hub:
        def whoami(self):
            return {"name": "testuser"}

        def repo_info(self, repo_id, repo_type=None):
            if repo_id not in repos:
                raise RuntimeError("not found")
            return SimpleNamespace(id=repo_id)

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
            repos.setdefault(repo_id, {})
            return SimpleNamespace(repo_id=repo_id)

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
            state["commits"] += 1
            repos.setdefault(repo_id, {})[path_in_repo] = bytes(path_or_fileobj)
            return SimpleNamespace(commit_hash=f"commit{state['commits']:012d}")

        def list_repo_files(self, repo_id, repo_type=None):
            if repo_id not in repos:
                raise RuntimeError("not found")
            return sorted(repos[repo_id])

        def get_paths_info(self, repo_id, paths, repo_type=None):
            repo = repos.get(repo_id, {})
            return [SimpleNamespace(path=p, size=len(repo[p])) for p in paths if p in repo]

        def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
            repo = repos.get(repo_id, {})
            if filename not in repo:
                raise RuntimeError("not found")
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            handle.write(repo[filename])
            handle.close()
            return handle.name

        def delete_file(self, path_in_repo, repo_id, repo_type=None):
            del repos[repo_id][path_in_repo]
            return SimpleNamespace(commit_hash="deleted")

    return Hub(), repos


class Sprint4ApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.hub, self.repos = _make_hub()
        self._patch = unittest.mock.patch.object(
            hf_module, "make_api", lambda credential_ref=None: self.hub
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)

        self.app = create_app(Path(self._tmp.name) / "drm.db")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.fixture = Path(self._tmp.name) / "fixture.bin"
        self.fixture.write_bytes(b"sprint 4 fixture bytes")
        self._connect("local-runner", "cpu", {})
        self.hf_store = self._connect("hf", "hf-private", {"repo_id": "testuser/drm-store"})
        self.hf_store_2 = self._connect("hf", "hf-private-2", {"repo_id": "testuser/drm-store-2"})
        self.local_store = self._connect(
            "filesystem", "local-store", {"root": str(Path(self._tmp.name) / "local")}
        )

    def _connect(self, adapter, name, config):
        response = self.client.post(
            "/providers", json={"adapter": adapter, "name": name, "config": config}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["provider"]

    def _ingest(self, source, store_id, verify=True, **extra):
        response = self.client.post(
            "/data/ingest", json={"source": source, "store": store_id, "verify": verify, **extra}
        )
        self.assertEqual(response.status_code, 200, response.text)
        return _wait_operation(self.client, response.json()["operation"]["id"])

    def test_two_hf_instances_are_independent(self):
        providers = {p["id"]: p for p in self.client.get("/providers").json()["providers"]}
        self.assertEqual(providers[self.hf_store["id"]]["config"]["repo_id"], "testuser/drm-store")
        self.assertEqual(providers[self.hf_store_2["id"]]["config"]["repo_id"], "testuser/drm-store-2")

    def test_private_release_publish_and_verified_readback(self):
        operation = self._ingest(str(self.fixture), self.hf_store["id"], verify=True)
        self.assertEqual(operation["state"], "complete", operation)
        copy = operation["result"]["copy"]
        self.assertEqual(copy["status"], "durable")
        self.assertTrue(copy["locator"].startswith("hf://datasets/testuser/drm-store/"))
        path_in_repo = copy["locator"].split("hf://datasets/testuser/drm-store/", 1)[1]
        self.assertIn(path_in_repo, self.repos["testuser/drm-store"])

        verify = self.client.post("/data/verify", json={"copy_id": copy["id"]})
        self.assertEqual(verify.status_code, 200)
        self.assertTrue(verify.json()["operation"]["result"]["verified"])

        probe = self.client.post(f"/providers/{self.hf_store['id']}/probe", json={})
        observed = probe.json()["probe"]["faces"]["storage"]["observed"]
        self.assertTrue(observed["authenticated"])
        self.assertEqual(observed["user"], "testuser")
        self.assertEqual(observed["repo_files"], 1)

    def test_hf_uri_ingest_via_executor(self):
        payload = b"pinned public corpus shard"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        handle.write(payload)
        handle.close()

        provenance = {
            "origin_uri": "hf://datasets/google/boolq@main/train.parquet",
            "repo_id": "google/boolq",
            "revision": "main",
            "path": "train.parquet",
            "producer": "hf:google/boolq@main",
        }
        # the executor runs in a background thread: keep the patch active
        # until the operation reaches a terminal state
        patcher = unittest.mock.patch.object(
            hf_module, "resolve_public_download", return_value=(Path(handle.name), provenance)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        response = self.client.post(
            "/data/ingest",
            json={
                "source": "hf://datasets/google/boolq@main/train.parquet",
                "store": self.local_store["id"],
                "verify": True,
                "release": "calib-v1",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        operation = _wait_operation(self.client, response.json()["operation"]["id"])
        self.assertEqual(operation["state"], "complete", operation)
        artifact = operation["result"]["artifact"]
        self.assertEqual(artifact["id"], "calib-v1")
        self.assertEqual(artifact["producer"], "hf:google/boolq@main")

    def test_hf_dry_run_plan_parses_uri_without_network(self):
        response = self.client.post(
            "/data/ingest",
            json={
                "source": "hf://datasets/google/boolq@refs%2Fconvert%2Fparquet/train.parquet",
                "store": self.local_store["id"],
                "dry_run": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        plan = response.json()["plan"]
        self.assertEqual(plan["origin"]["repo_id"], "google/boolq")
        self.assertEqual(plan["origin"]["revision"], "refs/convert/parquet")

    def test_invalid_hf_uri_is_validation_error(self):
        response = self.client.post(
            "/data/ingest",
            json={"source": "hf://datasets/owner", "store": self.local_store["id"], "dry_run": True},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "validation")

    def test_private_publish_through_second_instance_lands_in_its_repo(self):
        operation = self._ingest(str(self.fixture), self.hf_store_2["id"], verify=True)
        self.assertEqual(operation["state"], "complete")
        locator = operation["result"]["copy"]["locator"]
        self.assertTrue(locator.startswith("hf://datasets/testuser/drm-store-2/"))
        self.assertIn("testuser/drm-store-2", self.repos)


if __name__ == "__main__":
    unittest.main()
