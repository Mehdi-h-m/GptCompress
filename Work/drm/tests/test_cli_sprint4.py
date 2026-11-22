"""CLI-level Sprint 4 acceptance with an injected fake hub (live server, in-process)."""

import json
import socket
import tempfile
import threading
import time
import unittest
import unittest.mock
from pathlib import Path

import uvicorn
from typer.testing import CliRunner

from drm.cli.main import app
from drm.daemon.app import create_app
from drm.providers import hf as hf_module

runner = CliRunner()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_hub():
    repos = {}

    class Hub:
        def whoami(self):
            return {"name": "cliuser"}

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
            repos.setdefault(repo_id, {})
            return object()

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
            repos.setdefault(repo_id, {})[path_in_repo] = bytes(path_or_fileobj)
            return object()

        def list_repo_files(self, repo_id, repo_type=None):
            return sorted(repos.get(repo_id, {}))

        def get_paths_info(self, repo_id, paths, repo_type=None):
            repo = repos.get(repo_id, {})
            return [type("Info", (), {"path": p, "size": len(repo[p])})() for p in paths if p in repo]

        def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
            import tempfile

            repo = repos.get(repo_id, {})
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            handle.write(repo[filename])
            handle.close()
            return handle.name

        def delete_file(self, path_in_repo, repo_id, repo_type=None):
            del repos[repo_id][path_in_repo]
            return object()

    return Hub(), repos


class Sprint4CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Path(self._tmp.name) / "fixture.bin"
        self.fixture.write_bytes(b"sprint 4 cli fixture")
        self.hub, self.repos = _make_hub()
        self._patch = unittest.mock.patch.object(
            hf_module, "make_api", lambda credential_ref=None: self.hub
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)

        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(Path(self._tmp.name) / "drm.db"),
                host="127.0.0.1", port=0, log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.addCleanup(self._shutdown)
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(self.server.started)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.env = {"DRM_URL": f"http://127.0.0.1:{port}"}

    def _shutdown(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def _invoke(self, *args, expect=0):
        result = runner.invoke(app, list(args), env=self.env)
        self.assertEqual(result.exit_code, expect, result.output)
        return result

    def test_hf_connect_probe_ingest_verify_list(self):
        store = json.loads(
            self._invoke(
                "providers", "connect", "hf",
                "--name", "hf-private",
                "--repo-id", "cliuser/drm-store",
                "--json",
            ).output
        )["provider"]
        self.assertEqual(store["adapter"], "hf")

        # duplicate connect returns the same instance (idempotent by name)
        again = json.loads(
            self._invoke("providers", "connect", "hf", "--name", "hf-private", "--repo-id", "cliuser/drm-store", "--json").output
        )
        self.assertTrue(again["already_connected"])

        probe = json.loads(self._invoke("providers", "probe", store["id"], "--json").output)
        observed = probe["probe"]["faces"]["storage"]["observed"]
        self.assertTrue(observed["authenticated"])
        self.assertEqual(observed["user"], "cliuser")
        self.assertIn("private_storage_free_gb", probe["probe"]["faces"]["storage"]["advertised"])

        operation = json.loads(
            self._invoke("data", "ingest", str(self.fixture), "--store", store["id"], "--verify", "--json").output
        )["operation"]
        self.assertEqual(operation["state"], "complete")
        copy = operation["result"]["copy"]
        self.assertTrue(copy["locator"].startswith("hf://datasets/cliuser/drm-store/"))
        path_in_repo = copy["locator"].split("hf://datasets/cliuser/drm-store/", 1)[1]
        self.assertIn(path_in_repo, self.repos["cliuser/drm-store"])

        verified = json.loads(self._invoke("data", "verify", copy["id"], "--json").output)
        self.assertTrue(verified["operation"]["result"]["verified"])

        listing = self._invoke("data", "list").output
        self.assertIn("hf://datasets/cliuser/drm-store/", listing)

    def test_hf_uri_public_ingest_with_release_name(self):
        local_store = json.loads(
            self._invoke(
                "providers", "connect", "filesystem",
                "--name", "local", "--root", str(Path(self._tmp.name) / "local"), "--json",
            ).output
        )["provider"]
        payload = b"public shard for cli"
        handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
        handle.write(payload)
        handle.close()
        provenance = {
            "origin_uri": "hf://datasets/demo/corpus@main/shard.jsonl",
            "repo_id": "demo/corpus", "revision": "main", "path": "shard.jsonl",
            "producer": "hf:demo/corpus@main",
        }
        with unittest.mock.patch.object(
            hf_module, "resolve_public_download", return_value=(Path(handle.name), provenance)
        ):
            operation = json.loads(
                self._invoke(
                    "data", "ingest",
                    "hf://datasets/demo/corpus@main/shard.jsonl",
                    "--store", local_store["id"],
                    "--release", "calib-v1",
                    "--json",
                ).output
            )["operation"]
        self.assertEqual(operation["state"], "complete")
        self.assertEqual(operation["result"]["artifact"]["id"], "calib-v1")
        self.assertEqual(operation["result"]["artifact"]["producer"], "hf:demo/corpus@main")

    def test_connect_requires_repo_id(self):
        result = self._invoke("providers", "connect", "hf", "--name", "no-repo", "--json", expect=2)
        self.assertEqual(json.loads(result.output)["error"]["code"], "validation")


if __name__ == "__main__":
    unittest.main()
