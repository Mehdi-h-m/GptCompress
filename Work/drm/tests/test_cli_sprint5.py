"""CLI-level Sprint 5 acceptance: assisted submit exits 3, import completes, exit 4 reconcile."""

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
from drm.workers import assisted

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
            return [type("I", (), {"path": p, "size": len(repo[p])})() for p in paths if p in repo]

        def hf_hub_download(self, repo_id, filename, repo_type=None, revision=None):
            import tempfile

            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bin")
            handle.write(repos[repo_id][filename])
            handle.close()
            return handle.name

        def delete_file(self, path_in_repo, repo_id, repo_type=None):
            del repos[repo_id][path_in_repo]
            return object()

    return Hub(), repos


class Sprint5CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
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

    def _setup(self):
        self.kaggle = json.loads(
            self._invoke("providers", "connect", "kaggle", "--name", "kaggle-a", "--json").output
        )["provider"]
        self.store = json.loads(
            self._invoke(
                "providers", "connect", "filesystem",
                "--name", "s5-store", "--root", str(Path(self._tmp.name) / "store"), "--json",
            ).output
        )["provider"]

    def test_assisted_submit_exits_3_with_instructions(self):
        self._setup()
        result = self._invoke(
            "run", "submit", "--task", "checkpoint_scan@1",
            "--param", 'items=["alpha","beta","gamma","delta"]',
            "--compute", self.kaggle["id"],
            "--store", self.store["id"],
            "--name", "assisted-live",
            "--json",
            expect=3,
        )
        body = json.loads(result.output)
        self.assertEqual(body["error"]["code"], "awaiting_user")
        self.assertIn("drm run import", body["error"]["message"])
        details = body["error"]["details"]
        self.assertTrue(Path(details["bundle"]).is_file())
        self.assertTrue(Path(details["instructions"]).is_file())
        self.assertEqual(details["outputs_expected"], ["summary.json"])
        self.assertEqual(details["items_in_bundle"], ["alpha", "beta", "gamma", "delta"])

        status = self._invoke("run", "status", details["job_id"]).output
        self.assertIn("awaiting_user", status)
        self.assertIn("bundle", status)

    def test_full_assisted_cycle_through_cli(self):
        self._setup()
        result = self._invoke(
            "run", "submit", "--task", "checkpoint_scan@1",
            "--param", 'items=["alpha","beta"]',
            "--compute", self.kaggle["id"],
            "--store", self.store["id"],
            "--json",
            expect=3,
        )
        details = json.loads(result.output)["error"]["details"]
        attempt_id, job_id = details["attempt_id"], details["job_id"]

        # the "notebook": run the generated task implementation over the bundle
        bundle_dir = Path(details["bundle_dir"])
        impl = assisted.load_task_impl(bundle_dir)
        manifest = json.loads((bundle_dir / "task_manifest.json").read_text())
        outputs = impl["run"](manifest, bundle_dir, push_checkpoint=lambda p, d: False)
        self.assertEqual(outputs[0]["name"], "summary.json")

        imported = self._invoke("run", "import", attempt_id, "--output", str(bundle_dir), "--json")
        body = json.loads(imported.output)
        self.assertFalse(body["import"]["reconciled"])
        published = body["import"]["published"][0]["artifact"]
        self.assertEqual(published["kind"], "result")

        status = json.loads(self._invoke("run", "status", job_id, "--json").output)
        self.assertEqual(status["job"]["state"], "complete")
        listing = json.loads(self._invoke("data", "list", "--json").output)["artifacts"]
        self.assertEqual(len(listing), 1)

    def test_import_reconcile_exits_4(self):
        self._setup()
        result = self._invoke(
            "run", "submit", "--task", "checkpoint_scan@1",
            "--param", 'items=["one"]',
            "--compute", self.kaggle["id"],
            "--store", self.store["id"],
            "--json",
            expect=3,
        )
        details = json.loads(result.output)["error"]["details"]
        attempt_id = details["attempt_id"]

        empty = Path(self._tmp.name) / "downloaded"
        empty.mkdir()
        reconcile = self._invoke("run", "import", attempt_id, "--output", str(empty), "--json", expect=4)
        body = json.loads(reconcile.output)
        self.assertEqual(body["error"]["code"], "reconcile_required")
        self.assertIn("re-import", body["error"]["message"])

        # after the notebook really finishes, the same import succeeds
        bundle_dir = Path(details["bundle_dir"])
        impl = assisted.load_task_impl(bundle_dir)
        manifest = json.loads((bundle_dir / "task_manifest.json").read_text())
        impl["run"](manifest, bundle_dir, push_checkpoint=lambda p, d: False)
        self._invoke("run", "import", attempt_id, "--output", str(bundle_dir), "--json")

    def test_plain_submit_still_uses_api_compute(self):
        self._setup()
        self._invoke("providers", "connect", "local-runner", "--name", "cpu", "--json")
        artifact_setup = self._invoke(
            "run", "submit", "--task", "noop@1",
            "--store", self.store["id"], "--max-attempts", "1", "--json",
        )
        body = json.loads(artifact_setup.output)
        operation_id = body["operation"]["id"]
        waited = json.loads(
            self._invoke("wait", operation_id, "--timeout", "60", "--json").output
        )["operation"]
        self.assertEqual(waited["state"], "complete")


if __name__ == "__main__":
    unittest.main()
