"""CLI-level Sprint 2 acceptance: preview → submit → wait → verified artifact."""

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import uvicorn
from typer.testing import CliRunner

from drm.cli.main import app
from drm.daemon.app import create_app

runner = CliRunner()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Sprint2CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fixture = Path(self._tmp.name) / "fixture.bin"
        self.fixture.write_bytes(b"sprint 2 cli fixture")
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

    def _setup_store_and_artifact(self):
        store = json.loads(
            self._invoke(
                "providers", "connect", "filesystem",
                "--name", "cli-run-store", "--root", str(Path(self._tmp.name) / "store"),
                "--json",
            ).output
        )["provider"]
        compute = json.loads(
            self._invoke("providers", "connect", "local-runner", "--name", "cli-cpu", "--json").output
        )["provider"]
        operation = json.loads(
            self._invoke("data", "ingest", str(self.fixture), "--store", store["id"], "--json").output
        )["operation"]
        return store, operation["result"]["artifact"]["id"]

    def test_preview_submit_wait_publish_flow(self):
        store, artifact_id = self._setup_store_and_artifact()

        preview = json.loads(
            self._invoke(
                "plan", "preview", "--task", "hash_inputs@1", "--input", artifact_id, "--json"
            ).output
        )["plan"]
        self.assertEqual(preview["blockers"], [])
        self.assertIsNotNone(preview["decision_id"])

        submit = json.loads(
            self._invoke(
                "run", "submit", "--task", "hash_inputs@1", "--input", artifact_id,
                "--name", "cli-job", "--json",
            ).output
        )
        operation_id = submit["operation"]["id"]
        self.assertEqual(submit["job"]["state"], "queued")

        waited = json.loads(
            self._invoke("wait", operation_id, "--timeout", "60", "--json").output
        )["operation"]
        self.assertEqual(waited["state"], "complete")
        published = waited["result"]["published"][0]["artifact"]
        self.assertEqual(published["kind"], "result")

        listing = json.loads(self._invoke("data", "list", "--json").output)["artifacts"]
        self.assertEqual(len(listing), 2)  # input fixture + published result

        explained = json.loads(self._invoke("explain", preview["decision_id"], "--json").output)["decision"]
        self.assertEqual(explained["chosen"]["output_store"], store["id"])

        human = self._invoke("run", "status", submit["job"]["id"]).output
        self.assertIn("complete", human)

    def test_submit_dry_run_dispatches_nothing(self):
        self._setup_store_and_artifact()
        result = self._invoke(
            "run", "submit", "--task", "hash_inputs@1", "--input", "art-gone", "--dry-run", "--json"
        )
        # dry run of a plan with blockers still exits 0 and lists them
        body = json.loads(result.output)
        self.assertTrue(body["dry_run"])
        self.assertTrue(body["plan"]["blockers"])

    def test_submit_blockers_exit_2(self):
        self._setup_store_and_artifact()
        result = self._invoke("run", "submit", "--task", "hash_inputs@1", "--input", "art-gone", "--json", expect=2)
        self.assertEqual(json.loads(result.output)["error"]["code"], "validation")

    def test_failed_job_retry_cycle(self):
        self._setup_store_and_artifact()
        artifact_id = json.loads(self._invoke("data", "list", "--json").output)["artifacts"][0]["id"]
        submit = json.loads(
            self._invoke(
                "run", "submit", "--task", "fail_always@1", "--input", artifact_id,
                "--max-attempts", "1", "--json",
            ).output
        )
        failed = self._invoke("wait", submit["operation"]["id"], "--timeout", "60", "--json", expect=1)
        self.assertEqual(json.loads(failed.output)["error"]["code"], "unexpected_error")

        status = self._invoke("run", "status", submit["job"]["id"]).output
        self.assertIn("failed", status)

        retried = json.loads(self._invoke("jobs", "retry", submit["job"]["id"], "--json").output)
        self.assertEqual(retried["job"]["state"], "queued")
        self._invoke("wait", retried["operation"]["id"], "--timeout", "60", "--json", expect=1)
        attempts = json.loads(self._invoke("run", "status", submit["job"]["id"], "--json").output)["attempts"]
        self.assertEqual(len(attempts), 2)

    def test_wait_unknown_operation_exits_1(self):
        result = self._invoke("wait", "op-nope", "--timeout", "5", "--json", expect=1)
        self.assertEqual(json.loads(result.output)["error"]["code"], "not_found")

    def test_explain_unknown_decision_exits_1(self):
        result = self._invoke("explain", "dec-nope", "--json", expect=1)
        self.assertEqual(json.loads(result.output)["error"]["code"], "not_found")

    def test_cancel_sleeping_job(self):
        self._setup_store_and_artifact()
        artifact_id = json.loads(self._invoke("data", "list", "--json").output)["artifacts"][0]["id"]
        submit = json.loads(
            self._invoke(
                "run", "submit", "--task", "sleep@1", "--input", artifact_id,
                "--param", "seconds=5", "--json",
            ).output
        )
        time.sleep(0.3)
        self._invoke("run", "cancel", submit["job"]["id"], "--json")
        waited = json.loads(
            self._invoke("wait", submit["operation"]["id"], "--timeout", "30", "--json").output
        )["operation"]
        self.assertEqual(waited["state"], "cancelled")
        human = self._invoke("run", "status", submit["job"]["id"]).output
        self.assertIn("cancelled", human)


if __name__ == "__main__":
    unittest.main()
