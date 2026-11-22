"""CLI-level Sprint 1 acceptance through a live daemon on an ephemeral port."""

import json
import socket
import tempfile
import threading
import time
import unittest
import uuid
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


class Sprint1CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.source = Path(self._tmp.name) / "fixture.bin"
        self.source.write_bytes(b"cli fixture bytes")
        self.store_root = Path(self._tmp.name) / "store"

        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(Path(self._tmp.name) / "drm.db"),
                host="127.0.0.1",
                port=0,
                log_level="warning",
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        self.addCleanup(self._shutdown_server)
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(self.server.started)
        port = self.server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.env = {"DRM_URL": self.url}

    def _shutdown_server(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)

    def _invoke(self, *args, expect=0):
        result = runner.invoke(app, list(args), env=self.env)
        self.assertEqual(result.exit_code, expect, result.output)
        return result

    def _connect_store(self):
        result = self._invoke(
            "providers", "connect", "filesystem",
            "--name", f"cli-store-{uuid.uuid4().hex[:8]}",
            "--root", str(self.store_root),
            "--json",
        )
        return json.loads(result.output)["provider"]["id"]

    def test_provider_connect_list_pause(self):
        store_id = self._connect_store()
        listed = json.loads(self._invoke("providers", "list", "--json").output)
        self.assertEqual(len(listed["providers"]), 1)

        paused = json.loads(self._invoke("providers", "pause", store_id, "--json").output)
        self.assertEqual(paused["provider"]["state"], "paused")
        human = self._invoke("providers", "list").output
        self.assertIn("paused", human)

        bad = self._invoke("providers", "connect", "ftp", "--name", "x", "--json", expect=2)
        self.assertEqual(json.loads(bad.output)["error"]["code"], "validation")

    def test_provider_connect_dry_run_creates_nothing(self):
        result = self._invoke(
            "providers", "connect", "filesystem", "--name", "dry-store",
            "--root", str(self.store_root), "--dry-run", "--json",
        )
        self.assertTrue(json.loads(result.output)["dry_run"])
        listed = json.loads(self._invoke("providers", "list", "--json").output)
        self.assertEqual(listed["providers"], [])

    def test_full_data_flow_with_verified_copy(self):
        store_id = self._connect_store()
        result = self._invoke("data", "ingest", str(self.source), "--store", store_id, "--json")
        body = json.loads(result.output)
        operation = body["operation"]
        self.assertEqual(operation["state"], "complete")
        artifact_id = operation["result"]["artifact"]["id"]
        copy_id = operation["result"]["copy"]["id"]

        human = self._invoke("data", "list").output
        self.assertIn(artifact_id, human)

        verified = json.loads(self._invoke("data", "verify", copy_id, "--json").output)
        self.assertTrue(verified["operation"]["result"]["verified"])

        # a custom id conflicts with the existing one
        conflict = self._invoke(
            "data", "ingest", str(self.source), "--store", store_id, "--id", artifact_id, "--json", expect=1
        )
        self.assertEqual(json.loads(conflict.output)["error"]["code"], "conflict")

    def test_ingest_dry_run_does_not_dispatch(self):
        store_id = self._connect_store()
        result = self._invoke("data", "ingest", str(self.source), "--store", store_id, "--dry-run", "--json")
        body = json.loads(result.output)
        self.assertTrue(body["dry_run"])
        self.assertEqual(json.loads(self._invoke("data", "list", "--json").output)["artifacts"], [])

    def test_idempotency_key_returns_original_operation(self):
        store_id = self._connect_store()
        first = json.loads(
            self._invoke("data", "ingest", str(self.source), "--store", store_id, "--idempotency-key", "same-key", "--json").output
        )
        second = json.loads(
            self._invoke("data", "ingest", str(self.source), "--store", store_id, "--idempotency-key", "same-key", "--json").output
        )
        self.assertTrue(second["reused"])
        self.assertEqual(second["operation"]["id"], first["operation"]["id"])

    def test_verify_failure_after_byte_flip_exits_1(self):
        store_id = self._connect_store()
        operation = json.loads(
            self._invoke("data", "ingest", str(self.source), "--store", store_id, "--json").output
        )["operation"]
        locator = operation["result"]["copy"]["locator"]
        target = self.store_root / locator
        data = bytearray(target.read_bytes())
        data[-1] ^= 0xFF
        target.write_bytes(bytes(data))

        copy_id = operation["result"]["copy"]["id"]
        result = self._invoke("data", "verify", copy_id, "--json", expect=1)
        body = json.loads(result.output)
        self.assertEqual(body["error"]["code"], "verify_failed")

    def test_pin_blocks_eviction_with_exit_2(self):
        store_id = self._connect_store()
        operation = json.loads(
            self._invoke("data", "ingest", str(self.source), "--store", store_id, "--json").output
        )["operation"]
        artifact_id = operation["result"]["artifact"]["id"]
        copy_id = operation["result"]["copy"]["id"]

        self._invoke("data", "pin", artifact_id, "--json")
        result = self._invoke("data", "evict", copy_id, "--json", expect=2)
        self.assertEqual(json.loads(result.output)["error"]["code"], "policy_blocked")

        dry = json.loads(self._invoke("data", "evict", copy_id, "--dry-run", "--json").output)
        self.assertIn("last durable copy of pinned artifact", dry["plan"]["blockers"])

    def test_verify_unknown_copy_exits_1(self):
        result = self._invoke("data", "verify", "copy-nope", "--json", expect=1)
        self.assertEqual(json.loads(result.output)["error"]["code"], "not_found")


if __name__ == "__main__":
    unittest.main()
