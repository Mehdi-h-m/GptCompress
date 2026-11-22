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
UNREACHABLE = "http://127.0.0.1:9"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class StatusUnreachableTests(unittest.TestCase):
    def test_json_error_record_with_exit_code_1(self):
        result = runner.invoke(app, ["status", "--json"], env={"DRM_URL": UNREACHABLE})
        self.assertEqual(result.exit_code, 1)
        body = json.loads(result.output)
        self.assertEqual(body["contract"], 1)
        self.assertEqual(body["error"]["code"], "daemon_unreachable")

    def test_human_mode_prints_error_line(self):
        result = runner.invoke(app, ["status"], env={"DRM_URL": UNREACHABLE})
        self.assertEqual(result.exit_code, 1)
        self.assertIn("daemon_unreachable", result.output)


class DaemonStopIdempotenceTests(unittest.TestCase):
    def test_stop_without_daemon_cleans_stale_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_file = Path(tmp) / "daemon.pid"
            pid_file.write_text("999999\n")
            result = runner.invoke(
                app,
                ["daemon", "stop", "--json"],
                env={"DRM_STATE_DIR": tmp, "DRM_URL": UNREACHABLE},
            )
            self.assertEqual(result.exit_code, 0, result.output)
            body = json.loads(result.output)
            self.assertEqual(body["status"], "stopped")
            self.assertFalse(body["was_running"])
            self.assertFalse(pid_file.exists())


class LiveDaemonStatusTests(unittest.TestCase):
    def test_status_against_live_daemon(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(Path(tmp) / "drm.db"),
                    host="127.0.0.1",
                    port=0,
                    log_level="warning",
                )
            )
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(server.started)
                port = server.servers[0].sockets[0].getsockname()[1]
                url = f"http://127.0.0.1:{port}"

                result = runner.invoke(app, ["status", "--json"], env={"DRM_URL": url})
                self.assertEqual(result.exit_code, 0, result.output)
                body = json.loads(result.output)
                self.assertEqual(body["contract"], 1)
                self.assertEqual(body["status"], "ok")
                self.assertEqual(body["scheduler"]["state"], "running")
                self.assertEqual(body["url"], url)

                result = runner.invoke(app, ["status"], env={"DRM_URL": url})
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("DRM daemon", result.output)
                self.assertIn("records", result.output)
            finally:
                server.should_exit = True
                thread.join(timeout=5)


class VersionTests(unittest.TestCase):
    def test_version_json(self):
        result = runner.invoke(app, ["version", "--json"])
        self.assertEqual(result.exit_code, 0)
        body = json.loads(result.output)
        self.assertEqual(body["contract"], 1)
        self.assertIn("version", body)


if __name__ == "__main__":
    unittest.main()
