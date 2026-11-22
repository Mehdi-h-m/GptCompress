"""End-to-end daemon lifecycle through the CLI in real subprocesses.

This is the Sprint 0 acceptance path: ``drm daemon start`` →
``drm status --json`` → ``drm daemon stop``, with the daemon detached and
state isolated to a temporary state dir.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_health(url: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        time.sleep(0.25)
    return False


class DaemonLifecycleTests(unittest.TestCase):
    def test_start_status_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            port = _free_port()
            env = dict(os.environ, DRM_STATE_DIR=tmp, DRM_PORT=str(port))
            url = f"http://127.0.0.1:{port}"
            cli = [sys.executable, "-m", "drm.cli"]
            try:
                started = subprocess.run(
                    cli + ["daemon", "start", "--json"],
                    env=env, capture_output=True, text=True, timeout=90,
                )
                self.assertEqual(started.returncode, 0, started.stdout + started.stderr)
                body = json.loads(started.stdout)
                self.assertEqual(body["status"], "running")
                self.assertIn("pid", body)
                self.assertTrue(_wait_health(url), "daemon /health never returned 200")

                shown = subprocess.run(
                    cli + ["status", "--json"],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(shown.returncode, 0, shown.stdout + shown.stderr)
                status_body = json.loads(shown.stdout)
                self.assertEqual(status_body["contract"], 1)
                self.assertEqual(status_body["status"], "ok")
                self.assertEqual(status_body["scheduler"]["state"], "running")
                self.assertEqual(status_body["counts"]["operations"], 0)

                stopped = subprocess.run(
                    cli + ["daemon", "stop", "--json"],
                    env=env, capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stdout + stopped.stderr)
                stop_body = json.loads(stopped.stdout)
                self.assertEqual(stop_body["status"], "stopped")
                self.assertTrue(stop_body["was_running"])
                self.assertFalse(_wait_health(url, timeout=2.0))
            finally:
                subprocess.run(
                    cli + ["daemon", "stop", "--json"],
                    env=env, capture_output=True, text=True, timeout=30,
                )


if __name__ == "__main__":
    unittest.main()
