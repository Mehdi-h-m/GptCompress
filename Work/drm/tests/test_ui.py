"""Offscreen Qt tests: the GUI drives the same flow as the CLI, over the daemon.

Sprint 3 acceptance — the Sprint 1–2 flow performed through the desktop shell
ends in the same daemon state the CLI produces. Windows are driven
programmatically under QT_QPA_PLATFORM=offscreen; no modal dialogs are
involved because outcomes go to the activity log.
"""

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import uvicorn
from PySide6.QtWidgets import QApplication

from drm.daemon.app import create_app
from drm.ui.app import MainWindow
from drm.ui.client import UiClient


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Server:
    def __init__(self, db_path: Path):
        self.server = uvicorn.Server(
            uvicorn.Config(create_app(db_path), host="127.0.0.1", port=0, log_level="warning")
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.server.started:
            raise AssertionError("test daemon never started")
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"

    def stop(self):
        self.server.should_exit = True
        self.thread.join(timeout=5)


class UiParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls._tmp.name) / "fixture.bin"
        cls.fixture.write_bytes(b"sprint 3 gui fixture")
        cls.server = _Server(Path(cls._tmp.name) / "drm.db")
        cls.qapp = QApplication.instance() or QApplication(sys.argv)
        cls.window = MainWindow(client=UiClient(cls.server.url))
        # seed a compute provider through the API, like an agent would via the CLI
        cls.window.client.connect_provider({"adapter": "local-runner", "name": "seed-cpu", "config": {}})

    @classmethod
    def tearDownClass(cls):
        cls.window.deleteLater()
        cls.server.stop()
        cls._tmp.cleanup()

    def _wait(self, predicate, timeout=30.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.qapp.processEvents()
            if predicate():
                return True
            time.sleep(0.05)
        return False

    def _select_row_with_value(self, table, value):
        for row in range(table.rowCount()):
            if table.item(row, 0).text() == value:
                table.selectRow(row)
                return True
        return False

    def test_connection_indicator_shows_down_for_unreachable_daemon(self):
        dead = MainWindow(client=UiClient("http://127.0.0.1:9"))
        dead._on_refresh_tick()
        self.assertTrue(dead.connection_label.text().startswith("not connected"))
        dead.deleteLater()
        self.assertTrue(self.window.connection_label.text().startswith("connected"))

    def test_provider_connect_list_pause_resume_through_ui(self):
        root = Path(self._tmp.name) / "ui-store"
        doc = self.window.connect_provider(
            {"adapter": "filesystem", "name": "ui-store", "config": {"root": str(root)}}
        )
        provider_id = doc["provider"]["id"]
        self.window.refresh_providers()
        self.assertGreaterEqual(self.window.providers_table.rowCount(), 2)

        self.assertTrue(self._select_row_with_value(self.window.providers_table, provider_id))
        self.window._on_pause_provider()
        states = {p["id"]: p["state"] for p in self.window.client.providers()}
        self.assertEqual(states[provider_id], "paused")
        self.window._on_resume_provider()
        states = {p["id"]: p["state"] for p in self.window.client.providers()}
        self.assertEqual(states[provider_id], "active")

    def test_full_gui_flow_ingest_plan_submit_publish(self):
        root = Path(self._tmp.name) / "flow-store"
        store = self.window.connect_provider(
            {"adapter": "filesystem", "name": "flow-store", "config": {"root": str(root)}}
        )["provider"]

        # GUI ingest, polled through the same signal path as interactive use
        op_id = self.window.ingest_artifact(
            {"source": str(self.fixture), "store": store["id"], "verify": True}
        )
        self.assertTrue(
            self._wait(lambda: self.window.client.operation(op_id)["operation"]["state"] in {"complete", "failed", "cancelled"})
        )
        ingest_op = self.window.client.operation(op_id)["operation"]
        self.assertEqual(ingest_op["state"], "complete", ingest_op)
        artifact_id = ingest_op["result"]["artifact"]["id"]
        self.assertTrue(self._wait(lambda: self.window.artifacts_table.rowCount() >= 1))
        artifact_ids = [
            self.window.artifacts_table.item(r, 0).text()
            for r in range(self.window.artifacts_table.rowCount())
        ]
        self.assertIn(artifact_id, artifact_ids)

        # GUI plan preview shows a decision reference
        plan_doc = self.window.preview_plan(
            {"task_type": "hash_inputs@1", "inputs": [artifact_id], "name": "gui-job"}
        )
        self.assertEqual(plan_doc["blockers"], [])
        self.assertIsNotNone(plan_doc["decision_id"])

        # GUI submit polled to completion; published artifact lands in the catalog
        run_op = self.window.submit_run(
            {"task_type": "hash_inputs@1", "inputs": [artifact_id], "name": "gui-job"}
        )
        self.assertTrue(
            self._wait(lambda: self.window.client.operation(run_op)["operation"]["state"] in {"complete", "failed", "cancelled"})
        )
        operation = self.window.client.operation(run_op)["operation"]
        self.assertEqual(operation["state"], "complete", operation)
        published = operation["result"]["published"][0]["artifact"]

        self.window.refresh_data()
        rows = [
            self.window.artifacts_table.item(r, 0).text()
            for r in range(self.window.artifacts_table.rowCount())
        ]
        self.assertIn(published["id"], rows)
        self.assertEqual(published["kind"], "result")

        # run monitor lists the job and shows its attempts
        self.window.refresh_runs()
        self.assertTrue(self._select_row_with_value(self.window.jobs_table, operation["result"]["job_id"]))
        self.assertIn("complete", self.window.run_detail.toPlainText())

    def test_verify_pin_evict_wiring_in_gui(self):
        root = Path(self._tmp.name) / "wiring-store"
        store = self.window.connect_provider(
            {"adapter": "filesystem", "name": "wiring-store", "config": {"root": str(root)}}
        )["provider"]
        op_id = self.window.ingest_artifact(
            {"source": str(self.fixture), "store": store["id"], "verify": True}
        )
        self.assertTrue(self._wait(lambda: self.window.client.operation(op_id)["operation"]["state"] != "queued"))
        artifact_id = self.window.client.operation(op_id)["operation"]["result"]["artifact"]["id"]

        self.window.refresh_data()
        self.assertTrue(self._select_row_with_value(self.window.artifacts_table, artifact_id))
        self.window._on_artifact_selected()
        copy_id = self.window.copies_table.item(0, 0).text()
        self.assertTrue(self._select_row_with_value(self.window.copies_table, copy_id))

        log_before = self.window.log.toPlainText()
        self.window._on_verify_copy()
        self.assertIn(f"verified {copy_id}", self.window.log.toPlainText())

        self.window._on_pin_artifact()
        self.assertTrue(self._wait(lambda: self._pinned(self.window, artifact_id)))

        log_len = len(self.window.log.toPlainText())
        self.window._on_evict_copy()  # last durable copy of a pinned artifact
        self.assertTrue(self._wait(lambda: len(self.window.log.toPlainText()) > log_len))
        self.assertIn("policy_blocked", self.window.log.toPlainText())

    @staticmethod
    def _pinned(window, artifact_id):
        for artifact in window.client.artifacts():
            if artifact["id"] == artifact_id:
                return bool(artifact.get("pinned"))
        return False


class UiDaemonControlTests(unittest.TestCase):
    def test_start_and_stop_daemon_from_the_app(self):
        port = _free_port()
        state_dir = tempfile.mkdtemp()
        old_env = {key: os.environ.get(key) for key in ("DRM_PORT", "DRM_STATE_DIR", "DRM_URL")}
        os.environ["DRM_PORT"] = str(port)
        os.environ["DRM_STATE_DIR"] = state_dir
        os.environ.pop("DRM_URL", None)
        try:
            self.qapp = QApplication.instance() or QApplication(sys.argv)
            window = MainWindow()
            try:
                self.assertTrue(window.start_daemon())
                self.assertIn("connected", window.connection_label.text())
            finally:
                self.assertTrue(window.stop_daemon())
            from drm.daemon import control

            self.assertFalse(control.is_healthy(f"http://127.0.0.1:{port}"))
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
