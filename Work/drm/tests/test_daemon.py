import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from drm.daemon.app import create_app
from drm.daemon.scheduler import Scheduler


class DaemonRouteTests(unittest.TestCase):
    def test_health_and_status_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(Path(tmp) / "drm.db")
            with TestClient(app) as client:
                body = client.get("/health").json()
                self.assertEqual(body["contract"], 1)
                self.assertEqual(body["status"], "ok")

                body = client.get("/status").json()
                self.assertEqual(body["contract"], 1)
                self.assertEqual(body["status"], "ok")
                self.assertIn("version", body)
                self.assertIn("db_path", body)
                self.assertEqual(body["scheduler"]["state"], "running")
                self.assertEqual(
                    set(body["counts"]),
                    {"providers", "artifacts", "artifact_copies", "jobs", "operations"},
                )

    def test_audit_log_records_startup(self):
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "drm.db"
            with TestClient(create_app(db)):
                pass
            conn = sqlite3.connect(db)
            try:
                rows = conn.execute("SELECT action FROM audit_log WHERE actor='daemon'").fetchall()
            finally:
                conn.close()
            self.assertEqual([row[0] for row in rows], ["started"])


class SchedulerTests(unittest.TestCase):
    def test_ticks_pause_resume_stop(self):
        scheduler = Scheduler(interval_seconds=0.02)
        scheduler.start()
        deadline = time.monotonic() + 3
        while scheduler.snapshot()["ticks"] < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreaterEqual(scheduler.snapshot()["ticks"], 3)

        scheduler.pause()
        self.assertEqual(scheduler.snapshot()["state"], "paused")
        frozen = scheduler.snapshot()["ticks"]
        time.sleep(0.08)
        self.assertEqual(scheduler.snapshot()["ticks"], frozen)

        scheduler.resume()
        deadline = time.monotonic() + 3
        while scheduler.snapshot()["ticks"] <= frozen and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertGreater(scheduler.snapshot()["ticks"], frozen)
        scheduler.stop()


if __name__ == "__main__":
    unittest.main()
