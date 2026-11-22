import threading
import tempfile
import unittest
from pathlib import Path

from drm.catalog import service
from drm.catalog.errors import CatalogError
from drm.providers.filesystem import FilesystemStorage
from drm.store import repos
from drm.store.db import connect, migrate
from drm.workers import runner
from drm.workers.tasks import TaskCancelled


class CtxStub:
    """Minimal daemon context for direct runner calls."""

    def __init__(self, conn, runs_dir: Path):
        self.conn = conn
        self.db_lock = threading.Lock()
        self.runs_dir = runs_dir
        self.cancelled = set()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)
        migrate(self.conn)
        self.input_root = Path(self._tmp.name) / "in"
        self.output_root = Path(self._tmp.name) / "out"
        self.input_store = repos.create_provider(
            self.conn, adapter="filesystem", display_name="in", kind="storage",
            execution="api", config={"root": str(self.input_root)},
        )
        self.output_store = repos.create_provider(
            self.conn, adapter="filesystem", display_name="out", kind="storage",
            execution="api", config={"root": str(self.output_root)},
        )
        self.compute = repos.create_provider(
            self.conn, adapter="local-runner", display_name="cpu", kind="compute",
            execution="api", config={},
        )
        fixture = Path(self._tmp.name) / "fixture.bin"
        fixture.write_bytes(b"runner fixture bytes")
        result = service.ingest_file(
            self.conn, FilesystemStorage(self.input_root),
            provider_id=self.input_store["id"], source=str(fixture),
        )
        self.artifact_id = result["artifact"]["id"]
        self.ctx = CtxStub(self.conn, Path(self._tmp.name) / "runs")

    def _submit_job(self, task_type="hash_inputs@1", **spec_overrides):
        spec = {
            "id": "task-1",
            "task_type": task_type,
            "task_version": 1,
            "max_attempts": 3,
            "timeout_seconds": 60,
            "partition": "single",
            "inputs": [self.artifact_id],
            "params": {},
            "output_store": self.output_store["id"],
            "compute_provider": self.compute["id"],
        }
        spec.update(spec_overrides)
        job = repos.create_job(self.conn, job_id=repos.new_id("job"), name="t", experiment_id=None, spec=spec)
        attempt = repos.create_attempt(
            self.conn, attempt_id=repos.new_id("att"), job_id=job["id"], provider_id=self.compute["id"]
        )
        return job, attempt, spec

    def test_execute_completes_and_publishes_verified_output(self):
        job, attempt, spec = self._submit_job()
        result = runner.execute_local_attempt(self.ctx, job, attempt, spec)
        self.assertEqual(result["result_manifest"]["task_type"], "hash_inputs@1")
        self.assertEqual(len(result["published"]), 1)
        published = result["published"][0]
        self.assertFalse(published["adopted"])
        artifact, copy = published["artifact"], published["copy"]
        self.assertEqual(artifact["kind"], "result")
        self.assertEqual(artifact["producer"], attempt["id"])
        stored = self.output_root / copy["locator"]
        self.assertTrue(stored.is_file())
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "complete")
        self.assertEqual(repos.get_attempt(self.conn, attempt["id"])["state"], "complete")
        self.assertTrue((self.ctx.runs_dir / job["id"] / attempt["id"] / "result_manifest.json").is_file())
        # staged input verified against the catalog checksum
        input_checksum = repos.get_artifact(self.conn, self.artifact_id)["content_hash"]
        self.assertEqual(result["result_manifest"]["staged_inputs"][0]["sha256"], input_checksum)

    def test_rerun_adopts_instead_of_duplicating_output(self):
        job, first, spec = self._submit_job()
        runner.execute_local_attempt(self.ctx, job, first, spec)
        second = repos.create_attempt(
            self.conn, attempt_id=repos.new_id("att"), job_id=job["id"], provider_id=self.compute["id"]
        )
        result = runner.execute_local_attempt(self.ctx, job, second, spec)
        self.assertTrue(result["published"][0]["adopted"])
        outputs = [
            a for a in repos.list_artifacts(self.conn)
            if a["kind"] == "result" and a["id"].startswith("art-")
        ]
        self.assertEqual(len(outputs), 1)  # exactly one accepted output for the logical output

    def test_conflicting_content_is_a_conflict_not_a_duplicate(self):
        job, attempt, spec = self._submit_job()
        runner.execute_local_attempt(self.ctx, job, attempt, spec)
        with self.assertRaises(CatalogError) as ctx:
            runner.publish_output(
                self.conn, FilesystemStorage(self.output_root), self.output_store["id"],
                job["id"], "summary.json", b"different bytes", producer=attempt["id"],
            )
        self.assertEqual(ctx.exception.code, "conflict")

    def test_cancelled_before_staging(self):
        job, attempt, spec = self._submit_job()
        self.ctx.cancelled.add(job["id"])
        with self.assertRaises(TaskCancelled):
            runner.execute_local_attempt(self.ctx, job, attempt, spec)
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "cancelled")
        self.assertEqual(repos.get_attempt(self.conn, attempt["id"])["state"], "cancelled")

    def test_unknown_task_type_is_validation_error(self):
        job, attempt, spec = self._submit_job(task_type="nope@9")
        with self.assertRaises(CatalogError) as ctx:
            runner.execute_local_attempt(self.ctx, job, attempt, spec)
        self.assertEqual(ctx.exception.code, "validation")

    def test_missing_input_copy_is_blocker(self):
        job, attempt, spec = self._submit_job(inputs=["art-gone"])
        with self.assertRaises(CatalogError) as ctx:
            runner.execute_local_attempt(self.ctx, job, attempt, spec)
        self.assertIn(ctx.exception.code, ("validation", "not_found"))


if __name__ == "__main__":
    unittest.main()
