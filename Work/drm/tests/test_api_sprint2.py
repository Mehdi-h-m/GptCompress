"""API-level Sprint 2 acceptance: plan, submit, wait, explain, cancel, retry, restart."""

import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from drm.daemon.app import create_app
from drm.providers.filesystem import FilesystemStorage
from drm.store import repos
from drm.store.db import connect as db_connect, migrate


def _wait_operation(client, op_id, timeout=25.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/operations/{op_id}").json()
        if body["operation"]["state"] in ("complete", "failed", "cancelled"):
            return body["operation"]
        time.sleep(0.1)
    raise AssertionError(f"operation {op_id} never reached a terminal state")


def _wait_job(client, job_id, states=("complete", "failed", "cancelled"), timeout=25.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/run/{job_id}").json()["job"]
        if job["state"] in states:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached a terminal state")


class Sprint2ApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.app = create_app(Path(self._tmp.name) / "drm.db")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        fixture = Path(self._tmp.name) / "fixture.bin"
        fixture.write_bytes(b"sprint 2 fixture")
        self.fixture = fixture
        self.store = self._connect("filesystem", "store", {"root": str(Path(self._tmp.name) / "store")})
        self.compute = self._connect("local-runner", "cpu", {})
        self.artifact_id = self._ingest_fixture()

    def _connect(self, adapter, name, config):
        response = self.client.post("/providers", json={"adapter": adapter, "name": name, "config": config})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["provider"]

    def _ingest_fixture(self):
        response = self.client.post(
            "/data/ingest",
            json={"source": str(self.fixture), "store": self.store["id"], "verify": True},
        )
        operation = _wait_operation(self.client, response.json()["operation"]["id"])
        self.assertEqual(operation["state"], "complete", operation)
        return operation["result"]["artifact"]["id"]

    def _submit(self, **overrides):
        payload = {
            "task_type": "hash_inputs@1",
            "inputs": [self.artifact_id],
            "name": "api-job",
            "max_attempts": 3,
            "timeout_seconds": 60,
        }
        payload.update(overrides)
        return self.client.post("/run/submit", json=payload)

    def test_preview_records_replayable_decision(self):
        response = self.client.post("/plan/preview", json={"task_type": "hash_inputs@1", "inputs": [self.artifact_id]})
        self.assertEqual(response.status_code, 200, response.text)
        plan_doc = response.json()["plan"]
        self.assertEqual(plan_doc["blockers"], [])
        self.assertEqual(plan_doc["chosen"]["output_store"], self.store["id"])
        self.assertIsNotNone(plan_doc["decision_id"])

        explained = self.client.get(f"/decisions/{plan_doc['decision_id']}").json()["decision"]
        self.assertEqual(explained["chosen"]["output_store"], self.store["id"])
        self.assertTrue(any(a.get("selected") for a in explained["alternatives"]))

    def test_full_run_flow_publishes_verified_artifact(self):
        submit = self._submit()
        self.assertEqual(submit.status_code, 200, submit.text)
        body = submit.json()
        operation_id = body["operation"]["id"]
        self.assertEqual(body["job"]["state"], "queued")

        operation = _wait_operation(self.client, operation_id)
        self.assertEqual(operation["state"], "complete", operation)
        published = operation["result"]["published"][0]
        artifact = published["artifact"]
        self.assertEqual(artifact["kind"], "result")
        self.assertEqual(artifact["producer"], body["attempt"]["id"])

        detail = self.client.get(f"/data/artifacts/{artifact['id']}").json()["artifact"]
        self.assertTrue(detail["copies"])
        stored = Path(self.store["config"]["root"]) / detail["copies"][0]["locator"]
        self.assertTrue(stored.is_file())

        job = self.client.get(f"/run/{body['job']['id']}").json()
        self.assertEqual(job["job"]["state"], "complete")
        self.assertEqual(job["attempts"][0]["state"], "complete")

    def test_submit_dry_run_creates_nothing(self):
        response = self._submit(dry_run=True)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["dry_run"])
        self.assertEqual(response.json()["plan"]["blockers"], [])
        counts = self.client.get("/status").json()["counts"]
        self.assertEqual(counts["jobs"], 0)
        self.assertEqual(counts["operations"], 1)  # only the ingest operation

    def test_submit_with_missing_input_is_validation_block(self):
        response = self._submit(inputs=["art-nope"])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "validation")

    def test_submit_with_unknown_task_type_is_validation(self):
        response = self._submit(task_type="nope@1")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "validation")

    def test_idempotency_key_reuses_original_run(self):
        first = self._submit(idempotency_key="run-key-1")
        _wait_operation(self.client, first.json()["operation"]["id"])
        second = self._submit(idempotency_key="run-key-1")
        self.assertTrue(second.json()["reused"])
        self.assertEqual(second.json()["operation"]["id"], first.json()["operation"]["id"])
        counts = self.client.get("/status").json()["counts"]
        self.assertEqual(counts["jobs"], 1)

    def test_cancel_running_job(self):
        submit = self._submit(task_type="sleep@1", params={"seconds": 5})
        self.assertEqual(submit.status_code, 200)
        job_id = submit.json()["job"]["id"]
        time.sleep(0.3)  # let the attempt start sleeping
        cancel = self.client.post(f"/run/{job_id}/cancel", json={})
        self.assertEqual(cancel.status_code, 200)
        operation_id = submit.json()["operation"]["id"]
        operation = _wait_operation(self.client, operation_id)
        self.assertEqual(operation["state"], "cancelled", operation)
        job = self.client.get(f"/run/{job_id}").json()["job"]
        self.assertEqual(job["state"], "cancelled")
        again = self.client.post(f"/run/{job_id}/cancel", json={})
        self.assertTrue(again.json()["already_cancelled"])

    def test_cancel_completed_job_conflicts(self):
        submit = self._submit()
        _wait_operation(self.client, submit.json()["operation"]["id"])
        response = self.client.post(f"/run/{submit.json()['job']['id']}/cancel", json={})
        self.assertEqual(response.status_code, 409)

    def test_failed_task_auto_retries_then_manual_retry(self):
        submit = self._submit(task_type="fail_always@1", max_attempts=2, name="flaky")
        job_id = submit.json()["job"]["id"]
        job = _wait_job(self.client, job_id)
        self.assertEqual(job["state"], "failed")

        attempts = self.client.get(f"/run/{job_id}").json()["attempts"]
        self.assertEqual(len(attempts), 2)  # one initial + one automatic retry
        self.assertTrue(all(a["error"]["code"] == "unexpected_error" for a in attempts))

        retry = self.client.post(f"/jobs/{job_id}/retry", json={})
        self.assertEqual(retry.status_code, 200, retry.text)
        operation = _wait_operation(self.client, retry.json()["operation"]["id"])
        self.assertEqual(operation["state"], "failed")
        attempts = self.client.get(f"/run/{job_id}").json()["attempts"]
        self.assertEqual(len(attempts), 3)  # manual retry overrides the automatic bound

    def test_retry_complete_job_conflicts(self):
        submit = self._submit()
        _wait_operation(self.client, submit.json()["operation"]["id"])
        response = self.client.post(f"/jobs/{submit.json()['job']['id']}/retry", json={})
        self.assertEqual(response.status_code, 409)


class RestartRecoveryS2Tests(unittest.TestCase):
    def _bootstrap(self, tmp):
        """Direct-db bootstrap: providers, input artifact, output store."""
        db = Path(tmp) / "drm.db"
        conn = db_connect(db)
        migrate(conn)
        store_root = Path(tmp) / "store"
        store = repos.create_provider(
            conn, adapter="filesystem", display_name="out", kind="storage",
            execution="api", config={"root": str(store_root)},
        )
        compute = repos.create_provider(
            conn, adapter="local-runner", display_name="cpu", kind="compute",
            execution="api", config={},
        )
        fixture = Path(tmp) / "fixture.bin"
        fixture.write_bytes(b"recovery fixture")
        result = service_ingest(conn, store, fixture)
        conn.close()
        return db, store, compute, result["artifact"]["id"]

    def test_adopts_surviving_result_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, store, compute, _ = self._bootstrap(tmp)
            conn = db_connect(db)
            try:
                spec = {
                    "id": "task-1", "task_type": "noop@1", "task_version": 1,
                    "max_attempts": 1, "timeout_seconds": 60,
                    "inputs": [], "params": {}, "output_store": store["id"],
                    "compute_provider": compute["id"],
                }
                job = repos.create_job(conn, job_id="job-adopt", name="x", experiment_id=None, spec=spec)
                attempt = repos.create_attempt(conn, attempt_id="att-adopt", job_id="job-adopt", provider_id=compute["id"])
                repos.set_job_state(conn, "job-adopt", "running")
                repos.set_attempt_state(conn, "att-adopt", "running")

                runs_dir = Path(tmp) / "runs" / "job-adopt" / "att-adopt"
                outputs = runs_dir / "outputs"
                outputs.mkdir(parents=True)
                content = json.dumps({"task_type": "noop@1", "params": {}}).encode()
                (outputs / "result.json").write_bytes(content)
                manifest = {
                    "outputs": [
                        {"name": "result.json", "sha256": hashlib.sha256(content).hexdigest(),
                         "byte_count": len(content)}
                    ]
                }
                (runs_dir / "result_manifest.json").write_text(json.dumps(manifest))
            finally:
                conn.close()

            app = create_app(db)
            with TestClient(app) as client:
                job = client.get("/run/job-adopt").json()["job"]
                self.assertEqual(job["state"], "complete")
                artifacts = client.get("/data/artifacts").json()["artifacts"]
                results = [a for a in artifacts if a["kind"] == "result"]
                self.assertEqual(len(results), 1)
                expected_id = "art-" + hashlib.sha256(b"job-adopt:result.json").hexdigest()[:12]
                self.assertEqual(results[0]["id"], expected_id)

            # restarting again must not duplicate the adopted output
            app2 = create_app(db)
            with TestClient(app2) as client:
                artifacts = client.get("/data/artifacts").json()["artifacts"]
                results = [a for a in artifacts if a["kind"] == "result"]
                self.assertEqual(len(results), 1)

    def test_reruns_open_job_without_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, store, compute, artifact_id = self._bootstrap(tmp)
            conn = db_connect(db)
            try:
                spec = {
                    "id": "task-1", "task_type": "hash_inputs@1", "task_version": 1,
                    "max_attempts": 3, "timeout_seconds": 60,
                    "inputs": [artifact_id], "params": {}, "output_store": store["id"],
                    "compute_provider": compute["id"],
                }
                repos.create_job(conn, job_id="job-rerun", name="x", experiment_id=None, spec=spec)
                repos.create_attempt(conn, attempt_id="att-rerun", job_id="job-rerun", provider_id=compute["id"])
                repos.set_job_state(conn, "job-rerun", "running")
                repos.set_attempt_state(conn, "att-rerun", "running")
            finally:
                conn.close()

            app = create_app(db)
            with TestClient(app) as client:
                job = _wait_job(client, "job-rerun")
                self.assertEqual(job["state"], "complete")
                attempts = client.get("/run/job-rerun").json()["attempts"]
                self.assertEqual(len(attempts), 2)  # interrupted attempt + rerun
                by_id = {a["id"]: a for a in attempts}
                self.assertEqual(by_id["att-rerun"]["state"], "failed")
                self.assertEqual(by_id["att-rerun"]["error"]["code"], "interrupted")
                results = [a for a in client.get("/data/artifacts").json()["artifacts"] if a["kind"] == "result"]
                self.assertEqual(len(results), 1)

    def test_fails_job_when_attempts_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            db, store, compute, _ = self._bootstrap(tmp)
            conn = db_connect(db)
            try:
                spec = {
                    "id": "task-1", "task_type": "noop@1", "task_version": 1,
                    "max_attempts": 1, "timeout_seconds": 60,
                    "inputs": [], "params": {}, "output_store": store["id"],
                    "compute_provider": compute["id"],
                }
                repos.create_job(conn, job_id="job-exhaust", name="x", experiment_id=None, spec=spec)
                repos.create_attempt(conn, attempt_id="att-exhaust", job_id="job-exhaust", provider_id=compute["id"])
                repos.set_job_state(conn, "job-exhaust", "running")
                repos.set_attempt_state(conn, "att-exhaust", "running")
            finally:
                conn.close()

            app = create_app(db)
            with TestClient(app) as client:
                job = client.get("/run/job-exhaust").json()["job"]
                self.assertEqual(job["state"], "failed")
                attempts = client.get("/run/job-exhaust").json()["attempts"]
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]["error"]["code"], "interrupted")


def service_ingest(conn, store_row, fixture: Path):
    from drm.catalog import service
    from drm.providers import registry

    storage = registry.storage_for_row(store_row)
    return service.ingest_file(conn, storage, provider_id=store_row["id"], source=str(fixture))


if __name__ == "__main__":
    unittest.main()
