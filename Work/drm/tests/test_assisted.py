"""Assisted execution cycle: bundle, simulated notebook, session loss, recovery.

The generated task_impl.py is executed for real (exec'd from the bundle), so
the recovery path exercises the same code the user runs in the notebook.
"""

import json
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

from drm.catalog.errors import CatalogError
from drm.daemon.executor import maybe_autoretry, run_job
from drm.planner.core import plan
from drm.providers import hf as hf_module
from drm.store import repos
from drm.store.db import connect, migrate
from drm.workers import assisted


class FakeExecutor:
    """Records submitted operations; tests run them explicitly."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.submitted = []

    def submit(self, op):
        self.submitted.append(dict(op))


class CtxStub:
    def __init__(self, conn, runs_dir: Path):
        self.conn = conn
        self.db_lock = threading.Lock()
        self.runs_dir = runs_dir
        self.cancelled = set()
        self.executor = FakeExecutor(self)


def _fake_hub_api():
    repos = {}
    state = {"n": 0}

    class Hub:
        def whoami(self):
            return {"name": "dur"}

        def create_repo(self, repo_id, repo_type=None, private=False, exist_ok=False):
            repos.setdefault(repo_id, {})
            return object()

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
            repos.setdefault(repo_id, {})[path_in_repo] = bytes(path_or_fileobj)
            state["n"] += 1
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


class AssistedCycleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)
        migrate(self.conn)
        self.hub, self.repos = _fake_hub_api()
        self._patch = unittest.mock.patch.object(
            hf_module, "make_api", lambda credential_ref=None: self.hub
        )
        self._patch.start()
        self.addCleanup(self._patch.stop)

        self.hf_store = repos.create_provider(
            self.conn, adapter="hf", display_name="durable", kind="storage",
            execution="api", config={"repo_id": "dur/checkpoints"},
        )
        self.local_store = repos.create_provider(
            self.conn, adapter="filesystem", display_name="out", kind="storage",
            execution="api", config={"root": str(Path(self._tmp.name) / "out")},
        )
        self.kaggle = repos.create_provider(
            self.conn, adapter="kaggle", display_name="kaggle-a", kind="compute",
            execution="assisted", config={},
        )
        self.ctx = CtxStub(self.conn, Path(self._tmp.name) / "runs")
        self.items = ["alpha", "beta", "gamma", "delta"]

    def _submit_checkpoint_job(self, max_attempts=3):
        spec = {
            "id": "task-1",
            "task_type": "checkpoint_scan@1",
            "task_version": 1,
            "max_attempts": max_attempts,
            "timeout_seconds": 300,
            "partition": "single",
            "inputs": [],
            "params": {"items": list(self.items)},
            "output_store": self.hf_store["id"],
            "compute_provider": self.kaggle["id"],
        }
        job = repos.create_job(self.conn, job_id=repos.new_id("job"), name="assist", experiment_id=None, spec=spec)
        attempt = repos.create_attempt(
            self.conn, attempt_id=repos.new_id("att"), job_id=job["id"],
            provider_id=self.kaggle["id"],
        )
        operation = repos.create_operation(
            self.conn, kind="run.job", idempotency_key=f"run-{job['id']}-1",
            payload={"job_id": job["id"], "attempt_id": attempt["id"]},
        )
        repos.set_attempt_operation(self.conn, attempt["id"], operation["id"])
        return job, attempt, spec, operation

    def _run_job_and_wait(self, job_id, attempt_id):
        result = run_job(self.ctx, {"job_id": job_id, "attempt_id": attempt_id})
        return result

    def _simulate_notebook(self, bundle_dir: Path, pusher=None, should_stop=None):
        """Run the generated task_impl.py exactly as the notebook would."""
        impl = assisted.load_task_impl(bundle_dir)
        manifest = json.loads((bundle_dir / "task_manifest.json").read_text())
        return impl["run"](manifest, bundle_dir, push_checkpoint=pusher, should_stop=should_stop)

    def test_prepare_bundle_parks_run_in_awaiting_user(self):
        job, attempt, spec, operation = self._submit_checkpoint_job()
        result = self._run_job_and_wait(job["id"], attempt["id"])
        self.assertTrue(result["awaiting_user"])
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "awaiting_user")
        self.assertEqual(repos.get_attempt(self.conn, attempt["id"])["state"], "awaiting_user")
        self.assertEqual(repos.get_operation(self.conn, operation["id"])["state"], "queued")

        bundle_dir = Path(result["bundle_dir"])
        self.assertTrue((bundle_dir / "task_impl.py").is_file())
        notebook = json.loads((bundle_dir / "notebook.ipynb").read_text())
        self.assertEqual(notebook["nbformat"], 4)
        self.assertTrue(any("from task_impl import run" in c["source"] for c in notebook["cells"] if c["cell_type"] == "code"))
        manifest = json.loads((bundle_dir / "task_manifest.json").read_text())
        self.assertEqual(manifest["params"]["items"], self.items)
        self.assertTrue(Path(result["bundle"]).is_file())

        attempt_data = repos.attempt_dict(repos.get_attempt(self.conn, attempt["id"]))
        instructions = Path(attempt_data["result"]["instructions"]).read_text()
        self.assertIn("drm run import", instructions)

    def test_generated_task_impl_runs_and_pushes_checkpoints(self):
        job, attempt, spec, _ = self._submit_checkpoint_job()
        result = self._run_job_and_wait(job["id"], attempt["id"])
        bundle_dir = Path(result["bundle_dir"])

        pusher_calls = []

        def pusher(path, data):
            pusher_calls.append(path)
            self.repos.setdefault("dur/checkpoints", {})[path] = data
            return True

        outputs = self._simulate_notebook(bundle_dir, pusher=pusher)
        self.assertEqual(outputs[0]["name"], "summary.json")
        summary = json.loads((bundle_dir / "outputs" / "summary.json").read_text())
        self.assertEqual([entry["item"] for entry in summary["items"]], self.items)
        self.assertTrue(all(entry["checkpoint_pushed"] for entry in summary["items"]))
        self.assertEqual(len(pusher_calls), len(self.items))
        pushed_prefix = assisted.checkpoint_prefix(job["id"])
        self.assertTrue(all(p.startswith(pushed_prefix) for p in pusher_calls))
        self.assertEqual(len(self.repos["dur/checkpoints"]), len(self.items))

        written_manifest = json.loads((bundle_dir / "result_manifest.json").read_text())
        self.assertEqual(written_manifest["outputs"][0]["sha256"], outputs[0]["sha256"])

    def test_session_loss_then_recovery_reuses_durable_checkpoints(self):
        job, attempt1, spec, _ = self._submit_checkpoint_job(max_attempts=3)
        result1 = self._run_job_and_wait(job["id"], attempt1["id"])
        bundle_dir = Path(result1["bundle_dir"])

        # simulate the notebook: two items done, then the session dies
        impl = assisted.load_task_impl(bundle_dir)
        manifest = json.loads((bundle_dir / "task_manifest.json").read_text())
        done = []

        def pusher(path, data):
            done.append(json.loads(data)["item"])
            self.repos.setdefault("dur/checkpoints", {})[path] = data
            return True

        stop_after = {"count": 2}

        def should_stop():
            stop_after["count"] -= 1
            return stop_after["count"] < 0

        with self.assertRaises(impl["SessionLost"]):
            impl["run"](manifest, bundle_dir, push_checkpoint=pusher, should_stop=should_stop)
        self.assertEqual(done, ["alpha", "beta"])

        assisted.record_session_lost(self.ctx, job["id"], attempt1["id"], "session expired")
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "failed")

        # the coordinator re-dispatches; recovery reads the durable store
        job_row = repos.get_job(self.conn, job["id"])
        maybe_autoretry(self.ctx, job_row, spec)
        submitted = self.ctx.executor.submitted[-1]
        attempt2 = repos.get_attempt(self.conn, json.loads(submitted["payload_json"])["attempt_id"])
        self.assertIsNotNone(attempt2)
        result2 = self._run_job_and_wait(job["id"], attempt2["id"])
        self.assertTrue(result2["awaiting_user"])
        self.assertEqual(result2["recovered_items"], ["alpha", "beta"])
        self.assertEqual(result2["items_in_bundle"], ["gamma", "delta"])

        manifest2 = json.loads((Path(result2["bundle_dir"]) / "task_manifest.json").read_text())
        impl2 = assisted.load_task_impl(Path(result2["bundle_dir"]))
        impl2["run"](manifest2, Path(result2["bundle_dir"]), push_checkpoint=lambda p, d: True)

        # import the completed notebook output
        import_dir = Path(result2["bundle_dir"])
        import_result = assisted.import_results(self.ctx, repos.job_dict(repos.get_job(self.conn, job["id"])), attempt2, str(import_dir))
        self.assertFalse(import_result["reconciled"])
        published = import_result["published"][0]["artifact"]
        self.assertEqual(published["id"].startswith("art-"), True)

        summary = json.loads((import_dir / "outputs" / "summary.json").read_text())
        self.assertEqual([e["item"] for e in summary["items"]], self.items)  # all four items
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "complete")
        self.assertEqual(repos.get_attempt(self.conn, attempt2["id"])["state"], "complete")

        # exactly one accepted output for the logical output
        results = [a for a in repos.list_artifacts(self.conn) if a["kind"] == "result"]
        self.assertEqual(len(results), 1)

    def test_import_rejects_undeclared_and_missing_manifest(self):
        job, attempt, spec, _ = self._submit_checkpoint_job()
        result = self._run_job_and_wait(job["id"], attempt["id"])
        bundle_dir = Path(result["bundle_dir"])

        # no manifest at all: reconcile_required, attempt parked in reconciling
        empty = Path(self._tmp.name) / "import-empty"
        empty.mkdir()
        with self.assertRaises(CatalogError) as ctx:
            assisted.import_results(self.ctx, repos.job_dict(repos.get_job(self.conn, job["id"])), repos.attempt_dict(repos.get_attempt(self.conn, attempt["id"])), str(empty))
        self.assertEqual(ctx.exception.code, "reconcile_required")
        self.assertEqual(repos.get_attempt(self.conn, attempt["id"])["state"], "reconciling")

        # run the notebook, then corrupt the downloaded summary: still reconciling
        self._simulate_notebook(bundle_dir, pusher=lambda p, d: True)
        manifest = json.loads((bundle_dir / "result_manifest.json").read_text())
        bad_dir = Path(self._tmp.name) / "import-bad"
        bad_dir.mkdir()
        (bad_dir / "result_manifest.json").write_text(json.dumps(manifest))
        (bad_dir / "outputs").mkdir()
        (bad_dir / "outputs" / "summary.json").write_bytes(b"tampered")
        with self.assertRaises(CatalogError) as ctx:
            assisted.import_results(self.ctx, repos.job_dict(repos.get_job(self.conn, job["id"])), repos.attempt_dict(repos.get_attempt(self.conn, attempt["id"])), str(bad_dir))
        self.assertEqual(ctx.exception.code, "reconcile_required")

        # the real bundle output imports cleanly afterwards
        good = assisted.import_results(self.ctx, repos.job_dict(repos.get_job(self.conn, job["id"])), repos.attempt_dict(repos.get_attempt(self.conn, attempt["id"])), str(bundle_dir))
        self.assertFalse(good["reconciled"])
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "complete")

        # second import is a no-op reconciliation, not a duplicate
        again = assisted.import_results(self.ctx, repos.job_dict(repos.get_job(self.conn, job["id"])), repos.attempt_dict(repos.get_attempt(self.conn, attempt["id"])), str(bundle_dir))
        self.assertTrue(again["already_complete"])
        results = [a for a in repos.list_artifacts(self.conn) if a["kind"] == "result"]
        self.assertEqual(len(results), 1)

    def test_import_ignores_undeclared_files(self):
        job, attempt, spec, _ = self._submit_checkpoint_job()
        result = self._run_job_and_wait(job["id"], attempt["id"])
        bundle_dir = Path(result["bundle_dir"])
        self._simulate_notebook(bundle_dir, pusher=lambda p, d: True)
        (bundle_dir / "outputs" / "sneaky.txt").write_bytes(b"not declared")
        good = assisted.import_results(self.ctx, repos.job_dict(repos.get_job(self.conn, job["id"])), repos.attempt_dict(repos.get_attempt(self.conn, attempt["id"])), str(bundle_dir))
        self.assertIn("sneaky.txt", good["undeclared_files_ignored"])
        published_names = {i["artifact"]["id"] for i in good["published"]}
        self.assertEqual(len(published_names), 1)

    def test_reconcile_leaves_awaiting_user_jobs_alone(self):
        from drm.daemon.reconcile import reconcile_jobs

        job, attempt, spec, _ = self._submit_checkpoint_job()
        self._run_job_and_wait(job["id"], attempt["id"])
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "awaiting_user")
        outcome = reconcile_jobs(self.ctx)
        self.assertEqual(outcome["adopted"], [])
        self.assertEqual(outcome["requeued"], [])
        self.assertEqual(outcome["failed"], [])
        self.assertEqual(repos.get_job(self.conn, job["id"])["state"], "awaiting_user")
        self.assertEqual(repos.get_attempt(self.conn, attempt["id"])["state"], "awaiting_user")

    def test_planner_ranks_api_before_assisted(self):
        repos.create_provider(
            self.conn, adapter="local-runner", display_name="cpu", kind="compute",
            execution="api", config={},
        )
        doc = plan(self.conn, {"task_type": "noop@1", "name": "p1", "max_attempts": 1, "timeout_seconds": 60})
        chosen = doc["chosen"]["compute"]
        providers = {p["id"]: p for p in repos.list_providers(self.conn)}
        self.assertEqual(providers[chosen]["adapter"], "local-runner")
        assisted_candidate = [c for c in doc["alternatives"] if c["adapter"] == "kaggle"]
        self.assertEqual(assisted_candidate[0]["execution"], "assisted")


if __name__ == "__main__":
    unittest.main()
