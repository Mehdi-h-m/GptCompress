import tempfile
import unittest
from pathlib import Path

from drm.catalog import service
from drm.catalog.errors import CatalogError
from drm.planner.core import build_spec, plan, record_decision
from drm.providers.filesystem import FilesystemStorage
from drm.store import repos
from drm.store.db import connect, migrate


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.conn = connect(Path(self._tmp.name) / "drm.db")
        self.addCleanup(self.conn.close)
        migrate(self.conn)
        self.input_store = repos.create_provider(
            self.conn, adapter="filesystem", display_name="input-store", kind="storage",
            execution="api", config={"root": str(Path(self._tmp.name) / "in")},
        )
        self.output_store = repos.create_provider(
            self.conn, adapter="filesystem", display_name="output-store", kind="storage",
            execution="api", config={"root": str(Path(self._tmp.name) / "out")},
        )
        compute = repos.create_provider(
            self.conn, adapter="local-runner", display_name="cpu", kind="compute",
            execution="api", config={},
        )
        self.compute_id = compute["id"]
        (Path(self._tmp.name) / "out").mkdir(parents=True, exist_ok=True)
        fixture = Path(self._tmp.name) / "fixture.bin"
        fixture.write_bytes(b"planner fixture")
        result = service.ingest_file(
            self.conn, FilesystemStorage(Path(self._tmp.name) / "in"),
            provider_id=self.input_store["id"], source=str(fixture),
        )
        self.artifact_id = result["artifact"]["id"]

    def _payload(self, **overrides):
        payload = {
            "task_type": "hash_inputs@1",
            "inputs": [self.artifact_id],
            "name": "planner-job",
            "max_attempts": 3,
            "timeout_seconds": 120,
        }
        payload.update(overrides)
        return payload

    def test_build_spec_normalizes_and_validates(self):
        spec = build_spec(self._payload())
        self.assertEqual(spec["partition"], "single")
        self.assertEqual(spec["max_attempts"], 3)
        with self.assertRaises(CatalogError) as ctx:
            build_spec({"task_type": "nope@1"})
        self.assertEqual(ctx.exception.code, "validation")
        with self.assertRaises(CatalogError):
            build_spec({"task_type": "hash_inputs@1", "max_attempts": 0})

    def test_plan_chooses_deterministic_store(self):
        doc = plan(self.conn, self._payload())
        self.assertEqual(doc["blockers"], [])
        expected_store = min(self.input_store["id"], self.output_store["id"])
        self.assertEqual(doc["chosen"]["output_store"], expected_store)
        self.assertEqual(doc["chosen"]["compute"], self.compute_id)
        self.assertEqual(doc["estimated_cost_range"]["max"], 0.0)
        self.assertEqual(doc["inputs"][0]["state"], "ok")

    def test_plan_prefers_requested_store_when_eligible(self):
        doc = plan(self.conn, self._payload(output_store=self.output_store["id"]))
        self.assertEqual(doc["chosen"]["output_store"], self.output_store["id"])

    def test_plan_reports_blockers_for_missing_inputs(self):
        doc = plan(self.conn, self._payload(inputs=["art-nope"]))
        self.assertIn("input artifact art-nope is not in the catalog", doc["blockers"])
        self.assertIsNone(doc["chosen"])

    def test_plan_blocks_when_no_storage(self):
        repos.set_provider_state(self.conn, self.output_store["id"], "paused")
        repos.set_provider_state(self.conn, self.input_store["id"], "paused")
        doc = plan(self.conn, self._payload())
        self.assertTrue(any("no writable storage" in b for b in doc["blockers"]))

    def test_decision_recorded_and_replayable(self):
        doc = plan(self.conn, self._payload(output_store=self.output_store["id"]))
        decision = record_decision(self.conn, "plan.preview", doc["spec"]["id"], doc)
        replay = repos.decision_dict(repos.get_decision(self.conn, decision["id"]))
        self.assertEqual(replay["subject"], "plan.preview")
        self.assertEqual(replay["chosen"]["output_store"], self.output_store["id"])
        selected = [a["provider_id"] for a in replay["alternatives"] if a.get("selected")]
        self.assertIn(self.output_store["id"], selected)
        self.assertIn("spec", replay["inputs"])


if __name__ == "__main__":
    unittest.main()
