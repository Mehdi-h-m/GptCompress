import json
import tempfile
import unittest
from pathlib import Path

from drm.workers.tasks import TASK_REGISTRY, TaskCancelled, TaskContext, run_hash_inputs, run_sleep


class TaskRegistryTests(unittest.TestCase):
    def test_registered_types(self):
        self.assertIn("noop@1", TASK_REGISTRY)
        self.assertIn("hash_inputs@1", TASK_REGISTRY)
        self.assertIn("sleep@1", TASK_REGISTRY)
        self.assertIn("fail_always@1", TASK_REGISTRY)

    def test_hash_inputs_writes_deterministic_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            inputs = workdir / "inputs"
            inputs.mkdir()
            (inputs / "a.bin").write_bytes(b"alpha")
            (inputs / "b.bin").write_bytes(b"beta")
            ctx = TaskContext(workdir=workdir, params={}, cancelled=lambda: False)
            result = run_hash_inputs(ctx)
            self.assertEqual(result["stats"]["inputs_hashed"], 2)
            summary = json.loads((workdir / "outputs" / "summary.json").read_text())
            self.assertEqual(len(summary["inputs"]), 2)
            first = summary["inputs"][0]
            self.assertEqual(first["sha256"], __import__("hashlib").sha256(b"alpha").hexdigest())

            # determinism: rerunning over the same inputs yields byte-identical output
            before = (workdir / "outputs" / "summary.json").read_bytes()
            run_hash_inputs(ctx)
            self.assertEqual((workdir / "outputs" / "summary.json").read_bytes(), before)

    def test_sleep_respects_cancellation_quickly(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = TaskContext(workdir=Path(tmp), params={"seconds": 5}, cancelled=lambda: True)
            start = __import__("time").monotonic()
            with self.assertRaises(TaskCancelled):
                run_sleep(ctx)
            self.assertLess(__import__("time").monotonic() - start, 1.0)

    def test_fail_always_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = TaskContext(workdir=Path(tmp), params={}, cancelled=lambda: False)
            with self.assertRaises(RuntimeError):
                TASK_REGISTRY["fail_always@1"](ctx)


if __name__ == "__main__":
    unittest.main()
