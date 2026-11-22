"""Comparison checks with synthetic missing and mismatched run records."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gptcompress.inspector.comparison_data import default_runs, matched_workload, relative_delta, resource_values


class ComparisonDataTests(unittest.TestCase):
    def test_resource_matching_validates_every_task_and_execution_setting(self):
        run = {"status": "complete", "protocol_hash": "p1", "device": "cpu", "cpu_threads": 4,
               "question_ids": {"task": [2, 8]}, "tasks": {"task": {"status": "complete", "n": 2}}}
        self.assertTrue(matched_workload(run, dict(run)))
        for change in ({"question_ids": {"task": [8, 2]}}, {"cpu_threads": 8}, {"device": "hybrid"},
                       {"protocol_hash": None}, {"tasks": {}}, {"status": "failed"}):
            self.assertFalse(matched_workload(run, {**run, **change}), change)
        alternate = {**run, "protocol_hash": "smoke", "question_ids": {}, "tasks": {}}
        self.assertFalse(matched_workload(alternate, alternate))
        self.assertEqual(default_runs([alternate, run], [alternate, run]), (run, run))

    def test_missing_values_are_distinct_from_zero(self):
        self.assertEqual(resource_values(None), (None, None, None))
        run = {"status": "complete", "seconds": 0, "resources": {"peak_vram_bytes": 0}}
        self.assertEqual(resource_values(run), (0, None, 0))
        self.assertEqual(resource_values({**run, "status": "failed"}), (None, None, None))
        self.assertEqual(relative_delta(8, 4), -50)
        self.assertEqual(relative_delta(4, 8), 100)
        self.assertIsNone(relative_delta(0, 8))
        self.assertIsNone(relative_delta(8, None))
