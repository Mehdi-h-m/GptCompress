"""Synthetic report fixtures exercise display and comparison correctness."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gptcompress.report import generate_report


class ReportTests(unittest.TestCase):
    def make_run(self, root, run_id, score, protocol="same", baseline=None, count=10):
        directory = root / "experiments" / run_id
        directory.mkdir(parents=True)
        manifest = {"run_id": run_id, "protocol_hash": protocol, "baseline_run_id": baseline,
                    "model_size_bytes": 8e9, "model_id": "synthetic-fixture"}
        metrics = {"tasks": {"gsm8k": {"score": score, "n": count, "status": "completed"}}, "resources": {}}
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        return directory

    def test_matching_protocol_delta_and_history_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self.make_run(root, "baseline", .7)
            self.make_run(root, "variant", .5, baseline="baseline")
            previous = (baseline / "metrics.json").read_bytes()
            report = generate_report(root).read_text(encoding="utf-8")
            self.assertIn("-20.00 pp", report)
            self.assertIn("<svg", report)
            self.assertEqual(previous, (baseline / "metrics.json").read_bytes())

    def test_incomparable_protocol_and_missing_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root, "baseline", .7)
            self.make_run(root, "different", .5, protocol="other", baseline="baseline")
            self.make_run(root, "missing", None, baseline="baseline")
            report = generate_report(root).read_text(encoding="utf-8")
            self.assertIn("Different or unknown protocol", report)
            self.assertIn("Not measured", report)
            self.assertNotIn("-20.00 pp", report)
            self.assertNotIn("<svg", report)

    def test_predictions_are_escaped_and_malformed_rows_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.make_run(root, "baseline", .7)
            prediction = {"task": "<script>alert(1)</script>", "response": "<img src=x onerror=alert(2)>"}
            (directory / "predictions.jsonl").write_text(json.dumps(prediction) + "\ninvalid json\n", encoding="utf-8")
            report = generate_report(root).read_text(encoding="utf-8")
            self.assertNotIn("<script>", report)
            self.assertNotIn("<img", report)
            self.assertIn("&lt;script&gt;", report)
            self.assertIn("predictions.jsonl line 2", report)

    def test_empty_workspace_and_partial_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIn("No benchmark measurements", generate_report(root).read_text(encoding="utf-8"))
            (root / "experiments" / "partial").mkdir(parents=True)
            report = generate_report(root).read_text(encoding="utf-8")
            self.assertIn("partial", report)
            self.assertIn("manifest.json", report)
            self.assertIn("metrics.json", report)

    def test_different_counts_suppress_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.make_run(root, "baseline", .7)
            self.make_run(root, "variant", .5, baseline="baseline", count=5)
            report = generate_report(root).read_text(encoding="utf-8")
            self.assertIn("Different sample counts", report)
            self.assertNotIn("-20.00 pp", report)


if __name__ == "__main__":
    unittest.main()
