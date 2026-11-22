"""Kaggle assisted execution — adapter surface and assisted-cycle pieces."""

import unittest

from drm.planner.core import plan
from drm.providers.base import ComputeAdapter
from drm.providers.kaggle import ADVERTISED_LIMITS, KaggleCompute
from drm.providers.registry import build_bundle


class KaggleAdapterTests(unittest.TestCase):
    def test_registry_declares_kaggle_as_assisted_compute(self):
        from drm.providers.registry import ADAPTERS

        self.assertEqual(ADAPTERS["kaggle"]["kind"], "compute")
        self.assertEqual(ADAPTERS["kaggle"]["execution"], "assisted")

    def test_adapter_satisfies_the_compute_contract(self):
        adapter = KaggleCompute({"name": "kaggle-a"})
        self.assertIsInstance(adapter, ComputeAdapter)

    def test_probe_records_documented_limits_as_advertised(self):
        probe = KaggleCompute().probe()
        self.assertEqual(probe["advertised"]["session_duration_limit_hours"], 9)
        self.assertEqual(probe["advertised"]["weekly_gpu_quota_hours"], 30)
        self.assertFalse(probe["advertised"]["inbound_connectivity"])
        self.assertIn("observed_at", probe)

    def test_quote_is_zero_with_bounded_quota_and_unknowns(self):
        quote = KaggleCompute().quote({})
        self.assertEqual(quote["prices"]["compute_per_hour"], 0.0)
        self.assertTrue(quote["unknowns"])

    def test_cancel_is_explicitly_manual(self):
        result = KaggleCompute().cancel("att-x")
        self.assertTrue(result["cancelled"])
        self.assertFalse(result["confirmed"])
        self.assertIn("manual_step", result)

    def test_build_bundle_yields_compute_face_only(self):
        bundle = build_bundle("kaggle", {})
        self.assertIsNotNone(bundle.compute)
        self.assertIsNone(bundle.storage)

    def test_notebook_instructions_name_bundle_and_import_command(self):
        text = KaggleCompute().notebook_instructions(
            {"bundle_dir": "d", "bundle_zip": "z", "attempt_id": "att-1"}
        )
        self.assertIn("drm run import att-1", text)
        self.assertIn("HF_TOKEN", text)


if __name__ == "__main__":
    unittest.main()
