"""Focused inspection contracts, using explicitly synthetic tensors."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gptcompress.inspector.inspection import (adapter_for, benchmark_deltas, component_mapping, matrix_compare,
                                              read_json, save_json, stats)
from gptcompress.inspector.recording import align_channels, check_tokenizers, comparison, trace


class ToyInference(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Identity()
        self.fail = False

    def forward(self, input_ids, use_cache, past_key_values):
        values = self.projection(torch.nn.functional.one_hot(input_ids, num_classes=4).float())
        if self.fail:
            raise RuntimeError("Synthetic forward failure")
        return SimpleNamespace(logits=values, past_key_values=None)


class InspectionTests(unittest.TestCase):
    def test_inventory_identity_and_factor_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_json(root / "config.json", {"model_type": "qwen3"})
            save_json(root / "factorization.json", {"modules": {"model.layers.0.mlp.up_proj": {"rank": 2}}})
            save_file({"model.layers.0.mlp.up_proj.left.weight": torch.zeros(6, 2),
                       "model.layers.0.mlp.up_proj.right.weight": torch.zeros(2, 4)}, root / "model.safetensors")
            first = adapter_for(root).describe()
            second = adapter_for(root).describe()
            self.assertEqual(first["identity"], second["identity"])
            self.assertEqual(first["parameters"], 20)
            self.assertEqual(first["modules"][0]["kind"], "factorized")
            self.assertEqual(len(first["modules"][0]["tensors"]), 2)
            self.assertEqual(first["modules"][0]["shape"], [6, 4])
            self.assertEqual(first["modules"][0]["matrix"]["left"], "model.layers.0.mlp.up_proj.left.weight")
            save_json(root / "config.json", {"model_type": "qwen3", "vocab_size": 42})
            self.assertNotEqual(first["identity"], adapter_for(root).describe()["identity"])
            save_json(root / "recording.json", {"status": "complete", "missing": None})
            self.assertIsNone(read_json(root / "recording.json")["missing"])

    def test_statistics_and_comparison(self):
        self.assertEqual(stats(np.zeros(4))["norm"], 0)
        self.assertEqual(stats([float("nan"), 2])["nonfinite"], 1)
        self.assertIsNone(stats([float("nan")])["mean"])
        self.assertEqual(comparison([0], [0])["relative_error"], 0)
        self.assertIsNone(comparison([0], [1])["relative_error"])
        self.assertEqual(comparison([1, 2], [1])["status"], "incompatible shapes")
        self.assertEqual(comparison([float("nan")], [1])["status"], "nonfinite values")

    def test_explicit_pruning_mapping(self):
        left = np.array([[1., 2., 3.]])
        right = np.array([[1., 3.]])
        aligned, other = align_channels(left, right, {"a_indices": [0, 2], "b_indices": [0, 1]})
        self.assertEqual(comparison(aligned, other)["relative_error"], 0)
        with self.assertRaises(ValueError):
            align_channels(left, right, {"a_indices": [9], "b_indices": [0]})
        a = {"id": "a", "identity": "identity-a"}
        b = {"id": "b", "identity": "identity-b", "mappings": {"comparisons": {
            "a": {"mlp.proj": {"b_name": "mlp.pruned", "a_indices": [0, 2], "b_indices": [0, 1]}}
        }}}
        self.assertEqual(component_mapping(a, b, "mlp.proj")["b_name"], "mlp.pruned")
        self.assertEqual(component_mapping(b, a, "mlp.pruned")["b_name"], "mlp.proj")
        self.assertEqual(component_mapping(b, a, "mlp.pruned")["b_indices"], [0, 2])

    def test_tokenizer_and_vocabulary_compatibility(self):
        a = {"tokenizer_identity": "same", "vocab_size": 2}
        check_tokenizers([a, dict(a)])
        with self.assertRaisesRegex(ValueError, "Tokenizer"):
            check_tokenizers([a, {**a, "tokenizer_identity": None}])
        with self.assertRaisesRegex(ValueError, "vocabularies"):
            check_tokenizers([a, {**a, "vocab_size": 3}])

    def test_delta_requires_actual_question_ids(self):
        first = {"id": "a", "protocol_hash": "hash", "question_ids": {"task": [1, 5]}, "tasks": {
            "task": {"status": "complete", "metric": "score", "n": 2, "score": .5}}}
        second = {**first, "id": "b", "question_ids": {"task": [2, 5]}, "tasks": {
            "task": {"status": "complete", "metric": "score", "n": 2, "score": .25}}}
        a = {"experiments": [first]}
        b = {"experiments": [second]}
        self.assertFalse(benchmark_deltas(a, b))
        second["question_ids"] = first["question_ids"]
        self.assertEqual(benchmark_deltas(a, b)[0]["delta_pp"], -25)

    def test_full_matrix_error_and_source_span(self):
        name = "model.layers.0.mlp.up_proj"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dense, factors = root / "dense", root / "factors"
            for path in (dense, factors):
                path.mkdir()
                save_json(path / "config.json", {"model_type": "qwen3"})
                save_json(path / "manifest.json", {"repo": "synthetic", "revision": "fixture"})
            save_file({name + ".weight": torch.diag(torch.tensor([2., 1.]))}, dense / "model.safetensors")
            save_file({name + ".left.weight": torch.tensor([[2.], [0.]]),
                       name + ".right.weight": torch.tensor([[1., 0.]])}, factors / "model.safetensors")
            save_json(factors / "factorization.json", {"modules": {name: {"rank": 1}}})
            entries = {key: adapter_for(path).describe() for key, path in (("a", dense), ("b", factors))}
        with (patch("gptcompress.inspector.inspection.STORE", root / "store"),
              patch("gptcompress.inspector.inspection.get_version", entries.__getitem__)):
                compared = matrix_compare({"a": "a", "b": "b", "module": name}, lambda *_: None, lambda: False)
            self.assertAlmostEqual(compared["relative_squared_reconstruction_error"], .2)
            self.assertAlmostEqual(compared["retained_spectral_energy"], .8)

    def test_trace_capture_parity_and_cleanup(self):
        model = ToyInference()
        settings = ([0, 1], [3], [1, 2])
        captures, _, _, fingerprints = trace(
            model, *settings, {"projection": "projection"}, lambda: False, lambda *_: None, parity=True)
        self.assertEqual(set(captures["projection"]["outputs"]), {1, 2})
        self.assertFalse(model.projection._forward_hooks)
        plain = trace(model, *settings, {}, lambda: False, lambda *_: None, parity=True)
        self.assertEqual(fingerprints, plain[3])
        with self.assertRaises(InterruptedError):
            trace(model, *settings, {"projection": "projection"}, lambda: True, lambda *_: None)
        self.assertFalse(model.projection._forward_hooks)
        model.fail = True
        with self.assertRaisesRegex(RuntimeError, "Synthetic forward failure"):
            trace(model, *settings, {"projection": "projection"}, lambda: False, lambda *_: None)
        self.assertFalse(model.projection._forward_hooks)


if __name__ == "__main__":
    unittest.main()
