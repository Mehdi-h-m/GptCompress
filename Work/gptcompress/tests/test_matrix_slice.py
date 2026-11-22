"""Check native tiled reads against the installed checkpoint reader."""

import sys
import unittest
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gptcompress.inspector import inspection
from gptcompress.inspector.matrix import heatmap_cells, inspect_slice, projection_sources


class MatrixSliceTests(unittest.TestCase):
    @staticmethod
    def version(name):
        return next((item for item in inspection.versions() if item["name"] == name), None)

    @staticmethod
    def reference(weight, rows, cols):
        with safe_open(weight.path, framework="pt", device="cpu") as checkpoint:
            return checkpoint.get_slice(weight.name)[rows, cols].float().numpy()

    def test_original_and_factor_tiles_match_reference(self):
        dense = self.version("Qwen3-4B")
        als = self.version("als-r256-mlp")
        if not dense or not als:
            self.skipTest("Requires both local Qwen3 checkpoints")
        module = "model.layers.0.mlp.down_proj"
        direct, = projection_sources(dense, module)
        logical, a, b = projection_sources(als, module)
        rows, cols = slice(101, 117), slice(213, 229)
        np.testing.assert_array_equal(direct.read(rows, cols), self.reference(direct.weights[0], rows, cols))
        np.testing.assert_array_equal(a.read(rows, cols), self.reference(a.weights[0], rows, cols))
        np.testing.assert_array_equal(b.read(rows, cols), self.reference(b.weights[0], rows, cols))
        left = self.reference(a.weights[0], rows, slice(None))
        right = self.reference(b.weights[0], slice(None), cols)
        np.testing.assert_allclose(logical.read(rows, cols), left @ right, rtol=1e-5, atol=1e-7)
        report = inspect_slice(logical, 101, 213, 16, True)
        self.assertEqual(report["shape"], (16, 16))
        self.assertEqual(report["step"], 1)
        self.assertEqual(sum(report["histogram"][0]), 256)
        self.assertEqual(len(report["spectrum"]), 16)
        self.assertEqual(inspect_slice(a, 2558, 254, 16)["shape"], (2, 2))

    def test_heatmap_preserves_signed_extrema_and_partial_tiles(self):
        tile = np.zeros((65, 67), dtype=np.float32)
        tile[0, 0], tile[1, 0], tile[64, 66] = -3, 2, 4
        cells, step = heatmap_cells(tile)
        self.assertEqual(step, 2)
        self.assertEqual(cells.shape, (33, 34))
        self.assertEqual(cells[0, 0], -3)
        self.assertEqual(cells[-1, -1], 4)


if __name__ == "__main__":
    unittest.main()
