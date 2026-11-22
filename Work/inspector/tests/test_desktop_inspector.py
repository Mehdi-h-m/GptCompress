"""Verify Sprint 1 with real checkpoint inventory and native Tk events."""

import sys
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gptcompress.inspector import inspection
from gptcompress.inspector.desktop_app import ArchitectureWindow


class DesktopInspectorTests(unittest.TestCase):
    def setUp(self):
        entries = {entry["name"]: entry for entry in inspection.versions()}
        names = ("Qwen3-4B", "als-r256-mlp")
        if not all(name in entries for name in names):
            self.skipTest("Requires the two registered local Qwen3 checkpoints")
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Requires a desktop display: {error}")
        self.addCleanup(self.root.destroy)
        self.app = ArchitectureWindow(self.root, [entries[name] for name in names])
        self.root.update()

    def click_node(self, target, *, expected=None):
        app = self.app
        tag = next(tag for tag, name in app.actions.items() if name == target)
        left, top, right, bottom = app.canvas.bbox(tag)
        x = (left + right) // 2 - int(app.canvas.canvasx(0))
        y = (top + bottom) // 2 - int(app.canvas.canvasy(0))
        app.canvas.event_generate("<ButtonPress-1>", x=x, y=y)
        app.canvas.event_generate("<ButtonRelease-1>", x=x, y=y)
        self.root.update()
        self.assertEqual(app.current, target if expected is None else expected)

    def labels(self):
        canvas = self.app.canvas
        return {canvas.itemcget(item, "text") for item in canvas.find_all() if canvas.type(item) == "text"}

    def test_navigate_dense_and_factorized_checkpoint(self):
        app = self.app
        self.assertEqual(len(app.blocks), 36)
        self.assertTrue({"Input token IDs", "Next-token logits", "Add attention residual", "Add MLP residual"}
                        <= self.labels())
        self.click_node("model.layers.0")
        self.assertTrue({"Input RMSNorm", "Post-attention RMSNorm", "Add attention residual"} <= self.labels())
        self.click_node("model.layers.0.self_attn")
        self.assertTrue({"Query projection", "Q RMSNorm", "K RMSNorm", "Rotary position embedding · Q",
                         "Causal mask", "32 query heads · 8 key/value heads · 128 features/head"} <= self.labels())
        app.go_back()
        self.click_node("model.layers.0.mlp")
        self.assertTrue({"SiLU activation", "Elementwise multiplication", "SiLU(gate) × up"} <= self.labels())
        projection = "model.layers.0.mlp.down_proj"
        self.click_node(projection)
        self.assertIn("W · Dense weight", self.labels())
        self.assertIn("2,560 × 9,728 · BF16", self.labels())
        app.select_version(1)
        self.assertEqual(app.current, projection)
        self.assertTrue({"B · Input factor", "A · Output factor",
                         "256 × 9,728 · BF16", "2,560 × 256 · BF16"} <= self.labels())

    def test_zoom_and_drag(self):
        app = self.app
        app.zoom(2)
        self.root.update()
        self.assertEqual(app.zoom_label.cget("text"), "200%")
        app.canvas.xview_moveto(0)
        app.canvas.event_generate("<ButtonPress-1>", x=450, y=400)
        app.canvas.event_generate("<B1-Motion>", x=250, y=400)
        app.canvas.event_generate("<ButtonRelease-1>", x=250, y=400)
        self.assertGreater(app.canvas.xview()[0], 0)
        app.reset_zoom()
        self.assertEqual(app.zoom_label.cget("text"), "100%")

    def test_scroll_decoder_and_open_active_stage(self):
        app = self.app
        canvas = app.canvas
        canvas.event_generate("<MouseWheel>", x=240, y=350, delta=-120)
        self.assertEqual(app.block_index, 1)
        self.assertEqual(app.zoom_label.cget("text"), "100%")
        self.assertIn("Block 1", self.labels())
        self.click_node("model.layers.1.self_attn")
        app.go_back()
        canvas.event_generate("<MouseWheel>", x=215, y=352, delta=-120)
        self.assertEqual(app.current, "model.layers.2")
        self.click_node("model.layers.2.mlp")
        app.open_module("")
        self.assertIn("Block 2", self.labels())
        canvas.event_generate("<Control-MouseWheel>", x=240, y=350, delta=120)
        self.assertGreater(app.scale, 1)
        self.assertEqual(app.block_index, 2)
        app.step_block(100)
        self.assertEqual(app.block_index, 35)

    def test_hover_animates_and_restores_node_color(self):
        app = self.app
        canvas = app.canvas
        tag = next(tag for tag, name in app.actions.items() if name == "model.layers.0.mlp")
        item, fill, _ = app.hover_nodes[tag]
        left, top, right, bottom = canvas.bbox(tag)
        canvas.event_generate("<Motion>", x=(left + right) // 2, y=(top + bottom) // 2)
        self.assertNotEqual(canvas.itemcget(item, "fill"), fill)
        self.assertEqual(canvas.cget("cursor"), "hand2")
        finished = tk.BooleanVar(master=self.root)
        self.root.after(150, finished.set, True)
        self.root.wait_variable(finished)
        self.assertEqual(canvas.itemcget(item, "fill"), "#cbdcff")
        canvas.event_generate("<Leave>")
        self.root.after(150, finished.set, True)
        self.root.wait_variable(finished)
        self.assertEqual(canvas.itemcget(item, "fill"), fill)

    def wait_for_slice(self, window):
        deadline = time.monotonic() + 8
        while window.result is None and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertIsNotNone(window.result, window.status.cget("text"))
        self.root.update()

    def test_matrix_window_dense_factors_spectrum_and_zoom(self):
        app = self.app
        projection = "model.layers.0.mlp.down_proj"
        app.open_module(projection)
        target = next(target for target in app.actions.values() if callable(target))
        self.click_node(target, expected=projection)
        window = app.matrix_window
        self.addCleanup(window.close)
        self.wait_for_slice(window)
        self.assertEqual(window.source_select.get(), "Dense weight W")
        self.assertIn("2,560 × 9,728", window.meta.cget("text"))
        self.assertTrue(window.heat.find_all() and window.image)
        x, y, width, height = window.map_bounds
        window.heat.event_generate("<Button-1>", x=x + width // 2, y=y + height // 2)
        self.wait_for_slice(window)
        self.assertEqual(window.side.get(), "256")
        row, col = window.row.get(), window.col.get()
        window.version_select.current(1)
        window.version_select.event_generate("<<ComboboxSelected>>")
        self.wait_for_slice(window)
        self.assertEqual((window.row.get(), window.col.get()), (row, col))
        self.assertEqual(window.source_select.get(), "Logical weight A × B")
        self.assertEqual(len(window.sources), 3)
        window.side.set("64")
        window.spectrum.set(True)
        window.source_select.current(1)
        window.source_select.event_generate("<<ComboboxSelected>>")
        self.wait_for_slice(window)
        self.assertIn("2,560 × 256", window.meta.cget("text"))
        self.assertEqual(len(window.result["spectrum"]), 64)
        window.row.set("99999")
        window.request()
        self.assertIn("Enter a row", window.status.cget("text"))
        self.assertIsNotNone(window.result)

    def test_compare_real_models_swap_runs_and_restore(self):
        app = self.app
        with tempfile.TemporaryDirectory() as directory, patch.object(inspection, "STORE", Path(directory)):
            compare_button = next(widget for frame in self.root.winfo_children() for widget in frame.winfo_children()
                                  if isinstance(widget, tk.ttk.Button) and widget.cget("text") == "Compare models")
            compare_button.invoke()
            self.root.update()
            window = app.comparison_window
            self.assertTrue(window.exists())
            if not window.score_rows:
                window.window.destroy()
                self.skipTest("Requires the matched local development runs")

            def labels(canvas):
                return {canvas.itemcget(item, "text") for item in canvas.find_all() if canvas.type(item) == "text"}

            self.assertTrue({"7.51 GiB", "3.13 GiB", "-58.3% relative to Version A"} <= labels(window.storage))
            self.assertTrue({"-80 pp (B − A)", "-50 pp (B − A)", "0.0%"} <= labels(window.scores))
            self.assertIn("Matched resource runs", window.resources_note.cget("text"))
            self.assertIn("Not measured", labels(window.resource_panels[2]))
            self.assertTrue({"2h 54m 53s", "1h 42m 15s"} <= labels(window.resource_panels[0]))
            for runs, picker in zip(window.run_lists, window.run_pickers):
                hybrid = next(index for index, row in enumerate(runs) if row["device"] == "hybrid")
                picker.current(hybrid)
                picker.event_generate("<<ComboboxSelected>>")
            self.assertIn("A generated 3 tokens · B generated 8 tokens", window.resources_note.cget("text"))
            self.assertIn("1.88 GiB", labels(window.resource_panels[2]))
            window.swap()
            self.assertTrue({"+80 pp (B − A)", "+50 pp (B − A)"} <= labels(window.scores))
            self.assertIn("A generated 8 tokens · B generated 3 tokens", window.resources_note.cget("text"))
            window.window.destroy()
            compare_button.invoke()
            restored = app.comparison_window
            self.assertEqual([selector.get() for selector in restored.pair], ["als-r256-mlp", "Qwen3-4B"])
            self.assertEqual([run["device"] for run in restored.chosen_runs()], ["hybrid", "hybrid"])
            restored.pair[1].current(restored.pair[0].current())
            restored.select_pair()
            self.assertEqual(restored.versions_in_view[0]["id"], restored.versions_in_view[1]["id"])


    def test_factorized_mlp_view_shows_factor_pairs(self):
        app = self.app
        app.select_version(1)
        app.open_module("model.layers.0.mlp")
        self.root.update()
        labels = self.labels()
        self.assertTrue({"B · Input factor", "A · Output factor", "256 × 2,560 · BF16", "9,728 × 256 · BF16",
                         "256 × 9,728 · BF16", "2,560 × 256 · BF16"} <= labels)
        self.assertTrue(any("rank 256" in label for label in labels))
        app.select_version(0)
        self.assertEqual(app.current, "model.layers.0.mlp")
        labels = self.labels()
        self.assertNotIn("B · Input factor", labels)
        self.assertTrue({"9,728 × 2,560", "2,560 × 9,728"} <= labels)

    def test_overview_storage_and_parameters_differ(self):
        app = self.app
        app.open_module("")
        self.assertIn("7.51 GiB", self.labels())
        self.assertIn("4,022,468,096", self.labels())
        app.select_version(1)
        self.assertEqual(app.current, "")
        self.assertIn("3.13 GiB", self.labels())
        self.assertIn("1,672,609,280", self.labels())


class GenericCheckpointTests(unittest.TestCase):
    """A checkpoint without Qwen3 structure must render through the generic adapter path alone."""

    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Requires a desktop display: {error}")
        self.addCleanup(self.root.destroy)

    def labels(self, app):
        canvas = app.canvas
        return {canvas.itemcget(item, "text") for item in canvas.find_all() if canvas.type(item) == "text"}

    def test_generic_checkpoint_renders_without_qwen3_structure(self):
        import json

        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps(
                {"model_type": "gpt2_custom", "vocab_size": 32, "hidden_size": 16}), encoding="utf-8")
            save_file({"model.embed_tokens.weight": torch.zeros(32, 16),
                       "lm_head.weight": torch.zeros(32, 16),
                       "h.0.attn.q.weight": torch.zeros(16, 16),
                       "h.0.attn.q.bias": torch.zeros(16)},
                      path / "model.safetensors", metadata={"format": "pt"})
            entry = inspection.CheckpointAdapter(path).describe()
            app = ArchitectureWindow(self.root, [entry])
            self.root.update()
            self.assertEqual(app.blocks, [])
            labels = self.labels(app)
            self.assertIn("Token embedding", labels)
            self.assertIn("Vocabulary head", labels)
            self.assertIn("lm_head", set(app.actions.values()))
            app.open_module("lm_head")
            self.root.update()
            self.assertIn("STORED TENSORS", self.labels(app))
            app.open_module("h.0.attn.q")
            self.root.update()
            self.assertTrue({"STORED TENSORS", "weight", "bias", "16 × 16 · F32", "16 · F32"}
                            <= self.labels(app))


if __name__ == "__main__":
    unittest.main()
