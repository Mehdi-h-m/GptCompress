import sys
import unittest
import tempfile
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gptcompress.compression.factorization import FactorizedLinear, factorize, relative_error, load_factorized_model


class FactorizationTests(unittest.TestCase):
    def test_checkpoint_reload(self):
        from transformers import Qwen3Config, AutoModelForCausalLM
        from safetensors.torch import save_file

        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, tie_word_embeddings=True)
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).eval()
        name = "model.layers.0.mlp.up_proj"
        original = model.get_submodule(name)
        left, right, _ = factorize(original.weight, 4)
        layer = FactorizedLinear(16, 24, 4)
        layer.left.weight.data.copy_(left)
        layer.right.weight.data.copy_(right)
        model.set_submodule(name, layer.bfloat16())
        inputs = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            expected = model(inputs).logits
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config.save_pretrained(path)
            state = {key: value.contiguous() for key, value in model.state_dict().items()
                     if key != "lm_head.weight"}
            save_file(state, path / "model.safetensors", metadata={"format": "pt"})
            (path / "factorization.json").write_text(json.dumps({"modules": {name: {"rank": 4}}}))
            restored = load_factorized_model(path)
            with torch.inference_mode():
                torch.testing.assert_close(expected, restored(inputs).logits, rtol=0, atol=0)
            self.assertIs(restored.lm_head.weight, restored.model.embed_tokens.weight)
            state.pop(name + ".left.weight")
            save_file(state, path / "model.safetensors", metadata={"format": "pt"})
            with self.assertRaisesRegex(ValueError, "Missing checkpoint parameter"):
                load_factorized_model(path)

    def test_known_rank_and_execution(self):
        torch.manual_seed(17)
        weight = torch.randn(40, 4) @ torch.randn(4, 30)
        left, right, record = factorize(weight, 4)
        self.assertLess(record["relative_squared_error"], 1e-10)
        layer = FactorizedLinear(30, 40, 4)
        layer.left.weight.data.copy_(left)
        layer.right.weight.data.copy_(right)
        inputs = torch.randn(2, 3, 30)
        torch.testing.assert_close(layer(inputs), inputs @ weight.T, atol=1e-4, rtol=1e-4)

    def test_deficient_and_zero(self):
        torch.manual_seed(2)
        weight = torch.randn(20, 2) @ torch.randn(2, 15)
        left, right, _ = factorize(weight, 5)
        self.assertLess(relative_error(weight, left, right), 1e-10)
        _, _, record = factorize(torch.zeros(20, 15), 5)
        self.assertEqual(record["relative_squared_error"], 0)

    def test_full_rank_error_and_validation(self):
        torch.manual_seed(3)
        weight = torch.randn(30, 20)
        _, _, record = factorize(weight, 5, max_iterations=40)
        singular = torch.linalg.svdvals(weight)
        optimum = (singular[5:].square().sum() / singular.square().sum()).item()
        self.assertGreaterEqual(record["relative_squared_error"] + 1e-6, optimum)
        self.assertLess(record["relative_squared_error"] - optimum, 0.001)
        with self.assertRaises(ValueError):
            factorize(weight, 0)
        with self.assertRaises(ValueError):
            factorize(torch.full((10, 10), float("nan")), 2)


if __name__ == "__main__":
    unittest.main()
