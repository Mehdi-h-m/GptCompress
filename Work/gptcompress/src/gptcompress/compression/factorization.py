"""Weight-only alternating least squares and executable low-rank projections."""

import time

import torch
from torch import nn


def relative_error(weight, left, right, rows=256):
    numerator = denominator = 0.0
    for start in range(0, weight.shape[0], rows):
        block = weight[start:start + rows].float()
        residual = block - left[start:start + rows].float() @ right.float()
        numerator += residual.double().square().sum().item()
        denominator += block.double().square().sum().item()
    return numerator / denominator if denominator else numerator


def factorize(weight, rank, seed=42, tolerance=1e-5, max_iterations=20):
    """Alternate rank-revealing CPU least-squares solves without normal equations."""
    if weight.ndim != 2 or not 0 < rank <= min(weight.shape):
        raise ValueError("Rank must be positive and bounded by both matrix dimensions")
    if tolerance < 0 or max_iterations < 1:
        raise ValueError("Invalid convergence settings")
    weight = weight.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(weight).all():
        raise ValueError("Weight contains nonfinite values")
    generator = torch.Generator().manual_seed(seed)
    right = torch.randn(rank, weight.shape[1], generator=generator)
    history = []
    started = time.perf_counter()
    for iteration in range(max_iterations):
        left = torch.linalg.lstsq(right.T, weight.T, driver="gelsy").solution.T.contiguous()
        right = torch.linalg.lstsq(left, weight, driver="gelsy").solution.contiguous()
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise RuntimeError("Least-squares solve produced nonfinite factors")
        error = relative_error(weight, left, right)
        history.append(error)
        if error <= 1e-10:
            break
        if len(history) > 1:
            improvement = history[-2] - error
            if improvement < -1e-5 * max(history[-2], 1e-8):
                raise RuntimeError("ALS reconstruction error increased materially")
            if improvement <= tolerance * max(history[-2], 1e-8):
                break
    return left, right, {"relative_squared_error": history[-1], "loss_history": history,
                         "iterations": len(history), "seconds": time.perf_counter() - started}


class FactorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.right = nn.Linear(in_features, rank, bias=False)
        self.left = nn.Linear(rank, out_features, bias=bias)

    def forward(self, inputs):
        return self.left(self.right(inputs))


def load_factorized_model(path, dtype=torch.bfloat16):
    import json
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    specification = json.loads((path / "factorization.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=dtype)
        for name, record in specification["modules"].items():
            original = model.get_submodule(name)
            replacement = FactorizedLinear(original.in_features, original.out_features,
                                           record["rank"], original.bias is not None)
            model.set_submodule(name, replacement)
    model.tie_weights()
    expected = model.state_dict()
    present = set()
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        filenames = sorted(set(index.values()))
    else:
        filenames = ["model.safetensors"]
    for filename in filenames:
        with safe_open(path / filename, framework="pt", device="cpu") as checkpoint:
            for name in checkpoint.keys():
                if name in present or name not in expected:
                    raise ValueError(f"Unexpected or duplicated checkpoint tensor {name}")
                if tuple(checkpoint.get_slice(name).get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"Checkpoint shape mismatch for {name}")
                present.add(name)
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    for names in aliases.values():
        if not present.intersection(names):
            raise ValueError(f"Missing checkpoint parameter {names[0]}")
    model = load_checkpoint_and_dispatch(model, str(path), device_map={"": "cpu"}, dtype=dtype)
    if (path / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(path, local_files_only=True)
    return model.eval()
