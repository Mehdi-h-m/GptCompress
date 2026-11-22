"""Bound the temporary FP32 workspace for BF16 CPU linear operations."""

from types import MethodType

import torch
from torch.nn import functional

TILE_BYTES = 64 * 1024**2


def _forward(layer, inputs):
    rows = max(1, TILE_BYTES // (4 * layer.in_features))
    fp32_inputs = inputs.float()
    outputs = []
    for start in range(0, layer.out_features, rows):
        end = start + rows
        bias = layer.bias[start:end].float() if layer.bias is not None else None
        output = functional.linear(fp32_inputs, layer.weight[start:end].float(), bias)
        outputs.append(output.to(inputs.dtype))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


def use_tiled_fp32_linear(model):
    """Use exact FP32 casts of BF16 weights and round each projection output to BF16."""
    count = 0
    for layer in model.modules():
        if (type(layer) is torch.nn.Linear and layer.weight.dtype == torch.bfloat16
                and layer.weight.device.type == "cpu"):
            forward = MethodType(_forward, layer)
            if hasattr(layer, "_hf_hook"):
                layer._old_forward = forward
            else:
                layer.forward = forward
            count += 1
    return count
