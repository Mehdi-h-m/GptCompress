"""Keep selected Qwen3 modules resident on CUDA and the remainder on CPU."""

import torch
from accelerate import dispatch_model


def place_qwen3(model, budget_mib):
    budget = budget_mib * 1024**2
    free_bytes, _ = torch.cuda.mem_get_info()
    if budget + 768 * 1024**2 > free_bytes:
        raise RuntimeError("GPU weight budget leaves less than 768 MiB free for inference. Lower --gpu-weights-mib.")

    def parameter_bytes(*modules):
        unique = {id(parameter): parameter for module in modules for parameter in module.parameters()}
        return sum(parameter.numel() * parameter.element_size() for parameter in unique.values())

    device_map = {"model.embed_tokens": 0, "lm_head": 0, "model.norm": 0, "model.rotary_emb": "cpu"}
    used = parameter_bytes(model.model.embed_tokens, model.lm_head, model.model.norm)
    if used > budget:
        raise ValueError("GPU weight budget cannot fit the tied embedding and output head")

    # Keep the last blocks next to the output head to limit activation transfers.
    for index in reversed(range(len(model.model.layers))):
        size = parameter_bytes(model.model.layers[index])
        device = 0 if used + size <= budget else "cpu"
        device_map[f"model.layers.{index}"] = device
        if device == 0:
            used += size
    dispatch_model(model, device_map, main_device="cpu", force_hooks=True,
                   skip_keys=model._skip_keys_device_placement)
    return device_map, used
