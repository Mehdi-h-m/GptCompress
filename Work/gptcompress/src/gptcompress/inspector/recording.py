"""Sequential CPU captures and aligned Qwen3 inference diagnostics."""

import gc
import hashlib
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version as package_version

import numpy as np
import psutil
import torch
from transformers import AutoTokenizer

from gptcompress.cpu_linear import use_tiled_fp32_linear
from gptcompress.compression.factorization import FactorizedLinear
from .inspection import (STORE, adapter_for, check_cancelled, component_mapping, corresponding_name,
                         get_version, save_json, stats)

MIB = 1024**2


def comparison(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return {"status": "incompatible shapes", "relative_error": None, "cosine": None}
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return {"status": "nonfinite values", "relative_error": None, "cosine": None}
    norm_left, norm_right = np.linalg.norm(left), np.linalg.norm(right)
    difference = np.linalg.norm(left - right)
    return {"status": "matched", "relative_error": float(difference / norm_left) if norm_left else (0.0 if not difference else None),
            "cosine": float(np.clip(np.sum(left * right) / (norm_left * norm_right), -1, 1)) if norm_left and norm_right else None,
            "zero_reference": bool(norm_left == 0)}


def align_channels(left, right, mapping):
    """Apply explicit paired final-axis indices; never infer pruning correspondences."""
    a_indices, b_indices = mapping.get("a_indices"), mapping.get("b_indices")
    if not isinstance(a_indices, list) or not isinstance(b_indices, list) or len(a_indices) != len(b_indices) or not a_indices:
        raise ValueError("Channel mapping needs equally sized nonempty paired index lists")
    for indices, width in ((a_indices, left.shape[-1]), (b_indices, right.shape[-1])):
        if any(type(i) is not int or not 0 <= i < width for i in indices) or len(set(indices)) != len(indices):
            raise ValueError("Invalid or duplicated channel mapping index")
    return left[..., a_indices], right[..., b_indices]


def check_tokenizers(selected):
    if len(selected) == 2:
        a, b = selected
        if not a["tokenizer_identity"] or a["tokenizer_identity"] != b["tokenizer_identity"]:
            raise ValueError("Tokenizer identities differ or are unavailable. Aligned token comparison is unavailable.")
        if not a["vocab_size"] or a["vocab_size"] != b["vocab_size"]:
            raise ValueError("Configured model vocabularies differ or are unavailable.")


def load_model(version):
    model = adapter_for(version["path"]).load()
    model.eval()
    use_tiled_fp32_linear(model)
    return model


def _row(tensor, position):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        return None
    return tensor[0, position].detach().float().cpu().numpy().copy()


def trace(model, tokens, continuation, positions, locations, cancelled, progress, parity=False):
    """Run one prefill and bounded cached decode, removing all hooks even on failure."""
    prompt = list(tokens)
    greedy = isinstance(continuation, int)
    steps = continuation if greedy else len(continuation)
    chosen = [] if greedy else list(continuation)
    handles = []
    captures = {key: {"inputs": {}, "outputs": {}} for key in locations}
    distributions = {}
    fingerprints = []
    offset = 0
    length = len(prompt)
    buffered_bytes = 0

    def hook(name):
        def capture(module, args, kwargs, value):
            nonlocal buffered_bytes
            check_cancelled(cancelled)
            value = value[0] if isinstance(value, tuple) else value
            first = args[0] if args else kwargs.get("hidden_states")
            for position in positions:
                if offset <= position < offset + length:
                    for field, tensor in (("inputs", first), ("outputs", value)):
                        row = _row(tensor, position - offset)
                        if row is not None:
                            buffered_bytes += row.nbytes
                            if buffered_bytes > 64 * MIB:
                                raise ValueError("Activation capture exceeds 64 MiB per version")
                            captures[name][field][position] = row
        return capture

    try:
        for key, name in locations.items():
            handles.append(model.get_submodule(name).register_forward_hook(hook(key), with_kwargs=True))
        with torch.inference_mode():
            cache = None
            current = prompt
            for step in range(steps + 1):
                check_cancelled(cancelled)
                length = len(current)
                progress("Prefill" if step == 0 else f"Decode {step}/{steps}", step / max(1, steps))
                output = model(input_ids=torch.tensor([current]), use_cache=bool(steps), past_key_values=cache)
                logits = output.logits[0]
                if not torch.isfinite(logits).all():
                    raise ValueError("Model returned nonfinite logits")
                if parity:
                    fingerprints.append(hashlib.sha256(logits.contiguous().view(torch.uint8).numpy()).hexdigest())
                for position in sorted(set(positions) | {offset + length - 1}):
                    if offset <= position < offset + length:
                        distributions[position] = logits[position - offset].float().softmax(-1).numpy()
                cache = output.past_key_values
                offset += length
                if step < steps:
                    if greedy:
                        chosen.append(int(logits[-1].argmax()))
                    current = [chosen[step]]
                del output, logits
    finally:
        for handle in handles:
            handle.remove()
    return captures, distributions, chosen, fingerprints


def _measure_version(version, locations, tokens, continuation, positions, request, replay, progress, cancelled):
    process = psutil.Process()
    peak = process.memory_info().rss
    finished = threading.Event()

    def sample():
        nonlocal peak
        while not finished.wait(.1):
            peak = max(peak, process.memory_info().rss)

    # Safetensors weights remain file-backed. Reserve RAM for private workspaces,
    # activations, cache and transient conversions in the tiled CPU kernel.
    required = 2 * 1024 * MIB
    if psutil.virtual_memory().available < required:
        raise ValueError(f"Free RAM is below the conservative {required / 1024**3:.2f} GiB estimate for {version['name']}")
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        progress(f"Loading {version['name']} on CPU", 0)
        model = load_model(version)
        check_cancelled(cancelled)
        modules = dict(model.named_modules())
        tracing = {key: name for key, name in locations.items() if name in modules}
        loaded = time.perf_counter()
        captures, probabilities, continuation, fingerprints = trace(
            model, tokens, continuation, positions, tracing, cancelled,
            lambda phase, amount: progress(f"{version['name']} · {phase}", .2 + .55 * amount),
            parity=request.get("verify_parity", False))
        recorded = time.perf_counter()
        parity_result = None
        if request.get("verify_parity"):
            progress(f"Checking bitwise capture parity for {version['name']}", .8)
            plain = trace(model, tokens, continuation, [], {}, cancelled, lambda *_: None, parity=True)[3]
            parity_result = {"exact": fingerprints == plain, "algorithm": "SHA-256 of all logits in each pass"}
        local = []
        if replay:
            progress(f"Replaying common projection inputs on {version['name']}", .9)
        for key, example in replay.items():
            check_cancelled(cancelled)
            name = tracing.get(key)
            if not name:
                local.append({"name": key, "status": "unmatched or unavailable"})
                continue
            module = model.get_submodule(name)
            input_width = module.right.in_features if isinstance(module, FactorizedLinear) else getattr(
                module, "in_features", None)
            inputs = example["inputs"]
            outputs = example["outputs"]
            indices = sorted(inputs.keys() & outputs.keys())
            if not indices or not input_width or any(inputs[i].size != input_width for i in indices):
                local.append({"name": key, "status": "incompatible or unavailable input width"})
                continue
            with torch.inference_mode():
                x = torch.from_numpy(np.stack([inputs[i] for i in indices])).to(torch.bfloat16).unsqueeze(0)
                predicted = module(x)[0].float().numpy()
            local.append({"name": key, **comparison(np.stack([outputs[i] for i in indices]), predicted),
                          "positions": indices, "basis": "Both projections receive Version A's captured BF16 input"})
        details = {"id": version["id"], "identity": version["identity"], "name": version["name"],
                   "load_seconds": loaded - started, "forward_seconds": recorded - loaded,
                   "seconds": time.perf_counter() - started, "capture_parity": parity_result,
                   "captures": capture_metadata(locations, captures, positions, request.get("raw", False))}
    finally:
        finished.set()
        monitor.join()
    details["peak_rss_bytes"] = max(peak, process.memory_info().rss)
    return details, captures, probabilities, continuation, local


def capture_metadata(locations, captures, positions, raw):
    metadata = {}
    for key, name in locations.items():
        entry = metadata[key] = {"module": name, "status": "captured", "positions": {}}
        data = captures.get(key)
        if not data or not data["outputs"]:
            entry["status"] = "unavailable capture"
            continue
        for position in positions:
            grouped = {}
            for kind, values in data.items():
                if position not in values:
                    continue
                value = values[position]
                grouped[kind] = {"shape": list(value.shape), "stats": stats(value)}
                if raw:
                    grouped[kind]["first_128_features"] = [float(x) if np.isfinite(x) else None for x in value[:128]]
            entry["positions"][position] = grouped
    return metadata


def activation_deltas(a, b, names, captured):
    results = []
    for name in names:
        other, status = corresponding_name(a, b, name)
        row = {"name": name, "b_module": other, "status": status, "by_position": []}
        if status == "matched":
            left = captured[0].get(name, {}).get("outputs", {})
            right = captured[1].get(name, {}).get("outputs", {})
            shared = sorted(left.keys() & right.keys())
            if not shared:
                row["status"] = "unavailable capture" if left or right else "unmatched or unavailable"
            else:
                x, y = np.stack([left[i] for i in shared]), np.stack([right[i] for i in shared])
                mapping = component_mapping(a, b, name)
                basis = "Aligned positions and native channels in whole-model forward passes"
                if mapping and (mapping.get("a_indices") is not None or mapping.get("b_indices") is not None):
                    try:
                        x, y = align_channels(x, y, mapping)
                    except ValueError as error:
                        row.update(status="incompatible channel mapping", reason=str(error))
                        results.append(row)
                        continue
                    basis = "Aligned positions and explicitly mapped channels in whole-model forward passes"
                row.update(comparison(x, y), basis=basis)
                row["by_position"] = [{"position": p, **comparison(left_row, right_row)}
                                      for p, left_row, right_row in zip(shared, x, y)]
        results.append(row)
    return results


def token_row(tokenizer, token, probability):
    return {"id": int(token), "text": tokenizer.decode([int(token)]), "probability": float(probability)}


def top_tokens(tokenizer, distribution):
    top = np.argsort(distribution, stable=True)[-10:][::-1]
    return [token_row(tokenizer, token, distribution[token]) for token in top]


def probability_deltas(tokenizer, a, b):
    rows = []
    for position in sorted(a.keys() & b.keys()):
        p, q = a[position], b[position]
        if p.shape != q.shape:
            rows.append({"position": position, "status": "incompatible vocabulary"})
            continue
        difference = q.astype(np.float64) - p
        indices = np.argsort(np.abs(difference), stable=True)[-12:][::-1]
        rows.append({"position": position, "status": "matched",
                     "total_variation": float(np.abs(difference).sum() / 2),
                     "largest_deltas": [{**token_row(tokenizer, t, q[t]), "a_probability": float(p[t]),
                                         "delta": float(difference[t])} for t in indices],
                     "a_top": top_tokens(tokenizer, p), "b_top": top_tokens(tokenizer, q)})
    return rows


def record(request, progress, cancelled):
    selected = [get_version(key) for key in (request["a"], request.get("b")) if key]
    if not selected or any("capture" not in v["capabilities"] for v in selected):
        raise ValueError("Activation capture requires a supported Qwen3 adapter")
    check_tokenizers(selected)
    if shutil.disk_usage(STORE).free < 512 * MIB:
        raise ValueError("Recording requires at least 512 MiB free disk")
    prompt = str(request.get("prompt", ""))
    if not prompt.strip():
        raise ValueError("Enter a prompt")
    budget = int(request.get("max_tokens", 32))
    decode = int(request.get("decode_tokens", 0))
    if not 1 <= budget <= 128 or not 0 <= decode <= 8:
        raise ValueError("Choose up to 128 prompt tokens and 8 decode tokens")
    tokenizer = AutoTokenizer.from_pretrained(selected[0]["path"], local_files_only=True, trust_remote_code=False)
    original_tokens = tokenizer.encode(prompt)
    tokens = original_tokens[:budget]
    if not tokens:
        raise ValueError("Prompt has no tokens")
    positions = sorted(set(int(p) for p in (request.get("positions") or range(min(len(tokens) + decode, 8)))))
    if len(positions) > 16 or any(p < 0 or p >= len(tokens) + decode for p in positions):
        raise ValueError(f"Choose at most 16 token positions between 0 and {len(tokens) + decode - 1}")
    blocks = [m for m, kind in selected[0]["capture_locations"].items() if kind == "transformer block"]
    defaults = list(dict.fromkeys(blocks[i] for i in (0, (len(blocks) - 1) // 2, len(blocks) - 1))) if blocks else []
    names = list(dict.fromkeys(request.get("modules") or defaults))
    if len(names) > 40 or any(not isinstance(name, str) for name in names):
        raise ValueError("Choose up to 40 named modules")
    if request.get("raw") and len(names) * len(positions) * 128 * 2 > 65536:
        raise ValueError("Reduce raw capture to at most 65,536 input/output values per version")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    now = datetime.now(timezone.utc)
    identifier = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    output = STORE / "recordings" / identifier / "recording.json"
    result = {"schema": 2, "id": identifier, "created": now.isoformat(), "a": selected[0]["id"],
              "b": selected[1]["id"] if len(selected) > 1 else None, "prompt": prompt, "positions": positions,
              "versions": [], "comparisons": [], "local_comparisons": [], "status": "running",
              "settings": {"max_tokens": budget, "decode_tokens": decode, "prompt_truncated": len(original_tokens) > budget,
                           "raw": bool(request.get("raw")), "tokenizer_identity": selected[0]["tokenizer_identity"],
                           "arithmetic": "CPU BF16 weights, tiled FP32 linear accumulation, BF16 projection output",
                           "threads": 4, "seed": 42, "tokenization": "Plain prompt without chat template",
                           "packages": {name: package_version(name) for name in ("torch", "transformers", "safetensors")},
                           "continuation": "Greedy Version A continuation replayed as fixed input for Version B",
                           "rss_sampling_ms": 100}, "notes": [
                  "Runtime includes capture work. Process RSS is sampled every 100 ms and can miss transient peaks.",
                  "Activation differences are diagnostic evidence. They do not establish cause.",
                  "Zero reference norm gives unavailable relative error except for equal zero-valued arrays."]}
    save_json(output, result)
    captures = []
    distributions = []
    continuation = decode
    try:
        for index, version in enumerate(selected):
            check_cancelled(cancelled)
            locations = {name: name for name in names}
            replay = {}
            if index:
                locations = {}
                for name in names:
                    other, status = corresponding_name(selected[0], version, name)
                    if status != "matched":
                        continue
                    locations[name] = other
                    if name.endswith("_proj") and not component_mapping(selected[0], version, name) and name in captures[0]:
                        replay[name] = captures[0][name]
            locations = {key: name for key, name in locations.items() if name in version["capture_locations"]}
            details, capture, probabilities, continuation, local = _measure_version(
                version, locations, tokens, continuation, positions, request, replay,
                lambda message, value: progress(message, (index + value) / len(selected)), cancelled)
            gc.collect()
            for name in names:
                details["captures"].setdefault(name, {"status": "unavailable or unmatched capture"})
            details["top_tokens"] = top_tokens(tokenizer, probabilities[max(probabilities)])
            result["versions"].append(details)
            captures.append(capture)
            distributions.append(probabilities)
            result["local_comparisons"].extend(local)
            sequence = tokens + continuation
            result["tokens"] = [{"id": t, "text": tokenizer.decode([t]), "position": i,
                                 "phase": "prefill" if i < len(tokens) else "decode"} for i, t in enumerate(sequence)]
            save_json(output, result)
        check_cancelled(cancelled)
        if len(selected) == 2:
            result["comparisons"] = activation_deltas(*selected, names, captures)
            result["probability_comparisons"] = probability_deltas(tokenizer, *distributions)
        result["status"] = "complete"
        return result
    except Exception as error:
        result.update(status="cancelled" if isinstance(error, InterruptedError) else "failed", error=str(error))
        raise
    finally:
        save_json(output, result)
