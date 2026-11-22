"""Read-only safetensors adapters, model registry, and bounded matrix inspection."""

import hashlib
import json
import math
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path

WORK = Path(__file__).resolve().parents[3]
STORE = WORK / "inspection"
REGISTRY_LOCK = threading.RLock()
TOKENIZER_ASSETS = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                    "vocab.json", "merges.txt", "tokenizer.model", "added_tokens.json", "chat_template.jinja")


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, encoding="utf-8", mode="w") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path, cancelled=lambda: False):
    digest_value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            check_cancelled(cancelled)
            digest_value.update(chunk)
    return digest_value.hexdigest()


def check_cancelled(cancelled):
    if cancelled():
        raise InterruptedError("Cancelled")


def asset_files(path):
    return sorted(file for file in path.iterdir() if file.is_file() and
                  (file.suffix == ".safetensors" or file.name.endswith(".safetensors.index.json")
                   or file.name in (*TOKENIZER_ASSETS, "config.json", "generation_config.json",
                                    "factorization.json", "component_mappings.json")))


def signature(path):
    metadata = [path / name for name in ("manifest.json", "reconstruction.json")]
    files = sorted(set(asset_files(path)) | {file for file in metadata if file.is_file()})
    return [[file.name, file.stat().st_size, file.stat().st_mtime_ns] for file in files]


def file_checksums(path, manifest, cancelled):
    known = {item["path"]: item for item in manifest.get("files", [])}
    checksums = {}
    for file in asset_files(path):
        stamp = file.stat()
        item = known.get(file.name, {})
        if (item.get("bytes") == stamp.st_size and item.get("mtime_ns") == stamp.st_mtime_ns
                and item.get("sha256")):
            checksums[file.name] = item["sha256"]
        else:
            checksums[file.name] = file_hash(file, cancelled)
    return checksums


def experiments(identity):
    if not identity:
        return []
    rows = []
    for path in sorted((WORK / "experiments").glob("*/manifest.json"), reverse=True):
        try:
            run = read_json(path, {})
            if run.get("model_identity") != identity:
                continue
            metrics = read_json(path.parent / "metrics.json", {})
        except (OSError, ValueError):
            continue
        resources = metrics.get("resources", {})
        rows.append({"id": path.parent.name, "protocol_hash": run.get("protocol_hash"),
                     "question_ids": {name: entry.get("question_ids") for name, entry in
                                      run.get("datasets", {}).items()},
                     "tasks": metrics.get("tasks", {}), "resources": resources,
                     "status": metrics.get("status"), "profile": run.get("profile"),
                     "device": run.get("device"), "cpu_threads": run.get("cpu_threads"),
                     "smoke_tokens": metrics.get("smoke", {}).get("tokens"),
                     "seconds": resources.get("elapsed_seconds", metrics.get("elapsed_seconds",
                                                                             metrics.get("wall_time_seconds")))})
    return rows


def benchmark_deltas(left, right):
    rows = []
    seen = set()
    for a in left.get("experiments", []):
        for b in right.get("experiments", []):
            protocol = a.get("protocol_hash")
            if not protocol or protocol != b.get("protocol_hash"):
                continue
            for task, first in a["tasks"].items():
                second = b["tasks"].get(task, {})
                ids = a["question_ids"].get(task)
                if (task, protocol) in seen or not ids or ids != b["question_ids"].get(task):
                    continue
                if (first.get("status") != "complete" or second.get("status") != "complete"
                        or first.get("n") != len(ids) or second.get("n") != len(ids)
                        or not first.get("metric") or first.get("metric") != second.get("metric")
                        or not all(isinstance(t.get("score"), (int, float))
                                   and math.isfinite(t["score"]) for t in (first, second))):
                    continue
                rows.append({"task": task, "a": first["score"], "b": second["score"], "n": len(ids),
                             "delta_pp": 100 * (second["score"] - first["score"]),
                             "profile": a.get("profile"), "protocol": protocol, "metric": first.get("metric"),
                             "runs": [a["id"], b["id"]]})
                seen.add((task, protocol))
    return rows


class CheckpointAdapter:
    """Generic safetensors inventory, with no execution or semantic alignment."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.config = read_json(self.path / "config.json", {})

    def inventory(self):
        tensors = []
        for file in sorted(self.path.glob("*.safetensors")):
            with file.open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                if not 0 < size <= 100_000_000:
                    raise ValueError(f"Invalid safetensors header size in {file.name}")
                header = json.loads(stream.read(size))
            for name, item in header.items():
                if name != "__metadata__":
                    start, end = item["data_offsets"]
                    tensors.append({"name": name, "shape": item["shape"], "dtype": item["dtype"],
                                    "bytes": end - start, "file": file.name})
        if not tensors:
            raise ValueError("No safetensors checkpoint found")
        if len({t["name"] for t in tensors}) != len(tensors):
            raise ValueError("Duplicate tensor names across checkpoint shards")
        return tensors

    def components(self, tensors):
        modules = {}
        for tensor in tensors:
            name = tensor["name"].rsplit(".", 1)[0]
            module = modules.setdefault(name, {"name": name, "kind": "module", "tensors": []})
            module["tensors"].append(tensor)
        return modules

    def capture_locations(self):
        return []

    def load(self):
        raise ValueError("Inference is unavailable for the generic adapter")

    def describe(self, cancelled=lambda: False):
        tensors = self.inventory()
        current_signature = signature(self.path)
        manifest = read_json(self.path / "manifest.json", {})
        checksums = file_checksums(self.path, manifest, cancelled)
        if signature(self.path) != current_signature:
            raise ValueError("Checkpoint files changed during import. Retry when the checkpoint is stable.")
        identity = digest(checksums)
        version_id = digest({"content": identity, "path": str(self.path)})[:16]
        known = {item["path"]: item["sha256"] for item in manifest.get("files", []) if "sha256" in item}
        evaluation_identity = None
        if known and all(known.get(name) == checksum for name, checksum in checksums.items()
                         if name != "component_mappings.json"):
            evaluation_identity = hashlib.sha256(json.dumps(sorted(known.items()), sort_keys=True).encode()).hexdigest()
        tokenizer = {name: checksums[name] for name in TOKENIZER_ASSETS if name in checksums}
        modules = self.components(tensors)
        for module in modules.values():
            module["parameters"] = sum(math.prod(t["shape"]) for t in module["tensors"])
            module["bytes"] = sum(t["bytes"] for t in module["tensors"])
        return {"id": version_id, "identity": identity, "signature": current_signature,
                "evaluation_identity": evaluation_identity, "path": str(self.path), "name": self.path.name,
                "family": self.config.get("model_type", "unknown"), "revision": manifest.get("revision"),
                "repository": manifest.get("repo"), "bytes": sum(p.stat().st_size for p in self.path.iterdir()
                                                               if p.is_file() and p.name != "progress.json"),
                "tensor_bytes": sum(t["bytes"] for t in tensors),
                "parameters": sum(math.prod(t["shape"]) for t in tensors), "modules": list(modules.values()),
                "tokenizer_identity": digest(tokenizer) if tokenizer else None, "vocab_size": self.config.get("vocab_size"),
                "transformation": manifest.get("compression"), "declared_parent": manifest.get("parent"),
                "mappings": read_json(self.path / "component_mappings.json", {}),
                "representation": "safetensors", "capture_locations": self.capture_locations(),
                "capabilities": ["inventory", "matrix"], "unsupported": ["inference and semantic alignment"]}


def _attach_flow(modules):
    """Attach child dataflow to container modules. Call after all children exist:
    in a factorized checkpoint the merged gate/up/down names only appear once
    FactorizedQwen3Adapter.components() has replaced the *.left/*.right leaves."""

    def flow(prefix, suffixes):
        return [prefix + "." + suffix for suffix in suffixes if prefix + "." + suffix in modules]

    for name, module in modules.items():
        if module["kind"] == "transformer block":
            children = flow(name, ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"))
            module["flow"] = children
            if children and modules[children[0]]["kind"] == "normalization":
                module["residual_after"] = [child for child in children
                                            if child in flow(name, ("self_attn", "mlp"))]
        elif module["kind"] == "self-attention":
            module["flow"] = flow(name, ("q_proj", "q_norm", "k_proj", "k_norm", "v_proj", "o_proj"))
        elif module["kind"] == "feed-forward network":
            module["flow"] = flow(name, ("gate_proj", "up_proj", "down_proj"))
    return modules


class Qwen3Adapter(CheckpointAdapter):
    """Logical Qwen3 hierarchy shared by dense and factorized checkpoints."""

    def components(self, tensors):
        modules = super().components(tensors)
        for name, module in modules.items():
            suffix = name.rsplit(".", 1)[-1]
            if suffix.endswith("_proj"):
                module["kind"] = "projection"
                weight = next((t for t in module["tensors"] if t["name"] == name + ".weight"), None)
                if weight and len(weight["shape"]) == 2:
                    module.update(shape=weight["shape"], matrix={"dense": weight["name"]})
            elif "norm" in suffix:
                module["kind"] = "normalization"
        for i in range(self.config.get("num_hidden_layers", 0)):
            block = f"model.layers.{i}"
            for name, kind in ((block, "transformer block"), (block + ".self_attn", "self-attention"),
                               (block + ".mlp", "feed-forward network")):
                modules[name] = {"name": name, "kind": kind, "tensors": [],
                                 "output_width": (self.config.get("hidden_size")
                                                  if kind == "transformer block" else None)}
        return _attach_flow(modules)

    def capture_locations(self):
        groups = {}
        for name, module in self.components(self.inventory()).items():
            if module["kind"] in ("transformer block", "self-attention", "feed-forward network",
                                  "projection", "factorized", "normalization"):
                groups[name] = module["kind"]
        return groups

    def describe(self, cancelled=lambda: False):
        entry = super().describe(cancelled)
        entry.update(capabilities=["inventory", "matrix", "capture", "comparison"],
                     unsupported=["attention probability capture with optimized attention"],
                     representation="dense")
        if self.config.get("tie_word_embeddings"):
            entry["shared_parameters"] = {"lm_head.weight": "model.embed_tokens.weight"}
        return entry

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(
            self.path, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            device_map={"": "cpu"}, low_cpu_mem_usage=True)


class FactorizedQwen3Adapter(Qwen3Adapter):
    def components(self, tensors):
        modules = super().components(tensors)
        data = read_json(self.path / "factorization.json", {})
        for name, spec in data.get("modules", {}).items():
            left = modules.pop(name + ".left", {}).get("tensors", [])
            right = modules.pop(name + ".right", {}).get("tensors", [])
            a = next((t for t in left if t["name"] == name + ".left.weight"), None)
            b = next((t for t in right if t["name"] == name + ".right.weight"), None)
            if not a or not b or len(a["shape"]) != 2 or len(b["shape"]) != 2:
                raise ValueError(f"Incomplete factor pair for {name}")
            rank = spec["rank"]
            if a["shape"][1] != rank or b["shape"][0] != rank:
                raise ValueError(f"Factor rank disagrees with stored shapes for {name}")
            modules[name] = {"name": name, "kind": "factorized", "rank": rank,
                             "shape": [a["shape"][0], b["shape"][1]], "tensors": left + right,
                             "matrix": {"left": a["name"], "right": b["name"]}}
        return _attach_flow(modules)

    def describe(self, cancelled=lambda: False):
        entry = super().describe(cancelled)
        entry["representation"] = "A(Bx) factorized projections with dense remainder"
        return entry

    def load(self):
        from gptcompress.compression.factorization import load_factorized_model

        return load_factorized_model(self.path)


def adapter_for(path):
    path = Path(path).resolve()
    config = read_json(path / "config.json", {})
    if config.get("model_type") == "qwen3" and not config.get("quantization_config"):
        return (FactorizedQwen3Adapter if (path / "factorization.json").exists() else Qwen3Adapter)(path)
    return CheckpointAdapter(path)


def _stored_versions():
    return read_json(STORE / "versions.json", [])


def discover():
    with REGISTRY_LOCK:
        entries = _stored_versions()
        paths = {Path(v["path"]): v for v in entries}
        for root in (WORK / "models/original", WORK / "models/compressed"):
            for config in sorted(root.glob("*/config.json")):
                paths.setdefault(config.parent, {})
        for path, previous in paths.items():
            if not path.is_dir() or (previous.get("signature") is not None):
                continue
            try:
                current = adapter_for(path).describe()
            except (ValueError, OSError):
                continue
            current["name"] = previous.get("name", current["name"])
            current["legacy_ids"] = [previous["id"]] if previous and previous["id"] != current["id"] else []
            entries = [v for v in entries if v["path"] != str(path)]
            entries.append(current)
        save_json(STORE / "versions.json", entries)


def versions():
    with REGISTRY_LOCK:
        entries = _stored_versions()
    for entry in entries:
        entry["experiments"] = experiments(entry.get("evaluation_identity"))
        entry["parent"] = entry.get("declared_parent")
        if not entry["parent"] and (entry.get("transformation") or {}).get("method") == "weight-only-als":
            entry["parent"] = next((other["id"] for other in entries if not other.get("transformation")
                                   and entry.get("revision") and other.get("revision") == entry["revision"]
                                   and other.get("repository") == entry.get("repository")), None)
    return entries


def add_version(path, name=None, cancelled=lambda: False):
    entry = adapter_for(path).describe(cancelled)
    if name:
        entry["name"] = str(name)[:200]
    with REGISTRY_LOCK:
        check_cancelled(cancelled)
        entries = _stored_versions()
        existing = next((v for v in entries if v["id"] == entry["id"]), {})
        entry["legacy_ids"] = existing.get("legacy_ids", [])
        entries = [v for v in entries if v["id"] != entry["id"]]
        entries.append(entry)
        save_json(STORE / "versions.json", entries)
    return {"id": entry["id"], "name": entry["name"]}


def get_version(identifier):
    for entry in versions():
        if identifier == entry["id"] or identifier in entry.get("legacy_ids", []):
            if not Path(entry["path"]).is_dir():
                raise ValueError(f"Checkpoint directory is unavailable for {entry['name']}")
            if signature(Path(entry["path"])) != entry.get("signature"):
                raise ValueError(f"Checkpoint files changed for {entry['name']}. Import the directory again.")
            return entry
    raise ValueError("Model version is unavailable")


def stats(array):
    import numpy as np

    array = np.asarray(array, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {"min": None, "max": None, "mean": None, "std": None, "norm": None, "nonfinite": int(array.size)}
    return {"min": float(finite.min()), "max": float(finite.max()), "mean": float(finite.mean()),
            "std": float(finite.std()), "norm": float(np.linalg.norm(finite)), "nonfinite": int(array.size - finite.size)}


def save_matrix_cache(path, result):
    # Keep recent on-demand results without allowing unbounded disk growth.
    save_json(path, result)
    entries = sorted(path.parent.glob("*.json"), key=lambda file: file.stat().st_mtime, reverse=True)
    for old in entries[128:]:
        old.unlink(missing_ok=True)


class TensorReader:
    """Scoped readers keep mappings open only for the tensors requested by a job."""

    def __init__(self, version, stack):
        self.path = Path(version["path"])
        self.stack = stack
        self.tensors = {t["name"]: t for m in version["modules"] for t in m["tensors"]}
        self.files = {}

    def view(self, name):
        from safetensors import safe_open

        tensor = self.tensors[name]
        filename = tensor["file"]
        if filename not in self.files:
            self.files[filename] = self.stack.enter_context(
                safe_open(self.path / filename, framework="pt", device="cpu"))
        return self.files[filename].get_slice(name)

    def matrix(self, spec, rows, columns):
        if "dense" in spec:
            return self.view(spec["dense"])[rows, columns].float()
        return self.view(spec["left"])[rows, :].float() @ self.view(spec["right"])[:, columns].float()


def matrix_inspect(request, progress, cancelled):
    import numpy as np

    version = get_version(request["version"])
    key = digest({"identity": version["identity"], "request": request, "algorithm": 2})
    cache = STORE / "matrix_cache" / f"{key}.json"
    if cache.exists():
        return read_json(cache)
    progress("Reading bounded tensor slice", .2)
    row = max(0, int(request.get("row_start", 0)))
    col = max(0, int(request.get("col_start", 0)))
    with ExitStack() as stack:
        reader = TensorReader(version, stack)
        name = request.get("tensor")
        if name is not None:
            if name not in reader.tensors:
                raise ValueError("Tensor is unavailable in this checkpoint")
            shape = reader.tensors[name]["shape"]
            if not shape or row >= shape[0] or (len(shape) == 2 and col >= shape[1]):
                raise ValueError("Slice starts outside tensor")
            view = reader.view(name)
            if len(shape) == 2:
                data = view[row:row + 512, col:col + 512].float().numpy()
            elif len(shape) == 1:
                data = view[row:row + 512].float().numpy().reshape(1, -1)
                col = 0
            else:
                raise ValueError("Matrix inspector supports vectors and matrices")
        else:
            name = request["module"]
            module = next((m for m in version["modules"] if m["name"] == name and m.get("matrix")), None)
            if not module:
                raise ValueError("Logical matrix is unavailable")
            shape = module["shape"]
            if row >= shape[0] or col >= shape[1]:
                raise ValueError("Slice starts outside tensor")
            data = reader.matrix(module["matrix"], slice(row, row + 512), slice(col, col + 512)).numpy()
            name += " · logical weight"
    if not data.size:
        raise ValueError("Slice starts outside tensor")
    check_cancelled(cancelled)
    finite = data[np.isfinite(data)].astype(np.float64)
    counts, edges = np.histogram(finite, bins=32) if finite.size else (np.zeros(32), np.arange(33))
    step = max(1, math.ceil(max(data.shape) / 64))
    heatmap = data[::step, ::step]
    valid = np.where(np.isfinite(data), data, 0).astype(np.float64)
    result = {"tensor": name, "shape": shape, "slice_shape": list(data.shape), "stats": stats(data),
              "row_start": row, "col_start": col, "heatmap_step": step,
              "sampling": f"FP32 slice from ({row}, {col}), at most 512 × 512. Heatmap shows every {step}th cell. Norms exclude nonfinite cells.",
              "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
              "heatmap": [[float(x) if np.isfinite(x) else None for x in line] for line in heatmap],
              "row_norms": np.linalg.norm(valid, axis=1).tolist(),
              "column_norms": np.linalg.norm(valid, axis=0).tolist(), "spectrum": []}
    if request.get("spectrum"):
        progress("Computing singular values of the bounded slice", .6)
        if not np.isfinite(data).all():
            raise ValueError("Spectrum requires finite data")
        result["spectrum"] = np.linalg.svd(data, compute_uv=False).tolist()
    check_cancelled(cancelled)
    save_matrix_cache(cache, result)
    return result


def component_mapping(left, right, name):
    forward = right.get("mappings", {}).get("comparisons", {})
    reverse = left.get("mappings", {}).get("comparisons", {})
    for version in (left["id"], left["identity"], *left.get("legacy_ids", [])):
        if name in forward.get(version, {}):
            return forward[version][name]
    for version in (right["id"], right["identity"], *right.get("legacy_ids", [])):
        for other_name, mapping in reverse.get(version, {}).items():
            if name == mapping.get("b_name", other_name):
                return {**mapping, "a_indices": mapping.get("b_indices"), "b_indices": mapping.get("a_indices"),
                        "a_name": name, "b_name": other_name}
    return None


def corresponding_name(left, right, name):
    mapping = component_mapping(left, right, name)
    if mapping:
        return mapping.get("b_name", name), mapping.get("status", "matched")
    if "comparison" not in left["capabilities"] or "comparison" not in right["capabilities"]:
        return name, "semantic alignment unavailable"
    if (left.get("family"), left.get("repository"), left.get("revision")) != (
            right.get("family"), right.get("repository"), right.get("revision")) or (
            left["identity"] != right["identity"] and not left.get("revision")):
        return name, "component correspondence unavailable"
    structural = {"pruned", "pruning", "structural_pruning"}
    for version in (left, right):
        method = (version.get("transformation") or {}).get("method")
        if method in structural or method not in (None, "weight-only-als"):
            return name, "explicit transformation mapping required"
    return name, "matched"


def matrix_compare(request, progress, cancelled):
    """Compare full logical matrices with bounded row/column tiles."""
    import numpy as np
    import torch

    a, b = get_version(request["a"]), get_version(request["b"])
    name = request["module"]
    other, status = corresponding_name(a, b, name)
    if status != "matched":
        return {"status": status, "module": name}
    modules = [next((m for m in v["modules"] if m["name"] == key), None) for v, key in ((a, name), (b, other))]
    if not all(modules):
        return {"status": "unmatched component", "module": name}
    if not all(module.get("matrix") for module in modules):
        return {"status": "logical matrix unavailable", "module": name}
    if modules[0]["shape"] != modules[1]["shape"] or component_mapping(a, b, name):
        return {"status": "incompatible logical matrices for native-coordinate comparison", "module": name}
    key = digest([a["identity"], b["identity"], name, "full-frobenius-and-factor-energy-v2"])
    cache = STORE / "matrix_cache" / f"{key}.json"
    if cache.exists():
        return read_json(cache)
    torch.set_num_threads(4)
    rows, cols = modules[0]["shape"]
    squared_a = squared_b = squared_diff = 0.0
    preview = {}
    specs = [module["matrix"] for module in modules]
    with ExitStack() as stack:
        readers = [TensorReader(v, stack) for v in (a, b)]
        for i in range(0, rows, 256):
            for j in range(0, cols, 512):
                check_cancelled(cancelled)
                bounds = slice(i, i + 256), slice(j, j + 512)
                x, y = (reader.matrix(spec, *bounds).double() for reader, spec in zip(readers, specs))
                if i == 0 and j == 0:
                    step = max(1, math.ceil(max(x.shape) / 64))
                    delta = (y[::step, ::step] - x[::step, ::step]).numpy()
                    preview = {"heatmap": np.where(np.isfinite(delta), delta, 0).tolist(),
                               "row_start": 0, "col_start": 0, "heatmap_step": step, "extent": list(x.shape)}
                squared_a += float(x.square().sum())
                squared_b += float(y.square().sum())
                squared_diff += float((x - y).square().sum())
            progress("Comparing logical weights in bounded tiles", .1 + .62 * min(1, (i + 256) / rows))
        if not all(map(math.isfinite, (squared_a, squared_b, squared_diff))):
            return {"status": "nonfinite matrix values", "module": name}
        result = {"status": "matched", "module": name, "shape": [rows, cols],
                  "relative_error": math.sqrt(squared_diff / squared_a) if squared_a else (0.0 if not squared_diff else None),
                  "cosine": ((squared_a + squared_b - squared_diff) / (2 * math.sqrt(squared_a * squared_b))
                             if squared_a and squared_b else None),
                  "basis": "Full logical matrices, same coordinates. Factor tiles use FP32 A×B with FP64 error reductions.",
                  "preview": preview}
        for source, compressed in ((0, 1), (1, 0)):
            if "dense" in specs[source] and "left" in specs[compressed]:
                energy = (squared_a, squared_b)[source]
                result["source_version"] = (a, b)[source]["name"]
                result["relative_squared_reconstruction_error"] = squared_diff / energy if energy else (
                    0.0 if not squared_diff else None)
                # Thin SVD of A identifies its actual span even when the factors are rank deficient.
                view = readers[compressed].view(specs[compressed]["left"])
                if math.prod(view.get_shape()) > 8_000_000:
                    result["retained_spectral_energy_status"] = "Factor exceeds span-analysis budget"
                    break
                left = view[:, :].float()
                basis, singular, _ = torch.linalg.svd(left, full_matrices=False)
                keep = singular > singular[0] * max(left.shape) * torch.finfo(left.dtype).eps
                basis = basis[:, keep].contiguous().T
                captured = 0.0
                for column in range(0, cols, 64):
                    check_cancelled(cancelled)
                    tile = readers[source].view(specs[source]["dense"])[:, column:column + 64].float()
                    captured += float((basis @ tile).double().square().sum())
                    progress("Measuring source energy in the factor output span", .73 + .25 * min(1, (column + 64) / cols))
                result["retained_spectral_energy"] = captured / energy if energy else None
                result["energy_basis"] = "Fraction of dense source energy in the stored A column span; thin SVD of A in FP32. Optimal truncated SVD can capture more."
                break
    check_cancelled(cancelled)
    save_matrix_cache(cache, result)
    return result
