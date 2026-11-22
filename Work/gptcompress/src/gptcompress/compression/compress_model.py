"""Create standalone fixed-rank ALS checkpoints or inspect representative matrices."""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gptcompress.evaluation.prepare_assets import ROOT, dump, sha256
from gptcompress.compression.factorization import factorize, relative_error

ORIGINAL = ROOT / "models/original/Qwen3-4B"
PROJECTIONS = {"mlp": ("gate_proj", "up_proj", "down_proj"),
               "attention": ("q_proj", "k_proj", "v_proj", "o_proj")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--target", choices=["mlp", "attention", "combined"], default="combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-factors", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    settings = {key: value for key, value in vars(args).items()
                if key not in ("output", "reuse_factors")}
    settings.update(method="weight-only-als", solver="gelsy", compute_dtype="float32",
                    storage_dtype="bfloat16", regularization=0)
    index = json.loads((ORIGINAL / "model.safetensors.index.json").read_text())
    projections = sum(PROJECTIONS.values(), ()) if args.target == "combined" else PROJECTIONS[args.target]
    modules, records, weight_map = {}, [], {}
    cached = {}
    for directory in args.reuse_factors:
        from evaluate import verified_asset

        verified_asset(directory)
        specification = json.loads((directory / "factorization.json").read_text())
        for key in ("rank", "seed", "tolerance", "max_iterations", "solver", "compute_dtype", "storage_dtype"):
            if specification["settings"][key] != settings[key]:
                raise ValueError(f"Incompatible factor cache setting {key}")
        cache_index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
        cache_records = json.loads((directory / "reconstruction.json").read_text())
        for record in cache_records:
            if "rank" in record:
                cached[record["name"]] = (directory, cache_index, record)
    started = time.perf_counter()
    dump(args.output / "progress.json", {"status": "running", "settings": settings})
    try:
        for name, shard in index["weight_map"].items():
            module = name.removesuffix(".weight")
            selected = name.endswith(".weight") and module.split(".")[-1] in projections
            if args.probe and (not selected or module.split(".")[2] not in ("0", "18", "35")
                               or module.split(".")[-1] not in ("k_proj", "down_proj")):
                continue
            with safe_open(ORIGINAL / shard, framework="pt", device="cpu") as handle:
                weight = handle.get_tensor(name)
                eligible = selected and args.rank * sum(weight.shape) < weight.numel()
                if selected and not eligible:
                    records.append({"name": name, "status": "skipped", "reason": "no storage saving"})
                if eligible:
                    if name in cached:
                        directory, cache_index, old_record = cached[name]
                        with safe_open(directory / cache_index[module + ".left.weight"], framework="pt") as cached_file:
                            left = cached_file.get_tensor(module + ".left.weight")
                            right = cached_file.get_tensor(module + ".right.weight")
                        record = {**old_record, "reused_from": str(directory.resolve())}
                    else:
                        left, right, record = factorize(weight, args.rank, args.seed,
                                                       args.tolerance, args.max_iterations)
                    left, right = left.bfloat16(), right.bfloat16()
                    record.update(name=name, shape=list(weight.shape), rank=args.rank,
                                  stored_relative_squared_error=relative_error(weight, left, right),
                                  original_parameters=weight.numel(),
                                  factor_parameters=left.numel() + right.numel())
                    if args.probe:
                        singular_values = torch.linalg.svdvals(weight.float())
                        record["exact_svd_relative_squared_error"] = (
                            singular_values[args.rank:].double().square().sum()
                            / singular_values.double().square().sum()).item()
                    tensors = {module + ".left.weight": left, module + ".right.weight": right}
                    modules[module] = {"rank": args.rank}
                    records.append(record)
                    print(json.dumps(record), flush=True)
                else:
                    tensors = {name: weight}
                if not args.probe:
                    filename = f"weights-{len(weight_map):05d}.safetensors"
                    save_file(tensors, args.output / filename, metadata={"format": "pt"})
                    weight_map.update({key: filename for key in tensors})
            memory = psutil.Process().memory_info()
            dump(args.output / "progress.json", {"status": "running", "settings": settings,
                 "matrices": records, "seconds": time.perf_counter() - started,
                 "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss)})
        if not args.probe:
            for filename in ("config.json", "generation_config.json", "tokenizer.json",
                             "tokenizer_config.json", "vocab.json", "merges.txt", "LICENSE"):
                if (ORIGINAL / filename).exists():
                    shutil.copyfile(ORIGINAL / filename, args.output / filename)
            dump(args.output / "factorization.json", {"settings": settings, "modules": modules})
            dump(args.output / "reconstruction.json", records)
            dump(args.output / "model.safetensors.index.json", {"metadata": {}, "weight_map": weight_map})
            files = [{"path": path.name, "bytes": path.stat().st_size,
                      "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256(path)}
                     for path in sorted(args.output.iterdir()) if path.name != "progress.json"]
            source = json.loads((ORIGINAL / "manifest.json").read_text())
            dump(args.output / "manifest.json", {"status": "complete", "repo": source["repo"],
                 "revision": source["revision"], "compression": settings, "files": files})
        dump(args.output / "progress.json", {"status": "complete", "settings": settings,
             "matrices": records, "seconds": time.perf_counter() - started,
             "peak_working_set_bytes": getattr(memory, "peak_wset", memory.rss)})
    except Exception as error:
        dump(args.output / "progress.json", {"status": "failed", "settings": settings,
             "matrices": records, "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
