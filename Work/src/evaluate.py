"""Run pinned local GSM8K and IFEval protocols with lm-evaluation-harness."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from assets import MODEL, REVISION, ROOT, dump, sha256
from gptcompress.report import generate_report

BASELINE = ROOT / "models/original/Qwen3-4B"
SEED = 42
CONTEXT = 2048
PILOT_CONTEXT = 512
TASKS = {
    "gsm8k": {"output_tokens": 192, "metric": "exact_match,flexible-extract"},
    "ifeval": {"output_tokens": 256, "metric": "prompt_level_strict_acc,none"},
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def verified_asset(path):
    asset = read(path / "manifest.json")
    if asset["status"] != "complete":
        raise RuntimeError(f"{asset['repo']} asset incomplete: {asset.get('error')}")
    for record in asset["files"]:
        file_path = path / record["path"]
        stat = file_path.stat()
        if stat.st_size != record["bytes"] or (
            stat.st_mtime_ns != record.get("mtime_ns") and sha256(file_path) != record["sha256"]
        ):
            raise ValueError(f"Asset differs from its manifest: {file_path}")
    return asset


def task_setup(names, profile):
    from datasets import load_dataset
    from lm_eval.tasks import __file__ as tasks_file
    from lm_eval.tasks._yaml_loader import load_yaml

    configs, datasets, samples = {}, {}, {}
    for name in names:
        path = ROOT / "datasets/benchmarks" / name
        asset = verified_asset(path)
        if name == "gsm8k":
            builder, split = "parquet", "test"
            files = {key: str(next((path / "main").glob(f"{key}-*.parquet")))
                     for key in ("train", "test")}
        else:
            builder, split = "json", "train"
            files = {split: str(next(path.glob("*.jsonl")))}
        source = Path(tasks_file).parent / name / f"{name}.yaml"
        config = load_yaml(source)
        config.update(dataset_path=builder, dataset_name=None, dataset_kwargs={"data_files": files})
        cap = TASKS[name]["output_tokens"]
        if profile == "pilot":
            cap = 24 if name == "gsm8k" else 48
            if name == "gsm8k":
                config["doc_to_text"] = "Question: {{question}}\nReply with only the final number."
                config["num_fewshot"] = 0
        config["generation_kwargs"]["max_gen_toks"] = cap
        rows = load_dataset(builder, data_files=files, split=split)
        order = random.Random(SEED).sample(range(len(rows)), len(rows))
        dev, heldout = order[:10], order[10:110]
        assert len(dev) == 10 and len(heldout) == 100 and not set(dev) & set(heldout)
        samples[name] = heldout if profile == "heldout" else dev
        datasets[name] = {
            "repo": asset["repo"], "revision": asset["revision"], "split": split,
            "files": {item["path"]: item["sha256"] for item in asset["files"]},
            "task_yaml_sha256": sha256(source), "question_ids": samples[name],
            "fewshot": config["num_fewshot"], "prompt_renderer": config["doc_to_text"],
            "generation_kwargs": config["generation_kwargs"],
        }
        configs[name] = config
    return configs, datasets, samples


def smoke(backend):
    import torch

    tokenizer = backend.tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Answer in one word. What color is a clear daytime sky?"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    started = time.perf_counter()
    with torch.inference_mode():
        result = backend.model.generate(
            **inputs.to(backend.device), do_sample=False, max_new_tokens=8,
            temperature=None, top_p=None, top_k=None,
        )
    tokens = result[0, inputs.input_ids.shape[1]:]
    return {"prompt": prompt, "response": tokenizer.decode(tokens, skip_special_tokens=True),
            "tokens": len(tokens), "seconds": time.perf_counter() - started}


def configure_local_caches():
    os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk-data"))
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"


def validate_profiles():
    configure_local_caches()
    from lm_eval import simple_evaluate
    from lm_eval.models.dummy import DummyLM
    from lm_eval.tasks import TaskManager

    configs, _, dev = task_setup(TASKS, "dev")
    _, _, heldout = task_setup(TASKS, "heldout")
    for name in TASKS:
        assert len(set(dev[name])) == 10 and len(set(heldout[name])) == 100
        assert not set(dev[name]).intersection(heldout[name])
    results = simple_evaluate(model=DummyLM(), tasks=list(configs.values()),
                              task_manager=TaskManager(include_defaults=False),
                              samples={name: ids[:1] for name, ids in dev.items()},
                              bootstrap_iters=0)
    for name, spec in TASKS.items():
        assert spec["metric"] in results["results"][name]
        assert {row["doc_id"] for row in results["samples"][name]} == {dev[name][0]}
    print("Both local harness tasks scored dummy responses. Development and held-out IDs are disjoint.")
    return 0


def run(args):
    configure_local_caches()
    from datasets import disable_progress_bars
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import psutil
    import torch

    disable_progress_bars()
    torch.set_num_threads(args.threads)
    model_path = args.model.resolve()
    asset = verified_asset(model_path)
    is_original = model_path == BASELINE
    if is_original and (asset["repo"], asset["revision"]) != (MODEL, REVISION):
        raise ValueError("The baseline path must contain the pinned Qwen3-4B checkpoint")
    if not is_original and not isinstance(asset.get("compression"), dict):
        raise ValueError("A variant must document its compression configuration in the model manifest")
    if args.parent_run:
        read(ROOT / "experiments" / args.parent_run / "manifest.json")
    configs, datasets, samples = task_setup(args.tasks, args.profile) if not args.smoke else ({}, {}, {})
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    versions = {key: importlib.metadata.version(key) for key in
                ("torch", "transformers", "accelerate", "safetensors", "lm-eval", "datasets")}
    protocol = {
        "profile": "smoke" if args.smoke else args.profile, "datasets": datasets,
        "seed": SEED, "batch_size": 1 if args.smoke else args.batch_size,
        "context_tokens": PILOT_CONTEXT if args.profile == "pilot" and not args.smoke else CONTEXT,
        "enable_thinking": False, "chat_template_sha256": digest(tokenizer.chat_template),
        "decode": "greedy", "versions": versions, "linear_kernel": args.linear_kernel,
    }
    if "ifeval" in configs:
        punkt = Path(os.environ["NLTK_DATA"]) / "tokenizers/punkt_tab/english"
        assets = {p.name: sha256(p) for p in punkt.glob("*") if p.is_file()}
        if not assets:
            raise RuntimeError("Install NLTK punkt_tab into .nltk-data before running IFEval")
        protocol["punkt_tab_english"] = assets
    protocol_hash = digest(protocol)
    identity = digest(sorted((item["path"], item["sha256"]) for item in asset["files"]))
    now = datetime.now(timezone.utc)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    directory = ROOT / "experiments" / run_id
    directory.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(directory / "execution.log", encoding="utf-8"), logging.StreamHandler(),
    ], force=True)
    log = logging.getLogger("gptcompress")
    log.info("Run %s", run_id)
    manifest = {
        **protocol, "run_id": run_id, "created_at": now.isoformat(),
        "model_id": asset["repo"], "model_revision": asset["revision"], "model_identity": identity,
        "model_role": "original" if is_original else "variant",
        "model_size_bytes": sum(item["bytes"] for item in asset["files"] if item["path"].endswith(".safetensors")),
        "asset_size_bytes": sum(item["bytes"] for item in asset["files"]),
        "protocol_hash": protocol_hash, "parent_run_id": args.parent_run,
        "compression": asset.get("compression"), "device": "cpu", "dtype": "bfloat16",
        "cpu_threads": args.threads, "python": sys.version,
        "available_ram_before_load_bytes": psutil.virtual_memory().available,
        "source_hash": digest({str(p.relative_to(ROOT)): sha256(p)
                               for p in (ROOT / "src").rglob("*.py")}),
    }
    for path in sorted(directory.parent.glob("*/manifest.json")):
        previous = read(path)
        same_protocol = previous.get("protocol_hash") == protocol_hash
        baseline = (previous.get("model_role") == "original" or (
            "model_role" not in previous and previous.get("compression") is None
            and (previous.get("model_id"), previous.get("model_revision")) == (MODEL, REVISION)
        ))
        if same_protocol and baseline:
            metrics_path = path.parent / "metrics.json"
            if metrics_path.is_file() and read(metrics_path).get("status") == "complete":
                if identity == previous.get("model_identity"):
                    log.info("Identical completed baseline available at %s", path.parent)
                    if not args.repeat:
                        dump(directory / "manifest.json", {**manifest, "baseline_run_id": previous["run_id"]})
                        dump(directory / "metrics.json", {"status": "reused", "tasks": {}})
                        directory.joinpath("predictions.jsonl").touch()
                        generate_report(ROOT)
                        return 0
                manifest["baseline_run_id"] = previous["run_id"]
                break
    tasks = {name: {"status": "pending", "n": 0} for name in configs}
    metrics = {"status": "running", "tasks": tasks}
    dump(directory / "manifest.json", manifest)
    dump(directory / "metrics.json", metrics)
    predictions = directory / "predictions.jsonl"
    predictions.touch()
    started = time.perf_counter()
    try:
        log.info("Loading BF16 checkpoint on CPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            device_map={"": "cpu"}, low_cpu_mem_usage=True, trust_remote_code=False,
        ).eval()
        if args.linear_kernel == "fp32-tiled":
            from gptcompress.cpu_linear import use_tiled_fp32_linear

            log.info("Enabled 64 MiB FP32 weight tiles for %s BF16 projections",
                     use_tiled_fp32_linear(model))
        backend = HFLM(pretrained=model, tokenizer=tokenizer, device="cpu",
                       batch_size=protocol["batch_size"], max_length=protocol["context_tokens"],
                       enable_thinking=False)
        metrics["model_load_seconds"] = time.perf_counter() - started
        log.info("Model load complete in %.2f seconds", metrics["model_load_seconds"])
        dump(directory / "metrics.json", metrics)
        if args.smoke:
            metrics["smoke"] = smoke(backend)
            log.info("Smoke result %s", metrics["smoke"])
            append(predictions, {"task": "smoke", "doc_id": "sky", **metrics["smoke"]})
        for name, config in configs.items():
            task_started = time.perf_counter()
            try:
                cache = directory / name
                if args.resume_from:
                    source = ROOT / "experiments" / args.resume_from
                    previous = read(source / "manifest.json")
                    if previous.get("protocol_hash") != protocol_hash or previous.get("model_identity") != identity:
                        raise ValueError("Resume requires an identical model and protocol")
                    from shutil import copyfile

                    for file in source.glob(f"{name}_rank*.db"):
                        copyfile(file, directory / file.name)
                result = simple_evaluate(
                    model=backend, tasks=[config], samples={name: samples[name]},
                    apply_chat_template=True, random_seed=SEED, numpy_random_seed=SEED,
                    torch_random_seed=SEED, fewshot_random_seed=SEED, bootstrap_iters=1000,
                    use_cache=str(cache), task_manager=TaskManager(include_defaults=False),
                )
                metric = TASKS[name]["metric"]
                key, filter_name = metric.split(",")
                scores = dict(result["results"][name])
                rows = result["samples"][name]
                for row in rows:
                    prompt = row["arguments"][0][0]
                    response = row["resps"][0][0]
                    cap = config["generation_kwargs"]["max_gen_toks"]
                    append(predictions, {
                        "task": name, "doc_id": row["doc_id"], "prompt": prompt,
                        "response": response, "target": row["target"] if name == "gsm8k" else row["doc"],
                        "score": {key: row[key] for key in row["metrics"]}, "harness": row,
                        "input_truncated": len(tokenizer.encode(prompt, add_special_tokens=False))
                        > protocol["context_tokens"] - cap,
                        "output_near_cap": len(tokenizer.encode(response, add_special_tokens=False)) >= cap - 1,
                    })
                tasks[name] = {
                    "status": "complete", "n": len({row["doc_id"] for row in rows}),
                    "metric": metric, "score": scores[metric],
                    "stderr": scores.get(f"{key}_stderr,{filter_name}"), "all_metrics": scores,
                    "elapsed_seconds": time.perf_counter() - task_started,
                }
                log.info("%s %s", name, tasks[name])
            except Exception as exc:
                tasks[name] = {"status": "failed", "n": 0, "error": str(exc)}
                log.exception("%s failed", name)
            dump(directory / "metrics.json", metrics)
        metrics["status"] = "complete" if all(t["status"] == "complete" for t in tasks.values()) else "incomplete"
    except BaseException as exc:
        metrics.update(status="failed", error=str(exc))
        log.exception("Run interrupted or failed")
    finally:
        memory = psutil.Process().memory_info()
        metrics["resources"] = {
            "peak_rss_bytes": getattr(memory, "peak_wset", None),
            "peak_vram_bytes": None, "elapsed_seconds": time.perf_counter() - started,
            "method": "Windows process peak working set, elapsed wall time; CPU-only inference",
        }
        dump(directory / "metrics.json", metrics)
        log.info("Report %s", generate_report(ROOT))
    return metrics["status"] != "complete"


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=BASELINE)
    parser.add_argument("--parent-run")
    parser.add_argument("--profile", choices=("pilot", "dev", "heldout"), default="dev")
    parser.add_argument("--tasks", nargs="+", choices=tuple(TASKS), default=list(TASKS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, choices=range(1, 11), default=None)
    parser.add_argument("--linear-kernel", choices=("native", "fp32-tiled"), default="native")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--resume-from", help="Reuse cached generations from a stopped run with the same protocol")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--report", action="store_true")
    action.add_argument("--validate-profiles", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(generate_report(ROOT))
        return 0
    if args.validate_profiles:
        return validate_profiles()
    if args.batch_size is None:
        args.batch_size = 10 if args.profile == "pilot" else 1
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
