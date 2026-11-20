"""Download immutable assets from Hugging Face and keep a local provenance manifest."""

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs"
MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
HUB = "https://huggingface.co"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metadata(kind, repo, revision):
    with urllib.request.urlopen(
        f"{HUB}/api/{kind}/{repo}/revision/{revision}?blobs=true", timeout=40
    ) as response:
        return json.load(response)


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.stat().st_size != expected:
            raise ValueError(f"Existing file has unexpected size: {path}")
        return
    partial = path.with_suffix(path.suffix + ".partial")
    if not partial.exists() or partial.stat().st_size < expected:
        result = subprocess.run(
            ["curl.exe", "-fsSL", "--retry", "2", "--connect-timeout", "20",
             "-C", "-", "-o", str(partial), url], capture_output=True, text=True,
        )
        if result.returncode:
            raise OSError(f"{url}: {result.stderr.strip() or f'curl exit {result.returncode}'}")
    if partial.stat().st_size != expected:
        raise ValueError(f"Downloaded file has unexpected size: {partial}")
    partial.replace(path)


def fetch(kind, repo, revision, folder, pattern, required=()):
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        revision = previous.get("revision") or revision
    else:
        previous = {}
    manifest = {
        "repo": repo, "revision": previous.get("revision"),
        "status": "incomplete", "files": previous.get("files", []),
    }
    try:
        info = metadata(kind, repo, revision)
        manifest.update(revision=info["sha"], gated=info.get("gated"),
                        license=(info.get("cardData") or {}).get("license"), files=[])
        files = [item for item in info["siblings"] if re.search(pattern, item["rfilename"])]
        if not files:
            raise ValueError(f"No files matched {pattern}")
        prefix = "/datasets" if kind == "datasets" else ""
        missing = 0
        for item in files:
            path = folder / item["rfilename"]
            if not path.exists():
                partial = path.with_suffix(path.suffix + ".partial")
                missing += item["size"] - (partial.stat().st_size if partial.exists() else 0)
        free = shutil.disk_usage(ROOT).free
        if missing + 128 * 1024**2 > free:
            raise OSError(f"Need {missing:,} download bytes plus 128 MiB reserve, have {free:,}")
        verified = {item["path"]: item for item in previous.get("files", [])}
        for item in sorted(files, key=lambda entry: entry["size"]):
            name = item["rfilename"]
            path = folder / name
            url = f"{HUB}{prefix}/{repo}/resolve/{info['sha']}/{urllib.parse.quote(name)}"
            download(url, path, item["size"])
            cached = verified.get(name, {})
            digest = (cached["sha256"] if cached.get("mtime_ns") == path.stat().st_mtime_ns
                      and cached.get("bytes") == item["size"] else sha256(path))
            upstream = (item.get("lfs") or {}).get("sha256")
            if upstream and digest != upstream:
                raise ValueError(f"SHA-256 mismatch for {path}")
            manifest["files"].append(dict(path=name, bytes=item["size"], sha256=digest,
                                          mtime_ns=path.stat().st_mtime_ns))
            dump(manifest_path, manifest)
        for prefix in required:
            if not any(item["path"].startswith(prefix) for item in manifest["files"]):
                raise ValueError(f"Missing expected data for {prefix}")
        manifest["status"] = "complete"
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        manifest["error"] = str(exc)
    dump(manifest_path, manifest)
    print(f"{repo} {manifest['status']} {manifest.get('error', '')}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("assets", choices=["model", "data", "evalplus"])
    parser.add_argument("tasks", nargs="*")
    args = parser.parse_args()
    if args.assets == "model":
        result = fetch("models", MODEL, REVISION, ROOT / "models/original/Qwen3-4B",
                       r"^(LICENSE|README\.md|.*\.(json|txt|safetensors))$",
                       ("config.json", "tokenizer.json", "model.safetensors.index.json",
                        "model-00001-", "model-00002-", "model-00003-"))
        return result["status"] != "complete"
    if args.assets == "evalplus":
        return download_evalplus()
    suite = json.loads((CONFIG / "benchmarks.json").read_text(encoding="utf-8"))
    selected = set(args.tasks)
    unknown = selected - suite.keys()
    if unknown:
        parser.error(f"Unknown benchmarks: {sorted(unknown)}")
    results = []
    for key, spec in suite.items():
        if not selected or key in selected:
            manifest = fetch("datasets", spec["repo"], spec.get("revision", "main"),
                             ROOT / "datasets/benchmarks" / key, spec["files"],
                             spec.get("required", ()))
            manifest["benchmark"] = spec
            dump(ROOT / "datasets/benchmarks" / key / "manifest.json", manifest)
            results.append(manifest["status"])
    return any(status != "complete" for status in results)


def download_evalplus():
    """Stage the dataset files expected by the pinned EvalPlus release."""
    sources = (
        ("humaneval_plus", "humanevalplus_release", "v0.1.10", "HumanEvalPlus"),
        ("mbpp_plus", "mbppplus_release", "v0.2.0", "MbppPlus"),
    )
    failed = False
    for benchmark, repo, tag, name in sources:
        folder = ROOT / "datasets/benchmarks" / benchmark / "evalplus"
        source = f"https://api.github.com/repos/evalplus/{repo}/releases/tags/{tag}"
        manifest = {"release": source, "tag": tag, "status": "incomplete"}
        try:
            with urllib.request.urlopen(source, timeout=30) as response:
                release = json.load(response)
            item = next(item for item in release["assets"] if item["name"] == f"{name}.jsonl.gz")
            compressed = folder / item["name"]
            download(item["browser_download_url"], compressed, item["size"])
            checksum = sha256(compressed)
            if item.get("digest") and item["digest"] != f"sha256:{checksum}":
                raise ValueError(f"SHA-256 mismatch for {compressed}")
            path = folder / f"{name}.jsonl"
            pending = path.with_suffix(".jsonl.tmp")
            with gzip.open(compressed, "rb") as original, pending.open("wb") as output:
                shutil.copyfileobj(original, output)
            pending.replace(path)
            manifest.update(status="complete", upstream_asset_id=item["id"],
                            archive_sha256=checksum, bytes=path.stat().st_size,
                            path=path.name, sha256=sha256(path))
        except (OSError, ValueError, StopIteration, KeyError) as exc:
            manifest["error"] = str(exc)
            failed = True
        dump(folder / "manifest.json", manifest)
        print(f"{benchmark} EvalPlus {manifest['status']} {manifest.get('error', '')}", flush=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
