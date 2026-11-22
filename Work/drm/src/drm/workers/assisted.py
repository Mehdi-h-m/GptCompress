"""Assisted execution: notebook task bundles, import validation, recovery.

The coordinator prepares a self-contained bundle (manifest, staged verified
inputs, a portable task implementation, instructions), the user runs it on
Kaggle, and import accepts **only the declared result files** after
recomputing checksums. Sessions that die mid-work leave item-level
checkpoints in the durable store; the next bundle re-dispatches only the
missing items and carries the recovered records forward, so the job produces
exactly one accepted output regardless of how many sessions it needed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from drm.catalog.errors import CatalogError
from drm.providers import registry
from drm.providers.base import AdapterError, sha256_hex
from drm.store import repos
from drm.workers import runner

TERMINAL_JOB_STATES = repos.TERMINAL_JOB_STATES


def checkpoint_prefix(job_id: str) -> str:
    return f"checkpoints/{job_id}/"


def _load_checkpoints(storage, job_id: str, items: list) -> tuple[list[str], list[dict]]:
    """Return (completed item ids, recovered records) from the durable store."""
    completed: list[str] = []
    records: list[dict] = []
    try:
        for locator in storage.list(checkpoint_prefix(job_id)):
            record = json.loads(storage.read(locator))
            item = record.get("item")
            if item in items:
                completed.append(item)
                records.append(record)
    except AdapterError:
        return [], []  # durable store unreachable: re-run everything
    return completed, records


def prepare_bundle(ctx, job_row, attempt_row, spec: dict) -> dict:
    """Build the notebook bundle for one attempt and park the run in awaiting_user.

    For ``checkpoint_scan@1`` the durable store is consulted first: completed
    items are dropped from the bundle's params and their records ride along in
    the manifest, so the final summary always covers every item.
    """
    from drm.providers.kaggle import KaggleCompute

    job_id, attempt_id = job_row["id"], attempt_row["id"]
    workdir = ctx.runs_dir / job_id / attempt_id
    bundle_dir = workdir / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    staged = runner._stage_inputs(ctx, bundle_dir, spec)

    params = dict(spec.get("params", {}))
    recovered: list[dict] = []
    if spec["task_type"] == "checkpoint_scan@1" and isinstance(params.get("items"), list):
        with ctx.db_lock:
            store_row = repos.get_provider(ctx.conn, spec.get("output_store", ""))
        if store_row is not None:
            storage = registry.storage_for_row(store_row)
            items = params["items"]
            completed, recovered = _load_checkpoints(storage, job_id, items)
            params["items"] = [item for item in items if item not in completed]

    with ctx.db_lock:
        store_row = repos.get_provider(ctx.conn, spec.get("output_store", ""))
    durable_repo = None
    if store_row is not None and store_row["adapter"] == "hf":
        durable_repo = json.loads(store_row["config_json"]).get("repo_id")

    manifest = {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "task_type": spec["task_type"],
        "task_version": spec.get("task_version", 1),
        "numerical_profile": spec.get("numerical_profile"),
        "params": params,
        "inputs": [
            {"name": item["name"], "sha256": item["sha256"], "byte_count": item["byte_count"]}
            for item in staged
        ],
        "checkpoint_prefix": checkpoint_prefix(job_id),
        "durable_store": {"repo_id": durable_repo},
        "recovered": recovered,
    }
    manifest_path = bundle_dir / "task_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (bundle_dir / "task_impl.py").write_text(TASK_IMPL_SOURCE, encoding="utf-8")
    (bundle_dir / "notebook.ipynb").write_text(
        json.dumps(_make_notebook(), indent=1), encoding="utf-8"
    )

    expected_outputs = _expected_outputs(spec["task_type"])
    bundle = {
        "bundle_dir": str(bundle_dir),
        "bundle_zip": str(_zip_bundle(bundle_dir, workdir / "bundle.zip")),
        "attempt_id": attempt_id,
        "expected_outputs": expected_outputs,
        "recovered_items": [r.get("item") for r in recovered],
    }
    instructions = KaggleCompute().notebook_instructions(bundle)
    (workdir / "instructions.txt").write_text(instructions, encoding="utf-8")

    result = {
        "awaiting_user": True,
        "attempt_id": attempt_id,
        "bundle": bundle["bundle_zip"],
        "bundle_dir": bundle["bundle_dir"],
        "instructions": str(workdir / "instructions.txt"),
        "outputs_expected": expected_outputs,
        "recovered_items": bundle["recovered_items"],
        "items_in_bundle": params.get("items"),
    }
    with ctx.db_lock:
        repos.set_job_state(ctx.conn, job_id, "awaiting_user")
        repos.set_attempt_state(ctx.conn, attempt_id, "awaiting_user", result=result)
        repos.audit(ctx.conn, "daemon", "assisted.bundle_prepared", job_id,
                    {"attempt": attempt_id, "recovered": len(recovered)})
    return result


def _expected_outputs(task_type: str) -> list[str]:
    return ["summary.json"] if task_type in ("noop@1", "hash_inputs@1", "checkpoint_scan@1") else []


_RUN_CELL = """\
import glob
import json
import os
import shutil
import sys

# locate the bundle whether files sit in the working dir or an attached dataset
hits = sorted(glob.glob("task_manifest.json") + glob.glob("/kaggle/input/*/task_manifest.json"))
if not hits:
    raise SystemExit("task_manifest.json not found - upload the bundle files first")
base = os.path.dirname(os.path.abspath(hits[0]))
work = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
for name in ("task_impl.py", "task_manifest.json"):
    src = os.path.join(base, name)
    if os.path.abspath(src) != os.path.abspath(os.path.join(work, name)):
        shutil.copy(src, os.path.join(work, name))
inputs_src = os.path.join(base, "inputs")
if os.path.isdir(inputs_src):
    os.makedirs(os.path.join(work, "inputs"), exist_ok=True)
    for f in glob.glob(os.path.join(inputs_src, "*")):
        shutil.copy(f, os.path.join(work, "inputs"))
os.chdir(work)
sys.path.insert(0, work)

from task_impl import run

manifest = json.load(open("task_manifest.json"))
outputs = run(manifest, work)
print(json.dumps(outputs, indent=2))
print("result_manifest.json written to the working dir - download it with outputs/")
"""


def _make_notebook() -> dict:
    """A ready-to-run notebook: open it on Kaggle, attach secrets, Run All."""
    def code(source: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

    def md(source: str) -> dict:
        return {"cell_type": "markdown", "metadata": {}, "source": source}

    return {
        "cells": [
            md(
                "# DRM assisted run\n\n"
                "Generated by the Distributed Resource Manager. Cells copy the bundle "
                "into the writable working directory, verify input checksums, run the "
                "bounded task, and write `result_manifest.json` + `outputs/`.\n\n"
                "If the task checkpoints to Hugging Face: **Add-ons \u2192 Secrets \u2192 add "
                "`HF_TOKEN`** (a write-scoped token). The token is resolved inside this "
                "notebook only \u2014 DRM never ships credentials in the bundle."
            ),
            code(
                "try:\n"
                "    import huggingface_hub  # noqa: F401\n"
                "except ImportError:\n"
                "    try:\n"
                "        get_ipython().system('pip install -q huggingface_hub')\n"
                "    except Exception:\n"
                "        pass"
            ),
            code(_RUN_CELL),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _zip_bundle(bundle_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle_dir).as_posix())
    return zip_path


def import_results(ctx, job_row, attempt_row, output_dir: str) -> dict:
    """Validate the imported notebook output and publish exactly one accepted result."""
    job_row = dict(job_row)
    attempt_row = dict(attempt_row)
    job_id, attempt_id = job_row["id"], attempt_row["id"]
    imported = Path(output_dir)
    if not imported.is_dir():
        raise CatalogError("validation", f"import output directory not found: {imported}")
    if attempt_row["state"] == "complete":
        return {"reconciled": True, "already_complete": True, "job_id": job_id}

    manifest_path = imported / "result_manifest.json"
    if not manifest_path.is_file():
        _mark_reconciling(ctx, attempt_id, "no result_manifest.json in the imported output")
        raise CatalogError(
            "reconcile_required",
            "imported output has no result manifest; confirm the notebook finished, then re-import",
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        _mark_reconciling(ctx, attempt_id, "unparsable result manifest")
        raise CatalogError("reconcile_required", f"result manifest is not valid JSON: {exc}")

    declared = manifest.get("outputs", [])
    if not declared:
        _mark_reconciling(ctx, attempt_id, "result manifest declares no outputs")
        raise CatalogError("reconcile_required", "result manifest declares no outputs")

    def _resolve_output(name: str) -> Path | None:
        for candidate in (imported / name, imported / "outputs" / name):
            if candidate.is_file():
                return candidate
        return None

    validated = []
    for item in declared:
        name = item.get("name")
        path = _resolve_output(name)
        if path is None:
            _mark_reconciling(ctx, attempt_id, f"declared output {name!r} missing from the import")
            raise CatalogError(
                "reconcile_required",
                f"declared output {name!r} is not in the imported folder;"
                " re-download the notebook output and re-import",
            )
        data = path.read_bytes()
        if sha256_hex(data) != item.get("sha256"):
            _mark_reconciling(ctx, attempt_id, f"output {name!r} fails checksum")
            raise CatalogError(
                "reconcile_required",
                f"declared output {name!r} does not match its manifest checksum;"
                " re-download the notebook output and re-import",
            )
        validated.append({"name": name, "data": data, "sha256": item["sha256"], "byte_count": len(data)})

    declared_names = {item["name"] for item in declared}
    scanned = [p for p in imported.iterdir() if p.is_file()]
    outputs_subdir = imported / "outputs"
    if outputs_subdir.is_dir():
        scanned += [p for p in outputs_subdir.iterdir() if p.is_file()]
    undeclared = [
        p.name for p in scanned
        if p.name not in declared_names | {"result_manifest.json"}
    ]

    with ctx.db_lock:
        store_row = repos.get_provider(ctx.conn, job_row.get("spec", {}).get("output_store", ""))
    if store_row is None:
        raise CatalogError("not_found", "job output store is gone")
    storage = registry.storage_for_row(store_row)

    published = []
    with ctx.db_lock:
        for item in validated:
            published.append(
                runner.publish_output(
                    ctx.conn, storage, store_row["id"], job_id, item["name"], item["data"],
                    producer=attempt_id,
                )
            )
    result_manifest = {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "task_type": manifest.get("task_type"),
        "numerical_profile": manifest.get("numerical_profile"),
        "outputs": [
            {"name": v["name"], "sha256": v["sha256"], "byte_count": v["byte_count"]} for v in validated
        ],
        "imported_from": str(imported),
        "undeclared_files_ignored": undeclared,
    }
    with ctx.db_lock:
        repos.set_attempt_state(ctx.conn, attempt_id, "complete", result=result_manifest)
        repos.set_job_state(ctx.conn, job_id, "complete")
        if attempt_row.get("operation_id"):
            repos.set_operation_state(ctx.conn, attempt_row["operation_id"], "complete", result=result_manifest)
        repos.audit(ctx.conn, "drm", "assisted.import_accepted", job_id,
                    {"attempt": attempt_id, "outputs": [v["name"] for v in validated]})
    return {
        "job_id": job_id,
        "attempt_id": attempt_id,
        "published": published,
        "undeclared_files_ignored": undeclared,
        "reconciled": False,
    }


def _mark_reconciling(ctx, attempt_id: str, reason: str) -> None:
    with ctx.db_lock:
        repos.set_attempt_state(
            ctx.conn, attempt_id, "reconciling",
            error={"code": "reconcile_required", "message": reason},
        )


def record_session_lost(ctx, job_id: str, attempt_id: str, message: str = "session expired") -> None:
    """Park a failed assisted attempt in the retry path (spec Section 9)."""
    with ctx.db_lock:
        repos.set_attempt_state(
            ctx.conn, attempt_id, "failed",
            error={"code": "session_lost", "message": message},
        )
        repos.set_job_state(ctx.conn, job_id, "failed")


def load_task_impl(bundle_dir: Path):
    """Load the generated portable task implementation for out-of-process execution."""
    namespace: dict = {}
    exec(compile((bundle_dir / "task_impl.py").read_text(encoding="utf-8"), "task_impl.py", "exec"), namespace)
    return namespace


TASK_IMPL_SOURCE = '''\
"""Standalone bundle task implementation (generated by DRM; do not edit)."""
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _get_secret():
    """The user's own HF token, resolved inside the notebook — never shipped by DRM."""
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


def _push_checkpoint(path_in_repo, data, repo_id):
    token = _get_secret()
    if not token or not repo_id:
        return False
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=data,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="drm checkpoint",
        )
        return True
    except Exception as exc:  # checkpoints are best-effort; the summary is authoritative
        print(f"checkpoint push failed (continuing): {exc}", file=sys.stderr)
        return False


class SessionLost(RuntimeError):
    pass


def run(manifest, workdir, push_checkpoint=None, should_stop=None):
    workdir = Path(workdir)
    outputs_dir = workdir / "outputs"
    checkpoints_dir = workdir / "checkpoints"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    task_type = manifest["task_type"]
    params = manifest.get("params", {})
    outputs = []

    def _write(name, payload):
        data = json.dumps(payload, indent=2, sort_keys=True).encode()
        (outputs_dir / name).write_bytes(data)
        outputs.append({"name": name, "sha256": _sha(data), "byte_count": len(data)})

    if task_type == "noop@1":
        _write("result.json", {"task_type": task_type, "params": params})

    elif task_type == "hash_inputs@1":
        entries = []
        for spec in manifest.get("inputs", []):
            data = (workdir / "inputs" / spec["name"]).read_bytes()
            if _sha(data) != spec["sha256"]:
                raise RuntimeError(f"staged input {spec['name']} fails checksum verification")
            entries.append({"name": spec["name"], "sha256": _sha(data), "byte_count": len(data)})
        _write("summary.json", {"inputs": entries, "task_type": task_type})

    elif task_type == "checkpoint_scan@1":
        items = list(params.get("items", []))
        repo_id = (manifest.get("durable_store") or {}).get("repo_id")
        prefix = manifest.get("checkpoint_prefix", "checkpoints/")
        results = list(manifest.get("recovered", []))
        for index, item in enumerate(items):
            if should_stop is not None and should_stop():
                raise SessionLost(f"session lost after {index} of {len(items)} remaining items")
            digest = _sha(item.encode())
            record = {"item": item, "sha256": digest}
            data = json.dumps(record, indent=2, sort_keys=True).encode()
            (checkpoints_dir / f"item-{digest[:12]}.json").write_bytes(data)
            record["checkpoint_pushed"] = False
            if push_checkpoint is not None:
                record["checkpoint_pushed"] = push_checkpoint(prefix + f"item-{digest[:12]}.json", data)
            elif repo_id:
                record["checkpoint_pushed"] = _push_checkpoint(prefix + f"item-{digest[:12]}.json", data, repo_id)
            results.append(record)
        _write("summary.json", {"items": results, "task_type": task_type})

    else:
        raise RuntimeError(f"unsupported task type {task_type}")

    (workdir / "result_manifest.json").write_text(
        json.dumps(
            {
                "task_type": task_type,
                "outputs": outputs,
                "numerical_profile": manifest.get("numerical_profile"),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return outputs
'''
