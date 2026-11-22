"""Versioned builtin task types for the local worker runtime (spec Section 3).

Workers are constrained to the app's registered task types; arbitrary code
execution is never accepted from a job manifest. Sprint 2 ships deterministic
builtin tasks so the execution machinery can be validated end to end; the
Qwen3 task pack arrives in Sprint 9.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class TaskCancelled(Exception):
    """The coordinator asked this job to cancel; unwind cooperatively."""


@dataclass
class TaskContext:
    """What a task implementation may touch: its workdir and parameters."""

    workdir: Path
    params: dict
    cancelled: Callable[[], bool]

    @property
    def inputs_dir(self) -> Path:
        return self.workdir / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.workdir / "outputs"


def _write_output(ctx: TaskContext, name: str, payload: dict) -> dict:
    ctx.outputs_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.outputs_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"name": name, "path": str(path)}


def run_noop(ctx: TaskContext) -> dict:
    """Minimal bounded task: writes one deterministic output document."""
    output = _write_output(ctx, "result.json", {"task_type": "noop@1", "params": ctx.params})
    return {"outputs": [output], "stats": {"kind": "noop"}}


def run_hash_inputs(ctx: TaskContext) -> dict:
    """Hash every staged input file; proves staging delivered verified bytes."""
    import hashlib

    entries = []
    for path in sorted(ctx.inputs_dir.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append({"name": path.name, "sha256": digest, "byte_count": path.stat().st_size})
    output = _write_output(ctx, "summary.json", {"inputs": entries, "task_type": "hash_inputs@1"})
    return {"outputs": [output], "stats": {"inputs_hashed": len(entries)}}


def run_sleep(ctx: TaskContext) -> dict:
    """Bounded wait that cooperates with cancellation every 50 ms."""
    seconds = float(ctx.params.get("seconds", 1))
    waited = 0.0
    while waited < seconds:
        if ctx.cancelled():
            raise TaskCancelled("cancel observed by sleep@1")
        time.sleep(min(0.05, seconds - waited))
        waited += 0.05
    output = _write_output(ctx, "result.json", {"task_type": "sleep@1", "seconds": seconds})
    return {"outputs": [output], "stats": {"slept_seconds": seconds}}


def run_checkpoint_scan(ctx: TaskContext) -> dict:
    """Scan items, writing one checkpoint file per item into workdir/checkpoints/.

    Used for the Sprint 5 assisted-recovery demonstration: the notebook bundle
    pushes these checkpoints to the durable store as it goes, so a lost session
    re-dispatches only the missing items.
    """
    items = list(ctx.params.get("items", []))
    checkpoints_dir = ctx.workdir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    import hashlib

    results = []
    for item in items:
        if ctx.cancelled():
            raise TaskCancelled("cancel observed by checkpoint_scan@1")
        digest = hashlib.sha256(item.encode()).hexdigest()
        record = {"item": item, "sha256": digest}
        data = json.dumps(record, indent=2, sort_keys=True).encode()
        (checkpoints_dir / f"item-{digest[:12]}.json").write_bytes(data)
        results.append(record)
    output = _write_output(ctx, "summary.json", {"items": results, "task_type": "checkpoint_scan@1"})
    return {"outputs": output and [output], "stats": {"items": len(results)}}


def run_fail(ctx: TaskContext) -> dict:
    """Always fails; exercises retry paths and attempt accounting in tests."""
    raise RuntimeError(ctx.params.get("message", "injected task failure"))


TASK_REGISTRY: dict[str, Callable[[TaskContext], dict]] = {
    "noop@1": run_noop,
    "hash_inputs@1": run_hash_inputs,
    "sleep@1": run_sleep,
    "checkpoint_scan@1": run_checkpoint_scan,
    "fail_always@1": run_fail,
}
