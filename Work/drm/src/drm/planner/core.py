"""Planner v0: single-job placement with recorded decisions (spec Section 7).

A greedy planner with later correction is adequate for the first
implementation. Sprint 2 places one job on the local runner with a chosen
output store; every decision records its inputs and alternatives so
``drm explain`` can replay it.
"""

from __future__ import annotations

import json

from drm.catalog.errors import CatalogError
from drm.contracts import task_manifest
from drm.providers import registry
from drm.store import repos
from drm.store.ids import new_id
from drm.workers.tasks import TASK_REGISTRY


def build_spec(payload: dict) -> dict:
    """Normalize a submission payload into the job spec stored on the job row."""
    spec = {
        "id": payload.get("name") or new_id("task"),
        "task_type": payload.get("task_type"),
        "task_version": int(payload.get("task_version", 1)),
        "max_attempts": int(payload.get("max_attempts", 3)),
        "timeout_seconds": int(payload.get("timeout_seconds", 300)),
        "partition": payload.get("partition", "single"),
        "inputs": list(payload.get("inputs", [])),
        "params": dict(payload.get("params", {})),
        "numerical_profile": payload.get("numerical_profile"),
        "output_store": payload.get("output_store"),
        "compute_provider": payload.get("compute_provider"),
    }
    problems = task_manifest.validate(spec)
    if problems:
        raise CatalogError("validation", "; ".join(problems))
    if spec["task_type"] not in TASK_REGISTRY:
        raise CatalogError(
            "validation",
            f"unknown task type {spec['task_type']!r}; registered: {sorted(TASK_REGISTRY)}",
        )
    return spec


def _input_resolution(conn, spec: dict) -> list[dict]:
    """Resolve each input artifact to its preferred copy; missing pieces become blockers."""
    resolution = []
    for artifact_id in spec.get("inputs", []):
        artifact = repos.get_artifact(conn, artifact_id)
        if artifact is None:
            resolution.append({"artifact_id": artifact_id, "state": "missing", "copy": None})
            continue
        copies = repos.list_copies(conn, artifact_id)
        if not copies:
            resolution.append({"artifact_id": artifact_id, "state": "no_copies", "copy": None})
            continue
        durable = [c for c in copies if c["status"] == "durable"]
        chosen = (durable or copies)[0]
        resolution.append({"artifact_id": artifact_id, "state": "ok", "copy": dict(chosen)})
    return resolution


def _storage_candidates(conn) -> list[dict]:
    """Active storage providers passing a live probe, deterministically ordered."""
    candidates = []
    for row in repos.list_providers(conn):
        if row["kind"] not in ("storage", "combined") or row["state"] != "active":
            continue
        try:
            probe = registry.probe(row)
        except Exception:
            continue  # an unprovable provider is not a candidate this round
        storage_face = probe.get("faces", {}).get("storage", {})
        if not storage_face.get("writable", False):
            continue
        candidates.append(
            {
                "provider_id": row["id"],
                "display_name": row["display_name"],
                "adapter": row["adapter"],
                "cost_estimate": 0.0,
                "writable": True,
            }
        )
    candidates.sort(key=lambda c: c["provider_id"])  # deterministic tie-break
    return candidates


def _compute_candidates(conn) -> list[dict]:
    candidates = []
    for row in repos.list_providers(conn):
        if row["kind"] not in ("compute", "combined") or row["state"] != "active":
            continue
        candidates.append(
            {
                "provider_id": row["id"],
                "display_name": row["display_name"],
                "adapter": row["adapter"],
                "cost_estimate": 0.0,
            }
        )
    candidates.sort(key=lambda c: c["provider_id"])
    return candidates


def plan(conn, payload: dict) -> dict:
    """Build a placement plan; blockers are reported, chosen only when unblocked."""
    spec = build_spec(payload)
    resolution = _input_resolution(conn, spec)
    blockers = []
    for item in resolution:
        if item["state"] == "missing":
            blockers.append(f"input artifact {item['artifact_id']} is not in the catalog")
        elif item["state"] == "no_copies":
            blockers.append(f"input artifact {item['artifact_id']} has no physical copies")

    compute = _compute_candidates(conn)
    stores = _storage_candidates(conn)
    if not compute:
        blockers.append("no active compute provider is registered")
    if not stores:
        blockers.append("no writable storage provider is registered")
    if payload.get("output_store"):
        stores = [c for c in stores if c["provider_id"] == payload["output_store"]]
        if not stores:
            blockers.append(f"requested output store {payload['output_store']} is not active or not writable")
    if payload.get("compute_provider"):
        compute = [c for c in compute if c["provider_id"] == payload["compute_provider"]]
        if not compute:
            blockers.append(f"requested compute provider {payload['compute_provider']} is not active")

    chosen = None
    reason = None
    if not blockers:
        chosen = {
            "compute": compute[0]["provider_id"],
            "output_store": stores[0]["provider_id"],
        }
        reason = (
            "free local execution; lowest estimated cost (0 USD);"
            " deterministic tie-break by provider id"
        )
    alternatives = [
        {"face": "compute", **candidate, "selected": bool(chosen) and candidate["provider_id"] == chosen["compute"]}
        for candidate in compute
    ] + [
        {"face": "storage", **candidate, "selected": bool(chosen) and candidate["provider_id"] == chosen["output_store"]}
        for candidate in stores
    ]
    plan_doc = {
        "spec": spec,
        "inputs": resolution,
        "candidates": {"compute": compute, "output_store": stores},
        "alternatives": alternatives,
        "estimated_cost_range": {"min": 0.0, "max": 0.0, "currency": "USD", "basis": "free local providers"},
        "duration_estimate": {"seconds": None, "certainty": "unmeasured"},
        "blockers": blockers,
        "chosen": chosen,
        "reason": reason,
    }
    return plan_doc


def record_decision(conn, subject: str, subject_id: str, plan_doc: dict) -> dict:
    """Persist a plan decision so ``drm explain`` can replay it."""
    reason = plan_doc.get("reason")
    if not reason and plan_doc["blockers"]:
        reason = "blocked: " + "; ".join(plan_doc["blockers"])
    return repos.create_decision(
        conn,
        subject=subject,
        subject_id=subject_id,
        chosen=json.dumps(plan_doc.get("chosen"), sort_keys=True) if plan_doc.get("chosen") else "none",
        inputs={
            "spec": plan_doc["spec"],
            "inputs": plan_doc["inputs"],
            "estimated_cost_range": plan_doc["estimated_cost_range"],
            "duration_estimate": plan_doc["duration_estimate"],
        },
        alternatives=plan_doc["alternatives"],
        reason=reason,
    )
