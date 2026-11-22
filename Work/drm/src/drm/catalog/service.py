"""Catalog flows: ingest, verify, pin, evict (spec Sections 3 and 6).

Ingest verifies content identity *before* any catalog row exists — a failed
verification leaves only the store bytes to clean up, never a half-registered
artifact. Copies publish atomically through the storage adapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from drm.catalog.errors import CatalogError
from drm.providers.base import AdapterError, sha256_hex
from drm.store import repos
from drm.store.db import transaction
from drm.store.ids import new_id

CHUNK_BYTES = 1024 * 1024


def hash_file(path: Path) -> tuple[str, int]:
    """Chunked SHA-256 of a file; returns (hexdigest, byte_count)."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def normalize_ingest_payload(body: dict) -> dict:
    source = body.get("source")
    if not source:
        raise CatalogError("validation", "ingest requires 'source'")
    if not body.get("store"):
        raise CatalogError("validation", "ingest requires 'store'")
    return {
        "source": str(source),
        "store": body["store"],
        "artifact_id": body.get("artifact_id") or None,
        "kind": body.get("kind") or "dataset_release",
        "format": body.get("format") or None,
        "split_identity": body.get("split_identity") or None,
        "access_class": body.get("access_class") or "private",
        "verify": bool(body.get("verify", True)),
    }


def derive_ingest_key(store_row, payload: dict) -> str:
    """Derive a stable idempotency key when the caller did not supply one."""
    material = "|".join(
        ["ingest", str(Path(payload["source"]).resolve()), store_row["id"], payload.get("artifact_id") or ""]
    )
    return "ingest-" + hashlib.sha256(material.encode()).hexdigest()[:16]


def ingest_plan(conn, store_row, payload: dict) -> dict:
    """Dry-run: what ingest would do, without creating anything."""
    source = Path(payload["source"])
    if not source.is_file():
        raise CatalogError("validation", f"source is not a file: {source}")
    artifact_id = payload.get("artifact_id") or new_id("art")
    return {
        "source": str(source),
        "source_size": source.stat().st_size,
        "store": {"id": store_row["id"], "display_name": store_row["display_name"]},
        "artifact_id": artifact_id,
        "locator": f"{artifact_id}/{source.name}",
        "verify": payload["verify"],
        "blockers": [],
    }


def ingest_file(conn, storage, *, provider_id: str, source: str, artifact_id: str | None = None,
                kind: str = "dataset_release", format: str | None = None,
                split_identity: str | None = None, access_class: str = "private",
                verify: bool = True) -> dict:
    path = Path(source)
    if not path.is_file():
        raise CatalogError("validation", f"source is not a file: {path}")
    artifact_id = artifact_id or new_id("art")
    if repos.get_artifact(conn, artifact_id) is not None:
        raise CatalogError("conflict", f"artifact {artifact_id} already exists")
    content_hash, byte_count = hash_file(path)
    locator = f"{artifact_id}/{path.name}"
    try:
        written = storage.write(locator, path.read_bytes(), checksum=content_hash)
    except AdapterError as exc:
        raise CatalogError("adapter_error", f"staging into {provider_id} failed: {exc}")
    if verify:
        try:
            stored = storage.read(locator)
        except AdapterError as exc:
            raise CatalogError("verify_failed", f"stored copy cannot be read back: {exc}")
        if sha256_hex(stored) != content_hash:
            try:
                storage.delete(locator)
            except AdapterError:
                pass
            raise CatalogError("verify_failed", "stored copy fails checksum verification")
    with transaction(conn):
        artifact = repos.create_artifact(
            conn,
            artifact_id=artifact_id,
            kind=kind,
            content_hash=content_hash,
            byte_count=byte_count,
            format=format,
            split_identity=split_identity,
            access_class=access_class,
            producer="drm data ingest",
        )
        copy = repos.create_copy(
            conn,
            copy_id=new_id("copy"),
            artifact_id=artifact["id"],
            provider_id=provider_id,
            locator=locator,
            version=content_hash[:12],
            checksum=written["checksum"],
            byte_count=byte_count,
            status="durable",
        )
    return {"artifact": artifact_dict(artifact), "copy": dict(copy)}


def verify_copy(conn, storage, copy_id: str) -> dict:
    copy = repos.get_copy(conn, copy_id)
    if copy is None:
        raise CatalogError("not_found", f"unknown copy {copy_id}")
    try:
        data = storage.read(copy["locator"])
    except AdapterError as exc:
        raise CatalogError("verify_failed", f"copy {copy_id} cannot be read: {exc}")
    actual = sha256_hex(data)
    if actual != copy["checksum"]:
        raise CatalogError(
            "verify_failed",
            f"copy {copy_id} content hash {actual} does not match recorded {copy['checksum']}",
        )
    return {"copy_id": copy_id, "verified": True, "checksum": actual, "byte_count": len(data)}


def evict_plan(conn, copy) -> dict:
    """Dry-run: eviction gates evaluated without touching bytes or rows."""
    artifact = repos.get_artifact(conn, copy["artifact_id"])
    if artifact is None:
        raise CatalogError("not_found", f"artifact {copy['artifact_id']} row missing")
    blockers = []
    if copy["status"] == "durable":
        durable = [c for c in repos.list_copies(conn, artifact["id"]) if c["status"] == "durable"]
        if len(durable) <= 1:
            state = "pinned artifact" if artifact["pinned"] else "the artifact"
            blockers.append(f"last durable copy of {state}")
    return {
        "copy_id": copy["id"],
        "artifact_id": artifact["id"],
        "locator": copy["locator"],
        "blockers": blockers,
    }


def pin_artifact(conn, artifact_id: str) -> dict:
    artifact = repos.get_artifact(conn, artifact_id)
    if artifact is None:
        raise CatalogError("not_found", f"unknown artifact {artifact_id}")
    with transaction(conn):
        repos.set_artifact_pinned(conn, artifact_id, 1)
    return dict(repos.get_artifact(conn, artifact_id))


def evict_copy(conn, storage, copy_id: str) -> dict:
    """Evict one copy, gated on the last-durable-copy rule (spec Section 6.5).

    A pinned artifact must retain a verified durable copy, so the gate is:
    a durable copy may be evicted only while another durable copy remains.
    """
    copy = repos.get_copy(conn, copy_id)
    if copy is None:
        raise CatalogError("not_found", f"unknown copy {copy_id}")
    artifact = repos.get_artifact(conn, copy["artifact_id"])
    if artifact is None:
        raise CatalogError("not_found", f"artifact {copy['artifact_id']} row missing")
    blockers = []
    if copy["status"] == "durable":
        durable = [c for c in repos.list_copies(conn, artifact["id"]) if c["status"] == "durable"]
        if len(durable) <= 1:
            state = "pinned artifact" if artifact["pinned"] else "the artifact"
            blockers.append(f"last durable copy of {state}")
    if blockers:
        raise CatalogError("policy_blocked", "; ".join(blockers))
    bytes_absent = False
    try:
        storage.delete(copy["locator"])
    except AdapterError:
        bytes_absent = True  # record removal proceeds even if the bytes were already gone
    with transaction(conn):
        repos.delete_copy(conn, copy_id)
    return {"copy_id": copy_id, "evicted": True, "bytes_absent": bytes_absent}


def artifact_dict(row) -> dict:
    data = dict(row)
    data["pinned"] = bool(data.get("pinned"))
    data["parents"] = json.loads(data.pop("parents_json"))
    return data


def list_artifacts_with_copies(conn) -> list[dict]:
    results = []
    for row in repos.list_artifacts(conn):
        data = artifact_dict(row)
        data["copies"] = [dict(copy) for copy in repos.list_copies(conn, row["id"])]
        results.append(data)
    return results
