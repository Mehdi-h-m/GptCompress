"""Provider registry: adapter types vs configured instances (spec Section 5).

The adapter type (``local-runner``, ``filesystem``, ``mock``) is separate
from configured instances: two stores are two instances of the filesystem
adapter, independently selectable. Adding an instance is a configuration
action, not a code change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from drm.catalog.errors import CatalogError
from drm.providers.base import ComputeAdapter, StorageAdapter
from drm.providers.filesystem import FilesystemStorage
from drm.providers.local import LocalCompute
from drm.providers.mock import MockCompute, MockStorage
from drm.store import repos

ADAPTERS: dict[str, dict] = {
    "local-runner": {"kind": "compute", "execution": "api", "faces": ["compute"]},
    "filesystem": {"kind": "storage", "execution": "api", "faces": ["storage"], "required_config": ["root"]},
    "hf": {"kind": "storage", "execution": "api", "faces": ["storage"], "required_config": ["repo_id"]},
    "kaggle": {"kind": "compute", "execution": "assisted", "faces": ["compute"]},
    "mock": {"kind": "combined", "execution": "api", "faces": ["compute", "storage"]},
}


@dataclass
class AdapterBundle:
    compute: ComputeAdapter | None
    storage: StorageAdapter | None


def provider_dict(row) -> dict:
    data = dict(row)
    data["config"] = json.loads(data.pop("config_json"))
    return data


def validate_config(adapter: str | None, config: dict) -> None:
    spec = ADAPTERS.get(adapter or "")
    if spec is None:
        raise CatalogError(
            "validation", f"unknown adapter type {adapter!r}; known: {sorted(ADAPTERS)}"
        )
    for field in spec.get("required_config", []):
        if not config.get(field):
            raise CatalogError("validation", f"adapter {adapter!r} requires config {field!r}")


def connect(conn, payload: dict) -> tuple[dict, bool]:
    """Register a provider instance; returns (provider, created)."""
    adapter = payload.get("adapter")
    name = payload.get("name")
    if not name:
        raise CatalogError("validation", "connect requires 'name'")
    config = payload.get("config") or {}
    validate_config(adapter, config)
    existing = repos.find_provider_by_name(conn, adapter, name)
    if existing is not None:
        return provider_dict(existing), False
    spec = ADAPTERS[adapter]
    if adapter == "filesystem":
        try:
            Path(config["root"]).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CatalogError("validation", f"cannot create store root {config['root']!r}: {exc}")
    row = repos.create_provider(
        conn,
        adapter=adapter,
        display_name=name,
        kind=spec["kind"],
        execution=spec["execution"],
        config=config,
        credential_ref=payload.get("credential_ref"),
        region=payload.get("region"),
    )
    repos.audit(conn, "drm", "provider.connected", row["id"], {"adapter": adapter, "name": name})
    return provider_dict(row), True


def build_bundle(adapter: str, config: dict, credential_ref: str | None = None) -> AdapterBundle:
    validate_config(adapter, config)
    config = config or {}
    if adapter == "filesystem":
        return AdapterBundle(compute=None, storage=FilesystemStorage(config["root"]))
    if adapter == "hf":
        from drm.providers.hf import HfStorage

        return AdapterBundle(compute=None, storage=HfStorage(config["repo_id"], credential_ref=credential_ref))
    if adapter == "local-runner":
        return AdapterBundle(compute=LocalCompute(config), storage=None)
    if adapter == "kaggle":
        from drm.providers.kaggle import KaggleCompute

        return AdapterBundle(compute=KaggleCompute(config), storage=None)
    if adapter == "mock":
        name = config.get("name", "mock")
        return AdapterBundle(compute=MockCompute(name), storage=MockStorage(name))
    raise CatalogError("validation", f"unknown adapter type {adapter!r}")


def bundle_for_row(row) -> AdapterBundle:
    return build_bundle(row["adapter"], json.loads(row["config_json"]), credential_ref=row["credential_ref"])


def storage_for_row(row) -> StorageAdapter:
    bundle = bundle_for_row(row)
    if bundle.storage is None:
        raise CatalogError("validation", f"provider {row['id']} ({row['adapter']}) has no storage face")
    return bundle.storage


def compute_for_row(row) -> ComputeAdapter:
    bundle = bundle_for_row(row)
    if bundle.compute is None:
        raise CatalogError("validation", f"provider {row['id']} ({row['adapter']}) has no compute face")
    return bundle.compute


def probe(row) -> dict:
    """Probe every configured face; observations carry timestamps (spec Section 5)."""
    bundle = bundle_for_row(row)
    result = {"adapter": row["adapter"], "provider": row["id"], "faces": {}}
    if bundle.compute is not None:
        result["faces"]["compute"] = bundle.compute.probe()
    if bundle.storage is not None:
        result["faces"]["storage"] = bundle.storage.probe()
    return result
