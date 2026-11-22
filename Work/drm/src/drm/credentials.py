"""Credential resolution: the store holds references, never copies of secrets.

References come in two forms:

- ``env:NAME`` resolves an environment variable at use time (the default for
  ``hf`` connections, so nothing sensitive touches the disk);
- bare reference ids (``cred-…``) resolve through ``<state dir>/credentials.json``,
  written best-effort with owner-only permissions.

Secrets never appear in manifests, logs, audit records or API responses; the
daemon only ever passes the resolved token into the provider SDK.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from drm.config import state_dir


class CredentialsError(ValueError):
    """A credential reference could not be resolved."""


def _store_path() -> Path:
    return state_dir() / "credentials.json"


def _load_store() -> dict:
    path = _store_path()
    if not path.exists():
        return {"refs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_store(data: dict) -> None:
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)  # best effort; Windows ignores POSIX modes
    except OSError:
        pass


def store_secret(ref_id: str, secret: str) -> str:
    """Persist a secret under a reference id; returns the reference id."""
    data = _load_store()
    data.setdefault("refs", {})[ref_id] = {"kind": "secret", "secret": secret}
    _write_store(data)
    return ref_id


def resolve(credential_ref: str | None) -> str | None:
    """Resolve a credential reference to a secret, or None for unauthenticated use."""
    if not credential_ref:
        return None
    if credential_ref.startswith("env:"):
        return os.environ.get(credential_ref[4:])
    entry = _load_store().get("refs", {}).get(credential_ref)
    if entry is None:
        raise CredentialsError(f"unknown credential reference {credential_ref!r}")
    return entry.get("secret")
