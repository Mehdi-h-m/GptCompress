"""Hugging Face Hub storage adapter (spec Section 5, 'Hugging Face Hub' profile).

Locator scheme: ``hf://datasets/<owner>/<repo>[@<revision>]/<path-in-repo>``.
Revisions containing ``/`` must be URL-encoded in the URI (e.g.
``refs%2Fconvert%2Fparquet``); the parser decodes them for the hub API.

- one private dataset repo per configured store instance;
- writes are content-checked before upload and versioned by commit hash;
- downloads are resumable (``hf_hub_download``); ranged reads arrive with
  planner staging in a later sprint;
- credentials are resolved through :mod:`drm.credentials` and never appear in
  manifests, logs or error records — hub errors are redacted before surfacing.

``make_api`` is a module-level factory so tests can inject a fake hub without
network access.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

from drm import credentials
from drm.providers.base import AdapterError, StorageAdapter, sha256_hex
from drm.store.db import utcnow

ADVERTISED_LIMITS = {
    "private_storage_free_gb": 100,
    "source": "https://huggingface.co/docs/hub/storage-limits",
    "documented_at": "2026-11-22",
}

_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{10,}")
_HF_URI = re.compile(
    r"^hf://datasets/(?P<owner>[^/@]+)/(?P<repo>[^/@]+)"
    r"(?:@(?P<revision>[^/]+))?(?:/(?P<path>.+))?$"
)


def _clean(message: str) -> str:
    """Redact anything that looks like a hub token from provider error text."""
    return _TOKEN_PATTERN.sub("hf_***", str(message))


def make_api(credential_ref: str | None = None):
    """Build an HfApi for a credential reference; token resolution happens here."""
    from huggingface_hub import HfApi

    return HfApi(token=credentials.resolve(credential_ref))


def parse_hf_uri(uri: str) -> dict:
    match = _HF_URI.match(uri.strip())
    if not match:
        raise AdapterError(
            f"unrecognized hf:// URI {uri!r}; expected hf://datasets/<owner>/<repo>[@<revision>]/<file>"
        )
    groups = match.groupdict()
    return {
        "repo_type": "dataset",
        "owner": groups["owner"],
        "repo": groups["repo"],
        "repo_id": f"{groups['owner']}/{groups['repo']}",
        "revision": unquote(groups["revision"]) if groups["revision"] else None,
        "path": unquote(groups["path"]) if groups["path"] else None,
    }


def resolve_public_download(uri: str, downloader=None) -> tuple[Path, dict]:
    """Fetch one pinned public file; returns (local_path, provenance).

    The download lands in the hub's own resumable cache; provenance records
    the origin URI and revision so the catalog can reproduce the fetch.
    """
    parsed = parse_hf_uri(uri)
    if not parsed["path"]:
        raise AdapterError(
            f"hf:// ingest needs a file path: hf://datasets/<owner>/<repo>@<revision>/<file> ({uri!r})"
        )
    if downloader is None:
        from huggingface_hub import hf_hub_download

        def downloader(**kwargs):
            return hf_hub_download(**kwargs)

    try:
        local = downloader(
            repo_id=parsed["repo_id"],
            filename=parsed["path"],
            repo_type="dataset",
            revision=parsed["revision"],
        )
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError(f"public fetch failed for {uri}: {_clean(exc)}")
    provenance = {
        "origin_uri": uri,
        "repo_id": parsed["repo_id"],
        "revision": parsed["revision"] or "unpinned",
        "path": parsed["path"],
        "producer": f"hf:{parsed['repo_id']}@{parsed['revision'] or 'unpinned'}",
    }
    return Path(local), provenance


class HfStorage(StorageAdapter):
    """Storage face of one HF dataset repo; the repo stays private."""

    adapter_name = "hf"

    def locator_for(self, artifact_id: str, filename: str) -> str:
        return f"hf://datasets/{self.repo_id}/{artifact_id}/{filename}"

    def __init__(self, repo_id: str, credential_ref: str | None = None, api=None) -> None:
        self.repo_id = repo_id
        self.credential_ref = credential_ref
        self._api = api

    @property
    def api(self):
        if self._api is None:
            self._api = make_api(self.credential_ref)
        return self._api

    def _parse(self, locator: str) -> dict:
        parsed = parse_hf_uri(locator)
        if parsed["repo_id"] != self.repo_id or not parsed["path"]:
            raise AdapterError(f"locator {locator!r} does not belong to store repo {self.repo_id}")
        return parsed

    def head(self, locator: str) -> dict:
        parsed = self._parse(locator)
        try:
            infos = self.api.get_paths_info(repo_id=self.repo_id, paths=[parsed["path"]], repo_type="dataset")
        except Exception as exc:
            raise AdapterError(f"head failed for {locator}: {_clean(exc)}")
        info = infos[0] if infos else None
        return {
            "locator": locator,
            "exists": info is not None,
            "byte_count": getattr(info, "size", None) if info is not None else None,
            "checksum": None,  # the catalog owns recorded checksums
        }

    def read(self, locator: str) -> bytes:
        parsed = self._parse(locator)
        try:
            local = self.api.hf_hub_download(
                repo_id=self.repo_id,
                filename=parsed["path"],
                repo_type="dataset",
                revision=parsed["revision"],
            )
            return Path(local).read_bytes()
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"read failed for {locator}: {_clean(exc)}")

    def write(self, locator: str, data: bytes, checksum: str | None = None) -> dict:
        parsed = self._parse(locator)
        digest = sha256_hex(data)
        if checksum is not None and checksum != digest:
            raise AdapterError(f"checksum mismatch writing {locator}: expected {checksum}, got {digest}")
        try:
            self.api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=True, exist_ok=True)
            commit = self.api.upload_file(
                path_or_fileobj=bytes(data),
                path_in_repo=parsed["path"],
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message="drm: store verified copy",
            )
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(f"write failed for {locator}: {_clean(exc)}")
        commit_hash = getattr(commit, "commit_hash", None) or digest
        return {
            "locator": locator,
            "checksum": digest,
            "byte_count": len(data),
            "version": str(commit_hash)[:12],
        }

    def list(self, prefix: str = "") -> list[str]:
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        except Exception as exc:
            raise AdapterError(f"list failed for repo {self.repo_id}: {_clean(exc)}")
        stem = f"hf://datasets/{self.repo_id}/"
        return sorted(stem + f for f in files if f.startswith(prefix))

    def delete(self, locator: str) -> dict:
        parsed = self._parse(locator)
        try:
            self.api.delete_file(path_in_repo=parsed["path"], repo_id=self.repo_id, repo_type="dataset")
        except Exception as exc:
            raise AdapterError(f"delete failed for {locator}: {_clean(exc)}")
        return {"locator": locator, "deleted": True}

    def probe(self) -> dict:
        observed = {
            "reachable": False,
            "authenticated": False,
            "user": None,
            "repo_exists": False,
            "repo_files": 0,
            "repo_bytes": 0,
        }
        try:
            info = self.api.whoami()
            observed["reachable"] = True
            observed["authenticated"] = True
            observed["writable"] = True  # an authenticated connection can write to its own repo
            observed["user"] = info.get("name") if isinstance(info, dict) else getattr(info, "name", None)
        except Exception:
            try:
                self.api.repo_info(repo_id=self.repo_id, repo_type="dataset")
                observed["reachable"] = True  # public reachability without credentials
            except Exception:
                return {
                    "adapter": self.adapter_name,
                    "provider": self.repo_id,
                    "advertised": dict(ADVERTISED_LIMITS),
                    "observed": observed,
                    "observed_at": utcnow(),
                }
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
            observed["repo_exists"] = True
            observed["repo_files"] = len(files)
            if files:
                infos = self.api.get_paths_info(repo_id=self.repo_id, paths=files[:100], repo_type="dataset")
                observed["repo_bytes"] = sum(getattr(i, "size", 0) or 0 for i in infos)
        except Exception:
            pass
        return {
            "adapter": self.adapter_name,
            "provider": self.repo_id,
            "advertised": dict(ADVERTISED_LIMITS),
            "writable": observed.get("writable", False),
            "observed": observed,
            "observed_at": utcnow(),
        }
