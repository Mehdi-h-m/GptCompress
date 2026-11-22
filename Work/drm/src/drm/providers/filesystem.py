"""Filesystem storage adapter: an approved local store (spec Section 5).

Writes are atomic — bytes land in a ``*.drm-partial`` staging file that is
``os.replace``d into place only after checksum verification — so a crash
never leaves a half-written object visible, and startup reconciliation can
safely sweep leftover staging files.
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

from drm.providers.base import AdapterError, StorageAdapter, sha256_hex
from drm.store.db import utcnow


class FilesystemStorage(StorageAdapter):
    adapter_name = "filesystem"

    PARTIAL_SUFFIX = ".drm-partial"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _path(self, locator: str) -> Path:
        candidate = (self.root / locator).resolve()
        if not candidate.is_relative_to(self.root):
            raise AdapterError(f"locator escapes store root: {locator}")
        return candidate

    def head(self, locator: str) -> dict:
        path = self._path(locator)
        exists = path.is_file()
        return {
            "locator": locator,
            "exists": exists,
            "byte_count": path.stat().st_size if exists else None,
            "checksum": None,  # the catalog owns recorded checksums for this adapter
        }

    def read(self, locator: str) -> bytes:
        path = self._path(locator)
        if not path.is_file():
            raise AdapterError(f"no object at {locator}")
        return path.read_bytes()

    def write(self, locator: str, data: bytes, checksum: str | None = None) -> dict:
        target = self._path(locator)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = sha256_hex(data)
        if checksum is not None and checksum != digest:
            raise AdapterError(
                f"checksum mismatch writing {locator}: expected {checksum}, got {digest}"
            )
        staging = target.parent / f"{target.name}.{uuid.uuid4().hex}{self.PARTIAL_SUFFIX}"
        try:
            with open(staging, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(staging, target)
        finally:
            if staging.exists():
                staging.unlink()
        return {"locator": locator, "checksum": digest, "byte_count": len(data)}

    def list(self, prefix: str = "") -> list[str]:
        if not self.root.is_dir():
            return []
        results = []
        for path in self.root.rglob("*"):
            if path.is_file():
                relative = path.relative_to(self.root).as_posix()
                if relative.startswith(prefix):
                    results.append(relative)
        return sorted(results)

    def delete(self, locator: str) -> dict:
        path = self._path(locator)
        if not path.is_file():
            raise AdapterError(f"no object at {locator}")
        path.unlink()
        return {"locator": locator, "deleted": True}

    def probe(self) -> dict:
        exists = self.root.is_dir()
        writable = False
        free_bytes = None
        if exists:
            try:
                probe_file = self.root / ".drm-probe"
                probe_file.write_bytes(b"")
                probe_file.unlink()
                writable = True
            except OSError:
                pass
            free_bytes = shutil.disk_usage(self.root).free
        return {
            "root": str(self.root),
            "exists": exists,
            "writable": writable,
            "free_bytes": free_bytes,
            "observed_at": utcnow(),
        }

    def cleanup_partials(self) -> int:
        """Remove leftover staging files from an interrupted process."""
        removed = 0
        if not self.root.is_dir():
            return removed
        for staging in self.root.rglob(f"*{self.PARTIAL_SUFFIX}*"):
            try:
                staging.unlink()
                removed += 1
            except OSError:
                pass
        return removed
